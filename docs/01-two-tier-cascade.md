# Khối 1 — Kiến Trúc Phân Tầng Lọc Sự Kiện (Two-tier Event-Driven Cascade)

## 1. Nguyên tắc thiết kế

Hai ràng buộc mâu thuẫn nhau:

- **DMS phải phản hồi < 300ms** (phát hiện ngủ gật, dùng điện thoại) — bắt buộc chạy trên mọi frame ở 5–10 FPS.
- **VLM không thể chạy 10 FPS** — một lần inference VLM quantized trên chip edge mất 1–4 giây, chạy liên tục sẽ quá nhiệt và tụt FPS toàn hệ thống.

Cách giải: **tách hẳn đường an toàn (safety path) khỏi đường ngữ cảnh (context path)**.

> **Nguyên tắc số 1: Mọi cảnh báo an toàn khẩn cấp do Tier 1 quyết định trực tiếp bằng rule, KHÔNG BAO GIỜ chờ VLM.** VLM chỉ làm nhiệm vụ phân tích ngữ cảnh sâu, diễn đạt lời nhắc, và đánh giá xu hướng sức khỏe — những việc chấp nhận độ trễ vài giây.

```
Camera 5-10 FPS
      │
      ▼
┌─────────────────────────────────────────────┐
│ TIER 1 — Lightweight CV (mọi frame, <20ms)  │
│  • Face Landmark → EAR, MAR, PERCLOS        │
│  • Head Pose (Euler: pitch/yaw/roll)        │
│  • Object Detection (phone, helmet, mask)   │
│  • Temporal State Machine (đếm thời gian)   │
└──────┬──────────────────────┬───────────────┘
       │                      │
  Rule khẩn cấp          Trigger event
  (< 300ms)              (debounced)
       │                      │
       ▼                      ▼
  🔊 CẢNH BÁO         ┌──────────────────────────────┐
  NGAY LẬP TỨC        │ TIER 2 — Edge VLM (event-    │
  (buzzer/TTS)        │ driven hoặc định kỳ 5 phút)  │
                      │  • Phân tích ngữ cảnh sâu    │
                      │  • So sánh health baseline   │
                      │  • Sinh lời nhắc tinh tế     │
                      │  → qua Guardrails (Khối 2)   │
                      └──────────────────────────────┘
```

## 2. Tier 1 — Lightweight CV (Real-time, mọi frame)

### 2.1 Mô hình và chỉ số

| Thành phần | Mô hình đề xuất | Chỉ số tính ra | Chi phí/frame |
|---|---|---|---|
| Face Landmark | MediaPipe Face Mesh (468 điểm) hoặc PFLD | EAR, MAR, vị trí mắt/môi | ~5–10ms CPU |
| Head Pose | solvePnP trên 6 landmark chuẩn → Euler angles | Pitch / Yaw / Roll | ~1ms (tính hình học, không cần model riêng) |
| Object Detection | YOLOv8n / YOLO-NAS-S quantized INT8 | phone, helmet_strap, mask, sunglasses | ~15–30ms NPU; chạy 1–2 FPS là đủ (không cần mọi frame) |

**Các chỉ số cụ thể:**

- **EAR (Eye Aspect Ratio)** = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). Mắt mở ~0.25–0.3; nhắm < 0.2. Dùng để đo thời lượng nhắm mắt liên tục.
- **PERCLOS** (Percentage of Eye Closure) = % thời gian EAR < ngưỡng trong cửa sổ trượt 60 giây. Chỉ số chuẩn công nghiệp đo mức buồn ngủ tích lũy — nhạy hơn việc chỉ bắt một lần nhắm mắt.
- **MAR (Mouth Aspect Ratio)**: MAR > 0.6 kéo dài > 2s = một lần ngáp. Đếm số lần ngáp/10 phút.
- **Head Pose Euler**: Pitch < −25° kéo dài > 1.5s = cúi nhìn xuống (điện thoại/ngủ gật). |Yaw| > 45° lặp > 3 lần/30s = quay đầu về sau liên tục.
- **Blink rate**: số lần chớp mắt/phút — vừa là tín hiệu mệt mỏi (chớp chậm dần), vừa là feature ghi vào health baseline (Khối 3).

### 2.2 Temporal State Machine (chống nhiễu)

Chỉ số 1 frame không đáng tin (chớp mắt bình thường cũng làm EAR tụt). Tier 1 duy trì state machine theo thời gian:

- **Hysteresis + debounce**: sự kiện chỉ được xác nhận khi điều kiện giữ liên tục N frame (ví dụ EAR < 0.2 liên tục ≥ 1.5s × 10 FPS = 15 frame).
- **Cửa sổ trượt** cho các chỉ số tích lũy: PERCLOS 60s, yawn count 10 phút, head-turn count 30s.
- **Cooldown**: sau mỗi lần trigger Tier 2, khóa trigger cùng loại trong 3–5 phút, tránh spam VLM và spam người lái.

### 2.3 Kiểm tra trước khi chạy (Pre-ride Check)

Chạy 1 lần khi bắt đầu hành trình (xe chưa lăn bánh, không có ràng buộc latency):

1. Object detection: quai mũ bảo hiểm chưa cài, chưa đeo khẩu trang/kính (kết hợp telematics thời tiết: chỉ nhắc kính khi trời nắng bụi).
2. Đây là thời điểm rẻ để gọi VLM 1 lần: chụp ảnh mặt, trích feature sức khỏe cho baseline (Khối 3), sinh lời chào + nhắc nhở đầu chuyến.

## 3. Bảng điều kiện Trigger — ai xử lý, xử lý thế nào

| # | Điều kiện phát hiện (Tier 1) | Loại | Hành động |
|---|---|---|---|
| T0 | EAR < 0.2 liên tục > 1.5s | **KHẨN CẤP** | Tier 1 phát cảnh báo âm thanh ngay (<300ms). KHÔNG gọi VLM trong critical path. VLM có thể được gọi *sau đó* để phân tích nguyên nhân. |
| T1 | Pitch < −25° > 1.5s **hoặc** phone detected | **KHẨN CẤP** | Như T0: cảnh báo rule-based ngay. |
| T2 | PERCLOS > 25% trong 60s | Trigger VLM | Buồn ngủ tích lũy — VLM phân tích khuôn mặt + telematics, sinh lời khuyên nghỉ ngơi. |
| T3 | ≥ 3 lần ngáp / 10 phút | Trigger VLM | Như T2. |
| T4 | |Yaw| > 45° lặp > 3 lần / 30s | Trigger VLM | Mất tập trung bất thường — VLM đánh giá ngữ cảnh (tìm đồ? chở trẻ em? có người phía sau?). |
| T5 | `continuous_driving_min` > 60 (telematics, không cần CV) | Trigger VLM | Nhắc nghỉ theo luật + đánh giá vẻ mệt mỏi hiện tại. |
| T6 | Định kỳ mỗi 5 phút (background, ưu tiên thấp) | Scheduled VLM | Trích feature sức khỏe cho baseline (Khối 3); chỉ chạy khi nhiệt độ chip cho phép, có thể skip. |
| T7 | Feature sức khỏe lệch baseline: z-score > 2 (Khối 3) | Trigger VLM | Sinh lời nhắc quan tâm tinh tế — bắt buộc qua Guardrails (Khối 2). |

**Tiêu chí chung để một sự kiện "đắt giá" đáng gọi Tier 2:** (1) không khẩn cấp tức thời — khẩn cấp thì rule Tier 1 tự xử; (2) cần hiểu ngữ cảnh mà CV thuần không phân biệt được (mệt thật vs đeo kính râm, cúi tìm đồ vs ngủ gật); (3) đã qua debounce + cooldown.

## 4. Tier 2 — Edge VLM (Event-Driven)

- **Model đề xuất**: Qwen2-VL-2B-Instruct hoặc SmolVLM-2B, quantized Q4 (GGUF chạy llama.cpp / ONNX Runtime tùy chip). Footprint ~1.5–2GB RAM, inference 1–3s/ảnh trên NPU edge.
- **Input**: 1 frame hiện tại (đã crop vùng mặt) + trigger_reason + telematics + delta so với baseline (dạng text, không gửi ảnh lịch sử).
- **Output**: 1 câu nhắc nhở tiếng Việt, bắt buộc đi qua `enforce_medical_guardrails` (Khối 2) trước khi phát ra loa/màn hình.
- **Quản lý tài nguyên**: hàng đợi 1 slot (trigger mới đè trigger cũ chưa chạy); theo dõi nhiệt độ chip, quá ngưỡng thì bỏ qua T6 định kỳ, chỉ giữ trigger sự kiện.

## 5. Ngân sách độ trễ (Latency Budget)

| Đường xử lý | Ngân sách | Đạt bằng cách |
|---|---|---|
| Tier 1 mỗi frame | < 20ms | Landmark CPU + head pose hình học; YOLO hạ xuống 1–2 FPS |
| Cảnh báo khẩn cấp (T0/T1) | < 300ms | Rule thuần trong Tier 1, phát buzzer/TTS pre-recorded, không đụng VLM |
| Phản hồi VLM (T2–T7) | 1–5s (chấp nhận được) | Không nằm trên safety path; người lái đã được Tier 1 bảo vệ |
