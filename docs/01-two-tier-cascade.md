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
| Object Detection (in-ride) | YOLOv8n quantized INT8, chỉ giữ class `phone` | phone (kèm vị trí tay/mặt) | ~15–30ms NPU, chạy **mọi frame** — phone là sự kiện khẩn cấp <300ms nên không được hạ FPS |
| Object Detection (pre-ride) | Cùng model YOLOv8n, bật đủ class | helmet_strap, mask, sunglasses | Chỉ chạy lúc xe chưa lăn bánh, không có ràng buộc latency |

**Các chỉ số cụ thể:**

- **EAR (Eye Aspect Ratio)** = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). Mắt mở ~0.25–0.3; nhắm < 0.2. Dùng để đo thời lượng nhắm mắt liên tục.
- **PERCLOS** (Percentage of Eye Closure) = % thời gian EAR < ngưỡng trong cửa sổ trượt 60 giây. Chỉ số chuẩn công nghiệp đo mức buồn ngủ tích lũy — nhạy hơn việc chỉ bắt một lần nhắm mắt.
- **MAR (Mouth Aspect Ratio)**: MAR > 0.6 kéo dài > 2s = một lần ngáp. Đếm số lần ngáp/10 phút.
- **Head Pose Euler**: Pitch < −25° kéo dài > 1.5s = cúi nhìn xuống (điện thoại/ngủ gật). |Yaw| > 45° lặp > 3 lần/30s = quay đầu về sau liên tục.
- **Blink rate**: số lần chớp mắt/phút — vừa là tín hiệu mệt mỏi (chớp chậm dần), vừa là feature ghi vào health baseline (Khối 3).

### 2.2 Temporal State Machine (chống nhiễu)

Chỉ số 1 frame không đáng tin (chớp mắt bình thường cũng làm EAR tụt). Tier 1 duy trì state machine theo thời gian:

- **Hysteresis + debounce**: sự kiện chỉ được xác nhận khi điều kiện giữ liên tục N frame (ví dụ EAR < 0.2 liên tục ≥ 1.5s × 10 FPS = 15 frame). Phone: confidence > 0.5 và xuất hiện ≥ 3 frame liên tiếp mới xác nhận; mất detection 10 frame liên tiếp mới coi là kết thúc sự kiện (tránh nhấp nháy bật/tắt cảnh báo).
- **Cửa sổ trượt** cho các chỉ số tích lũy: PERCLOS 60s, yawn count 10 phút, head-turn count 30s.
- **Cooldown**: sau mỗi lần trigger Tier 2, khóa trigger cùng loại trong 3–5 phút, tránh spam VLM và spam người lái.

### 2.3 Kiểm tra trước khi chạy (Pre-ride Check)

Chạy 1 lần khi bắt đầu hành trình (xe chưa lăn bánh, không có ràng buộc latency):

1. Object detection: quai mũ bảo hiểm chưa cài (luôn kiểm tra, confidence > 0.6, lấy đa số phiếu trên 10 frame liên tiếp để tránh kết luận từ 1 frame mờ); khẩu trang và kính chỉ nhắc khi telematics báo trời nắng/bụi (chỉ số bụi AQI cao hoặc trời khô nắng) — trời mưa/râm thì không nhắc.
2. Đây là thời điểm rẻ để gọi VLM 1 lần: chụp ảnh mặt, trích feature sức khỏe cho baseline (Khối 3), sinh lời chào + nhắc nhở đầu chuyến.

## 3. Bảng điều kiện Trigger — ai xử lý, xử lý thế nào

| # | Điều kiện phát hiện (Tier 1) | Loại | Hành động |
|---|---|---|---|
| T0 | EAR < 0.2 liên tục > 1.5s | **KHẨN CẤP** | Tier 1 phát cảnh báo âm thanh ngay (<300ms). KHÔNG gọi VLM trong critical path. VLM có thể được gọi *sau đó* để phân tích nguyên nhân. |
| T1 | Pitch < −25° > 1.5s **hoặc** phone detected | **KHẨN CẤP** | Như T0: cảnh báo rule-based ngay. |
| T2 | PERCLOS > 25% trong 60s | Cảnh báo nhẹ + Trigger VLM | Tier 1 phát **ngay** câu TTS duyệt sẵn ("Bạn có vẻ buồn ngủ, chú ý tập trung nhé"); VLM chạy sau để sinh lời khuyên ngữ cảnh đầy đủ hơn. |
| T3 | ≥ 3 lần ngáp / 10 phút | Cảnh báo nhẹ + Trigger VLM | Như T2 — nhắc nhẹ tức thời trước, VLM bổ sung sau. |
| T4 | \|Yaw\| > 45° lặp > 3 lần / 30s | Cảnh báo nhẹ + Trigger VLM | Tier 1 nhắc "chú ý quan sát phía trước" ngay; VLM đánh giá ngữ cảnh sau (tìm đồ? chở trẻ em? có người phía sau?). |
| T5 | `continuous_driving_min` > 60 (telematics, không cần CV) | Trigger VLM | Nhắc nghỉ theo luật + đánh giá vẻ mệt mỏi hiện tại. Không khẩn cấp nên không cần cảnh báo tức thời. |
| T6 | Định kỳ mỗi 5 phút (background, ưu tiên thấp) | Scheduled (CV, không phải VLM) | Trích feature sức khỏe bằng landmark + thống kê màu của Tier 1 (xem Khối 3 — **không cần VLM để trích feature**), ghi vào baseline. Chỉ chạy khi nhiệt độ chip cho phép, có thể skip. |
| T7 | Anomaly baseline (Khối 3): ≥ 2 feature cùng hướng xấu với \|z\| > 2, hoặc 1 feature \|z\| > 3 lặp lại ≥ 2 phiên liên tiếp | Trigger VLM | Sinh lời nhắc quan tâm tinh tế — bắt buộc qua Guardrails (Khối 2). Ngưỡng định nghĩa một chỗ duy nhất trong config, hai tài liệu 01/03 cùng tham chiếu. |

**Nguyên tắc bổ sung cho T2–T4**: sự kiện đã qua debounce là sự kiện an toàn có thật — người lái phải được nhắc **ngay** bằng câu TTS tĩnh duyệt sẵn (không rủi ro guardrail vì không phải text sinh bởi model), VLM chỉ đóng vai trò *nâng cấp chất lượng lời nhắc* sau đó vài giây, không phải điều kiện để được nhắc.

**Tiêu chí chung để một sự kiện "đắt giá" đáng gọi Tier 2:** (1) không khẩn cấp tức thời — khẩn cấp thì rule Tier 1 tự xử; (2) cần hiểu ngữ cảnh mà CV thuần không phân biệt được (mệt thật vs đeo kính râm, cúi tìm đồ vs ngủ gật); (3) đã qua debounce + cooldown.

## 4. Tier 2 — Edge VLM (Event-Driven)

- **Model đề xuất**: Qwen2-VL-2B-Instruct hoặc SmolVLM-2B, quantized Q4 (GGUF chạy llama.cpp / ONNX Runtime tùy chip). Footprint ~1.5–2GB RAM, inference 1–3s/ảnh trên NPU edge.
- **Input**: 1 frame hiện tại (đã crop vùng mặt) + trigger_reason + telematics + delta so với baseline (dạng text, không gửi ảnh lịch sử).
- **Output**: 1 câu nhắc nhở tiếng Việt, bắt buộc đi qua `enforce_medical_guardrails` (Khối 2) trước khi phát ra loa/màn hình.
- **Quản lý tài nguyên**: hàng đợi 1 slot (trigger mới đè trigger cũ chưa chạy); theo dõi nhiệt độ chip, quá ngưỡng thì bỏ qua T6 định kỳ, chỉ giữ trigger sự kiện.

## 5. Ngân sách độ trễ (Latency Budget)

SLA 300ms đo từ **frame chứa bằng chứng cuối cùng hoàn tất điều kiện** (ví dụ frame thứ 15 của chuỗi nhắm mắt 1.5s) đến lúc loa phát — không tính thời gian debounce, vì debounce là một phần của định nghĩa sự kiện.

| Đường xử lý | Ngân sách | Phân bổ thực tế @ 10 FPS |
|---|---|---|
| Tier 1 mỗi frame | < 50ms | Capture + landmark ~10ms, head pose ~1ms, YOLO phone (NPU, mọi frame) ~15–30ms — chạy song song với landmark trên 2 đơn vị tính toán |
| Cảnh báo khẩn cấp (T0/T1) | < 300ms | Frame cuối của chuỗi debounce ~50ms xử lý + phát buzzer/TTS pre-recorded ~50ms. Tổng ~100ms, còn dư 200ms dự phòng cho queue/jitter |
| Cảnh báo nhẹ tức thời (T2–T4) | < 500ms | TTS tĩnh duyệt sẵn, phát ngay khi debounce xác nhận |
| Phản hồi VLM (T2–T7) | 1–5s (chấp nhận được) | Không nằm trên safety path; người lái đã được cảnh báo tức thời trước đó |

## 6. Telematics–VLM Fusion (kết hợp ngữ cảnh phương tiện)

Telematics tham gia ở **cả 3 điểm** của pipeline:

**(a) Điều kiện trigger** — telematics tự tạo trigger (T5) và điều biến ngưỡng: lái đêm hoặc `continuous_driving_min` > 90 thì hạ ngưỡng PERCLOS từ 25% xuống 20% (mệt tích lũy làm rủi ro tăng, cần nhạy hơn).

**(b) Input contract của VLM** — Tier 2 nhận khối telematics chuẩn hóa, VLM dùng để giải thích *vì sao* nên nghỉ (lời nhắc có căn cứ thuyết phục hơn):

```json
{
  "vehicle_state": "moving" | "stopped",   // suy từ speed_kmh
  "speed_kmh": 52,
  "continuous_driving_min": 125,
  "ambient_temp_c": 35,
  "weather": "sunny_dusty" | "rain" | "normal",
  "time_of_day": "afternoon"
}
```

**(c) Chính sách phát cảnh báo theo trạng thái xe** — cùng một phát hiện, hành vi khác nhau:

| Trạng thái | Kênh phát | Nội dung |
|---|---|---|
| `moving`, tốc độ cao | Âm thanh ngắn gọn, không màn hình | 1 câu tối đa, không bắt người lái đọc chữ ("Bạn đã lái hơn 2 tiếng dưới trời nắng nóng, phía trước có chỗ nghỉ thì tấp vào chút nhé") |
| `moving`, tốc độ thấp/kẹt xe | Âm thanh, câu đầy đủ hơn | Có thể kèm gợi ý cụ thể (uống nước, mở thoáng khí) |
| `stopped`/đỗ | Âm thanh + màn hình | Lời nhắc đầy đủ, có thể kèm tóm tắt chuyến ("Hôm nay bạn ngáp nhiều hơn thường ngày, nghỉ thêm chút rồi hãy đi tiếp") |

Ví dụ fusion đầy đủ: `continuous_driving_min=125` + `ambient_temp_c=35` + `weather=sunny_dusty` + VLM thấy mặt bóng dầu/mệt → lời nhắc dẫn cả 3 căn cứ ("lái hơn 2 tiếng", "trời nắng 35 độ", "trông bạn thấm mệt") thay vì câu chung chung — tính thuyết phục đến từ việc lời nhắc *khớp trải nghiệm thật* của người lái ngay lúc đó.
