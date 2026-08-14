# Khối 1 — Kiến trúc phân tầng lọc sự kiện (Two-tier Event-Driven Cascade)

## 1. Nguyên tắc thiết kế

Bài toán có hai ràng buộc kéo về hai phía ngược nhau. DMS phải phản hồi dưới 300ms khi phát hiện ngủ gật hay dùng điện thoại, nghĩa là phải theo dõi mọi frame ở 5–10 FPS. Trong khi đó VLM không có cách nào chạy nổi tần suất ấy: một lần inference model quantized trên chip edge mất 1–4 giây, ép chạy liên tục thì chip quá nhiệt và cả hệ thống tụt FPS.

Cách giải là tách hẳn đường an toàn (safety path) khỏi đường ngữ cảnh (context path). Mọi cảnh báo an toàn khẩn cấp do Tier 1 quyết định trực tiếp bằng rule, không bao giờ chờ VLM. VLM chỉ lo ba việc chấp nhận được độ trễ vài giây: phân tích ngữ cảnh sâu, diễn đạt lời nhắc, và đánh giá xu hướng sức khỏe. Nguyên tắc này lặp lại ở mọi quyết định thiết kế phía dưới.

```
Camera 5-10 FPS
      │
      ▼
┌─────────────────────────────────────────────┐
│ TIER 1 — Lightweight CV (mọi frame, <50ms)  │
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

## 2. Tier 1 — Lightweight CV (real-time, mọi frame)

### 2.1 Mô hình và chỉ số

| Thành phần | Mô hình | Chỉ số tính ra | Chi phí/frame |
|---|---|---|---|
| Face Landmark | MediaPipe FaceLandmarker (478 điểm, đã chạy thật — xem docs/04) | EAR, MAR, vị trí mắt/môi | ~5–10ms CPU |
| Head Pose | solvePnP trên 6 landmark chuẩn → Euler angles | Pitch / Yaw / Roll | ~1ms (tính hình học, không cần model riêng) |
| Object Detection (in-ride) | YOLO26n quantized INT8, chỉ giữ class `phone` | phone (kèm vị trí tay/mặt) | ~15–30ms NPU, chạy mọi frame — phone là sự kiện khẩn cấp <300ms nên không được hạ FPS |
| Object Detection (pre-ride) | Cùng model YOLO26n, bật đủ class | helmet_strap, mask, sunglasses | Chỉ chạy lúc xe chưa lăn bánh, không có ràng buộc latency |

Chọn YOLO26n thay các bản YOLO cũ hơn vì kiến trúc NMS-free/DFL-free cho latency tất định và không rớt accuracy khi quantize INT8 — hai tính chất đáng tiền trên chip edge.

Các chỉ số cụ thể:

- EAR (Eye Aspect Ratio) = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). Mắt mở thường đo được 0.25–0.3, nhắm thì tụt dưới 0.2. Dùng để đo thời lượng nhắm mắt liên tục.
- PERCLOS (Percentage of Eye Closure): % thời gian EAR dưới ngưỡng trong cửa sổ trượt 60 giây. Đây là chỉ số chuẩn công nghiệp đo buồn ngủ tích lũy, nhạy hơn nhiều so với chỉ bắt một lần nhắm mắt.
- MAR (Mouth Aspect Ratio): MAR cao kéo dài quá 2 giây tính là một lần ngáp, đếm số lần trong 10 phút. Ngưỡng lý thuyết hay gặp là 0.6, nhưng đo thật trên FL3D với bộ landmark MediaPipe thì ngáp thật chỉ đạt MAR ~0.5 nên hạ về 0.35 (chi tiết docs/04).
- Head Pose Euler: pitch dưới −25° kéo dài hơn 1.5 giây là cúi nhìn xuống (điện thoại hoặc gục đầu); |yaw| quá 45° lặp hơn 3 lần trong 30 giây là quay đầu về sau liên tục.
- Blink rate: số lần chớp mắt mỗi phút. Vừa là tín hiệu mệt mỏi (chớp chậm dần), vừa là feature ghi vào health baseline (Khối 3).

### 2.2 Temporal state machine (chống nhiễu)

Chỉ số của một frame đơn lẻ không đáng tin — chớp mắt bình thường cũng làm EAR tụt. Vì vậy Tier 1 duy trì state machine theo thời gian, gồm ba cơ chế:

Thứ nhất là hysteresis và debounce: sự kiện chỉ được xác nhận khi điều kiện giữ liên tục đủ lâu, ví dụ EAR dưới 0.2 liên tục 1.5 giây (15 frame ở 10 FPS) mới tính là nhắm mắt. Phone cũng vậy: confidence trên 0.5 phải xuất hiện ít nhất 3 frame liên tiếp mới xác nhận, và phải mất detection 10 frame liên tiếp mới coi là kết thúc — tránh cảnh báo nhấp nháy bật tắt.

Thứ hai là cửa sổ trượt cho các chỉ số tích lũy: PERCLOS 60 giây, đếm ngáp 10 phút, đếm quay đầu 30 giây.

Thứ ba là cooldown: sau mỗi lần trigger Tier 2, khóa trigger cùng loại trong 3–5 phút để không spam VLM lẫn người lái.

### 2.3 Kiểm tra trước khi chạy (pre-ride check)

Chạy một lần khi bắt đầu hành trình, lúc xe chưa lăn bánh nên không có ràng buộc latency. Object detection kiểm tra quai mũ bảo hiểm (luôn kiểm, confidence trên 0.6, lấy đa số phiếu trên 10 frame liên tiếp để không kết luận từ một frame mờ); khẩu trang và kính chỉ nhắc khi telematics báo trời nắng bụi — trời mưa hay râm thì thôi. Đây cũng là thời điểm rẻ để gọi VLM một lần: chụp ảnh mặt, trích feature sức khỏe cho baseline (Khối 3), sinh lời chào đầu chuyến.

## 3. Bảng điều kiện trigger — ai xử lý, xử lý thế nào

| # | Điều kiện phát hiện (Tier 1) | Loại | Hành động |
|---|---|---|---|
| T0 | EAR < 0.2 liên tục > 1.5s | KHẨN CẤP | Tier 1 phát cảnh báo âm thanh ngay (<300ms). Không gọi VLM trong critical path; VLM có thể được gọi sau đó để phân tích nguyên nhân. |
| T1 | Pitch < −25° > 1.5s hoặc phone detected | KHẨN CẤP | Như T0: cảnh báo rule-based ngay. |
| T2 | PERCLOS > 25% trong 60s | Cảnh báo nhẹ + trigger VLM | Tier 1 phát ngay câu TTS duyệt sẵn ("Bạn có vẻ buồn ngủ, chú ý tập trung nhé"); VLM chạy sau để sinh lời khuyên ngữ cảnh đầy đủ hơn. |
| T3 | ≥ 3 lần ngáp / 10 phút | Cảnh báo nhẹ + trigger VLM | Như T2 — nhắc nhẹ tức thời trước, VLM bổ sung sau. |
| T4 | \|Yaw\| > 45° lặp > 3 lần / 30s | Cảnh báo nhẹ + trigger VLM | Tier 1 nhắc "chú ý quan sát phía trước" ngay; VLM đánh giá ngữ cảnh sau (tìm đồ? chở trẻ em? có người phía sau?). |
| T5 | `continuous_driving_min` > 60 (telematics, không cần CV) | Trigger VLM | Nhắc nghỉ theo luật + đánh giá vẻ mệt mỏi hiện tại. Không khẩn cấp nên không cần cảnh báo tức thời. |
| T6 | Định kỳ mỗi 5 phút (background, ưu tiên thấp) | Scheduled (CV, không phải VLM) | Trích feature sức khỏe bằng landmark + thống kê màu của Tier 1 (Khối 3 — không cần VLM để trích feature), ghi vào baseline. Chỉ chạy khi nhiệt độ chip cho phép, skip được. |
| T7 | Anomaly baseline (Khối 3): ≥ 2 feature cùng hướng xấu với \|z\| > 2, hoặc 1 feature \|z\| > 3 lặp ≥ 2 phiên liên tiếp | Trigger VLM | Sinh lời nhắc quan tâm — bắt buộc qua Guardrails (Khối 2). Ngưỡng định nghĩa một chỗ duy nhất trong config, hai tài liệu 01/03 cùng tham chiếu. |

Với nhóm T2–T4 có một điểm cần nói rõ: sự kiện đã qua debounce là sự kiện an toàn có thật, người lái phải được nhắc ngay bằng câu TTS tĩnh duyệt sẵn (không có rủi ro guardrail vì không phải text model sinh). VLM chỉ đóng vai trò nâng chất lượng lời nhắc vài giây sau, chứ không phải điều kiện để được nhắc.

Còn tiêu chí chung để một sự kiện "đắt giá" đáng gọi Tier 2: nó không khẩn cấp tức thời (khẩn cấp thì rule Tier 1 tự xử), nó cần hiểu ngữ cảnh mà CV thuần không phân biệt được (mệt thật hay đeo kính râm? cúi tìm đồ hay ngủ gật?), và nó đã qua debounce cộng cooldown.

## 4. Tier 2 — Edge VLM (event-driven)

Model là Qwen3.5-2B-Instruct quantized Q4_K_M (GGUF 1.28GB + mmproj 0.67GB, chạy qua llama-server của llama.cpp) — đã chạy thật, xem docs/04 §5. Footprint khoảng 2GB RAM, inference 4–7 giây mỗi ảnh trên CPU, nhanh hơn đáng kể trên NPU/GPU edge. Phương án B là Gemma 4 E2B nếu roadmap về sau cần thêm audio native.

Input mỗi lần gọi gồm một frame hiện tại (đã crop vùng mặt), trigger_reason, telematics, và delta so với baseline ở dạng text — không gửi ảnh lịch sử. Output là một câu nhắc tiếng Việt, bắt buộc đi qua `enforce_medical_guardrails` (Khối 2) trước khi ra loa hay màn hình.

Về tài nguyên: hàng đợi chỉ một slot (trigger mới đè trigger cũ chưa kịp chạy), và có theo dõi nhiệt độ chip — quá ngưỡng thì bỏ chu kỳ T6 định kỳ, chỉ giữ trigger sự kiện.

## 5. Ngân sách độ trễ (latency budget)

SLA 300ms đo từ frame chứa bằng chứng cuối cùng hoàn tất điều kiện (ví dụ frame thứ 15 của chuỗi nhắm mắt 1.5s) đến lúc loa phát. Không tính thời gian debounce vào SLA, vì debounce là một phần của định nghĩa sự kiện.

| Đường xử lý | Ngân sách | Phân bổ thực tế @ 10 FPS |
|---|---|---|
| Tier 1 mỗi frame | < 50ms | Capture + landmark ~10ms, head pose ~1ms, YOLO phone (NPU, mọi frame) ~15–30ms — chạy song song với landmark trên 2 đơn vị tính toán |
| Cảnh báo khẩn cấp (T0/T1) | < 300ms | Frame cuối của chuỗi debounce ~50ms xử lý + phát buzzer/TTS pre-recorded ~50ms. Tổng ~100ms, dư 200ms dự phòng cho queue/jitter |
| Cảnh báo nhẹ tức thời (T2–T4) | < 500ms | TTS tĩnh duyệt sẵn, phát ngay khi debounce xác nhận |
| Phản hồi VLM (T2–T7) | 1–5s (chấp nhận được) | Không nằm trên safety path; người lái đã được cảnh báo tức thời trước đó |

## 6. Telematics–VLM fusion (kết hợp ngữ cảnh phương tiện)

Telematics tham gia ở cả ba điểm của pipeline.

Điểm thứ nhất là điều kiện trigger. Telematics tự tạo trigger riêng (T5) và điều biến ngưỡng của trigger khác: lái đêm hoặc `continuous_driving_min` quá 90 thì hạ ngưỡng PERCLOS từ 25% xuống 20%, vì mệt tích lũy làm rủi ro tăng nên cần nhạy hơn.

Điểm thứ hai là input contract của VLM. Tier 2 nhận khối telematics chuẩn hóa, VLM dùng nó để giải thích vì sao nên nghỉ — lời nhắc có căn cứ thì thuyết phục hơn:

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

Điểm thứ ba là chính sách phát cảnh báo theo trạng thái xe — cùng một phát hiện nhưng hành vi khác nhau:

| Trạng thái | Kênh phát | Nội dung |
|---|---|---|
| `moving`, tốc độ cao | Âm thanh ngắn gọn, không màn hình | Tối đa 1 câu, không bắt người lái đọc chữ ("Bạn đã lái hơn 2 tiếng dưới trời nắng nóng, phía trước có chỗ nghỉ thì tấp vào chút nhé") |
| `moving`, tốc độ thấp/kẹt xe | Âm thanh, câu đầy đủ hơn | Có thể kèm gợi ý cụ thể (uống nước, mở thoáng khí) |
| `stopped`/đỗ | Âm thanh + màn hình | Lời nhắc đầy đủ, kèm được tóm tắt chuyến ("Hôm nay bạn ngáp nhiều hơn thường ngày, nghỉ thêm chút rồi hãy đi tiếp") |

Một ví dụ fusion đầy đủ: `continuous_driving_min=125`, nhiệt độ 35°C, trời nắng bụi, và VLM thấy mặt bóng dầu mệt mỏi. Lời nhắc khi đó dẫn cả ba căn cứ — "lái hơn 2 tiếng", "trời nắng 35 độ", "trông bạn thấm mệt" — thay vì một câu chung chung. Tính thuyết phục đến từ chỗ lời nhắc khớp đúng trải nghiệm người lái đang có ngay lúc đó.
