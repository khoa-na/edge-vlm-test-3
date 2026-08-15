# Khối 1 — Kiến trúc phân tầng lọc sự kiện (Two-tier Event-Driven Cascade)

## 1. Nguyên tắc thiết kế

Hai yêu cầu của bài toán tạo ra một đánh đổi rõ ràng. Hệ thống DMS phải theo dõi liên tục ở 5–10 FPS và phát cảnh báo trong vòng 300 ms, trong khi một lần suy luận VLM trên CPU mất khoảng 4–7 giây. Nếu đưa VLM vào vòng xử lý từng frame, hệ thống vừa không đáp ứng được độ trễ, vừa tạo tải nhiệt không cần thiết.

Tôi tách pipeline thành hai đường độc lập. Đường an toàn do Tier 1 xử lý bằng các phép đo CV và luật tất định; cảnh báo khẩn cấp được phát ngay tại đây. Tier 2 chỉ nhận những sự kiện cần thêm ngữ cảnh và có thể chấp nhận chờ vài giây. Nhờ cách tách này, VLM giúp lời nhắc phù hợp hơn nhưng không trở thành điều kiện để hệ thống phát hiện nguy hiểm.

```
Camera 5-10 FPS
      │
      ▼
┌─────────────────────────────────────────────┐
│ TIER 1 — Lightweight CV (mọi frame, <50ms)  │
│  • Face Landmark → EAR, MAR, PERCLOS        │
│  • Head Pose (ước lượng pitch/yaw)          │
│  • Object Detection (phone, helmet, mask)   │
│  • Temporal State Machine (đếm thời gian)   │
└──────┬──────────────────────┬───────────────┘
       │                      │
  Rule khẩn cấp          Trigger event
  (< 300ms)              (debounced)
       │                      │
       ▼                      ▼
     CẢNH BÁO         ┌──────────────────────────────┐
  NGAY LẬP TỨC        │ TIER 2 — Edge VLM (event-    │
  (buzzer/TTS)        │ driven hoặc định kỳ 5 phút)  │
                      │  • Phân tích ngữ cảnh sâu    │
                      │  • So sánh health baseline   │
                      │  • Chọn observation/severity │
                      │  → qua Guardrails (Khối 2)   │
                      └──────────────────────────────┘
```

## 2. Tier 1 — Lightweight CV (real-time, mọi frame)

### 2.1 Mô hình và chỉ số

| Thành phần | Mô hình | Chỉ số tính ra | Chi phí/frame |
|---|---|---|---|
| Face Landmark | MediaPipe FaceLandmarker (478 điểm, đã chạy thật — xem docs/04) | EAR, MAR, vị trí mắt/môi | ~5–10 ms CPU |
| Head Pose | Tỉ lệ hình học landmark 2D + calibration góc camera | Pitch / Yaw xấp xỉ | <1 ms sau landmark |
| Object Detection (in-ride) | YOLO26n COCO `.pt` FP32, chỉ giữ class `cell phone` | confidence phone toàn frame | ~44 ms/lần CPU @384; chạy stride 3 và cache nên trung bình ~15 ms/frame |
| Object Detection (pre-ride) | Interface `Yolo26PreRideDetector` | helmet_strap, mask, sunglasses | Cần weights finetune riêng; repo hiện dùng mock cho nhánh này |

Tôi dùng YOLO26n vì model nhỏ, kiến trúc NMS-free/DFL-free và đã có class điện thoại trong COCO. Bản hiện tại chạy weights `.pt` FP32 trên CPU. Việc export INT8 hay chuyển sang NPU phụ thuộc vào thiết bị đích nên chưa được tính là phần đã triển khai.

Các chỉ số cụ thể:

- EAR (Eye Aspect Ratio) = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). Mắt mở thường đo được 0.25–0.3, nhắm thì tụt dưới 0.2. Dùng để đo thời lượng nhắm mắt liên tục.
- PERCLOS (Percentage of Eye Closure): % thời gian EAR dưới ngưỡng trong cửa sổ trượt 60 giây. Đây là chỉ số chuẩn công nghiệp đo buồn ngủ tích lũy, nhạy hơn nhiều so với chỉ bắt một lần nhắm mắt.
- MAR (Mouth Aspect Ratio): MAR cao kéo dài quá 2 giây tính là một lần ngáp, đếm số lần trong 10 phút. Ngưỡng lý thuyết hay gặp là 0.6, nhưng đo thật trên FL3D với bộ landmark MediaPipe thì ngáp thật chỉ đạt MAR ~0.5 nên hạ về 0.35 (chi tiết docs/04).
- Head pose xấp xỉ: pitch dưới −25° kéo dài hơn 1.5 giây là cúi nhìn xuống (điện thoại hoặc gục đầu); |yaw| quá 45° lặp hơn 3 lần trong 30 giây là quay đầu về sau liên tục. Bản production nên thay bằng `solvePnP` với camera matrix để có đủ Euler pitch/yaw/roll.
- Blink rate: số lần chớp mắt mỗi phút. Vừa là tín hiệu mệt mỏi (chớp chậm dần), vừa là feature ghi vào health baseline (Khối 3).

### 2.2 Temporal state machine (chống nhiễu)

Một frame riêng lẻ không đủ để kết luận hành vi. Chẳng hạn, một lần chớp mắt bình thường cũng làm EAR giảm mạnh. Tier 1 vì thế giữ trạng thái theo thời gian thay vì ra quyết định ngay trên từng frame.

Với các sự kiện ngắn, pipeline dùng hysteresis và debounce. EAR phải dưới 0.2 liên tục 1.5 giây, tương đương 15 frame ở 10 FPS, mới được xem là nhắm mắt kéo dài. Điện thoại cần xuất hiện ba frame liên tiếp để bật trạng thái và vắng mặt mười frame liên tiếp để tắt. Khoảng trễ này giúp cảnh báo không nhấp nháy khi detector mất dấu trong chốc lát.

Các tín hiệu tích lũy được tính trên cửa sổ trượt: PERCLOS trong 60 giây, số lần ngáp trong 10 phút và số lần quay đầu trong 30 giây.

Sau khi một loại sự kiện đã kích hoạt Tier 2, cooldown ngăn cùng sự kiện lặp lại trong vài phút. Cơ chế này vừa giảm số lần gọi model, vừa tránh nhắc người lái liên tục về cùng một vấn đề.

### 2.3 Kiểm tra trước khi chạy (pre-ride check)

`pre_ride_check()` nhận một cụm frame khi xe chưa lăn bánh rồi lấy kết quả theo đa số, thay vì kết luận từ một ảnh có thể bị mờ. Quai mũ luôn được kiểm tra; khẩu trang và kính chỉ được nhắc khi telematics báo trời nắng bụi. Repo đã có logic và test với detector giả lập. Phần weights finetune cũng như việc nối API này vào vòng đời vận hành thực tế của phương tiện vẫn nằm ngoài hai runner webcam/video hiện tại.

## 3. Bảng điều kiện trigger — ai xử lý, xử lý thế nào

| # | Điều kiện phát hiện (Tier 1) | Loại | Hành động |
|---|---|---|---|
| T0 | EAR < 0.2 liên tục > 1.5s | KHẨN CẤP | Tier 1 phát cảnh báo âm thanh ngay; không gọi VLM. |
| T1 | Pitch < −25° > 1.5s hoặc phone detected | KHẨN CẤP | Cảnh báo rule-based ngay; câu nói phân biệt phone với head-down để không khẳng định sai hành vi. |
| T2 | PERCLOS > 25% trong 60s | Cảnh báo nhẹ + trigger VLM | Tier 1 phát ngay câu TTS duyệt sẵn ("Bạn có vẻ buồn ngủ, chú ý tập trung nhé"); VLM chạy sau để sinh lời khuyên ngữ cảnh đầy đủ hơn. |
| T3 | ≥ 3 lần ngáp / 10 phút | Cảnh báo nhẹ + trigger VLM | Như T2 — nhắc nhẹ tức thời trước, VLM bổ sung sau. |
| T4 | \|Yaw\| > 45° lặp ≥ 3 lần / 30s | Cảnh báo nhẹ + trigger VLM | Tier 1 nhắc "chú ý quan sát phía trước" ngay; VLM chọn mức nhắc trong schema fatigue đóng. |
| T5 | `continuous_driving_min` > 60 (telematics, không cần CV) | Câu tĩnh + trigger VLM | Nhắc nghỉ bằng câu duyệt sẵn ngay; VLM chạy nền để bổ sung ngữ cảnh, không chặn frame loop. |
| T6 | Định kỳ mỗi 5 phút | Scheduled (CV, không phải VLM) | Trích feature bằng landmark + thống kê màu; bỏ frame quá tối và feature temporal chưa đủ cửa sổ trước khi ghi baseline. Sensor nhiệt/skip policy là hook production chưa triển khai. |
| T7 | Anomaly baseline (Khối 3): ≥ 2 feature cùng hướng xấu với \|z\| > 2, hoặc 1 feature \|z\| > 3 lặp ≥ 2 phiên liên tiếp | Câu tĩnh + trigger VLM | Ghi cờ ngày anomaly, nhắc nhẹ ngay và chạy VLM nền; ngưỡng nằm một chỗ trong `health_baseline.py`. |

Ở T2–T4, người lái được nghe một câu TTS đã duyệt ngay khi sự kiện vượt qua debounce. VLM chạy sau để bổ sung ngữ cảnh, chứ không quyết định việc có cảnh báo hay không. Cách làm này giữ được phản hồi nhanh ngay cả khi model đang bận hoặc gặp lỗi.

Một sự kiện chỉ được chuyển lên Tier 2 khi đã qua debounce và cooldown, đồng thời cần nhiều ngữ cảnh hơn các chỉ số CV thô có thể cung cấp. Những tình huống khẩn cấp không thỏa điều kiện này vì Tier 1 đã xử lý trực tiếp.

## 4. Tier 2 — Edge VLM (event-driven)

Tier 2 được thử nghiệm với Qwen3.5-2B-Instruct Q4_K_M, gồm file GGUF 1.28 GB và mmproj 0.67 GB, phục vụ qua llama-server của llama.cpp. Trên CPU của máy thử nghiệm, mỗi ảnh mất khoảng 4–7 giây. Gemma 4 E2B là một phương án thay thế nếu phiên bản sau cần xử lý audio trực tiếp; phần đánh giá model hiện tại nằm ở `docs/04-evaluation.md`.

Mỗi request gồm frame hiện tại, `trigger_reason`, telematics và phần chênh lệch so với baseline dưới dạng text. Ảnh lịch sử không được gửi vào model. Runner hiện dùng nguyên frame; crop vùng mặt là tối ưu có thể bổ sung khi triển khai. VLM chỉ trả JSON enum, còn câu tiếng Việt do renderer chọn từ template bank và được kiểm tra lần cuối bằng `enforce_medical_guardrails`.

Worker VLM chỉ xử lý một tác vụ tại một thời điểm. Nếu model đang chạy, trigger mới không được xếp thành hàng dài vì đến lúc xử lý xong thì frame và ngữ cảnh có thể đã cũ. Việc theo dõi nhiệt độ chip và bỏ chu kỳ T6 khi thiết bị quá nóng cần được nối với cảm biến của SoC thực tế; repo Python chưa có backend này.

## 5. Ngân sách độ trễ (latency budget)

SLA 300ms đo từ frame chứa bằng chứng cuối cùng hoàn tất điều kiện (ví dụ frame thứ 15 của chuỗi nhắm mắt 1.5s) đến lúc loa phát. Không tính thời gian debounce vào SLA, vì debounce là một phần của định nghĩa sự kiện.

| Đường xử lý | Ngân sách | Phân bổ thực tế @ 10 FPS |
|---|---|---|
| Tier 1 mỗi frame | < 50ms mục tiêu | MediaPipe + YOLO CPU đo trung bình khoảng 31ms/frame trên máy phát triển; YOLO chạy stride 3, không phải mọi frame |
| Cảnh báo khẩn cấp (T0/T1) | < 300ms mục tiêu | Câu WAV đã pre-render, phát trên thread riêng và có quyền preempt câu thường; chưa có phép đo end-to-end trên SoC đích |
| Cảnh báo nhẹ tức thời (T2–T4) | < 500ms | TTS tĩnh duyệt sẵn, phát ngay khi debounce xác nhận |
| Phản hồi VLM (T2–T7) | 4–7s CPU đã đo | Worker nền, không nằm trên safety path; T2–T5/T7 có câu tĩnh khi cần |

## 6. Telematics–VLM fusion (kết hợp ngữ cảnh phương tiện)

Telematics không chỉ được đính kèm vào prompt mà còn tham gia trực tiếp vào logic. Thời gian lái liên tục tạo trigger T5; nếu chuyến đi kéo dài quá 90 phút hoặc diễn ra ban đêm, ngưỡng PERCLOS được hạ từ 25% xuống 20% để hệ thống nhạy hơn.

Khi gọi Tier 2, pipeline truyền một khối telematics đã chuẩn hóa. Nhờ đó lời nhắc có thể gắn với hoàn cảnh cụ thể thay vì chỉ nói chung rằng người lái có vẻ mệt:

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

Trạng thái xe cũng quyết định kênh phát cảnh báo:

| Trạng thái | Kênh phát | Nội dung |
|---|---|---|
| `moving`, tốc độ cao | Âm thanh ngắn gọn, không màn hình | Tối đa 1 câu, không bắt người lái đọc chữ ("Bạn đã lái hơn 2 tiếng dưới trời nắng nóng, phía trước có chỗ nghỉ thì tấp vào chút nhé") |
| `moving`, tốc độ thấp/kẹt xe | Âm thanh, câu đầy đủ hơn | Có thể kèm gợi ý cụ thể (uống nước, mở thoáng khí) |
| `stopped`/đỗ | Âm thanh + màn hình | Lời nhắc đầy đủ, kèm được tóm tắt chuyến ("Hôm nay bạn ngáp nhiều hơn thường ngày, nghỉ thêm chút rồi hãy đi tiếp") |

Ví dụ, với `continuous_driving_min=125`, nhiệt độ 35°C và xe đang chạy, hệ thống có thể chọn câu nhắc nghỉ ở nơi mát thay vì một lời khuyên chung chung. Nếu xe đã dừng, template đổi sang khuyên nghỉ thêm trước khi đi tiếp. Phần quan trọng ở đây là các dữ kiện như thời gian lái, nhiệt độ và trạng thái xe đều do code lấy từ telematics; model không tự suy đoán chúng từ ảnh.
