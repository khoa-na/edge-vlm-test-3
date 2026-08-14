# Câu hỏi giải trình (Documentation)

## Câu 1 — Thiết kế kiến trúc tháp 2 tầng (Two-tier Cascade Latency Optimization)

Toàn bộ phân chia trách nhiệm quy về một nguyên tắc: an toàn tức thời thuộc Tier 1, hiểu ngữ cảnh thuộc Tier 2.

Tier 1 chạy mọi frame, dưới 50ms, trên CPU/NPU. MediaPipe FaceLandmarker (478 điểm) cho ra EAR (nhắm mắt, ngưỡng 0.20), MAR (ngáp, ngưỡng 0.35 — hiệu chỉnh trên dataset FL3D), PERCLOS (% nhắm mắt trong cửa sổ 60s) và head pose Euler (pitch dưới −25° là cúi nhìn điện thoại, |yaw| quá 45° là quay đầu). Phone detection dùng YOLO26n INT8, chạy mọi frame vì điện thoại là sự kiện khẩn cấp cần phản hồi dưới 300ms; chọn YOLO26n vì NMS-free/DFL-free cho latency tất định và không rớt accuracy khi quantize INT8. Đè lên các chỉ số thô là temporal state machine: debounce (EAR thấp phải giữ 15 frame liên tục mới tính nhắm mắt, phone cần 3 frame xác nhận), cửa sổ trượt, cooldown. Cảnh báo khẩn cấp T0/T1 phát trực tiếp bằng TTS tĩnh duyệt sẵn, không bao giờ chờ VLM.

Tier 2 là edge VLM quantized, event-driven, độ trễ 1–5s, chỉ chạy khi Tier 1 phát trigger "đắt giá". Một sự kiện đáng gọi VLM khi thỏa cả ba: không khẩn cấp tức thời (khẩn cấp thì rule tự xử), cần ngữ cảnh mà CV thuần không phân biệt được (mệt thật hay kính râm? cúi tìm đồ hay ngủ gật?), và đã qua debounce cộng cooldown 3 phút. Cụ thể các trigger: PERCLOS > 25%, ngáp ≥ 3 lần/10 phút, quay đầu ≥ 3 lần/30s, lái liên tục quá 60 phút, lệch health baseline, và một chu kỳ nền 5 phút có thể skip khi chip nóng.

Kiểm chứng trên FL3D (20,806 frame, 8 sequence): chỉ bằng chỉ số hình học, Tier 1 đạt episode recall 85% trên các đoạn microsleep ≥ 1.5s với false alarm 0.65% — đủ gánh toàn bộ phần an toàn, VLM không nằm trên critical path. Chi tiết: `01-two-tier-cascade.md`, `04-evaluation.md`.

## Câu 2 — Kiểm soát rủi ro y tế & pháp lý (Medical Guardrails)

Xuất phát điểm: không thể đạt "100%" bằng cách kiểm duyệt free text, nên phải đổi bài toán — model không bao giờ được viết lời văn.

Bốn lớp, trong đó hai lớp giữa là tất định:

1. Prompt engineering: system prompt định nghĩa allowlist hành vi (chỉ quan sát bề ngoài và khuyến nghị nghỉ ngơi) kèm few-shot GOOD/BAD. Lớp này nâng chất lượng chọn intent, không gánh trách nhiệm an toàn.
2. Grammar/negative constraints — lớp bảo đảm chính: GBNF grammar ép VLM chỉ xuất JSON enum đóng `{observation, severity, trip_factor, vehicle_state}`, chừng vài chục tổ hợp. Validator tất định từ chối mọi key/value ngoài schema. Renderer bằng code ánh xạ tổ hợp sang câu trong ngân hàng template người viết, đã duyệt pháp lý. Từ cấm không thể xuất hiện, đơn giản vì lời văn không do model sinh.
3. Deterministic post-filtering, lưới thứ hai: quét banned list (cả bản không dấu như "thieu mau"), pattern y tế (mmHg, bpm, số đo "120/80"), trần độ dài — áp lên mọi text trước khi ra loa. Vi phạm thì thay toàn bộ câu bằng fallback template, không vá từng phần vì câu vá dễ giữ nguyên hàm ý chẩn đoán.
4. Fail-closed: mọi nhánh lỗi (VLM timeout, JSON hỏng, filter exception) đổ về template an toàn duyệt sẵn. Audit log không chứa ảnh, đủ để chứng minh tuân thủ.

Vì sao dám nói "không bao giờ": mọi câu tới người dùng hoặc là template duyệt sẵn, hoặc không tồn tại. Việc kiểm chứng thu về review từng template đúng một lần, thay vì kiểm duyệt không gian ngôn ngữ tự nhiên vô hạn tại runtime. Test tự động xác nhận 100% banned term bị chặn. Chi tiết: `02-medical-guardrails.md`.

## Câu 3 — Bảo vệ quyền riêng tư (Privacy-Preserving Health Baseline)

Nguyên tắc gói trong một câu: lưu SỐ, không lưu ẢNH. Mỗi phiên đo (5 phút một lần) trích trên thiết bị khoảng 8 feature vô hướng — quầng thâm (độ sáng hốc mắt so với má), độ mở mắt, sưng mí, độ tái (kênh a Lab má so với trán), sắc môi (a_môi/a_má), blink rate, PERCLOS, yawn rate — rồi hủy frame ngay trong RAM. Các feature màu đều là tỉ lệ giữa hai vùng trên cùng khuôn mặt nên tự chuẩn hóa ánh sáng, cộng thêm bucket theo dải lux để chỉ so sánh giữa các phiên cùng điều kiện sáng.

Baseline tổng hợp theo ngày (median các phiên, chống lệch theo thời lượng chuyến) rồi lấy mean/std đúng nghĩa trên 7 ngày hợp lệ gần nhất, lưu trong SQLite mã hóa at-rest, khóa theo `profile_id` cục bộ chọn tay — không dùng face embedding để phân biệt người. Anomaly bật khi có 2 feature cùng hướng xấu |z| > 2, hoặc 1 feature |z| > 3 bền qua 2 phiên liên tiếp. Ngày bị cờ anomaly bị loại khỏi cửa sổ baseline, tránh chuyện ốm lâu thành bình thường mới. VLM chỉ nhận delta dạng text — ảnh lịch sử không tồn tại để mà lộ.

Về tuân thủ GDPR: data minimization (8 số float, không ảnh, không embedding — embedding vẫn là dữ liệu sinh trắc theo Art. 9), retention 14 ngày tự xóa, xử lý 100% on-device, opt-in riêng, nút xóa hiệu lực tức thời. Các feature vô hướng không đảo ngược được thành khuôn mặt, nên hệ thống nhớ "trạng thái mọi ngày" mà không giữ dữ liệu nhận dạng nào. Chi tiết: `03-health-baseline.md`.

## Câu 4 — Tích hợp ngữ cảnh phương tiện (Telematics-VLM Fusion)

Telematics tham gia ở ba điểm của pipeline. Thứ nhất, điều biến trigger: lái đêm hoặc quá 90 phút liên tục thì hạ ngưỡng PERCLOS từ 25% xuống 20% (rủi ro tích lũy cao thì phải nhạy hơn); thời tiết quyết định pre-ride check có nhắc khẩu trang, kính hay không. Thứ hai, input contract của VLM: khối JSON chuẩn hóa `{vehicle_state, speed_kmh, continuous_driving_min, ambient_temp_c, weather}` để VLM chọn `trip_factor` và lời nhắc dẫn được căn cứ cụ thể. Thứ ba, chính sách phát theo trạng thái xe: đang chạy tốc độ cao thì một câu âm thanh ngắn, không màn hình; kẹt xe chạy chậm thì câu đầy đủ hơn; đang dừng đỗ thì lời nhắc đầy đủ kèm màn hình và tóm tắt chuyến. Cùng một phát hiện, hành vi khác nhau.

Với ví dụ của đề bài — lái 2 tiếng, trời 35°C, VLM thấy mặt mệt — tổ hợp `(looks_more_tired_than_usual, recommend_rest_now, long_drive_hot_weather, moving)` render thành: *"Bạn đã lái hơn hai tiếng dưới trời nắng nóng rồi, phía trước có chỗ mát thì tấp vào uống chút nước nghỉ vài phút nhé"* — dẫn cả ba căn cứ, phát bằng âm thanh ngắn vì xe đang chạy. Nếu xe đang đỗ, câu chuyển thành khuyên nghỉ thêm trước khi khởi hành và hiển thị lên màn hình. Tính thuyết phục đến từ chỗ lời nhắc khớp đúng trải nghiệm người lái ngay thời điểm đó, thay vì một câu chung chung. Chi tiết: `01-two-tier-cascade.md` §6.
