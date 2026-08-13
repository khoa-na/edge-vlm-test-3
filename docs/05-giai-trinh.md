# Câu Hỏi Giải Trình (Documentation)

## Câu 1 — Thiết Kế Kiến Trúc Tháp 2 Tầng (Two-tier Cascade Latency Optimization)

**Phân chia trách nhiệm theo một nguyên tắc duy nhất: an toàn tức thời thuộc Tier 1, hiểu ngữ cảnh thuộc Tier 2.**

- **Tier 1 (mọi frame, <50ms, CPU/NPU):** MediaPipe FaceLandmarker (478 điểm) tính **EAR** (nhắm mắt, ngưỡng 0.20), **MAR** (ngáp, ngưỡng 0.35 — hiệu chỉnh trên dataset FL3D), **PERCLOS** (% nhắm mắt cửa sổ 60s), **Head Pose Euler** (pitch < −25° = cúi nhìn điện thoại, |yaw| > 45° = quay đầu), **YOLOv8n INT8** phát hiện điện thoại (mọi frame — vì là sự kiện khẩn cấp <300ms). Trên các chỉ số thô là **temporal state machine**: debounce (EAR thấp ≥15 frame liên tục mới là "nhắm mắt", phone cần 3 frame xác nhận), cửa sổ trượt, cooldown. Cảnh báo khẩn cấp (T0/T1) phát trực tiếp bằng TTS tĩnh duyệt sẵn — **không bao giờ chờ VLM**.

- **Tier 2 (event-driven, 1–5s):** Edge VLM quantized, chỉ chạy khi Tier 1 phát trigger "đắt giá". Tiêu chí gọi: (1) không khẩn cấp tức thời — khẩn cấp thì rule tự xử; (2) cần ngữ cảnh mà CV thuần không phân biệt được (mệt thật vs kính râm, cúi tìm đồ vs ngủ gật); (3) đã qua debounce + cooldown 3 phút. Cụ thể: PERCLOS > 25%, ≥3 ngáp/10 phút, quay đầu ≥3 lần/30s, lái liên tục >60 phút, lệch health baseline, và chu kỳ nền 5 phút (skip được khi chip nóng).

Kiểm chứng thực tế trên FL3D (19,407 frame): riêng chỉ số hình học Tier 1 đạt microsleep recall 95% — đủ gánh toàn bộ phần an toàn, VLM không nằm trên critical path. Chi tiết: `01-two-tier-cascade.md`, `04-evaluation.md`.

## Câu 2 — Kiểm Soát Rủi Ro Y Tế & Pháp Lý (Medical Guardrails)

**Không thể đạt "100%" bằng cách kiểm duyệt free text — nên đổi bài toán: model không bao giờ được viết lời văn.**

4 lớp, hai lớp giữa là tất định:

1. **Prompt Engineering:** system prompt định nghĩa allowlist hành vi (chỉ quan sát bề ngoài + khuyến nghị nghỉ ngơi) + few-shot GOOD/BAD — nâng chất lượng chọn intent, không gánh trách nhiệm an toàn.
2. **Grammar/Negative Constraints (lớp bảo đảm chính):** GBNF grammar ép VLM chỉ xuất **JSON enum đóng** `{observation, severity, trip_factor, vehicle_state}` (~vài chục tổ hợp). Validator tất định từ chối mọi key/value ngoài schema. **Renderer bằng code** ánh xạ tổ hợp → câu trong ngân hàng template người viết, đã duyệt pháp lý. Từ cấm không thể xuất hiện vì lời văn không do model sinh.
3. **Deterministic Post-filtering (lưới thứ hai):** quét banned list (cả bản không dấu: "thieu mau"), pattern y tế (mmHg, bpm, số đo "120/80"), trần độ dài — trên **mọi** text trước khi ra loa. Vi phạm = thay toàn bộ câu bằng fallback template, không vá từng phần (câu vá giữ nguyên hàm ý chẩn đoán).
4. **Fail-closed:** mọi nhánh lỗi (VLM timeout, JSON hỏng, filter exception) đổ về template an toàn duyệt sẵn. Audit log (không ảnh) chứng minh tuân thủ.

Vì sao "không bao giờ": mọi câu tới người dùng hoặc là template duyệt sẵn, hoặc không tồn tại — kiểm chứng bằng review từng template một lần, thay vì kiểm duyệt không gian ngôn ngữ tự nhiên vô hạn tại runtime. Test tự động xác nhận 100% banned term bị chặn. Chi tiết: `02-medical-guardrails.md`.

## Câu 3 — Bảo Vệ Quyền Riêng Tư (Privacy-Preserving Health Baseline)

**Lưu SỐ, không lưu ẢNH:** mỗi phiên đo (5 phút/lần) trích trên thiết bị ~8 feature vô hướng — quầng thâm (độ sáng hốc mắt/má), độ mở mắt, sưng mí, độ tái (kênh a Lab má/trán), sắc môi (a_môi/a_má), blink rate, PERCLOS, yawn rate — rồi **hủy frame ngay trong RAM**. Các feature màu đều là tỉ lệ giữa 2 vùng trên cùng khuôn mặt (tự chuẩn hóa ánh sáng) + bucket theo dải lux, chỉ so sánh cùng điều kiện sáng.

Cơ chế baseline: tổng hợp theo **ngày** (median các phiên — chống lệch theo thời lượng chuyến), baseline = mean/std đúng nghĩa trên **7 ngày hợp lệ gần nhất** trong SQLite mã hóa at-rest, khóa theo `profile_id` cục bộ (chọn tay — không dùng face embedding để phân biệt người). Anomaly: ≥2 feature cùng hướng xấu |z|>2, hoặc 1 feature |z|>3 bền ≥2 phiên; ngày bị cờ loại khỏi cửa sổ baseline (tránh "ốm lâu thành bình thường mới"). VLM chỉ nhận **delta dạng text** — ảnh lịch sử không tồn tại để mà lộ.

Tuân thủ GDPR: data minimization (8 số float, không ảnh/embedding — embedding vẫn là dữ liệu sinh trắc Art. 9), retention 14 ngày tự xóa, xử lý 100% on-device, opt-in riêng, nút xóa tức thời. Feature vô hướng **không đảo ngược được thành khuôn mặt** — hệ thống nhớ "trạng thái mọi ngày" mà không giữ dữ liệu nhận dạng. Chi tiết: `03-health-baseline.md`.

## Câu 4 — Tích Hợp Ngữ Cảnh Phương Tiện (Telematics-VLM Fusion)

Telematics tham gia **3 điểm** của pipeline:

1. **Điều biến trigger:** lái đêm hoặc >90 phút liên tục → hạ ngưỡng PERCLOS 25%→20% (rủi ro tích lũy cao thì nhạy hơn); thời tiết quyết định pre-ride check có nhắc khẩu trang/kính không.
2. **Input contract của VLM:** khối JSON chuẩn hóa `{vehicle_state, speed_kmh, continuous_driving_min, ambient_temp_c, weather}` — VLM dùng để chọn `trip_factor`, lời nhắc dẫn được căn cứ cụ thể.
3. **Chính sách phát theo trạng thái xe:** đang chạy tốc độ cao → 1 câu âm thanh ngắn, không màn hình; kẹt xe/chậm → câu đầy đủ hơn; **đang dừng đỗ → lời nhắc đầy đủ + màn hình + tóm tắt chuyến**. Cùng một phát hiện, hành vi khác nhau.

Ví dụ đề bài: lái 2 tiếng + 35°C + VLM thấy mặt mệt → tổ hợp `(looks_more_tired_than_usual, recommend_rest_now, long_drive_hot_weather, moving)` render thành: *"Bạn đã lái hơn hai tiếng dưới trời nắng nóng rồi, phía trước có chỗ mát thì tấp vào uống chút nước nghỉ vài phút nhé"* — dẫn cả 3 căn cứ, phát âm thanh ngắn vì xe đang chạy. Nếu xe đang đỗ: câu khuyên nghỉ thêm trước khi khởi hành + hiển thị màn hình. Tính thuyết phục đến từ lời nhắc **khớp trải nghiệm thật** của người lái ngay thời điểm đó, thay vì câu chung chung. Chi tiết: `01-two-tier-cascade.md` §6.
