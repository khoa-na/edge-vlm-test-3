# Câu hỏi giải trình

## Câu 1 — Vì sao chọn kiến trúc tháp hai tầng?

Tôi tách hệ thống theo mức độ khẩn cấp của quyết định: Tier 1 chịu trách
nhiệm cho cảnh báo tức thời, còn Tier 2 chỉ được gọi khi cần hiểu thêm ngữ
cảnh. Nhờ vậy, cảnh báo quan trọng không phụ thuộc vào tốc độ của VLM.

Tier 1 xử lý mọi frame trên CPU. MediaPipe FaceLandmarker cung cấp 478 điểm
để tính EAR, MAR, PERCLOS và góc quay đầu gần đúng. Ngưỡng EAR là 0,20; ngưỡng
MAR được hiệu chỉnh trên FL3D xuống 0,35. Phone detector dùng YOLO26n COCO
`.pt` FP32, chạy mỗi ba frame và dùng lại confidence ở các frame xen giữa.
Trên máy phát triển, riêng YOLO mất khoảng 44 ms mỗi lần chạy; khi ghép với
MediaPipe, pipeline trung bình khoảng 31 ms mỗi frame. Debounce, cửa sổ trượt
và cooldown giúp phân biệt một trạng thái kéo dài với nhiễu ở một frame. T0
và T1 dùng câu TTS tĩnh, có thể ngắt một lời nhắc thường đang phát và không
bao giờ chờ VLM. `solvePnP`, roll đầy đủ và INT8/NPU vẫn là phần cần làm khi
chuyển sang phần cứng production, chưa phải khả năng của prototype này.

Tier 2 dùng edge VLM quantized và chỉ chạy theo sự kiện. Phép đo hiện tại
trên CPU mất khoảng 4–7 giây mỗi lần gọi, nên model chạy trên worker nền một
slot. Những trigger cần thêm ngữ cảnh gồm PERCLOS cao, ngáp lặp lại, quay đầu
nhiều, lái liên tục quá lâu và lệch health baseline. Trong lúc chờ model,
T2–T5/T7 vẫn có thể trả một câu tĩnh đã duyệt nếu tình huống cần phản hồi sớm.
Theo dõi nhiệt độ chip và bỏ bớt chu kỳ suy luận khi thiết bị nóng mới chỉ là
điểm tích hợp dự kiến, chưa được triển khai trong repo.

Trên 20.806 frame của tám sequence FL3D, Tier 1 đạt recall 85% ở mức episode
cho các đoạn microsleep dài từ 1,5 giây, với 0,65% frame alert làm trạng thái
T0 bật. Kết quả chưa đủ để coi là chứng nhận production, nhưng ủng hộ quyết
định giữ VLM ra khỏi critical path. Chi tiết nằm trong
[`01-two-tier-cascade.md`](01-two-tier-cascade.md) và
[`04-evaluation.md`](04-evaluation.md).

## Câu 2 — Làm thế nào hạn chế rủi ro y tế và pháp lý?

Tôi không để model tự viết câu đưa tới người dùng. VLM chỉ chọn một intent
trong schema đóng; code mới là phần ánh xạ intent đó sang câu đã được viết và
duyệt trước.

Thiết kế có bốn lớp:

1. System prompt giới hạn nhiệm vụ ở quan sát bề ngoài và khuyến nghị nghỉ
   ngơi, kèm ví dụ tốt/xấu. Prompt giúp model chọn đúng intent nhưng không
   được xem là lớp bảo đảm an toàn.
2. Grammar ép đầu ra thành JSON gồm bốn enum: `observation`, `severity`,
   `trip_factor` và `vehicle_state`. Validator từ chối key hoặc value nằm
   ngoài schema. Sau đó renderer chọn câu trong template bank; model không
   có đường để đưa free text thẳng ra loa.
3. Post-filter quét từ cấm, cả dạng không dấu, mẫu số đo y tế như `mmHg`,
   `bpm`, `120/80`, cùng giới hạn độ dài. Nếu vi phạm, toàn bộ câu được thay
   bằng fallback an toàn thay vì cố sửa từng từ.
4. Hệ thống fail closed: timeout, JSON hỏng hoặc lỗi filter đều quay về
   template đã duyệt. Audit log chỉ lưu metadata cần thiết, không lưu ảnh.

Mức bảo đảm ở đây đến từ cấu trúc đầu ra đóng, không phải từ lời hứa rằng
model ngôn ngữ sẽ luôn nghe prompt. Bộ test hiện kiểm tra mọi từ cấm trong
danh sách cấu hình đều bị chặn; khi triển khai thực tế, template và banned
list vẫn cần được pháp chế rà soát theo ngôn ngữ và thị trường sử dụng. Xem
thêm [`02-medical-guardrails.md`](02-medical-guardrails.md).

## Câu 3 — Health baseline bảo vệ quyền riêng tư ra sao?

Nguyên tắc tôi dùng là “lưu số, không lưu ảnh”. Cứ mỗi 5 phút, pipeline trích
tám feature ngay trên thiết bị: độ tối vùng dưới mắt, độ mở mắt, độ sưng mí,
độ nhợt của da, chỉ số màu môi, tần suất chớp mắt, PERCLOS và tần suất ngáp.
Frame chỉ tồn tại trong RAM trong lúc xử lý. Các feature màu dùng tỷ lệ giữa
hai vùng trên cùng khuôn mặt, đồng thời mỗi mẫu được gắn `light_bucket` để
chỉ so sánh trong điều kiện sáng tương đương.

Các phiên trong ngày được gộp bằng median, rồi baseline lấy mean và độ lệch
chuẩn từ tối đa 7 ngày hợp lệ gần nhất. Dữ liệu nằm trong SQLite cục bộ và
gắn với `profile_id` do người dùng chọn, không dùng face embedding. Frame quá
tối bị bỏ; điều kiện sáng lấy từ sensor hoặc ước lượng thành `light_bucket`.
Feature temporal chưa đủ cửa sổ được lưu `NULL`. Hệ thống cảnh báo khi hai
feature cùng lệch theo hướng cần chú ý với |z| > 2, hoặc một feature có |z| >
3 trong hai phiên liên tiếp; noise floor theo từng feature tránh z-score tăng
vọt khi baseline có độ lệch chuẩn bằng 0. Ngày đã bị cảnh báo được loại khỏi
baseline để trạng thái bất thường không dần bị học thành bình thường. Nếu cần
gọi VLM, model chỉ nhận phần chênh lệch dạng text và frame hiện tại; không có
ảnh lịch sử để gửi.

Prototype đã có retention 14 ngày cho sample, dữ liệu tổng hợp và cờ anomaly.
SQLite hiện chưa mã hóa, và repo cũng chưa có màn hình opt-in hay chức năng
xóa theo profile. SQLCipher/Keystore, consent và quyền xóa là các yêu cầu bắt
buộc khi tích hợp vào sản phẩm. Dù không phải ảnh, các feature vẫn là dữ liệu
cá nhân khi gắn với hồ sơ người dùng. Chi tiết ở
[`03-health-baseline.md`](03-health-baseline.md).

## Câu 4 — Telematics được kết hợp với VLM như thế nào?

Telematics tham gia vào cả logic phát hiện lẫn cách đưa lời nhắc. Khi lái ban
đêm hoặc liên tục quá 90 phút, ngưỡng PERCLOS giảm từ 25% xuống 20%. Thời
tiết cũng được dùng để quyết định có nhắc khẩu trang hoặc kính ở bước
pre-ride hay không. Các dữ kiện như `speed_kmh`, `continuous_driving_min`,
`ambient_temp_c` và `weather` được code chuyển thành `trip_factor` và
`vehicle_state`; VLM không tự suy đoán những thông tin này từ ảnh.

Sau khi có nội dung, `delivery_channel()` chọn `audio_short`, `audio_full`
hoặc `audio_and_screen` theo tốc độ xe. Runner hiện mới log channel, còn việc
ẩn hoặc hiện nội dung trên màn hình cần do UI production thực thi.

Với tình huống trong đề — đã lái hai tiếng, nhiệt độ 35°C và VLM thấy người
lái trông mệt hơn thường ngày — tổ hợp
`(looks_more_tired_than_usual, recommend_rest_now, long_drive_hot_weather, moving)`
được render đúng theo template hiện tại:

> Bạn đã lái liên tục khá lâu dưới trời nắng nóng và trông khá mệt, hãy tấp
> vào chỗ mát nghỉ ngơi rồi hãy đi tiếp nhé.

Khi xe đang chạy, câu được phát dưới dạng âm thanh ngắn. Nếu xe đã dừng,
renderer chọn một template phù hợp hơn và channel có thể kèm màn hình. Lời
nhắc nhờ vậy phản ánh đúng thời gian lái, thời tiết và quan sát hiện tại mà
không để model bịa thêm dữ kiện. Phần luồng xử lý được mô tả ở mục 6 của
[`01-two-tier-cascade.md`](01-two-tier-cascade.md).
