# Khối 3 — Long-term Health Memory Index (Privacy-Preserving Baseline)

## 1. Bài toán

Mục tiêu của khối này không phải kết luận người lái đang mắc bệnh, mà là nhận ra hôm nay họ có khác đáng kể so với trạng thái thường ngày hay không, chẳng hạn mắt sưng hơn, quầng thâm rõ hơn hoặc sắc mặt nhợt hơn. Việc so sánh như vậy cần dữ liệu của nhiều ngày, nhưng không vì thế mà phải lưu ảnh khuôn mặt. Tôi chọn giữ toàn bộ xử lý trên thiết bị và chỉ lưu những chỉ số cần thiết, phù hợp với nguyên tắc giảm thiểu dữ liệu của GDPR và Nghị định 13/2023/NĐ-CP.

## 2. Nguyên tắc: lưu SỐ, không lưu ẢNH

```
Frame (RAM) ──► Trích xuất feature vô hướng (on-device, <1s)
                        │
                        ▼
              8 chỉ số số học phục vụ đúng bài toán
                        │
                        ▼
              SQLite local — rolling 7-14 ngày
                        │
Frame bị hủy ngay       ▼
sau khi trích     So sánh z-score với baseline → cờ anomaly → Trigger T7 (Khối 1)
(không ghi disk)
```

Pipeline dùng lại landmark từ Tier 1, lấy các vùng nhỏ quanh mắt, má và môi rồi tính thêm một số thống kê màu đơn giản. T6 lấy mẫu mỗi 5 phút. Khi chạy webcam hoặc video, dữ liệu mặc định được lưu vào SQLite; với các lần replay chỉ để thử nghiệm, có thể dùng `--db-path :memory:` để không ghi ra đĩa.

## 3. Bộ feature trích xuất (mỗi phiên đo)

| Feature | Cách tính (on-device) | Bắt bất thường gì |
|---|---|---|
| `eye_darkness` | Độ sáng trung bình (kênh L trong Lab) vùng dưới hốc mắt, chia cho độ sáng má (tự chuẩn hóa ánh sáng) | Quầng thâm đậm hơn ngày thường |
| `eye_openness` | EAR trung bình khi mắt "mở" trong phiên | Mắt sưng / lờ đờ (mở không hết) |
| `eye_puffiness` | Khoảng cách trung bình giữa mí dưới và gò má, chia cho khoảng cách hai mắt để giảm ảnh hưởng của độ phân giải và khoảng cách camera | Mắt sưng |
| `skin_paleness` | Giá trị kênh a trong Lab (trục xanh lục–đỏ; a thấp = ít sắc hồng = tái) vùng má, tỉ lệ với vùng trán để tự chuẩn hóa | Sắc mặt tái nhợt |
| `lip_color_index` | a_môi / (a_má + ε) — sắc đỏ môi tương đối so với má, ε tránh chia gần 0 | Môi tím tái (a_môi tụt về phía âm/xanh) |
| `blink_rate` | Số lần chớp/phút (từ Tier 1) | Mệt mỏi, khô mắt |
| `perclos` | % thời gian mắt nhắm trong phiên | Buồn ngủ tích lũy |
| `yawn_rate` | Số lần ngáp/10 phút | Mệt mỏi |
| `light_bucket` | Điều kiện sáng do camera, cảm biến hoặc telematics cung cấp, gom thành các dải tối / trong nhà / ngoài trời / nắng gắt | Biến kiểm soát — không phải feature sức khỏe |

Ánh sáng là nguồn sai lệch lớn nhất của nhóm chỉ số màu: cùng một khuôn mặt có thể trông nhợt dưới đèn trắng nhưng hoàn toàn bình thường dưới nắng chiều. Vì vậy, các chỉ số màu được tính tương đối giữa hai vùng trên cùng khuôn mặt, như môi so với má hoặc hốc mắt so với má. Mỗi mẫu còn đi kèm `light_bucket`, và baseline chỉ so sánh các phiên có điều kiện sáng tương đương. Nếu ảnh quá tối để trích xuất đáng tin cậy, phiên đó bị bỏ thay vì cố ghi một giá trị nhiễu.

Trước khi lưu, mẫu phải qua một quality gate: pipeline bỏ phiên nếu không thấy mặt, nếu `landmark_confidence < 0.7`, hoặc nếu chưa đến một nửa số feature có giá trị hợp lệ. Feature riêng lẻ trích không được sẽ để `NULL`. Phần nhận diện khẩu trang và kính râm chưa được nối vào nhánh health vì repo chưa có weights pre-ride thật; do đó prototype chưa thể đảm bảo mọi vùng bị che đều được loại chính xác.

## 4. Schema SQLite hiện thực

```sql
-- Mỗi phiên đo (5 phút/lần khi lái + pre-ride): 1 dòng, chỉ chứa số.
-- profile_id: định danh hồ sơ cục bộ (chọn tay khi ghép thiết bị / đăng nhập app),
-- KHÔNG dùng face embedding để phân biệt người — embedding là dữ liệu sinh trắc.
CREATE TABLE health_samples (
    sample_id       INTEGER PRIMARY KEY,
    profile_id      INTEGER NOT NULL,
    ts              INTEGER NOT NULL,      -- epoch seconds
    light_bucket    INTEGER NOT NULL,      -- 0=tối 1=trong nhà 2=ngoài trời 3=nắng gắt
    eye_darkness    REAL, eye_openness REAL, eye_puffiness REAL,   -- NULL = ROI bị che
    skin_paleness   REAL, lip_color_index REAL,
    blink_rate      REAL, perclos REAL, yawn_rate REAL
);
-- Retention: xóa dòng > 14 ngày (job hằng ngày). Không có cột ảnh/embedding.

-- Tổng hợp theo NGÀY (chống lệch theo thời lượng chuyến: ngày lái 4 tiếng có 48 phiên,
-- ngày lái 20 phút có 4 phiên — nếu tính baseline thẳng từ phiên, ngày dài áp đảo).
-- Cập nhật cuối mỗi chuyến: mỗi (ngày, bucket, feature) = median các phiên trong ngày.
CREATE TABLE health_daily (
    profile_id      INTEGER NOT NULL,
    day             TEXT    NOT NULL,      -- 'YYYY-MM-DD'
    light_bucket    INTEGER NOT NULL,
    feature_name    TEXT    NOT NULL,
    day_value       REAL    NOT NULL,      -- median các phiên trong ngày
    session_count   INTEGER NOT NULL,
    is_anomalous    INTEGER NOT NULL DEFAULT 0,  -- ngày bị cờ: loại khỏi baseline
    PRIMARY KEY (profile_id, day, light_bucket, feature_name)
);

-- Cờ được ghi ngay khi T7 nổ, kể cả daily chưa aggregate; cuối chuyến cờ
-- được chép vào health_daily.is_anomalous để ngày đó không nhiễm baseline.
CREATE TABLE health_anomaly_days (
    profile_id      INTEGER NOT NULL,
    day             TEXT NOT NULL,
    PRIMARY KEY (profile_id, day)
);
```

Mean và độ lệch chuẩn của 7 ngày được tính trực tiếp từ `health_daily` khi cần, thay vì cache thêm một bảng `user_health_baseline`. Với quy mô chỉ vài chục dòng của prototype, cách này đơn giản hơn và tránh tạo ra hai nguồn dữ liệu có thể lệch nhau.

Trong các lần chạy thử, cơ sở dữ liệu chỉ ở mức vài chục KB nên chi phí lưu trữ không đáng kể. SQLite cũng giúp việc tổng hợp theo ngày và xóa dữ liệu cũ rõ ràng hơn so với một file JSON tự quản lý.

Chính sách lưu 14 ngày áp dụng đồng thời cho `health_samples`, `health_daily` và `health_anomaly_days`. Khoảng này dài hơn cửa sổ baseline 7 ngày để vẫn gom đủ dữ liệu cho người không lái xe hằng ngày. Nếu không đủ ngày hợp lệ, hệ thống dùng baseline tạm thời với ít ngày hơn thay vì giữ dữ liệu quá hạn.

## 5. Phát hiện độ lệch (anomaly detection)

Mỗi phiên đo mới, với từng feature *f* có dữ liệu, trong cùng `light_bucket` và `profile_id`:

```
z(f) = (value_phiên_này(f) − mean_7d(f)) / max(std_7d(f), ε)
```

Mỗi feature có một hướng lệch cần chú ý, được định nghĩa trong `health_baseline.py` và dùng chung với trigger T7 của Khối 1:

| Feature | Hướng xấu |
|---|---|
| `eye_darkness` | z âm (vùng mắt tối hơn = quầng thâm đậm hơn) |
| `eye_openness` | z âm (mắt mở kém hơn) |
| `eye_puffiness` | z dương (sưng hơn) |
| `skin_paleness` | z âm (kênh a tụt = tái hơn) |
| `lip_color_index` | z âm (môi mất sắc đỏ = tím tái) |
| `blink_rate`, `perclos`, `yawn_rate` | z dương (chớp/nhắm/ngáp nhiều hơn) |

Cờ anomaly bật theo một trong hai cách. Cách thứ nhất là có ít nhất hai feature cùng lệch theo hướng cần chú ý với |z| > 2.0. Cách thứ hai là một feature lệch mạnh với |z| > 3.0 và lặp lại trong hai phiên liên tiếp. Nhánh thứ hai giúp không bỏ sót trường hợp chỉ một dấu hiệu thay đổi rõ, còn yêu cầu lặp lại giúp loại bớt nhiễu của một lần đo đơn lẻ.

Một số quy tắc giúp baseline không tự học nhầm dữ liệu bất thường:

- Cold start: hệ thống chỉ bắt đầu so sánh sau khi có dữ liệu của ít nhất 3 ngày. Baseline chưa đủ 7 ngày mang cờ `is_provisional` và chỉ dùng cho lời nhắc nhẹ, không dùng để khẳng định người lái "khác hẳn ngày thường".
- Baseline là rolling mean/std trên 7 ngày hợp lệ gần nhất, nên có thể thích nghi dần với những thay đổi tự nhiên. Tuy nhiên, ngày đã bị cảnh báo được đánh dấu `is_anomalous=1` và loại khỏi cửa sổ; nếu không, một trạng thái bất thường kéo dài vài ngày có thể vô tình trở thành mức bình thường mới.
- Khi phát hiện độ lệch, pipeline tạo trigger T7. VLM chỉ nhận phần chênh lệch dưới dạng text, chẳng hạn `eye_darkness ... 2.4σ`, cùng với frame hiện tại còn ở trong RAM. Ảnh lịch sử không được đưa vào VLM vì chúng chưa từng được ghi lại.

## 6. Checklist privacy: cái đã có và cái cần tích hợp production

| Nguyên tắc | Prototype hiện tại | Production cần thêm |
|---|---|---|
| Data minimization | Chỉ 8 số float/phiên; không lưu ảnh/video/face embedding | Review từng feature theo DPIA |
| Storage limitation | Tự xóa sample, daily và anomaly marker quá 14 ngày | Job nền theo lifecycle của thiết bị |
| On-device processing | DB và CV ở local; VLM gọi localhost khi bật | Chặn endpoint ngoài thiết bị bằng deployment policy |
| Security | SQLite thường, file nằm dưới `data/` | SQLCipher/EncryptedFile + khóa Keystore/TPM |
| Right to erasure | Có thể xóa file DB thủ công | API/UI xóa theo profile và audit sự kiện xóa |
| Consent | Chưa có UI trong repo Python | Opt-in riêng cho health trend, tắt được bất kỳ lúc nào |
| Transparency | README/schema công khai dữ liệu lưu | Màn hình settings hiển thị feature và retention |

Điểm cốt lõi của thiết kế là chỉ giữ dữ liệu phục vụ trực tiếp cho phép so sánh, đồng thời loại bỏ ảnh và face embedding khỏi nơi lưu trữ. Điều đó làm giảm đáng kể rủi ro nhận dạng hoặc tái dựng so với ảnh gốc, nhưng không biến các feature thành dữ liệu vô danh. Khi đã gắn với `profile_id`, chúng vẫn là dữ liệu cá nhân và phải được mã hóa, kiểm soát quyền truy cập cũng như cho phép người dùng xóa trong bản production.
