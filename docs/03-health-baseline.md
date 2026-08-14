# Khối 3 — Long-term Health Memory Index (Privacy-Preserving Baseline)

## 1. Bài toán

Cần phát hiện "hôm nay bất thường so với chính người này mọi ngày" — mắt sưng hơn, sắc mặt tái hơn, quầng thâm đậm hơn — tức là cần bộ nhớ dài hạn nhiều ngày. Ràng buộc: không được lưu bất kỳ ảnh khuôn mặt nào, chạy hoàn toàn trên thiết bị, tuân thủ GDPR và Nghị định 13/2023/NĐ-CP về dữ liệu cá nhân.

## 2. Nguyên tắc: lưu SỐ, không lưu ẢNH

```
Frame (RAM) ──► Trích xuất feature vô hướng (on-device, <1s)
                        │
                        ▼
              8-12 con số float (không đảo ngược được thành mặt)
                        │
                        ▼
              SQLite (mã hóa at-rest) — rolling 7-14 ngày
                        │
Frame bị hủy ngay       ▼
sau khi trích     So sánh z-score với baseline → cờ anomaly → Trigger T7 (Khối 1)
(không ghi disk)
```

Pipeline trích feature dùng lại chính landmark và crop vùng mặt của Tier 1, cộng thêm ít thống kê màu đơn giản — chi phí không đáng kể, gắn vào chu kỳ T6 (5 phút một lần) và pre-ride check.

## 3. Bộ feature trích xuất (mỗi phiên đo)

| Feature | Cách tính (on-device) | Bắt bất thường gì |
|---|---|---|
| `eye_darkness` | Độ sáng trung bình (kênh L trong Lab) vùng dưới hốc mắt, chia cho độ sáng má (tự chuẩn hóa ánh sáng) | Quầng thâm đậm hơn ngày thường |
| `eye_openness` | EAR trung bình khi mắt "mở" trong phiên | Mắt sưng / lờ đờ (mở không hết) |
| `eye_puffiness` | Diện tích vùng mí dưới / (khoảng cách hai đồng tử)² — tử và mẫu cùng đơn vị pixel² nên tỉ lệ không phụ thuộc độ phân giải và khoảng cách camera | Mắt sưng |
| `skin_paleness` | Giá trị kênh a trong Lab (trục xanh lục–đỏ; a thấp = ít sắc hồng = tái) vùng má, tỉ lệ với vùng trán để tự chuẩn hóa | Sắc mặt tái nhợt |
| `lip_color_index` | a_môi / (a_má + ε) — sắc đỏ môi tương đối so với má, ε tránh chia gần 0 | Môi tím tái (a_môi tụt về phía âm/xanh) |
| `blink_rate` | Số lần chớp/phút (từ Tier 1) | Mệt mỏi, khô mắt |
| `perclos` | % thời gian mắt nhắm trong phiên | Buồn ngủ tích lũy |
| `yawn_rate` | Số lần ngáp/10 phút | Mệt mỏi |
| `ambient_light` | Ước lượng lux từ exposure camera / cảm biến | Biến kiểm soát — không phải feature sức khỏe |

Chuẩn hóa ánh sáng là chuyện bắt buộc phải làm cho tử tế: "tái" dưới đèn trắng và "tái" dưới nắng chiều là hai thứ khác hẳn nhau. Ở đây phòng hai tuyến. Một là các feature màu đều tính theo tỉ lệ tương đối giữa hai vùng trên cùng khuôn mặt (môi so với má, hốc mắt so với má) thay vì giá trị tuyệt đối. Hai là lưu kèm `ambient_light` và chỉ so sánh baseline giữa các phiên có điều kiện sáng tương đương (bucket theo dải lux); phiên thiếu sáng quá thì bỏ luôn, không ghi.

Còn một quality gate cho che khuất (ROI occlusion): chính yêu cầu pre-ride — đeo khẩu trang, đeo kính — lại làm mất ROI, vì khẩu trang che môi và má, kính râm che mắt và quầng mắt. May là mỗi phiên đo Tier 1 đã có sẵn kết quả detect mask/sunglasses và landmark confidence. Feature nào có ROI bị che hoặc confidence dưới 0.7 thì ghi `NULL` thay vì ghi giá trị rác; các feature còn lại vẫn ghi bình thường. Anomaly detection chỉ chạy trên feature có dữ liệu, và phiên nào mất quá nửa số feature thì bỏ cả phiên.

## 4. Schema `User_Health_Baseline` (SQLite, mã hóa at-rest)

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

-- Baseline = mean/std ĐÚNG NGHĨA trên 7 ngày gần nhất có dữ liệu hợp lệ
-- (rolling window trên health_daily, loại ngày is_anomalous=1). Bảng này là
-- cache tính lại được, không phải nguồn sự thật.
CREATE TABLE user_health_baseline (
    profile_id      INTEGER NOT NULL,
    light_bucket    INTEGER NOT NULL,
    feature_name    TEXT    NOT NULL,
    mean_7d         REAL NOT NULL,         -- mean của day_value 7 ngày hợp lệ gần nhất
    std_7d          REAL NOT NULL,
    day_count       INTEGER NOT NULL,      -- số ngày thực có dữ liệu (3-7)
    is_provisional  INTEGER NOT NULL,      -- 1 nếu day_count < 7 (độ tin cậy thấp)
    updated_at      INTEGER NOT NULL,
    PRIMARY KEY (profile_id, light_bucket, feature_name)
);
```

Toàn bộ DB chỉ vài chục KB, vừa với mọi thiết bị nhúng. Không có SQLite thì thay bằng file JSON cũng được, khái niệm schema giữ nguyên.

Con số retention 14 ngày cho `health_samples` (dài hơn cửa sổ baseline 7 ngày) là có chủ đích: người không lái xe hằng ngày cần biên độ để gom đủ 7 ngày có dữ liệu. Quá 14 ngày thì chấp nhận baseline ít ngày hơn (provisional) chứ không giữ dữ liệu lâu thêm — chỗ này cân với nguyên tắc data minimization.

## 5. Phát hiện độ lệch (anomaly detection)

Mỗi phiên đo mới, với từng feature *f* có dữ liệu, trong cùng `light_bucket` và `profile_id`:

```
z(f) = (value_phiên_này(f) − mean_7d(f)) / max(std_7d(f), ε)
```

Mỗi feature định nghĩa sẵn hướng xấu trong config (dùng chung với trigger T7 của Khối 1):

| Feature | Hướng xấu |
|---|---|
| `eye_darkness` | z âm (vùng mắt tối hơn = quầng thâm đậm hơn) |
| `eye_openness` | z âm (mắt mở kém hơn) |
| `eye_puffiness` | z dương (sưng hơn) |
| `skin_paleness` | z âm (kênh a tụt = tái hơn) |
| `lip_color_index` | z âm (môi mất sắc đỏ = tím tái) |
| `blink_rate`, `perclos`, `yawn_rate` | z dương (chớp/nhắm/ngáp nhiều hơn) |

Cờ anomaly bật khi một trong hai điều kiện đúng: hoặc có ít nhất 2 feature cùng theo hướng xấu với |z| > 2.0 (nhiều tín hiệu vừa đồng thuận), hoặc có 1 feature theo hướng xấu với |z| > 3.0 lặp lại ít nhất 2 phiên đo liên tiếp. Điều kiện thứ hai để bắt được trường hợp "chỉ mỗi môi tím tái" mà không cần tín hiệu khác — yêu cầu bền qua 2 phiên chính là bộ lọc nhiễu.

Vài quy tắc vận hành đi kèm:

- Cold start: cần tối thiểu 3 ngày có dữ liệu mới bật so sánh. Baseline có `day_count` dưới 7 mang cờ `is_provisional`, chỉ dùng cho nhắc nhở nhẹ nhất, không dùng cho kết luận "khác hẳn ngày thường".
- Cập nhật baseline có bảo vệ: baseline là rolling mean/std đúng nghĩa trên 7 ngày hợp lệ gần nhất (khớp yêu cầu đề bài), tự trôi theo người dùng khi cửa sổ trượt — rám nắng dần hay thay kính mới đều thích nghi được. Nhưng ngày bị cờ anomaly được đánh dấu `is_anomalous=1` và loại khỏi cửa sổ, để tránh chuyện "ốm 3 ngày liền thì ốm thành bình thường mới".
- Khi có cờ, phát trigger T7 (Khối 1). VLM nhận delta ở dạng text ("eye_darkness cao hơn baseline 2.4σ, lip_color thấp hơn 2.1σ") cộng frame hiện tại đang trong RAM, sinh lời nhắc, đi qua Guardrails (Khối 2). VLM không bao giờ thấy ảnh lịch sử — đơn giản vì ảnh lịch sử không tồn tại.

## 6. Checklist tuân thủ privacy (GDPR / luật dữ liệu cá nhân)

| Nguyên tắc | Thực hiện |
|---|---|
| Data minimization | Chỉ 8 số float/phiên; không ảnh, không video, không embedding khuôn mặt (embedding vẫn là dữ liệu sinh trắc — cũng không lưu) |
| Storage limitation | Rolling 14 ngày, tự xóa; baseline chỉ giữ mean/std |
| On-device processing | Không có network call nào trong pipeline; dữ liệu không rời thiết bị |
| Security | SQLite mã hóa at-rest (SQLCipher / Android EncryptedFile), khóa trong Keystore/TPM |
| Right to erasure | Nút "Xóa dữ liệu sức khỏe" trong settings → DROP cả 3 bảng (`health_samples`, `health_daily`, `user_health_baseline`), hiệu lực tức thời |
| Consent | Opt-in riêng cho tính năng health trend khi onboarding (tách khỏi consent DMS an toàn); tắt được bất kỳ lúc nào |
| Transparency | Màn hình settings hiển thị đúng những gì đang lưu (8 con số, ngày gần nhất) |

Điểm quan trọng nhất của cả thiết kế: các feature vô hướng này không thể đảo ngược thành khuôn mặt, khác về bản chất với face embedding (vector nhận dạng, thuộc dữ liệu sinh trắc học theo GDPR Art. 9). Nhờ vậy hệ thống nhớ được "trạng thái mọi ngày" của người dùng mà không hề giữ dữ liệu nhận dạng sinh trắc nào.
