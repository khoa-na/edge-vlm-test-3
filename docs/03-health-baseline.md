# Khối 3 — Long-term Health Memory Index (Privacy-Preserving Baseline)

## 1. Bài toán

Phát hiện "hôm nay bất thường so với *chính người này* mọi ngày" (mắt sưng, sắc mặt tái, môi tím, quầng thâm đậm hơn) — nghĩa là cần bộ nhớ dài hạn nhiều ngày. Ràng buộc: **không lưu bất kỳ ảnh khuôn mặt nào**, chạy hoàn toàn trên thiết bị, tuân thủ GDPR/Nghị định 13/2023/NĐ-CP về dữ liệu cá nhân.

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

Pipeline trích feature dùng chính landmark + crop vùng mặt của Tier 1, cộng thêm thống kê màu đơn giản — chi phí không đáng kể, gắn vào chu kỳ T6 (5 phút/lần) và pre-ride check.

## 3. Bộ feature trích xuất (mỗi phiên đo)

| Feature | Cách tính (on-device) | Bắt bất thường gì |
|---|---|---|
| `eye_darkness` | Độ sáng trung bình (kênh L trong Lab) vùng dưới hốc mắt, chia cho độ sáng má (tự chuẩn hóa ánh sáng) | Quầng thâm đậm hơn ngày thường |
| `eye_openness` | EAR trung bình khi mắt "mở" trong phiên | Mắt sưng / lờ đờ (mở không hết) |
| `eye_puffiness` | Diện tích vùng mí dưới / (khoảng cách hai đồng tử)² — cả tử và mẫu cùng đơn vị pixel², tỉ lệ không phụ thuộc độ phân giải và khoảng cách camera | Mắt sưng |
| `skin_paleness` | Giá trị kênh a trong Lab (trục xanh lục–đỏ; a thấp = ít sắc hồng = tái) vùng má, lấy tỉ lệ với vùng trán để tự chuẩn hóa | Sắc mặt tái nhợt |
| `lip_color_index` | a_môi / (a_má + ε) — sắc đỏ môi tương đối so với má, ε tránh chia gần 0 | Môi tím tái (a_môi tụt về phía âm/xanh) |
| `blink_rate` | Số lần chớp/phút (từ Tier 1) | Mệt mỏi, khô mắt |
| `perclos` | % thời gian mắt nhắm trong phiên | Buồn ngủ tích lũy |
| `yawn_rate` | Số lần ngáp/10 phút | Mệt mỏi |
| `ambient_light` | Ước lượng lux từ exposure camera / cảm biến | **Biến kiểm soát** — không phải feature sức khỏe |

**Chuẩn hóa ánh sáng là bắt buộc**: "tái" dưới đèn trắng và dưới nắng chiều khác hẳn nhau. Hai tuyến phòng: (1) các feature màu đều là *tỉ lệ tương đối* giữa hai vùng trên cùng khuôn mặt (môi/má, hốc mắt/má) thay vì giá trị tuyệt đối; (2) lưu kèm `ambient_light`, chỉ so sánh baseline giữa các phiên có điều kiện sáng tương đương (bucket theo dải lux), phiên thiếu sáng quá thì bỏ, không ghi.

**Quality gate che khuất (ROI occlusion)**: chính yêu cầu pre-ride (đeo khẩu trang, kính) làm mất ROI — khẩu trang che môi + má, kính râm che mắt + quầng mắt. Mỗi phiên đo, Tier 1 đã có sẵn kết quả detect mask/sunglasses và landmark confidence: feature nào có ROI bị che hoặc landmark confidence < 0.7 thì ghi `NULL` cho feature đó (không ghi giá trị rác), các feature còn lại vẫn ghi bình thường. Anomaly detection chỉ chạy trên feature có dữ liệu; phiên mất quá nửa số feature thì bỏ cả phiên.

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

Toàn bộ DB vài chục KB — vừa với mọi thiết bị nhúng. Có thể thay bằng file JSON nếu không có SQLite; schema giữ nguyên khái niệm.

Retention 14 ngày cho `health_samples` (dài hơn cửa sổ baseline 7 ngày) có chủ đích: người không lái xe hằng ngày cần biên độ để gom đủ **7 ngày có dữ liệu**; quá 14 ngày thì chấp nhận baseline ít ngày hơn (provisional) thay vì giữ dữ liệu lâu thêm — cân bằng với nguyên tắc data minimization.

## 5. Phát hiện độ lệch (Anomaly Detection)

Mỗi phiên đo mới, với từng feature *f* có dữ liệu, trong cùng `light_bucket` và `profile_id`:

```
z(f) = (value_phiên_này(f) − mean_7d(f)) / max(std_7d(f), ε)
```

Mỗi feature định nghĩa sẵn **hướng xấu** trong config (dùng chung với trigger T7 của Khối 1):

| Feature | Hướng xấu |
|---|---|
| `eye_darkness` | z âm (vùng mắt tối hơn = quầng thâm đậm hơn) |
| `eye_openness` | z âm (mắt mở kém hơn) |
| `eye_puffiness` | z dương (sưng hơn) |
| `skin_paleness` | z âm (kênh a tụt = tái hơn) |
| `lip_color_index` | z âm (môi mất sắc đỏ = tím tái) |
| `blink_rate`, `perclos`, `yawn_rate` | z dương (chớp/nhắm/ngáp nhiều hơn) |

**Cờ anomaly** (điều kiện HOẶC — bắt được cả bất thường đơn lẻ rõ rệt lẫn suy giảm tổng thể):
- ≥ 2 feature cùng theo hướng xấu với |z| > 2.0 (đồng thuận nhiều tín hiệu vừa), **hoặc**
- 1 feature theo hướng xấu với |z| > 3.0 **lặp lại ≥ 2 phiên đo liên tiếp** (một tín hiệu mạnh + bền — bắt trường hợp "chỉ môi tím tái" mà không cần tín hiệu khác, persistence lọc nhiễu 1 phiên).

Các quy tắc khác:
- **Cold start**: cần tối thiểu 3 ngày có dữ liệu trước khi bật so sánh; baseline có `day_count` < 7 mang cờ `is_provisional` — chỉ dùng cho nhắc nhở nhẹ nhất (severity thấp), không dùng cho kết luận "khác hẳn ngày thường".
- **Cập nhật baseline có bảo vệ**: baseline là rolling mean/std đúng nghĩa trên 7 ngày hợp lệ gần nhất (khớp yêu cầu đề bài), tự trôi theo người dùng khi cửa sổ trượt (rám nắng dần, thay kính mới); **ngày bị cờ anomaly đánh dấu `is_anomalous=1` và loại khỏi cửa sổ** (tránh "ốm 3 ngày liền thành bình thường mới").
- Khi có cờ → phát Trigger T7 (Khối 1) → VLM nhận **delta dạng text** ("eye_darkness cao hơn baseline 2.4σ, lip_color thấp hơn 2.1σ") + frame hiện tại trong RAM → sinh lời nhắc → qua Guardrails (Khối 2). VLM không bao giờ thấy ảnh lịch sử — vì ảnh lịch sử không tồn tại.

## 6. Checklist tuân thủ Privacy (GDPR / luật dữ liệu cá nhân)

| Nguyên tắc | Thực hiện |
|---|---|
| Data minimization | Chỉ 8 số float/phiên; không ảnh, không video, không embedding khuôn mặt (embedding vẫn là dữ liệu sinh trắc — cũng không lưu) |
| Storage limitation | Rolling 14 ngày, tự xóa; baseline chỉ giữ mean/std |
| On-device processing | Không có network call nào trong pipeline; dữ liệu không rời thiết bị |
| Security | SQLite mã hóa at-rest (SQLCipher / Android EncryptedFile), khóa trong Keystore/TPM |
| Right to erasure | Nút "Xóa dữ liệu sức khỏe" trong settings → DROP cả 2 bảng, hiệu lực tức thời |
| Consent | Opt-in riêng cho tính năng health trend khi onboarding (tách khỏi consent DMS an toàn); tắt được bất kỳ lúc nào |
| Transparency | Màn hình settings hiển thị đúng những gì đang lưu (8 con số, ngày gần nhất) |

Điểm mấu chốt để trình bày với người chấm: **các feature vô hướng này không thể đảo ngược thành khuôn mặt** — khác về bản chất với face embedding (vector nhận dạng, thuộc dữ liệu sinh trắc học theo GDPR Art. 9). Vì vậy hệ thống nhớ được "trạng thái mọi ngày" của người dùng mà không hề giữ dữ liệu nhận dạng sinh trắc.
