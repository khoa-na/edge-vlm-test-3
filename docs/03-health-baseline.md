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
| `eye_puffiness` | Tỉ lệ diện tích vùng mí dưới / khoảng cách hai mắt (từ landmark) | Mắt sưng |
| `skin_paleness` | Độ bão hòa màu (kênh a trong Lab — sắc hồng) vùng má, chuẩn hóa theo ước lượng ánh sáng môi trường | Sắc mặt tái nhợt |
| `lip_color_index` | Tỉ lệ đỏ/xanh (a/b trong Lab) vùng môi so với má | Môi tím tái |
| `blink_rate` | Số lần chớp/phút (từ Tier 1) | Mệt mỏi, khô mắt |
| `perclos` | % thời gian mắt nhắm trong phiên | Buồn ngủ tích lũy |
| `yawn_rate` | Số lần ngáp/10 phút | Mệt mỏi |
| `ambient_light` | Ước lượng lux từ exposure camera / cảm biến | **Biến kiểm soát** — không phải feature sức khỏe |

**Chuẩn hóa ánh sáng là bắt buộc**: "tái" dưới đèn trắng và dưới nắng chiều khác hẳn nhau. Hai tuyến phòng: (1) các feature màu đều là *tỉ lệ tương đối* giữa hai vùng trên cùng khuôn mặt (môi/má, hốc mắt/má) thay vì giá trị tuyệt đối; (2) lưu kèm `ambient_light`, chỉ so sánh baseline giữa các phiên có điều kiện sáng tương đương (bucket theo dải lux), phiên thiếu sáng quá thì bỏ, không ghi.

## 4. Schema `User_Health_Baseline` (SQLite, mã hóa at-rest)

```sql
-- Mỗi phiên đo (5 phút/lần khi lái + pre-ride): 1 dòng, chỉ chứa số
CREATE TABLE health_samples (
    sample_id       INTEGER PRIMARY KEY,
    ts              INTEGER NOT NULL,      -- epoch seconds
    light_bucket    INTEGER NOT NULL,      -- 0=tối 1=trong nhà 2=ngoài trời 3=nắng gắt
    eye_darkness    REAL, eye_openness REAL, eye_puffiness REAL,
    skin_paleness   REAL, lip_color_index REAL,
    blink_rate      REAL, perclos REAL, yawn_rate REAL
);
-- Retention: xóa dòng > 14 ngày (job hằng ngày). Không có cột ảnh/embedding.

-- Baseline tổng hợp: cập nhật cuối mỗi chuyến, per light_bucket
CREATE TABLE user_health_baseline (
    light_bucket    INTEGER NOT NULL,
    feature_name    TEXT    NOT NULL,      -- 'eye_darkness', ...
    mean_7d         REAL NOT NULL,         -- trung bình 7 ngày gần nhất
    std_7d          REAL NOT NULL,
    sample_count    INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    PRIMARY KEY (light_bucket, feature_name)
);
```

Toàn bộ DB vài chục KB — vừa với mọi thiết bị nhúng. Có thể thay bằng file JSON nếu không có SQLite; schema giữ nguyên khái niệm.

## 5. Phát hiện độ lệch (Anomaly Detection)

Mỗi phiên đo mới, với từng feature *f* trong cùng `light_bucket`:

```
z(f) = (value_hôm_nay(f) − mean_7d(f)) / max(std_7d(f), ε)
```

- **Cờ anomaly** khi có ≥ 2 feature cùng hướng xấu với |z| > 2.0 (một feature lệch đơn lẻ dễ là nhiễu — đòi hỏi đồng thuận).
- **Cold start**: cần tối thiểu 3 ngày × 5 phiên đo trước khi bật so sánh; trước đó hệ thống chỉ thu thập, không cảnh báo.
- **Cập nhật baseline có bảo vệ**: dùng EMA (α ≈ 0.15/ngày) thay vì mean cứng — baseline trôi từ từ theo người dùng (rám nắng dần, thay kính mới) nhưng **phiên bị cờ anomaly không được ghi vào baseline** (tránh "ốm 3 ngày liền thành bình thường mới").
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
