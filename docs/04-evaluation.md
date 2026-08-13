# Đánh giá trên dữ liệu thật — Dataset FL3D

## 1. Dataset

**FL3D (Frame-Level Driver Drowsiness Detection)** — Kaggle
`matjazmuc/frame-level-driver-drowsiness-detection-fl3d` (~600MB, 53,331 frame
từ video cabin thật, nguồn NITYMED). Nhãn **từng frame**: `alert` /
`microsleep` / `yawning` — khớp trực tiếp với hai hành vi Tier 1 phải bắt
(mắt nhắm kéo dài → T0, ngáp → T3).

Tải (không cần token):

```python
import kagglehub
kagglehub.dataset_download("matjazmuc/frame-level-driver-drowsiness-detection-fl3d")
```

## 2. Cách chạy

```bash
# tải model landmark (1 lần, ~3.7MB)
curl -sL -o models/face_landmarker.task --create-dirs \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

.venv/bin/python -m src.eval_fl3d --limit-seq 8   # ~10 phút CPU
```

Tier 1 chạy **MediaPipe FaceLandmarker thật** (478 landmark, XNNPACK CPU),
không mock. Replay frame theo thứ tự trong mỗi sequence @ 25 FPS giả định,
qua đúng `Tier1Analyzer` của pipeline (cùng ngưỡng, cùng state machine).

## 3. Kết quả (8 sequence đầu theo thứ tự tên — không chọn lọc, 20,806 frame)

**Frame-level** (chất lượng chỉ số EAR/MAR tức thời):

| nhãn \ dự đoán | alert | microsleep | yawning | no_face |
|---|---|---|---|---|
| **alert** (16,881) | **90%** | 7% | 0% | 3% |
| **microsleep** (3,483) | 6% | **91%** | 0% | 3% |
| **yawning** (442) | 9% | 8% | **81%** | 2% |

Frame-level accuracy: **89.8%**.

**Episode-level** (điều hệ thống thật sự phải làm — bắt *đoạn* ngủ gật,
không phải từng frame):

- **Episode recall: 34/40 = 85%** — đoạn microsleep ≥ 1.5s có ít nhất 1 lần
  cảnh báo T0 nổ trong đoạn.
- **False T0: 113/16,881 frame alert (0.67%)** — các frame báo giả tập trung
  thành vài cụm nheo mắt kéo dài bị đọc nhầm, không rải đều; production giảm
  tiếp bằng per-user EAR calibration (ngưỡng 0.20 cố định không hợp mọi
  hình dạng mắt).

## 4. Nhận xét & hiệu chỉnh rút ra từ số liệu thật

1. **EAR tách lớp rất rõ**: microsleep EAR ≈ 0.05, alert ≈ 0.22–0.27 —
   ngưỡng 0.20 đạt 91% frame recall và 85% episode recall chỉ bằng chỉ số
   hình học, đúng luận điểm "Tier 1 nhẹ đủ gánh phần an toàn".
2. **MAR phải hiệu chỉnh trên dữ liệu thật**: ngưỡng lý thuyết 0.6 quá cao
   với bộ landmark MediaPipe (ngáp thật MAR ≈ 0.5); hạ về 0.35 sau khi đo.
   Bài học: mọi ngưỡng Tier 1 cần calibration trên camera/dataset thực tế.
3. **Ngáp kèm nhắm mắt**: xét EAR trước MAR làm đa số frame ngáp bị gán nhầm
   microsleep; đổi thứ tự ưu tiên (MAR trước) đưa yawning lên 81%. Với DMS
   thì nhầm lẫn này lành tính — cả hai đều là tín hiệu mệt mỏi, đều dẫn tới
   nhắc nghỉ — nhưng thứ tự đúng giúp đếm ngáp (T3) chính xác.
4. **7% alert bị gán microsleep ở mức frame** nhưng chỉ 0.67% frame alert
   gây báo giả T0 ở mức pipeline — debounce 1.5s lọc gần hết nheo mắt/nhìn
   xuống tự nhiên; đây chính là lý do temporal state machine bắt buộc phải có.
5. **6 đoạn microsleep bị sót (recall 85%)**: chủ yếu người lái nhắm hờ
   (EAR ~0.21, sát trên ngưỡng) — hướng cải thiện là per-user EAR calibration
   khi bắt đầu chuyến (đo EAR mở mắt bình thường 30s đầu, đặt ngưỡng tương
   đối 70% mức đó) thay vì ngưỡng tuyệt đối chung.

## 5. Giới hạn của đánh giá (khai báo minh bạch)

- **Ngưỡng MAR hiệu chỉnh trên chính dataset này** (in-sample calibration) —
  số yawning nên đọc là "khả năng của chỉ số sau calibration", không phải
  kết quả out-of-sample. EAR 0.20 là ngưỡng văn liệu chuẩn, không tinh chỉnh.
- **8/số sequence đầu theo thứ tự tên** — không chọn lọc theo nhãn (đã bỏ
  cách chọn ưu tiên sequence nhiều sự kiện ở bản đánh giá đầu), nhưng vẫn
  chưa phải toàn bộ dataset; chạy full bằng `--limit-seq 0` (~40 phút CPU).
- FPS nguồn giả định 25 (NITYMED); sai số FPS ảnh hưởng định nghĩa đoạn
  ≥ 1.5s ở mức ±1 frame.
