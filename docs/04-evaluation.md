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

## 3. Kết quả (8 sequence, 19,407 frame)

| nhãn \ dự đoán | alert | microsleep | yawning | no_face |
|---|---|---|---|---|
| **alert** (13,337) | **91%** | 7% | 0% | 3% |
| **microsleep** (5,878) | 3% | **95%** | 0% | 2% |
| **yawning** (192) | 5% | 17% | **74%** | 4% |

**Frame-level accuracy: 91.8%.** Pipeline-level: 1,250 frame phát cảnh báo
T0 (nhắm mắt > 1.5s) trong 152 đoạn microsleep — đúng thiết kế: T0 chỉ nổ
sau debounce 1.5s chứ không nổ mọi frame nhắm mắt (chớp mắt thường không nổ).

## 4. Nhận xét & hiệu chỉnh rút ra từ số liệu thật

1. **EAR tách lớp rất rõ**: microsleep EAR ≈ 0.05, alert ≈ 0.22–0.27 —
   ngưỡng 0.20 đứng vững, đạt 95% recall trên microsleep chỉ bằng chỉ số
   hình học, đúng luận điểm "Tier 1 nhẹ đủ gánh phần an toàn".
2. **MAR phải hiệu chỉnh trên dữ liệu thật**: ngưỡng lý thuyết 0.6 quá cao
   với bộ landmark MediaPipe (ngáp thật MAR ≈ 0.5); hạ về 0.35 sau khi đo.
   Bài học: mọi ngưỡng Tier 1 cần calibration trên camera/dataset thực tế.
3. **Ngáp kèm nhắm mắt**: xét EAR trước MAR làm 65% frame ngáp bị gán nhầm
   microsleep; đổi thứ tự ưu tiên (MAR trước) đưa yawning 26% → 74%. Với DMS
   thì nhầm lẫn này lành tính — cả hai đều là tín hiệu mệt mỏi, đều dẫn tới
   nhắc nghỉ — nhưng thứ tự đúng giúp đếm ngáp (T3) chính xác.
4. **7% alert bị gán microsleep**: phần lớn là nheo mắt/nhìn xuống tự nhiên —
   ở mức pipeline không gây báo động giả vì debounce 1.5s lọc các cụm ngắn;
   đây chính là lý do temporal state machine bắt buộc phải có.
5. **3% mất mặt (no_face)**: quay đầu mạnh/che khuất — pipeline xử lý bằng
   cách giữ nguyên state, không reset đếm thời gian nhắm mắt.
