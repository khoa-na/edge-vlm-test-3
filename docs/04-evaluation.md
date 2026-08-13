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

## 5. Tier 2 VLM thật — Qwen3.5-2B qua llama-server

Tier 2 đã chạy model thật: **Qwen3.5-2B-Instruct Q4_K_M** (1.28GB) + mmproj
F16 (0.67GB), phục vụ qua **llama-server** (llama.cpp gốc, API
OpenAI-compatible). Chọn llama-server thay vì binding `llama-cpp-python` vì
binding trễ hỗ trợ model mới (issue Qwen3.5 còn mở), còn llama.cpp gốc hỗ
trợ day-1 — và đây cũng là cách deploy edge thực tế (server process riêng,
pipeline gọi HTTP localhost).

```bash
llama-server -m models/qwen3.5-2b-q4_k_m.gguf \
  --mmproj models/qwen3.5-2b-mmproj-f16.gguf --port 8090 -c 4096 \
  --chat-template-kwargs '{"enable_thinking": false}'
python -m src.run_video --input clip.mp4 --vlm-url http://127.0.0.1:8090
```

Kết quả trên frame FL3D thật (CPU, 8 thread):

| Case | VLM trả (JSON enum) | Câu ra loa | Latency |
|---|---|---|---|
| Frame microsleep + delta baseline + lái 125' trời 35°C | `eyes_heavy \| recommend_rest_now` | "Bạn lái đã lâu dưới trời nóng và mắt có vẻ mỏi, tấp vào chỗ mát nghỉ vài phút cho tỉnh táo nhé." (audio_short) | ~4–7s |
| Frame alert, không delta | `looks_normal \| none` | (không nhắc — đúng) | ~4s |

Schema enum được ép server-side (`response_format: json_schema` → grammar
tại decoder) — model **không thể** trả free text, đúng thiết kế docs/02.

Ba bài học khi cắm model thật (đã sửa trong code):

1. **Model hybrid reasoning đốt token vào thinking** → content rỗng; phải tắt
   bằng `--chat-template-kwargs '{"enable_thinking": false}'`.
2. **Prompt "unsure thì chọn looks_normal" làm model 2B chọn normal cho cả
   frame nhắm mắt** — bias an toàn quá đà giết utility. Thay bằng decision
   rule 4 bước có thứ tự; an toàn y tế đã do schema + template bank gánh,
   không cần model tự kiềm chế.
3. **`context_slots` là fact từ telematics** (speed → moving/stopped) — code
   điền tất định sau khi model trả, model chỉ quyết observation + severity
   (phần thật sự cần nhìn ảnh). Model từng đoán "stopped" khi speed 52.

Latency 4–7s/call CPU khớp ngân sách Tier 2 (1–5s, không nằm trên safety
path); trên NPU/GPU edge thực tế sẽ nhanh hơn đáng kể.

## 6. Giới hạn của đánh giá (khai báo minh bạch)

- **Ngưỡng MAR hiệu chỉnh trên chính dataset này** (in-sample calibration) —
  số yawning nên đọc là "khả năng của chỉ số sau calibration", không phải
  kết quả out-of-sample. EAR 0.20 là ngưỡng văn liệu chuẩn, không tinh chỉnh.
- **8/số sequence đầu theo thứ tự tên** — không chọn lọc theo nhãn (đã bỏ
  cách chọn ưu tiên sequence nhiều sự kiện ở bản đánh giá đầu), nhưng vẫn
  chưa phải toàn bộ dataset; chạy full bằng `--limit-seq 0` (~40 phút CPU).
- FPS nguồn giả định 25 (NITYMED); sai số FPS ảnh hưởng định nghĩa đoạn
  ≥ 1.5s ở mức ±1 frame.
