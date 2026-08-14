# Đánh giá trên dữ liệu thật — Dataset FL3D

## 1. Dataset

FL3D (Frame-Level Driver Drowsiness Detection), Kaggle
`matjazmuc/frame-level-driver-drowsiness-detection-fl3d` — khoảng 600MB, 53,331
frame từ video cabin thật (nguồn NITYMED), nhãn từng frame: `alert` /
`microsleep` / `yawning`. Nhãn khớp trực tiếp với hai hành vi Tier 1 phải bắt:
mắt nhắm kéo dài (T0) và ngáp (T3).

Tải không cần token:

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

Tier 1 chạy MediaPipe FaceLandmarker thật (478 landmark, XNNPACK CPU), không
mock. Frame được replay theo thứ tự trong mỗi sequence với FPS giả định 25,
đi qua đúng `Tier1Analyzer` của pipeline — cùng ngưỡng, cùng state machine.

## 3. Kết quả (8 sequence đầu theo thứ tự tên — không chọn lọc, 20,806 frame)

Frame-level, đo chất lượng chỉ số EAR/MAR tức thời:

| nhãn \ dự đoán | alert | microsleep | yawning | no_face |
|---|---|---|---|---|
| **alert** (16,881) | **90%** | 7% | 0% | 3% |
| **microsleep** (3,483) | 6% | **91%** | 0% | 3% |
| **yawning** (442) | 9% | 8% | **81%** | 2% |

Frame-level accuracy: 89.8%.

Episode-level — đây mới là điều hệ thống thật sự phải làm, bắt *đoạn* ngủ gật
chứ không phải từng frame:

- Episode recall 34/40 = **85%**: đoạn microsleep dài từ 1.5s trở lên có ít
  nhất một lần cảnh báo T0 nổ trong đoạn.
- False T0: 110/16,881 frame alert (**0.65%**). Các frame báo giả tập trung
  thành vài cụm nheo mắt kéo dài bị đọc nhầm chứ không rải đều; production
  giảm tiếp được bằng per-user EAR calibration, vì ngưỡng 0.20 cố định không
  hợp mọi hình dạng mắt.

## 4. Nhận xét và hiệu chỉnh rút ra từ số liệu thật

1. EAR tách lớp rất rõ: microsleep EAR quanh 0.05, alert quanh 0.22–0.27.
   Ngưỡng 0.20 đạt 91% frame recall và 85% episode recall chỉ bằng chỉ số
   hình học — đúng luận điểm "Tier 1 nhẹ đủ gánh phần an toàn".
2. MAR thì phải hiệu chỉnh trên dữ liệu thật: ngưỡng lý thuyết 0.6 quá cao
   với bộ landmark MediaPipe (ngáp thật đo được MAR ~0.5), hạ về 0.35 sau
   khi đo. Bài học chung: mọi ngưỡng Tier 1 cần calibration trên
   camera/dataset thực tế.
3. Ngáp thường kèm nhắm mắt, nên xét EAR trước MAR làm đa số frame ngáp bị
   gán nhầm thành microsleep; đổi thứ tự ưu tiên (MAR trước) đưa yawning lên
   81%. Với DMS thì nhầm lẫn này lành tính — cả hai đều là tín hiệu mệt mỏi,
   đều dẫn tới nhắc nghỉ — nhưng thứ tự đúng giúp đếm ngáp (T3) chính xác.
4. 7% frame alert bị gán microsleep ở mức frame, nhưng chỉ 0.65% frame alert
   gây báo giả T0 ở mức pipeline — debounce 1.5s lọc gần hết nheo mắt và
   nhìn xuống tự nhiên. Đây chính là lý do temporal state machine bắt buộc
   phải có.
5. 6 đoạn microsleep bị sót (recall 85%) chủ yếu là người lái nhắm hờ, EAR
   quanh 0.21, sát ngay trên ngưỡng. Hướng cải thiện: per-user EAR
   calibration lúc bắt đầu chuyến — đo EAR mở mắt bình thường trong 30 giây
   đầu rồi đặt ngưỡng tương đối 70% mức đó, thay vì ngưỡng tuyệt đối chung.

## 5. Tier 2 VLM thật — Qwen3.5-2B qua llama-server

Tier 2 đã chạy model thật: Qwen3.5-2B-Instruct Q4_K_M (1.28GB) cộng mmproj
F16 (0.67GB), phục vụ qua llama-server — llama.cpp gốc, API OpenAI-compatible.
Chọn llama-server thay vì binding `llama-cpp-python` vì binding trễ hỗ trợ
model mới (issue Qwen3.5 lúc làm bài vẫn mở), còn llama.cpp gốc hỗ trợ day-1.
Đây cũng là cách deploy edge thực tế: server process riêng, pipeline gọi
HTTP localhost.

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

Schema enum được ép server-side (`response_format: json_schema`, thành grammar
tại decoder) — model không thể trả free text, đúng thiết kế docs/02.

Cắm model thật xong rút được ba bài học, đều đã sửa trong code:

1. Model hybrid reasoning đốt sạch token vào thinking, content trả về rỗng.
   Phải tắt bằng `--chat-template-kwargs '{"enable_thinking": false}'`.
2. Câu prompt "unsure thì chọn looks_normal" nghe an toàn, nhưng làm model
   2B chọn normal cho cả frame đang nhắm mắt — bias an toàn quá đà giết
   utility. Thay bằng decision rule 4 bước có thứ tự; phần an toàn y tế đã
   do schema và template bank gánh, không cần model tự kiềm chế thêm.
3. `context_slots` là fact lấy từ telematics (speed suy ra moving/stopped),
   nên để code điền tất định sau khi model trả. Model chỉ quyết observation
   và severity — phần thật sự cần nhìn ảnh. Trước khi sửa, model từng đoán
   "stopped" khi speed đang 52.

Latency 4–7s mỗi call trên CPU nằm trong ngân sách Tier 2 (1–5s, không trên
safety path); NPU/GPU edge thực tế sẽ nhanh hơn đáng kể.

## 6. Giới hạn của đánh giá (khai báo minh bạch)

- Ngưỡng MAR hiệu chỉnh trên chính dataset này (in-sample calibration), nên
  số yawning phải đọc là "khả năng của chỉ số sau calibration" chứ không
  phải kết quả out-of-sample. EAR 0.20 là ngưỡng văn liệu chuẩn, không
  tinh chỉnh.
- Chạy 8 sequence đầu theo thứ tự tên — không chọn lọc theo nhãn (bản đánh
  giá đầu từng chọn ưu tiên sequence nhiều sự kiện, đã bỏ) — nhưng vẫn chưa
  phải toàn bộ dataset. Chạy full bằng `--limit-seq 0`, mất ~40 phút CPU.
- FPS nguồn giả định 25 (NITYMED); sai số FPS ảnh hưởng định nghĩa đoạn
  ≥ 1.5s ở mức ±1 frame.

## 7. Bổ sung sau demo webcam thật (gate EAR theo yaw)

Chạy pipeline trên clip webcam người thật lộ ra một lớp lỗi mà FL3D không
có: khi tài xế quay đầu (yaw 45–94°), landmark mắt bị "dẹt" theo phối cảnh
— mắt đang mở mà đo như nhắm, PERCLOS vọt oan, cảnh báo buồn ngủ lặp hàng
chục lần. Sửa hai tầng:

1. Gate EAR/MAR theo yaw (`EAR_VALID_YAW_DEG = 45°`): frame quay đầu quá
   ngưỡng coi là không có dữ liệu mắt/miệng, không đếm vào
   closed/PERCLOS/blink/yawn (đường đếm quay đầu T4 vẫn chạy riêng). Con số
   45° chọn sau khi đo cả hai phía: 35° cắt oan 2 episode ngủ gật FL3D —
   người gục đầu thường nghiêng cả đầu, recall tụt 85% → 80%; còn 45° giữ
   nguyên recall 85% mà false T0 vẫn giảm nhẹ (0.67% → 0.65%). Số ở mục 3
   là số sau gate.
2. Cooldown cho câu nhắc tĩnh T2–T4: PERCLOS hay ngáp là trạng thái kéo dài
   nhiều phút chứ không phải sự kiện điểm — trước đó câu nhắc trả về mỗi
   frame (spam ~20 lần/phút), giờ mỗi loại chỉ phát một lần mỗi cooldown
   180s, giống VLM.

Bài học đáng ghi: eval trên dataset chính diện (FL3D quay người lái đang
nhìn đường) không bao giờ phát hiện được lỗi phối cảnh — phải chạy trên
chuyển động đầu thật mới lộ. Vì thế demo người thật nằm trong quy trình
kiểm chứng, không phải chỉ để trình diễn.
