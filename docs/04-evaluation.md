# Đánh giá trên dữ liệu thật — Dataset FL3D

## 1. Dataset

Tôi dùng FL3D (Frame-Level Driver Drowsiness Detection) trên Kaggle,
`matjazmuc/frame-level-driver-drowsiness-detection-fl3d`. Bộ dữ liệu có dung
lượng khoảng 600 MB, gồm 53.331 frame từ video cabin NITYMED và có nhãn theo
từng frame: `alert`, `microsleep`, `yawning`. Hai nhãn sau tương ứng trực tiếp
với mắt nhắm kéo dài (T0) và ngáp (T3), nên phù hợp để kiểm tra nhánh Tier 1.

Tải không cần token:

```python
import kagglehub
kagglehub.dataset_download("matjazmuc/frame-level-driver-drowsiness-detection-fl3d")
```

## 2. Cách chạy

```bash
# cài dependency đánh giá và tải model landmark
pip install -r requirements-eval.txt
./scripts/setup_models.sh tier1

.venv/bin/python -m src.eval_fl3d --limit-seq 8   # ~10 phút CPU
```

Phép đánh giá này dùng MediaPipe FaceLandmarker thật với 478 landmark và
XNNPACK CPU, không dùng mock. Các frame được phát lại đúng thứ tự trong từng
sequence ở FPS giả định là 25, sau đó đi qua chính `Tier1Analyzer` của
pipeline với cùng ngưỡng và state machine như lúc chạy demo.

## 3. Kết quả (8 sequence đầu theo thứ tự tên — không chọn lọc, 20,806 frame)

Frame-level, đo chất lượng chỉ số EAR/MAR tức thời:

| nhãn \ dự đoán | alert | microsleep | yawning | no_face |
|---|---|---|---|---|
| **alert** (16,881) | **90%** | 7% | 0% | 3% |
| **microsleep** (3,483) | 6% | **91%** | 0% | 3% |
| **yawning** (442) | 9% | 8% | **81%** | 2% |

Frame-level accuracy: 89.8%.

Ở mức episode, phép đo quan tâm hệ thống có bắt được cả *đoạn* ngủ gật hay
không, thay vì yêu cầu mọi frame trong đoạn đều được phân loại đúng:

- Episode recall 34/40 = **85%**: trong 40 đoạn microsleep dài từ 1,5 giây,
  34 đoạn có ít nhất một cảnh báo T0.
- False T0: 110/16,881 frame alert (**0.65%**). Các frame báo giả tập trung
  thành vài cụm nheo mắt kéo dài bị đọc nhầm chứ không rải đều; production
  giảm tiếp được bằng per-user EAR calibration, vì ngưỡng 0.20 cố định không
  hợp mọi hình dạng mắt.

`0.65%` ở đây là tỷ lệ frame mà state T0 đang bật, tương đương khoảng 586
frame-state/giờ nếu ngoại suy thô ở 25 FPS; nó **không phải** 586 lần phát loa
độc lập vì các frame nằm trong cụm và runner có dedupe/cooldown. Để công bố
false-alert SLA cần đếm episode cảnh báo độc lập trên nhiều giờ lái tự nhiên,
phép đo này chưa thay thế được.

## 4. Những điều rút ra từ số liệu

1. EAR phân tách hai lớp khá rõ: microsleep tập trung quanh 0,05, còn alert
   thường nằm trong khoảng 0,22–0,27. Với ngưỡng 0,20, Tier 1 đạt 91% recall
   theo frame và 85% recall theo episode. Kết quả này cho thấy một nhánh CV
   nhẹ đã có thể đảm nhiệm cảnh báo tức thời mà không phải chờ VLM.
2. MAR cần được hiệu chỉnh theo đúng landmark và camera sử dụng. Ngưỡng 0,6
   ban đầu quá cao vì các frame ngáp trong FL3D thường chỉ có MAR quanh 0,5;
   sau khi đo, tôi hạ ngưỡng xuống 0,35. Đây cũng là lý do các ngưỡng Tier 1
   cần được hiệu chuẩn lại trước khi chuyển sang phần cứng và camera khác.
3. Khi ngáp, người lái thường nhắm mắt cùng lúc. Nếu xét EAR trước MAR, nhiều
   frame ngáp bị gán thành microsleep; ưu tiên MAR trước đưa độ chính xác của
   lớp yawning lên 81%. Sự nhầm lẫn này không làm mất hoàn toàn tín hiệu mệt
   mỏi, nhưng ảnh hưởng đến bộ đếm ngáp T3 nên vẫn cần sửa.
4. Có 7% frame alert bị phân loại tức thời thành microsleep, trong khi tỷ lệ
   frame alert thực sự làm T0 bật chỉ còn 0,65%. Debounce 1,5 giây đã loại
   phần lớn các lần nheo mắt hoặc nhìn xuống ngắn, cho thấy state machine theo
   thời gian là một phần thiết yếu của hệ thống chứ không chỉ là lớp phụ.
5. Sáu episode bị bỏ sót chủ yếu là trường hợp người lái nhắm hờ với EAR
   quanh 0,21, ngay phía trên ngưỡng. Một hướng cải thiện hợp lý là đo EAR khi
   mắt mở trong 30 giây đầu chuyến đi, rồi đặt ngưỡng cho từng người ở khoảng
   70% mức đó thay vì dùng một giá trị tuyệt đối cho tất cả.

## 5. Tier 2 VLM thật — Qwen3.5-2B qua llama-server

Tier 2 đã được thử với model thật Qwen3.5-2B-Instruct Q4_K_M (1,28 GB) và
mmproj F16 (0,67 GB), phục vụ qua `llama-server` bằng API tương thích OpenAI.
Tôi chọn chạy server thành một process riêng để pipeline chỉ cần gọi HTTP
trên localhost; cách tách này cũng giúp lỗi hoặc độ trễ của VLM không giữ
vòng xử lý camera.

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

Đầu ra được ràng buộc ở phía server bằng `response_format: json_schema`, sau
đó chuyển thành grammar tại decoder. Vì vậy model chỉ chọn các giá trị enum,
không trực tiếp viết câu sẽ phát cho người lái như mô tả trong docs/02.

Việc chạy model thật làm lộ ra ba vấn đề không xuất hiện với mock; cả ba đã
được xử lý trong code:

1. Chế độ reasoning dùng hết ngân sách token cho phần thinking và đôi khi để
   `content` rỗng. Khi chạy server cần tắt chế độ này bằng
   `--chat-template-kwargs '{"enable_thinking": false}'`.
2. Chỉ dẫn "nếu không chắc thì chọn `looks_normal`" khiến model 2B chọn bình
   thường ngay cả khi frame cho thấy mắt đang nhắm. Tôi thay nó bằng quy tắc
   quyết định bốn bước có thứ tự. An toàn y tế vẫn do schema đóng và template
   bank đảm nhiệm, nên không cần đẩy trách nhiệm đó sang model.
3. `context_slots` là dữ kiện lấy từ telematics, ví dụ tốc độ quyết định xe
   đang chạy hay dừng. Code hiện điền các trường này sau khi model trả về;
   VLM chỉ chọn `observation` và `severity`, là hai phần thực sự cần nhìn ảnh.
   Trước thay đổi này, model từng trả `stopped` dù tốc độ đầu vào là 52 km/h.

Độ trễ đo được trên CPU là khoảng 4–7 giây cho mỗi lần gọi. T2–T7 hiện chạy
trên worker nền; T0/T1 không gọi VLM, còn các trigger thường có thể dùng câu
tĩnh đã duyệt trước. Do đó độ trễ trên không chặn vòng đọc camera. NPU hoặc
GPU có thể cải thiện tốc độ, nhưng repo này chưa có benchmark để khẳng định
mức cải thiện.

## 6. Giới hạn của đánh giá (khai báo minh bạch)

- Ngưỡng MAR được hiệu chỉnh trên chính FL3D, nên kết quả yawning phản ánh
  hiệu năng sau in-sample calibration, chưa phải đánh giá out-of-sample.
  Ngưỡng EAR 0,20 không được tinh chỉnh theo bộ dữ liệu này.
- Chạy 8 sequence đầu theo thứ tự tên — không chọn lọc theo nhãn (bản đánh
  giá đầu từng chọn ưu tiên sequence nhiều sự kiện, đã bỏ) — nhưng vẫn chưa
  phải toàn bộ dataset. Chạy full bằng `--limit-seq 0`, mất ~40 phút CPU.
- FPS nguồn giả định 25 (NITYMED); sai số FPS ảnh hưởng định nghĩa đoạn
  ≥ 1.5s ở mức ±1 frame.

## 7. Bổ sung sau khi thử bằng webcam

Clip webcam người thật cho thấy một lỗi mà FL3D không bộc lộ. Khi người lái
quay đầu với yaw khoảng 45–94°, landmark mắt bị dẹt theo phối cảnh; mắt vẫn
mở nhưng EAR giảm như đang nhắm, kéo PERCLOS tăng sai và làm lời nhắc buồn
ngủ lặp lại. Tôi xử lý ở hai chỗ:

1. Gate EAR/MAR theo yaw (`EAR_VALID_YAW_DEG = 45°`). Frame vượt ngưỡng được
   coi là thiếu dữ liệu mắt và miệng, nên không đi vào các bộ đếm
   closed/PERCLOS/blink/yawn; đường đếm quay đầu T4 vẫn chạy độc lập. Thử
   ngưỡng 35° làm mất hai episode FL3D và kéo recall từ 85% xuống 80%, còn
   45° giữ recall ở 85% đồng thời giảm false T0 từ 0,67% xuống 0,65%. Các số
   ở mục 3 là kết quả sau khi thêm gate này.
2. Thêm cooldown 180 giây cho lời nhắc tĩnh T2–T4. PERCLOS cao hoặc ngáp
   nhiều thường kéo dài qua nhiều frame, nên trước đó cùng một câu có thể
   được trả về khoảng 20 lần mỗi phút. Hiện mỗi loại chỉ phát một lần trong
   một khoảng cooldown, tương tự nhánh VLM.

FL3D chủ yếu ghi người lái ở góc chính diện nên khó làm lộ lỗi phối cảnh khi
quay đầu. Vì vậy, clip tự quay không chỉ phục vụ trình diễn mà còn bổ sung
một góc kiểm thử khác với dataset và đã trực tiếp dẫn tới thay đổi trong
pipeline.
