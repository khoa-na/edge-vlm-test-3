# Edge VLM Test #3 — Driver Safety & Health Monitor

Bài làm cho **Bài kiểm tra năng lực Edge VLM & Multimodal AI — Số 3**. Pipeline mock đáp ứng đầy đủ luồng bắt buộc của đề. Ngoài phần đó, repo còn có bản chạy với MediaPipe, YOLO và Qwen3.5-2B, kết quả đánh giá trên FL3D cùng một video webcam tự quay.

## Tóm tắt bài làm

| Phạm vi | Trạng thái | Bằng chứng |
|---|---|---|
| Two-tier event-driven cascade | Hoàn thành | `src/pipeline.py`, `docs/01-two-tier-cascade.md` |
| System prompt + anti-medical guardrails | Hoàn thành | `src/guardrails.py`, `src/vlm_backend.py`, `docs/02-medical-guardrails.md` |
| Health baseline 7 ngày, không lưu ảnh | Hoàn thành ở mức prototype | `src/health_baseline.py`, `docs/03-health-baseline.md` |
| TODO 1–4 trong skeleton | Hoàn thành | `src/pipeline.py` và các module `src/` |
| Bốn câu giải trình | Hoàn thành | `docs/05-giai-trinh.md` |
| Model/data/video thật — phần làm thêm | Có | `docs/04-evaluation.md`, `demo_webcam.mp4` |

Các kết quả có thể chạy lại từ repo:

- **58 unit test** cho temporal trigger, guardrails, async latency, persistence, retention, audio priority, TTS manifest và regression.
- **FL3D, 20.806 frame:** frame accuracy 89,8%; bắt 34/40 episode microsleep ≥1,5 giây, recall 85%.
- **Red-team mặc định:** 45 case post-filter, 0 vượt rào, 0 false-positive. Khi có llama-server: thêm 23 case prompt injection qua VLM thật, tổng 68 case.
- **Demo người thật:** [demo_webcam.mp4](demo_webcam.mp4), 143 giây, 640×480 @ 10 FPS, có audio cảnh báo.

## Chạy nhanh

Chỉ cần Python, không cần camera hay model:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m src.demo
# hoặc chạy cả test + demo smoke:
./scripts/smoke_test.sh
```

Chạy Tier 1 thật trên webcam/video:

```bash
.venv/bin/pip install -r requirements-demo.txt
./scripts/setup_models.sh tier1
.venv/bin/python -m src.run_webcam --audio
# hoặc không ghi baseline khi chỉ replay demo:
.venv/bin/python -m src.run_video --input clip.mp4 --db-path :memory:
```

## Thử với video của bạn

Sau khi cài `requirements-demo.txt` và chạy `./scripts/setup_models.sh tier1`
như ở trên, truyền đường dẫn video vào `--input`. Nên nhìn thẳng trong khoảng
5 giây đầu để pipeline hiệu chỉnh góc đặt camera.

Chạy MediaPipe và YOLO thật, đồng thời xuất video có overlay EAR, MAR,
PERCLOS, pitch/yaw và khung cảnh báo:

```bash
.venv/bin/python -m src.run_video \
  --input "/duong-dan/video-cua-ban.mp4" \
  --output result.mp4 \
  --db-path :memory:
```

`result.mp4` được ghi tại thư mục hiện tại. Console cũng in thời điểm và nội
dung từng cảnh báo. `--db-path :memory:` giúp lần thử không lưu health
baseline xuống máy.

Để ghép các WAV cảnh báo đã duyệt vào audio track của video kết quả, cài
`ffmpeg` và thêm `--audio-mux`:

```bash
ffmpeg -version
.venv/bin/python -m src.run_video \
  --input "/duong-dan/video-cua-ban.mp4" \
  --output result-with-audio.mp4 \
  --audio-mux \
  --db-path :memory:
```

Runner cũng nhận một thư mục ảnh bằng `--input /duong-dan/frames/ --fps 10`.
Nếu đã khởi động llama-server theo phần Tier 2 bên dưới, thêm
`--vlm-url http://127.0.0.1:8090` để dùng VLM thật; nếu không, Tier 1 vẫn chạy
đầy đủ và Tier 2 tự dùng mock.

Đánh giá lại FL3D:

```bash
.venv/bin/pip install -r requirements-eval.txt
.venv/bin/python -c "import kagglehub; kagglehub.dataset_download('matjazmuc/frame-level-driver-drowsiness-detection-fl3d')"
.venv/bin/python -m src.eval_fl3d --limit-seq 8
```

Chạy Tier 2 thật qua llama-server:

```bash
# yêu cầu binary llama-server của llama.cpp có trong PATH
llama-server --version
./scripts/setup_models.sh vlm  # khoảng 2 GB, file GGUF được .gitignore
llama-server -m models/qwen3.5-2b-q4_k_m.gguf \
  --mmproj models/qwen3.5-2b-mmproj-f16.gguf --port 8090 -c 4096 \
  --chat-template-kwargs '{"enable_thinking": false}'
.venv/bin/python -m src.run_video --input clip.mp4 \
  --vlm-url http://127.0.0.1:8090 --db-path :memory:
```

Kiểm tra model nào đang có:

```bash
./scripts/setup_models.sh check
```

## Kiến trúc

```text
Camera 5–10 FPS ──► Tier 1: MediaPipe + YOLO + temporal rules
                         │
             ┌───────────┴────────────┐
             │                        │
       T0/T1 khẩn cấp            T2–T7 event
       WAV duyệt sẵn             câu tĩnh ngay nếu cần
       không chờ VLM                    │
                                        ▼
                              Worker VLM một slot, async
                                        │
                              JSON enum constrained
                                        │
                              template bank + post-filter
```

Tier 1 chạy trên mọi frame. MediaPipe trích EAR/MAR và ước lượng pitch/yaw; YOLO26n FP32 `.pt` chạy mỗi ba frame rồi dùng lại confidence ở các frame xen giữa để giảm tải CPU. State machine theo thời gian dùng duration, debounce, cửa sổ trượt và cooldown để tránh kết luận từ một frame nhiễu.

Mọi lần gọi VLM từ T2–T5 và T7 đều chạy nền; T6 chỉ trích feature bằng CV. T0/T1 đi thẳng tới cảnh báo tĩnh; T5 lái lâu và T7 lệch baseline cũng có thể trả câu đã duyệt trước khi worker hoàn tất. Nếu loa đang phát một lời nhắc thường, T0/T1 có quyền ngắt để phát cảnh báo khẩn cấp, nhưng cùng một cảnh báo đang đọc sẽ không tự khởi động lại ở mỗi frame.

Tier 2 chỉ quyết định `observation` và `severity`. Code điền `context_slots` từ telematics rồi chọn câu tiếng Việt trong template bank; model không trực tiếp viết lời phát ra loa.

## Demo webcam — mốc xem nhanh

| Thời gian | Tình huống |
|---:|---|
| 00:06 | T5 — lái liên tục quá 60 phút, VLM fusion |
| 00:17 | T0 — nhắm mắt kéo dài |
| 00:47 | T3 — ngáp nhiều |
| 00:57 | T1 — điện thoại |
| 01:10 | T2 — PERCLOS cao |
| 01:28 | T4 — quay đầu lặp lại |

Video dùng MediaPipe, YOLO26n và Qwen3.5-2B trên một clip webcam tự quay; dữ liệu telematics được mô phỏng qua CLI. Phần overlay hiển thị EAR/MAR/pitch/yaw, còn audio track chứa các câu WAV đã duyệt sẵn.

## Kết quả và cách đọc số liệu

Phép đánh giá FL3D dùng tám sequence đầu theo thứ tự tên, không chọn theo nhãn. Script phát lại các frame đúng trình tự qua `Tier1Analyzer` thật:

| Chỉ số | Kết quả |
|---|---:|
| Frame accuracy | 89,8% |
| Microsleep episode recall | 34/40 = 85% |
| False T0 trên frame có nhãn alert | 110/16.881 = 0,65% |

`0,65%` là tỷ lệ **frame đang ở trạng thái T0**, không phải số lần loa phát cảnh báo độc lập. Các frame này tập trung thành cụm. Trước khi đặt SLA production vẫn cần đo số episode báo nhầm trên mỗi giờ bằng những chuyến lái tự nhiên dài hơn. MAR cũng được hiệu chỉnh trên chính FL3D, nên kết quả yawning là in-sample. Phần diễn giải và giới hạn đầy đủ nằm trong `docs/04-evaluation.md`.

Độ trễ khoảng 31 ms/frame của Tier 1 và 4–7 giây/lần gọi VLM được đo trên máy phát triển, không đại diện cho mọi SoC edge.

## Bộ nhớ trạng thái và quyền riêng tư

Runner thật mặc định lưu dữ liệu vào `data/health_baseline.sqlite3`. Có thể chọn hồ sơ cục bộ bằng `--profile-id`, hoặc dùng `--db-path :memory:` nếu không muốn ghi khi replay. Mỗi phiên chỉ giữ tối đa tám chỉ số, không lưu ảnh hay face embedding. Frame quá tối bị bỏ; `light_bucket` lấy từ sensor/CLI nếu có, nếu không sẽ được ước lượng từ độ sáng ảnh. Dữ liệu được tổng hợp theo ngày; baseline lấy tối đa bảy ngày hợp lệ gần nhất và tự xóa dữ liệu quá 14 ngày.

Prototype đã có giảm thiểu dữ liệu, lưu bền vững, giới hạn thời gian lưu và loại ngày bất thường khỏi baseline. Mã hóa bằng SQLCipher/Keystore, màn hình xin consent và quyền xóa trong ứng dụng mới chỉ là yêu cầu cho bản production, chưa được triển khai trong repo Python này.

## Khi có và khi thiếu model

| Thành phần | Mặc định khi đủ dependency/model | Khi thiếu |
|---|---|---|
| Face landmark | MediaPipe FaceLandmarker | MockLandmarkBackend |
| Phone detection | YOLO26n COCO, stride 3 | confidence từ backend/mock |
| Pre-ride helmet/mask/glasses | Có interface `Yolo26PreRideDetector`; cần weights finetune riêng | MockObjectDetector |
| Tier 2 | llama-server localhost, hoặc llama-cpp-python + mmproj | MockVLMBackend |
| TTS | WAV Piper pre-render trong `assets/tts/` | Chạy tiếp không tiếng |

Head pose hiện là ước lượng pitch/yaw nhanh từ landmark 2D, có calibration góc camera và yaw gate. `solvePnP` với camera matrix cùng roll đầy đủ là hướng nâng cấp production, không phải claim của implementation hiện tại. YOLO `.pt` hiện chạy FP32; export INT8/NPU cũng là bước triển khai theo phần cứng đích.

## Cấu trúc repo

```text
docs/                         Thiết kế, guardrails, baseline, eval, giải trình
THIRD_PARTY_NOTICES.md        Nguồn và giấy phép thành phần bên thứ ba
src/pipeline.py               Cascade, trigger, async VLM, pre-ride
src/tier1.py                  Landmark metrics + temporal state machine
src/object_detector.py        YOLO phone + interface pre-ride
src/guardrails.py             Schema validator, renderer, post-filter
src/health_baseline.py        SQLite rolling baseline và anomaly detection
src/vlm_backend.py            llama-server / llama-cpp / mock
src/audio_alerts.py           Player WAV có ưu tiên cảnh báo khẩn cấp
src/run_video.py              Runner video và xuất demo annotate
src/run_webcam.py             Runner webcam
src/eval_fl3d.py              Eval dữ liệu thật
src/red_team.py               Red-team hai lớp
assets/tts/                   28 WAV duyệt sẵn + manifest
scripts/setup_models.sh       Tải/check model có checksum
.github/workflows/tests.yml   CI cho test, demo mock và red-team
tests/test_pipeline.py        Unit và regression tests
```

## Những giới hạn còn lại

- Weights pre-ride cho quai mũ/khẩu trang/kính chưa được train trong repo.
- Phone detector chưa được benchmark trên một bộ cabin-phone có nhãn riêng và chưa ràng buộc phone với đúng tài xế.
- Các feature màu/sưng mắt của health baseline là heuristic prototype, chưa được đánh giá lâm sàng và tuyệt đối không dùng để chẩn đoán.
- SQLite trong prototype chưa mã hóa at-rest.
- GGUF, YOLO weights và voice model lớn không commit vào Git; script setup tải chúng từ nguồn công khai và kiểm checksum.
- Nguồn và điều kiện giấy phép của model/voice được ghi trong `THIRD_PARTY_NOTICES.md`; mã nguồn bài làm chưa được cấp license riêng.
