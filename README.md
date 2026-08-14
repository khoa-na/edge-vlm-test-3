# Edge VLM Test #3 — Driver Safety & Health Monitor

Bài làm cho **Bài kiểm tra năng lực Edge VLM & Multimodal AI — Số 3**: Real-time Driver Safety, Health Trend Detection & Ethical AI Guardrails.

🎬 **Demo người thật**: [`demo_webcam.mp4`](demo_webcam.mp4) — clip webcam 2.5 phút diễn đủ 6 kịch bản (nhắm mắt, ngáp, điện thoại, lim dim PERCLOS, quay đầu, lái >60'), chạy pipeline thật end-to-end (MediaPipe + YOLO26n + Qwen3.5-2B) với overlay chỉ số và 10 câu cảnh báo TTS tiếng Việt ghi thẳng vào audio track. Lệnh tạo video ở mục [Cách chạy](#cách-chạy), bước 5.

## Kết quả nổi bật

- Chạy model thật chứ không chỉ thiết kế: Tier 1 dùng MediaPipe FaceLandmarker, Tier 2 dùng Qwen3.5-2B Q4 qua llama-server, kiểm chứng end-to-end trên frame cabin thật lẫn webcam người thật.
- Đánh giá định lượng trên dataset FL3D (20,806 frame có nhãn): episode recall 85% trên các đoạn ngủ gật ≥ 1.5s, false-alarm 0.65% frame alert, kèm mục khai báo thẳng các giới hạn của phép đánh giá.
- Guardrails y tế thiết kế để không thể vi phạm: VLM chỉ trả JSON enum đóng (ép grammar tại decoder), lời văn tới người dùng 100% lấy từ template bank người viết đã duyệt — không tồn tại kênh free text.
- Phone detection thật (YOLO26n) và cảnh báo TTS tiếng Việt offline (Piper): Tier 1 đầy đủ MediaPipe + YOLO chạy ~31ms/frame CPU; mọi câu ra loa là WAV pre-render từ tập đóng duyệt sẵn, không synthesize runtime.
- Guardrails đo bằng red-team định lượng: 65 tấn công (prompt injection vào VLM thật + câu y tế bắn thẳng post-filter), 0% vượt rào, và chính red-team lộ ra một false-positive thật ("an toàn" bị chặn nhầm) đã sửa — chi tiết `docs/02` §7.
- 43 unit test phủ trigger logic, guardrails, red-team regression, baseline, async latency, phone detector, TTS manifest, pose calibration; 2 vòng review độc lập (Codex), sửa 19/22 finding vòng cuối.

## Trả lời yêu cầu đề bài

| Yêu cầu | Deliverable |
|---|---|
| Khối 1 — Two-tier Event-Driven Cascade | [`docs/01-two-tier-cascade.md`](docs/01-two-tier-cascade.md) |
| Khối 2 — System Prompt & Medical Guardrails | [`docs/02-medical-guardrails.md`](docs/02-medical-guardrails.md) |
| Khối 3 — Long-term Health Memory (privacy-preserving) | [`docs/03-health-baseline.md`](docs/03-health-baseline.md) |
| Mục 3 — Python code skeleton (TODO 1–4) | [`src/pipeline.py`](src/pipeline.py) + các module `src/` |
| Mục 4 — 4 câu hỏi giải trình | [`docs/05-giai-trinh.md`](docs/05-giai-trinh.md) |
| Kiểm chứng bằng model + dữ liệu thật | [`docs/04-evaluation.md`](docs/04-evaluation.md) |

## Kiến trúc trong 30 giây

```
Camera 5-10 FPS ──► TIER 1: CV nhẹ, mọi frame, <50ms
                    EAR / MAR / PERCLOS / Head Pose / Phone
                    + temporal state machine (debounce, cửa sổ trượt)
                         │
        ┌────────────────┴──────────────────┐
   KHẨN CẤP (T0/T1)                  TRIGGER "đắt giá" (T2-T7)
   TTS tĩnh duyệt sẵn, <300ms         │  async, cooldown 3'
   KHÔNG BAO GIỜ chờ VLM              ▼
                              TIER 2: Edge VLM (Qwen3.5-2B Q4)
                              chỉ trả JSON enum {observation, severity}
                                      │
                              context_slots điền tất định từ telematics
                                      ▼
                              Template bank duyệt sẵn ──► post-filter ──► loa
```

Nguyên tắc xuyên suốt: an toàn tức thời thuộc Tier 1 (rule tất định), hiểu ngữ cảnh thuộc Tier 2 (VLM, chấp nhận trễ vài giây) — VLM không bao giờ nằm trên safety path. Health baseline (Khối 3) chỉ lưu 8 feature số mỗi phiên, không ảnh, không embedding, và so sánh z-score với chính người dùng trong 7 ngày gần nhất.

## Cấu trúc repo

```
docs/
  01-two-tier-cascade.md     Khối 1 — cascade 2 tầng, bảng trigger, telematics fusion
  02-medical-guardrails.md   Khối 2 — guardrails 4 lớp, structured output
  03-health-baseline.md      Khối 3 — schema baseline, anomaly, GDPR
  04-evaluation.md           Eval FL3D + Tier 2 model thật + giới hạn
  05-giai-trinh.md           Trả lời 4 câu hỏi Mục 4
src/
  pipeline.py                Pipeline chính (TODO 1-4), pre-ride check, async VLM
  tier1.py                   EAR/MAR/head pose + temporal state machine + feature màu Lab
  object_detector.py         YOLO26n phone detection (stride + cache) + pre-ride detector
  guardrails.py              Schema validator + renderer + post-filter tất định
  audio_alerts.py            Phát cảnh báo TTS pre-render (chỉ câu trong manifest)
  tts_prerender.py           Build-time: render template bank -> assets/tts/*.wav (Piper)
  health_baseline.py         SQLite baseline 7 ngày, z-score anomaly
  vlm_backend.py             LlamaServer / LlamaCpp / Mock — cắm được
  demo.py                    Demo kịch bản (không cần camera/model)
  run_video.py               Chạy trên video file / thư mục frame
  run_webcam.py              Chạy với webcam
  eval_fl3d.py               Đánh giá trên dataset FL3D có nhãn
  config/guardrails_config.json   Banned list + template bank duyệt sẵn
tests/test_pipeline.py       43 unit test
  red_team.py                Red-team guardrails định lượng (2 lớp, xuất bảng)
assets/tts/                  28 câu cảnh báo tiếng Việt pre-render (WAV + manifest)
assets/                      Đề bài gốc
```

## Cách chạy

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Demo kịch bản — chạy được mọi nơi, không cần camera/model
.venv/bin/python -m src.demo

# 2. Unit tests
.venv/bin/python -m pytest tests/ -q

# 3. Tier 1 thật (MediaPipe) trên dataset FL3D có nhãn
.venv/bin/pip install mediapipe opencv-python kagglehub
curl -sL -o models/face_landmarker.task --create-dirs \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
.venv/bin/python -c "import kagglehub; kagglehub.dataset_download('matjazmuc/frame-level-driver-drowsiness-detection-fl3d')"
.venv/bin/python -m src.eval_fl3d --limit-seq 8   # ~10 phút CPU

# 4. Phone detection thật (YOLO26n, tự tải weights 5.3MB lần đầu)
.venv/bin/pip install ultralytics

# 5. Cảnh báo TTS tiếng Việt (WAV pre-render sẵn trong assets/tts/)
.venv/bin/pip install sounddevice   # + sudo apt install libportaudio2
.venv/bin/python -m src.run_webcam --audio
#    Xuất video demo có giọng cảnh báo baked-in (cần ffmpeg):
.venv/bin/python -m src.run_video --input clip.mp4 --output demo.mp4 --audio-mux
#    Re-render khi sửa template bank (cần Piper + voice 63MB, offline):
#    pip install piper-tts && python -m src.tts_prerender

# 6. Tier 2 thật (Qwen3.5-2B qua llama-server của llama.cpp)
#    Model: unsloth/Qwen3.5-2B-GGUF -> Q4_K_M (1.28GB) + mmproj-F16 (0.67GB)
llama-server -m models/qwen3.5-2b-q4_k_m.gguf \
  --mmproj models/qwen3.5-2b-mmproj-f16.gguf --port 8090 -c 4096 \
  --chat-template-kwargs '{"enable_thinking": false}'
.venv/bin/python -m src.run_video --input clip.mp4 --vlm-url http://127.0.0.1:8090
```

Mọi thành phần đều pluggable và fail-safe: có model thì chạy thật, không có thì fallback mock, nên demo và test chạy được ở mọi môi trường. Thứ tự ưu tiên Tier 2: llama-server → llama-cpp-python → mock.

## Ghi chú lựa chọn model Tier 2

Qwen3.5-2B được chọn sau khi so với Gemma 4 E2B: nó là 2B thật (GGUF Q4 1.28GB, RAM ~2GB) trong khi E2B raw cỡ ~5B (~3GB); llama.cpp hỗ trợ day-1; và mạnh tiếng Việt. Gemma 4 E2B để làm phương án B nếu roadmap về sau cần thêm audio (phân tích giọng mệt mỏi) — E2B có audio native. Chi tiết tích hợp cùng 3 bài học khi cắm model thật (thinking mode, prompt bias, fact vs judgment): `docs/04` §5.

## Roadmap

Các nâng cấp tiếp theo, giữ nguyên phạm vi đề bài:

**Giai đoạn 1 — Xóa các thành phần mock còn lại** ✅

- [x] **YOLO26n thật cho phone detection** — `src/object_detector.py`, tự cắm vào `Tier1Analyzer.phone_detector` khi Tier 1 chạy backend thật. Số đo được: YOLO26n 5.3MB, ~44ms mỗi lần chạy CPU @384, stride 3 frame nên trung bình ~15ms/frame; cả Tier 1 gồm MediaPipe + YOLO hết ~31ms/frame. Chọn YOLO26n thay SSDLite-MobileNet vì NMS-free nhanh hơn trên CPU edge và mAP COCO ~40 so với ~22 ở cùng cỡ ~5MB. Pre-ride helmet/mask cần weights finetune riêng (`Yolo26PreRideDetector`), chưa có weights nên slot đó vẫn mock.
- [x] **TTS tiếng Việt offline** — Piper `vi_VN-vais1000-medium` (63MB, ONNX, offline hoàn toàn) pre-render cả 28 câu duyệt sẵn thành WAV (`python -m src.tts_prerender` → `assets/tts/`, 4.4MB). Runtime chỉ phát file qua `AlertSpeaker` (sounddevice, non-blocking): 0ms synthesize, giữ SLA <300ms. Bật bằng `--audio` trong `run_video`/`run_webcam`. Player từ chối text ngoài manifest — không synthesize runtime, đúng nguyên tắc "không tồn tại kênh free text" của docs/02. Không dùng model omni nói thẳng (Qwen2.5-Omni) vì không hỗ trợ TTS tiếng Việt, và audio free-form không grammar-constrain được.

**Giai đoạn 2 — Bằng chứng end-to-end**

- [x] **Clip demo webcam người thật** — [`demo_webcam.mp4`](demo_webcam.mp4): 6/6 kịch bản nổ đúng trên clip webcam 2.5 phút (T5 lái dài→VLM t=6s, T0 nhắm mắt t=17s, T3 ngáp→VLM t=47s, T1 điện thoại t=57s, T2 PERCLOS→VLM t=70s, T4 quay đầu t=88s); overlay chỉ số + 10 câu TTS trong audio track (câu giống hệt trong ~2s được gộp — chuỗi T0 lim dim nhấp nháy chỉ nhắc lại tuần hoàn chứ không đè 7 bản); telematics mô phỏng qua CLI (`--driving-min 59.9`, cộng dồn theo thời gian clip). Quá trình này lộ ra và sửa được 3 bug thật: EAR sai phối cảnh khi quay đầu (thêm yaw-gate 45°), câu nhắc tĩnh không cooldown, và T2 PERCLOS che mất T4 trong chuỗi elif — những bug mà dataset frontal như FL3D không bao giờ lộ được.
- [x] **Red-team guardrails định lượng** — `src/red_team.py`: 65 câu tấn công (dụ chẩn đoán y tế, prompt injection qua delta_text/telematics/chữ trong frame) bắn vào cả VLM thật (23) lẫn post-filter (42), đo tỉ lệ vượt rào 0.0% và false-positive 0. Quá trình lộ ra và sửa được một false-positive thật: từ cấm "toa" khớp substring chặn nhầm "an toàn"; nay khớp theo biên từ, thêm 3 test hồi quy. Bảng kết quả: `docs/02` §7.
