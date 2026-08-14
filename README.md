# Edge VLM Test #3 — Driver Safety & Health Monitor

Bài làm cho **Bài kiểm tra năng lực Edge VLM & Multimodal AI — Số 3**: Real-time Driver Safety, Health Trend Detection & Ethical AI Guardrails.

## Kết quả nổi bật

- **Chạy model thật, không chỉ thiết kế**: Tier 1 dùng MediaPipe FaceLandmarker, Tier 2 dùng **Qwen3.5-2B Q4** qua llama-server — kiểm chứng end-to-end trên frame cabin thật.
- **Đánh giá định lượng trên dataset FL3D** (20,806 frame có nhãn): episode recall **85%** trên các đoạn ngủ gật ≥ 1.5s, false-alarm **0.67%** frame alert, kèm mục khai báo giới hạn đánh giá.
- **Guardrails y tế kiểu "không thể vi phạm"**: VLM chỉ trả JSON enum đóng (ép grammar tại decoder), lời văn tới người dùng 100% từ template bank người viết đã duyệt — không tồn tại kênh free text.
- **29 unit test** phủ trigger logic, guardrails, baseline, async latency; 2 vòng review độc lập (Codex), sửa 19/22 finding vòng cuối.

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

Nguyên tắc xuyên suốt: **an toàn tức thời thuộc Tier 1 (rule tất định), hiểu ngữ cảnh thuộc Tier 2 (VLM, chấp nhận trễ vài giây)** — VLM không bao giờ nằm trên safety path. Health baseline (Khối 3) chỉ lưu 8 feature số/phiên, không ảnh, không embedding; so sánh z-score với chính người dùng 7 ngày gần nhất.

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
  guardrails.py              Schema validator + renderer + post-filter tất định
  health_baseline.py         SQLite baseline 7 ngày, z-score anomaly
  vlm_backend.py             LlamaServer / LlamaCpp / Mock — cắm được
  demo.py                    Demo kịch bản (không cần camera/model)
  run_video.py               Chạy trên video file / thư mục frame
  run_webcam.py              Chạy với webcam
  eval_fl3d.py               Đánh giá trên dataset FL3D có nhãn
  config/guardrails_config.json   Banned list + template bank duyệt sẵn
tests/test_pipeline.py       29 unit test
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

# 4. Tier 2 thật (Qwen3.5-2B qua llama-server của llama.cpp)
#    Model: unsloth/Qwen3.5-2B-GGUF -> Q4_K_M (1.28GB) + mmproj-F16 (0.67GB)
llama-server -m models/qwen3.5-2b-q4_k_m.gguf \
  --mmproj models/qwen3.5-2b-mmproj-f16.gguf --port 8090 -c 4096 \
  --chat-template-kwargs '{"enable_thinking": false}'
.venv/bin/python -m src.run_video --input clip.mp4 --vlm-url http://127.0.0.1:8090
```

Mọi thành phần đều **pluggable + fail-safe**: có model thì chạy thật, không có thì fallback mock — demo và test chạy được ở mọi môi trường, thứ tự ưu tiên Tier 2: llama-server → llama-cpp-python → mock.

## Ghi chú lựa chọn model Tier 2

**Qwen3.5-2B** được chọn sau khi so với Gemma 4 E2B: 2B thật (GGUF Q4 1.28GB, RAM ~2GB) so với E2B raw ~5B (~3GB); llama.cpp hỗ trợ day-1; mạnh tiếng Việt. Gemma 4 E2B là phương án B nếu roadmap cần thêm **audio** (phân tích giọng mệt mỏi) — E2B có audio native. Chi tiết tích hợp + 3 bài học khi cắm model thật (thinking mode, prompt bias, fact vs judgment): `docs/04` §5.

## Roadmap

Các nâng cấp tiếp theo, giữ nguyên phạm vi đề bài:

**Giai đoạn 1 — Xóa các thành phần mock còn lại**

- [ ] **YOLO26n thật cho phone/helmet/mask detection** — thay `MockObjectDetector` trong `src/pipeline.py`. Pretrained COCO có sẵn class `cell phone`; helmet/mask dùng model finetune sẵn. Lý do chọn YOLO26n thay SSDLite-MobileNet: NMS-free nhanh hơn trên CPU edge, mAP COCO ~40 so với ~22 cùng cỡ ~5MB.
- [ ] **TTS tiếng Việt offline** — pre-render toàn bộ template bank thành WAV lúc build (edge-tts), runtime chỉ phát file: 0MB model, latency ~0ms, giữ cam kết cảnh báo <300ms. Khớp thiết kế "TTS tĩnh duyệt sẵn" trong docs/01. Không dùng model omni nói thẳng (Qwen2.5-Omni): không hỗ trợ TTS tiếng Việt, và audio free-form không grammar-constrain được — phá nguyên tắc "không tồn tại kênh free text" của guardrails (docs/02).

**Giai đoạn 2 — Bằng chứng end-to-end**

- [ ] **Clip demo webcam người thật** — diễn các kịch bản: nhắm mắt >1.5s (cảnh báo T0 tức thì), ngáp liên tục, cầm điện thoại, quay đầu, lim dim kéo dài (PERCLOS trigger Tier 2 VLM); overlay chỉ số + cảnh báo TTS phát tiếng; telematics mô phỏng qua CLI. Nhúng clip đầu README.
- [ ] **Red-team guardrails định lượng** — bộ 50-100 câu tấn công (dụ chẩn đoán y tế, prompt injection qua nội dung frame/telematics) bắn vào VLM thật, đo tỉ lệ vượt guardrail (mục tiêu: 0%), xuất bảng kết quả vào docs/02.
