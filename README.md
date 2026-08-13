# Edge VLM Test #3 — Driver Safety & Health Monitor

Bài kiểm tra năng lực Edge VLM & Multimodal AI — Số 3.

**Chủ đề:** Real-time Driver Safety, Health Trend Detection & Ethical AI Guardrails.

## Cấu trúc

```
docs/
  01-two-tier-cascade.md     Khối 1 — kiến trúc cascade 2 tầng + telematics fusion
  02-medical-guardrails.md   Khối 2 — system prompt + guardrails 4 lớp
  03-health-baseline.md      Khối 3 — health baseline privacy-preserving
src/
  pipeline.py                Pipeline chính (hoàn thiện TODO 1-4 của đề)
  tier1.py                   Tier 1: EAR/MAR/head pose + temporal state machine
  guardrails.py              Structured-output renderer + post-filter tất định
  health_baseline.py         SQLite baseline 7 ngày, z-score anomaly
  vlm_backend.py             Tier 2: MockVLM + LlamaCppVLM (GGUF)
  demo.py                    Demo kịch bản (không cần camera/model)
  run_webcam.py              Chạy thật với webcam (cần mediapipe + opencv)
  config/guardrails_config.json   Banned list + template bank duyệt sẵn
tests/
  test_pipeline.py           21 unit tests: trigger, guardrails, baseline
assets/                      Đề bài gốc
```

## Chạy

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Demo kịch bản (mock backend — chạy được mọi nơi)
.venv/bin/python -m src.demo

# Unit tests
.venv/bin/python -m pytest tests/ -q

# Chạy thật với webcam (Tier 1 MediaPipe real-time)
.venv/bin/pip install mediapipe opencv-python
.venv/bin/python -m src.run_webcam

# Chạy thật Tier 2 với model GGUF (tùy chọn)
.venv/bin/pip install llama-cpp-python
.venv/bin/python -m src.run_webcam --vlm qwen2-vl-2b-q4.gguf
```

## Kiến trúc tổng quan

Two-tier cascade — tách safety path khỏi context path:

- **Tier 1 (mọi frame, <50ms):** Face Landmark (EAR/MAR/PERCLOS), Head Pose Euler, phone detection. Cảnh báo khẩn cấp (mắt nhắm >1.5s, điện thoại) phát trực tiếp bằng TTS tĩnh duyệt sẵn — **không bao giờ chờ VLM** (SLA <300ms).
- **Tier 2 (event-driven / định kỳ 5 phút):** Edge VLM quantized. Chỉ trả **JSON enum theo schema đóng** — lời văn tới người dùng luôn lấy từ template bank người viết đã duyệt; model không có kênh phát free text (bảo đảm không vi phạm chẩn đoán y tế).
- **Health baseline:** chỉ lưu feature vô hướng (không ảnh/embedding), rolling mean/std 7 ngày theo ngày và light bucket, z-score anomaly với hướng xấu định nghĩa sẵn.

Cả 2 tầng đều pluggable: MediaPipe/llama.cpp khi có, mock khi không — demo và test chạy được ở mọi môi trường.
