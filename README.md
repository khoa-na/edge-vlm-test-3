# Edge VLM Test #3 — Driver Safety & Health Monitor

Bài kiểm tra năng lực Edge VLM & Multimodal AI — Số 3.

**Chủ đề:** Real-time Driver Safety, Health Trend Detection & Ethical AI Guardrails.

## Cấu trúc

- `docs/` — Thiết kế 3 khối (cascade architecture, guardrails, health baseline) + câu trả lời giải trình
- `src/` — Code pipeline Python (hoàn thiện skeleton từ đề bài)
- `assets/` — Đề bài gốc

## Kiến trúc tổng quan

Two-tier cascade:
- **Tier 1 (10 FPS, <20ms):** Lightweight CV — Face Landmark (EAR/MAR), Head Pose, Phone Detection
- **Tier 2 (event-driven):** Quantized Edge VLM — phân tích ngữ cảnh sâu khi có trigger hoặc định kỳ

## Giả định (chờ xác nhận từ người ra đề)

- [ ] Code chạy mock hay demo thật với webcam?
- [ ] VLM load model thật hay mock response?
- [ ] Format nộp: repo + README hay file doc riêng?
