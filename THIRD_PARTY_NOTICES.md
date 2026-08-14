# Third-party notices

Repository này dùng hoặc cung cấp cách tải một số thành phần bên thứ ba.
Danh sách dưới đây nhằm ghi rõ nguồn và không thay thế nội dung giấy phép gốc.

## MediaPipe Face Landmarker

- Nguồn model: Google MediaPipe Face Landmarker
- File được phân phối trong repo: `models/face_landmarker.task`
- Dự án MediaPipe: https://github.com/google-ai-edge/mediapipe
- Giấy phép dự án: Apache License 2.0

## Piper và giọng `vi_VN-vais1000-medium`

- Nguồn voice model: https://huggingface.co/rhasspy/piper-voices/tree/main/vi/vi_VN/vais1000/medium
- Các file trong `assets/tts/` được tạo từ voice model này.
- Kho Piper voices khai báo giấy phép MIT.
- Model card ghi nguồn dữ liệu là **VAIS-1000 Vietnamese Speech Synthesis
  Corpus**, giấy phép Creative Commons Attribution 4.0:
  https://creativecommons.org/licenses/by/4.0/

## Ultralytics YOLO

- Package và weights không được commit vào repo; `scripts/setup_models.sh`
  tải `yolo26n.pt` khi người dùng yêu cầu.
- Dự án: https://github.com/ultralytics/ultralytics
- Bản community được phát hành theo GNU AGPL-3.0; Ultralytics cũng cung cấp
  giấy phép enterprise. Người tích hợp cần chọn giấy phép phù hợp với cách
  phân phối sản phẩm của mình.

## Qwen3.5 GGUF

- GGUF và multimodal projector không được commit vào repo.
- Script tải bản quantized từ:
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- Model card khai báo giấy phép Apache License 2.0.

## Đề bài tuyển dụng

File đề bài trong `assets/` được giữ lại để reviewer đối chiếu phạm vi bài
làm. Quyền đối với nội dung đề bài vẫn thuộc về tác giả hoặc đơn vị phát hành
ban đầu.

## Mã nguồn của bài làm

Repository chưa gắn giấy phép cho phần mã nguồn do ứng viên viết. Chủ sở hữu
cần chọn giấy phép trước khi cho phép bên khác sao chép, sửa đổi hoặc phân
phối mã nguồn ngoài mục đích review bài tuyển dụng.
