# BÀI KIỂM TRA NĂNG LỰC EDGE VLM & MULTIMODAL AI — SỐ 3

> **Chủ đề:** Real-time Driver Safety, Health Trend Detection & Ethical AI Guardrails
> *(Kiểm tra an toàn, phát hiện mệt mỏi/trạng thái bất thường và thiết kế rào cản an toàn AI)*

**Mục tiêu đánh giá:**

- Năng lực xử lý luồng video thời gian thực (Streaming / Temporal Frames)
- Kỹ thuật phát hiện hành vi nguy hiểm (DMS — Driver Monitoring System)
- Thiết kế AI Guardrails tránh đưa ra chẩn đoán y tế sai luật (Medical Diagnosis Disclaimer)
- Cơ chế tích hợp với hệ thống cảnh báo của thiết bị nhúng

---

## PHẦN 1. BỐI CẢNH DỰ ÁN & YÊU CẦU KỸ THUẬT (SPECIFICATIONS)

### 1.1. Mô tả module

Bạn chịu trách nhiệm xây dựng **Module Safety & Driver Health Monitor** (Giám sát an toàn & Cảnh báo sức khỏe người dùng) tích hợp trên thiết bị di chuyển. Module chạy **liên tục hoặc theo chu kỳ ngắn** (Every N seconds) trong suốt hành trình.

### 1.2. Đầu vào (Input)

| Nguồn dữ liệu | Mô tả |
|---|---|
| `Frame_Stream` | Luồng video/ảnh chụp từ camera trước/cabin (5–10 FPS) |
| `Vehicle_Telematics` | Dữ liệu vận hành xe: tốc độ, thời gian lái xe liên tục, thời tiết, nhiệt độ môi trường |

### 1.3. Bài toán kỹ thuật

#### A. Real-time Safety Check & DMS (Driver Monitoring System)

- **Kiểm tra trước khi chạy:**
  - Phát hiện quai mũ bảo hiểm chưa cài
  - Chưa đeo khẩu trang/kính khi trời nắng bụi
- **Trong khi vận hành — phát hiện hành vi nguy cơ cao:**
  - Nhìn xuống điện thoại
  - Ngáp liên tục
  - Mắt nhắm quá **1.5 giây** (buồn ngủ/lơ đãng)
  - Quay đầu về sau liên tục

#### B. Non-Medical Health Trend Memory (Nhận biết trạng thái bất thường)

So sánh với **lịch sử nhiều ngày của chính người dùng** để nhận biết sự bất thường:

- Mắt sưng
- Sắc mặt tái nhạt
- Môi tím tái
- Quầng thâm mắt đậm hơn hẳn ngày thường

#### C. Strict Ethical & Medical Guardrails

| Tuyệt đối KHÔNG | Chỉ được phép |
|---|---|
| Đưa ra chẩn đoán y tế. Ví dụ: *"Bạn bị thiếu máu"*, *"Bạn bị đột quỵ"*, *"Bạn bị bệnh tim"* | Nhắc nhở nhẹ nhàng, quan tâm tinh tế + khuyến nghị an toàn giao thông. Ví dụ: *"Hôm nay trông bạn có vẻ hơi mệt, hãy cân nhắc nghỉ ngơi vài phút trước khi tiếp tục hành trình"* |

#### D. Ràng buộc phần cứng & thời gian thực (Real-time Constraint)

- Tác vụ phát hiện mất tập trung/buồn ngủ (DMS) yêu cầu phản hồi **dưới 300ms** (latency cực thấp).
- VLM lớn **không thể** chạy 10 FPS liên tục (gây nóng máy, quá tải chip Edge).
- Bắt buộc thiết kế **Tháp Kiến Trúc 2 Tầng (Two-tier Cascade Architecture):**
  - **Tier 1:** Mạng CV truyền thống mỏng nhẹ làm *Trigger Filter*
  - **Tier 2:** VLM chỉ chạy khi có sự kiện đặc biệt hoặc chạy định kỳ background

---

## PHẦN 2. NHIỆM VỤ THIẾT KẾ (DESIGN TASKS)

### Khối 1 — Kiến Trúc Phân Tầng Lọc Sự Kiện (Two-tier Event-Driven Cascade)

- **Nhiệm vụ:** Thiết kế luồng xử lý kết hợp giữa:
  - **Tier 1 (Real-time):** Lightweight CV Models (MobileNet / YOLO / FaceLandmark)
  - **Tier 2 (Event-Driven):** Edge VLM
- **Ràng buộc:** Xác định rõ **điều kiện/tiêu chí nào ở Tier 1** sẽ kích hoạt (trigger) Tier 2 gọi VLM phân tích ngữ cảnh sâu.

### Khối 2 — System Prompt & Medical Guardrails (Anti-Medical Diagnosis)

- **Nhiệm vụ:** Viết bộ quy tắc Guardrails và System Prompt để **kiểm soát tuyệt đối đầu ra của VLM**, đảm bảo không vi phạm quy định về chẩn đoán y tế tự động.

### Khối 3 — Xử Lý Trạng Thái Bất Thường Theo Chuỗi Thời Gian (Long-term Health Memory Index)

- **Nhiệm vụ:** Thiết kế cấu trúc bảng ghi `User_Health_Baseline`:
  - Lưu trữ chỉ số trạng thái trung bình **7 ngày gần nhất** để phát hiện độ lệch (Anomaly Detection)
  - **KHÔNG lưu trữ dữ liệu ảnh nhạy cảm** của người dùng (Privacy-preserving)

---

## PHẦN 3. THỬ THÁCH CODING & CẤU TRÚC (PYTHON CODE SKELETON)

Hoàn thiện các khối logic `# TODO:` trong pipeline dưới đây:

```python
import time
import numpy as np
from typing import Dict, Any, List, Optional

class SafetyAndHealthMonitorPipeline:
    def __init__(self, edge_vlm_path: str, device: str = "cuda"):
        self.device = device
        self.init_cascade_models(edge_vlm_path)
        # Lưu trữ chỉ số nền (Baseline) 7 ngày của người dùng
        self.user_baseline = {
            "avg_eye_darkness": 0.2,
            "avg_skin_paleness": 0.15,
            "normal_blink_rate": 18  # lần/phút
        }

    def init_cascade_models(self, vlm_path: str):
        """
        Khởi tạo Tier 1 (Lightweight Face & Pose Detector)
        và Tier 2 (Quantized Edge VLM Engine).
        """
        print(f"Loading Tier 1 (CV) & Tier 2 (Edge VLM) on {self.device}...")
        # TODO 1: Init Fast Face Landmark (EAR - Eye Aspect Ratio), Head Pose, và VLM
        pass

    def tier1_fast_stream_check(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        [Tier 1 - Chạy 10 FPS]
        Dùng mô hình CV siêu nhẹ tính toán nhanh:
        - EAR (Eye Aspect Ratio) để đo nhắm mắt
        - Pitch/Yaw/Roll để đo gật gụ/quay đầu
        - Phone Detection
        """
        # TODO 2: Trả về kết quả phân tích nhanh
        return {
            "eyes_closed_duration_sec": 0.2,
            "head_tilted_down": False,
            "using_phone": False,
            "trigger_vlm_needed": False  # Flag kích hoạt Tier 2
        }

    def enforce_medical_guardrails(self, raw_vlm_text: str) -> str:
        """
        Bộ lọc Guardrail kiểm tra và loại bỏ các từ khóa chẩn đoán y khoa cấm.
        """
        banned_medical_terms = [
            "bệnh", "đột quỵ", "thiếu máu", "chẩn đoán", "triệu chứng",
            "suy nhược", "viêm", "nhiễm trùng", "sốt"
        ]

        sanitized_text = raw_vlm_text
        # TODO 3: Viết logic quét và biến đổi câu chữ nếu phát hiện từ cấm.
        # Đảm bảo câu trả lời chuyển hướng sang nhắc nhở an toàn / nghỉ ngơi.

        return sanitized_text

    def tier2_run_vlm_context_analysis(
        self,
        frame: np.ndarray,
        trigger_reason: str,
        telematics: Dict[str, Any]
    ) -> str:
        """
        [Tier 2 - Chỉ chạy khi có Trigger hoặc định kỳ mỗi 5 phút]
        VLM phân tích ngữ cảnh sâu và đưa ra phản hồi.
        """
        system_prompt = """You are a Caring Vehicle AI Companion.
CRITICAL RULE: NEVER give medical diagnoses or use clinical medical terms.
Your goal is ONLY to remind the driver about safety and suggest taking a rest if they look tired.

Context: Driver looks unusually pale or fatigued compared to their usual baseline.
Trip Duration: {duration} mins.
Issue Triggered: {reason}

Formulate a gentle, supportive 1-sentence warning in Vietnamese.
"""
        # TODO 4: Truyền Image + Context vào VLM và đi qua hàm enforce_medical_guardrails
        print(f"Running Tier 2 VLM due to trigger: {trigger_reason}")

        raw_response = "Hôm nay trông bạn có vẻ hơi mệt hơn thường ngày. Bạn nên đỗ xe nghỉ ngơi vài phút nhé."
        safe_response = self.enforce_medical_guardrails(raw_response)

        return safe_response

    def process_stream_frame(
        self,
        frame: np.ndarray,
        telematics: Dict[str, Any]
    ) -> Optional[str]:
        """
        Hàm chính xử lý từng Frame theo kiến trúc Cascade
        """
        # 1. Tier 1 Fast Check (Latency < 20ms)
        t1_result = self.tier1_fast_stream_check(frame)

        # Cảnh báo khẩn cấp ngay lập tức nếu ngủ gật hoặc dùng điện thoại
        if t1_result["eyes_closed_duration_sec"] > 1.5:
            return "CẢNH BÁO: Báo động! Hãy tập trung lái xe!"

        # 2. Kiểm tra điều kiện kích hoạt Tier 2 (VLM)
        if t1_result["trigger_vlm_needed"] or telematics.get("continuous_driving_min", 0) > 60:
            trigger_reason = "Lái xe liên tục > 60 phút hoặc phát hiện mệt mỏi"
            vlm_feedback = self.tier2_run_vlm_context_analysis(frame, trigger_reason, telematics)
            return vlm_feedback

        return None


# Simulation
if __name__ == "__main__":
    monitor = SafetyAndHealthMonitorPipeline(edge_vlm_path="quantized_vlm.gguf")
    # telematics_data = {"continuous_driving_min": 65, "speed_kmh": 35}
    # result = monitor.process_stream_frame(dummy_frame, telematics_data)
    # print(result)
```

**Danh sách TODO cần hoàn thiện:**

| TODO | Vị trí | Nội dung |
|---|---|---|
| TODO 1 | `init_cascade_models` | Init Fast Face Landmark (EAR), Head Pose, và VLM |
| TODO 2 | `tier1_fast_stream_check` | Trả về kết quả phân tích nhanh Tier 1 |
| TODO 3 | `enforce_medical_guardrails` | Logic quét & biến đổi câu chữ khi phát hiện từ cấm, chuyển hướng sang nhắc nhở an toàn/nghỉ ngơi |
| TODO 4 | `tier2_run_vlm_context_analysis` | Truyền Image + Context vào VLM và đi qua `enforce_medical_guardrails` |

---

## PHẦN 4. CÂU HỎI GIẢI TRÌNH (DOCUMENTATION)

> Ứng viên trả lời **ngắn gọn, đi thẳng vào bản chất kỹ thuật**.

### Câu 1 — Thiết Kế Kiến Trúc Tháp 2 Tầng (Two-tier Cascade Latency Optimization)

Trong hệ thống giám sát người lái theo thời gian thực, việc gọi VLM liên tục ở mọi frame là bất khả thi.

- Bạn phân chia trách nhiệm giữa **Tier 1 (Lightweight CV)** và **Tier 2 (VLM)** như thế nào?
- Kể tên các thuật toán/chỉ số cụ thể (Ví dụ: **EAR, MAR, Head Pose Euler Angles, YOLO**) sử dụng ở Tier 1 để quyết định thời điểm "đắt giá" cần gọi Tier 2.

### Câu 2 — Kiểm Soát Rủi Ro Y Tế & Pháp Lý (Medical Disclaimer & Ethical Guardrails)

Nếu người dùng có dấu hiệu sức khỏe bất thường rõ rệt (sắc mặt rất tái, mắt lờ đờ), VLM rất dễ tự đưa ra dự đoán y khoa (Ví dụ: *"Bạn có dấu hiệu tụt huyết áp/suy nhược"*).

- Làm thế nào để đảm bảo **100%** mô hình KHÔNG BAO GIỜ vi phạm ranh giới này, thông qua kết hợp:
  1. **Prompt Engineering**
  2. **Grammar / Negative Constraints**
  3. **Deterministic Rule-based Post-filtering**

### Câu 3 — Bảo Vệ Quyền Riêng Tư & Dữ Liệu Nhạy Cảm (Privacy-Preserving Health Baseline)

Để phát hiện trạng thái "mệt mỏi hơn ngày thường", hệ thống cần so sánh với lịch sử nhiều ngày.

- Làm sao xây dựng bộ nhớ lịch sử (Long-term Baseline) trên thiết bị nhúng mà **KHÔNG lưu ảnh khuôn mặt riêng tư** vào bộ nhớ máy, tuân thủ **GDPR / Privacy Laws**?

### Câu 4 — Tích Hợp Ngữ Cảnh Phương Tiện (Telematics-VLM Fusion)

- Làm thế nào kết hợp dữ liệu cảm biến phương tiện (`speed`, `continuous_driving_time`, `weather`) với phân tích hình ảnh của VLM để đưa ra lời nhắc thuyết phục cao?
- Ví dụ: Lái xe liên tục 2 tiếng + trời nắng 35°C + VLM thấy mặt bóng dầu/mệt mỏi → AI đưa ra lời khuyên gì **khác** so với khi xe đang dừng đỗ?
