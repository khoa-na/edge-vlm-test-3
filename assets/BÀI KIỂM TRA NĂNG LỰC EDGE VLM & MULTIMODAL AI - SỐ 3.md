# **BÀI KIỂM TRA NĂNG LỰC EDGE VLM & MULTIMODAL AI \- SỐ 3**

> * **Chủ đề:** Real-time Driver Safety, Health Trend Detection & Ethical AI Guardrails (Kiểm tra an toàn, phát hiện mệt mỏi/trạng thái bất thường và thiết kế rào cản an toàn AI).  
> * **Mục tiêu:** Đánh giá năng lực xử lý luồng video thời gian thực (Streaming/Temporal Frames), kỹ thuật phát hiện hành vi nguy hiểm (DMS), thiết kế AI Guardrails để tránh đưa ra các chẩn đoán y tế sai luật (Medical Diagnosis Disclaimer), và cơ chế tích hợp với hệ thống cảnh báo của thiết bị nhúng.

## **1\. BỐI CẢNH DỰ ÁN & YÊU CẦU KỸ THUẬT (SPECIFICATIONS)**

Bạn chịu trách nhiệm xây dựng **Module Safety & Driver Health Monitor (Giám sát an toàn & Cảnh báo sức khỏe người dùng)** tích hợp trên thiết bị di chuyển. Module này chạy liên tục hoặc theo chu kỳ ngắn (Every N seconds) trong suốt hành trình.

> * **Đầu vào (Input):**  
  * Frame\_Stream: Luồng video/ảnh chụp từ camera trước/cabin (5-10 FPS).  
  * Vehicle\_Telematics: Dữ liệu vận hành xe (Tốc độ, thời gian lái xe liên tục, thời tiết, nhiệt độ môi trường).  
> * **Bài toán kỹ thuật:**  
  1. **Real-time Safety Check & DMS (Driver Monitoring System):**  
     * *Kiểm tra trước khi chạy:* Phát hiện quai mũ bảo hiểm chưa cài, chưa đeo khẩu trang/kính khi trời nắng bụi.  
     * *Trong khi vận hành:* Phát hiện các hành vi nguy cơ cao: nhìn xuống điện thoại, ngáp liên tục, mắt nhắm quá 1.5 giây (buồn ngủ/lơ đãng), quay đầu về sau liên tục.  
  2. **Non-Medical Health Trend Memory (Nhận biết trạng thái bất thường):**  
     * So sánh với lịch sử nhiều ngày của chính người dùng đó để nhận biết sự bất thường: Mắt sưng, sắc mặt tái nhạt, môi tím tái, quầng thâm mắt đậm hơn hẳn ngày thường.  
  3. **Strict Ethical & Medical Guardrails:**  
     * Tuyệt đối KHÔNG đưa ra chẩn đoán y tế (Ví dụ: KHÔNG được nói *"Bạn bị thiếu máu"*, *"Bạn bị đột quỵ"*, *"Bạn bị bệnh tim"*).  
     * Chỉ được phép nhắc nhở nhẹ nhàng mang tính quan tâm tinh tế và đưa ra khuyến nghị an toàn giao thông (Ví dụ: *"Hôm nay trông bạn có vẻ hơi mệt, hãy cân nhắc nghỉ ngơi vài phút trước khi tiếp tục hành trình"*).  
> * **Ràng buộc phần cứng & Thời gian thực (Real-time Constraint):**  
  * Tác vụ phát hiện mất tập trung/buồn ngủ (DMS) yêu cầu phản hồi dưới **300ms** (Latency cực thấp).  
  * Mô hình VLM lớn không thể chạy 10 FPS liên tục vì sẽ gây nóng máy và quá tải chip Edge. Cần thiết kế **Tháp Kiến Trúc 2 Tầng (Two-tier Cascade Architecture)**: Mạng CV truyền thống mỏng nhẹ làm Trigger Filter, VLM chỉ chạy khi có sự kiện đặc biệt hoặc chạy định kỳ background.

## **2\. NHIỆM VỤ THIẾT KẾ (DESIGN TASKS)**

### **Khối 1: Kiến Trúc Phân Tầng Lọc Sự Kiện (Two-tier Event-Driven Cascade)**

> * **Nhiệm vụ:** Thiết kế luồng xử lý kết hợp giữa Lightweight CV Models (MobileNet/YOLO/FaceLandmark) chạy ở Tier 1 (Real-time) và Edge VLM chạy ở Tier 2 (Event-Driven).  
> * **Ràng buộc:** Xác định rõ điều kiện/tiêu chí nào ở Tier 1 sẽ kích hoạt (trigger) Tier 2 gọi VLM phân tích ngữ cảnh sâu.

### **Khối 2: Thiết Kế System Prompt & Medical Guardrails (Anti-Medical Diagnosis)**

> * **Nhiệm vụ:** Viết bộ quy tắc Guardrails và System Prompt để kiểm soát tuyệt đối đầu ra của VLM, đảm bảo không vi phạm các quy định về chẩn đoán y tế tự động.

### **Khối 3: Xử Lý Trạng Thái Bất Thường Theo Chuỗi Thời Gian (Long-term Health Memory Index)**

> * **Nhiệm vụ:** Thiết kế cấu trúc bảng ghi User\_Health\_Baseline lưu trữ chỉ số trạng thái trung bình trong 7 ngày gần nhất để phát hiện độ lệch (Anomaly Detection) mà không lưu trữ dữ liệu ảnh nhạy cảm của người dùng (Bảo mật quyền riêng tư \- Privacy-preserving).

## **3\. THỬ THÁCH CODING & CẤU TRÚC (PYTHON CODE SKELETON)**

Hãy viết mã Python phác thảo Pipeline xử lý cho Module Safety & Health Monitoring bằng cách hoàn thiện các khối logic \# TODO:

Python  
import time  
import numpy as np  
from typing import Dict, Any, List, Optional

class SafetyAndHealthMonitorPipeline:  
    def \_\_init\_\_(self, edge\_vlm\_path: str, device: str \= "cuda"):  
        self.device \= device  
        self.init\_cascade\_models(edge\_vlm\_path)  
        \# Lưu trữ chỉ số nền (Baseline) 7 ngày của người dùng  
        self.user\_baseline \= {  
            "avg\_eye\_darkness": 0.2,  
            "avg\_skin\_paleness": 0.15,  
            "normal\_blink\_rate": 18 \# lần/phút  
        }

    def init\_cascade\_models(self, vlm\_path: str):  
        """  
        Khởi tạo Tier 1 (Lightweight Face & Pose Detector)   
        và Tier 2 (Quantized Edge VLM Engine).  
        """  
        print(f"Loading Tier 1 (CV) & Tier 2 (Edge VLM) on {self.device}...")  
        \# TODO 1: Init Fast Face Landmark (EAR \- Eye Aspect Ratio), Head Pose, và VLM  
        pass

    def tier1\_fast\_stream\_check(self, frame: np.ndarray) \-\> Dict\[str, Any\]:  
        """  
        \[Tier 1 \- Chạy 10 FPS\]   
        Dùng mô hình CV siêu nhẹ tính toán nhanh:  
        \- EAR (Eye Aspect Ratio) để đo nhắm mắt  
        \- Pitch/Yaw/Roll để đo gật gụ/quay đầu  
        \- Phone Detection  
        """  
        \# TODO 2: Trả về kết quả phân tích nhanh  
        return {  
            "eyes\_closed\_duration\_sec": 0.2,  
            "head\_tilted\_down": False,  
            "using\_phone": False,  
            "trigger\_vlm\_needed": False \# Flag kích hoạt Tier 2  
        }

    def enforce\_medical\_guardrails(self, raw\_vlm\_text: str) \-\> str:  
        """  
        Bộ lọc Guardrail kiểm tra và loại bỏ các từ khóa chẩn đoán y khoa cấm.  
        """  
        banned\_medical\_terms \= \[  
            "bệnh", "đột quỵ", "thiếu máu", "chẩn đoán", "triệu chứng",   
            "suy nhược", "viêm", "nhiễm trùng", "sốt"  
        \]  
          
        sanitized\_text \= raw\_vlm\_text  
        \# TODO 3: Viết logic quét và biến đổi câu chữ nếu phát hiện từ cấm.  
        \# Đảm bảo câu trả lời chuyển hướng sang nhắc nhở an toàn / nghỉ ngơi.  
          
        return sanitized\_text

    def tier2\_run\_vlm\_context\_analysis(  
        self,   
        frame: np.ndarray,   
        trigger\_reason: str,  
        telematics: Dict\[str, Any\]  
    ) \-\> str:  
        """  
        \[Tier 2 \- Chỉ chạy khi có Trigger hoặc định kỳ mỗi 5 phút\]  
        VLM phân tích ngữ cảnh sâu và đưa ra phản hồi.  
        """  
        system\_prompt \= """You are a Caring Vehicle AI Companion.  
CRITICAL RULE: NEVER give medical diagnoses or use clinical medical terms.   
Your goal is ONLY to remind the driver about safety and suggest taking a rest if they look tired.

Context: Driver looks unusually pale or fatigued compared to their usual baseline.  
Trip Duration: {duration} mins.  
Issue Triggered: {reason}

Formulate a gentle, supportive 1-sentence warning in Vietnamese.  
"""  
        \# TODO 4: Truyền Image \+ Context vào VLM và đi qua hàm enforce\_medical\_guardrails  
        print(f"Running Tier 2 VLM due to trigger: {trigger\_reason}")  
          
        raw\_response \= "Hôm nay trông bạn có vẻ hơi mệt hơn thường ngày. Bạn nên đỗ xe nghỉ ngơi vài phút nhé."  
        safe\_response \= self.enforce\_medical\_guardrails(raw\_response)  
          
        return safe\_response

    def process\_stream\_frame(  
        self,   
        frame: np.ndarray,   
        telematics: Dict\[str, Any\]  
    ) \-\> Optional\[str\]:  
        """  
        Hàm chính xử lý từng Frame theo kiến trúc Cascade  
        """  
        \# 1\. Tier 1 Fast Check (Latency \< 20ms)  
        t1\_result \= self.tier1\_fast\_stream\_check(frame)  
          
        \# Cảnh báo khẩn cấp ngay lập tức nếu ngủ gật hoặc dùng điện thoại  
        if t1\_result\["eyes\_closed\_duration\_sec"\] \> 1.5:  
            return "CẢNH BÁO: Báo động\! Hãy tập trung lái xe\!"  
              
        \# 2\. Kiểm tra điều kiện kích hoạt Tier 2 (VLM)  
        if t1\_result\["trigger\_vlm\_needed"\] or telematics.get("continuous\_driving\_min", 0\) \> 60:  
            trigger\_reason \= "Lái xe liên tục \> 60 phút hoặc phát hiện mệt mỏi"  
            vlm\_feedback \= self.tier2\_run\_vlm\_context\_analysis(frame, trigger\_reason, telematics)  
            return vlm\_feedback  
              
        return None

\# Simulation  
if \_\_name\_\_ \== "\_\_main\_\_":  
    monitor \= SafetyAndHealthMonitorPipeline(edge\_vlm\_path="quantized\_vlm.gguf")  
    \# telematics\_data \= {"continuous\_driving\_min": 65, "speed\_kmh": 35}  
    \# result \= monitor.process\_stream\_frame(dummy\_frame, telematics\_data)  
    \# print(result)

## **4\. CÂU HỎI GIẢI TRÌNH (DOCUMENTATION)**

Ứng viên trả lời ngắn gọn, đi thẳng vào bản chất kỹ thuật cho các câu hỏi sau:

> 1. **Thiết Kế Kiến Trúc Tháp 2 Tầng (Two-tier Cascade Latency Optimization):**  
   * Trong một hệ thống giám sát người lái theo thời gian thực (Real-time Stream), việc gọi VLM liên tục ở mọi frame là điều bất khả thi. Bạn phân chia trách nhiệm giữa Tier 1 (Lightweight CV) và Tier 2 (VLM) như thế nào? Kể tên các thuật toán/chỉ số cụ thể (Ví dụ: EAR, MAR, Head Pose Euler Angles, YOLO) mà bạn sẽ sử dụng ở Tier 1 để quyết định thời điểm "đắt giá" cần gọi Tier 2\.  
> 2. **Kiểm Soát Rủi Ro Y Tế & Pháp Lý (Medical Disclaimer & Ethical Guardrails):**  
   * Nếu người dùng có các dấu hiệu sức khỏe bất thường rõ rệt (Ví dụ: sắc mặt rất tái, mắt lờ đờ), VLM rất dễ tự động đưa ra các dự đoán mang tính y khoa (Ví dụ: *"Bạn có dấu hiệu tụt huyết áp/suy nhược"*). Làm thế nào để bạn đảm bảo 100% mô hình KHÔNG BAO GIỜ vi phạm các ranh giới này thông qua kết hợp giữa **Prompt Engineering**, **Grammar/Negative Constraints**, và **Deterministic Rule-based Post-filtering**?  
> 3. **Bảo Vệ Quyền Riêng Tư & Dữ Liệu Nhạy Cảm (Privacy-Preserving Health Baseline):**  
   * Để phát hiện trạng thái "mệt mỏi hơn ngày thường", hệ thống cần so sánh với lịch sử nhiều ngày của người dùng. Làm sao bạn xây dựng được bộ nhớ lịch sử này (Long-term Baseline) trên thiết bị nhúng mà KHÔNG cần phải lưu trữ các bức ảnh khuôn mặt riêng tư của người dùng vào bộ nhớ máy, tuân thủ các quy định bảo mật dữ liệu cá nhân (GDPR / Privacy Laws)?  
> 4. **Tích Hợp Ngữ Cảnh Phương Tiện (Telematics-VLM Fusion):**  
   * Làm thế nào để kết hợp giữa dữ liệu cảm biến phương tiện (speed, continuous\_driving\_time, weather) và phân tích hình ảnh của VLM để đưa ra các lời nhắc có tính thuyết phục cao? (Ví dụ: Nếu lái xe liên tục 2 tiếng \+ trời nắng 35 độ C \+ VLM thấy mặt bóng dầu/mệt mỏi \-\> AI sẽ đưa ra lời khuyên gì khác so với khi xe đang dừng đỗ?).