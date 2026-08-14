# Khối 2 — System Prompt & Medical Guardrails (Anti-Medical Diagnosis)

## 1. Mô hình mối đe dọa

Khi nhìn thấy khuôn mặt tái hoặc đôi mắt lờ đờ, VLM rất dễ suy diễn sang những câu như "có dấu hiệu thiếu máu" hay "tụt huyết áp". Đây không chỉ là vấn đề câu chữ: thiết bị không phải thiết bị y tế được cấp phép, còn một nhận định sai có thể khiến người dùng lo lắng hoặc chủ quan không đúng lúc.

Vì vậy tôi không giao toàn bộ trách nhiệm cho system prompt. Prompt giúp model chọn đúng ý định, nhưng ranh giới an toàn cuối cùng phải được thực thi bằng code tất định. Pipeline dùng nhiều lớp độc lập để một lỗi ở model không đi thẳng tới loa hoặc màn hình.

```
VLM input ──► [Lớp 1: System Prompt] ──► [Lớp 2: Constrained Decoding]
                                                     │
Loa/màn hình ◄── [Lớp 4: Safe Fallback] ◄── [Lớp 3: Rule-based Post-filter]
                        ▲ (chỉ khi Lớp 3 phát hiện vi phạm)
```

## 2. Lớp 1 — System Prompt (kiểm soát hành vi)

```text
You are a Caring Vehicle AI Companion assessing a driver's fatigue.

## IDENTITY & HARD LIMITS (NON-NEGOTIABLE)
1. You are NOT a doctor, nurse, or medical device. You CANNOT and MUST NOT:
   - Name any disease, medical condition, or clinical state
     (NO: thiếu máu, đột quỵ, tụt huyết áp, suy nhược, sốt, viêm, bệnh...)
   - Use clinical/diagnostic verbs: chẩn đoán, triệu chứng, dấu hiệu bệnh lý
   - Suggest medication, treatment, or that the user "has" anything medical
2. You only assess how the driver LOOKS (fatigue/alertness) and whether
   they should rest. Safety reminders only — never medical claims.

## INPUT CONTEXT (provided each call)
- Trigger reason, baseline deltas (text), and vehicle telematics:
  vehicle_state (moving/stopped), speed_kmh, continuous_driving_min,
  ambient_temp_c, weather. Use them to pick severity and trip_factor:
  e.g. long_drive + hot_weather + tired face => "recommend_rest_now".

## DECISION RULE (apply in order)
1. Eyes closed/heavy or head droops -> observation="eyes_heavy".
2. Measured fatigue deltas and a consistent face
   -> "looks_more_tired_than_usual".
3. Long drive and visible tiredness -> "signs_of_long_trip_fatigue".
4. Clearly alert face and no meaningful deltas -> "looks_normal".

## OUTPUT CONTRACT
- Respond ONLY with a JSON object matching the provided schema
  (observation / severity / context_slots). No prose, no explanations.

## EXAMPLES
Input: pale face vs baseline, driving 125 min, 35°C sunny, moving
GOOD: {"observation":"looks_more_tired_than_usual","severity":"recommend_rest_now",
       "context_slots":{"trip_factor":"long_drive_hot_weather","vehicle_state":"moving"}}
BAD (NEVER): any free text, any medical wording, any field outside the schema.
```

Các ví dụ về câu nói cụ thể được để trong template bank, vì model không còn nhiệm vụ viết lời nhắc. Nó chỉ cần chọn đúng trạng thái và mức độ.

Prompt dùng allowlist để nói rõ model được phép làm gì: quan sát vẻ mệt mỏi và khuyến nghị nghỉ ngơi. Cách này rõ ràng hơn việc cố liệt kê mọi cách diễn đạt cần cấm. Đầu ra được yêu cầu ở dạng JSON để lớp sau có thể kiểm tra bằng máy. Với model 2B, ví dụ GOOD/BAD ngắn cũng giúp định hình hành vi tốt hơn một đoạn hướng dẫn dài và trừu tượng.

## 3. Lớp 2 — Constrained Decoding (kiểm soát tại lúc sinh token)

Prompt vẫn chỉ là hướng dẫn xác suất, đặc biệt với model nhỏ đã quantize. Vì thế đầu ra của VLM được giới hạn vào một JSON schema gồm các enum cố định. Model chọn ý định; nó không có trường nào để viết câu tự do cho người dùng.

```json
{
  "observation": "looks_more_tired_than_usual" | "looks_normal"
               | "eyes_heavy" | "signs_of_long_trip_fatigue",
  "severity":    "none" | "gentle" | "recommend_rest_now",
  "context_slots": {
    "trip_factor":    "none" | "long_drive" | "hot_weather" | "long_drive_hot_weather",
    "vehicle_state":  "moving" | "stopped"
  }
}
```

Từ decoder đến đầu ra cuối cùng có bốn chốt kiểm soát:

1. Grammar constraint (GBNF của llama.cpp): grammar sinh tự động từ JSON schema trên, mọi giá trị là enum đóng — model không có token nào để viết chữ tự do. Đây là ràng buộc thực thi tại decoder, không phải quy ước trong prompt.
2. Validator tất định sau decode: parse JSON, từ chối mọi key/value ngoài schema. Validator là lưới độc lập, vẫn đứng đó kể cả khi grammar lỗi hoặc backend không hỗ trợ GBNF.
3. Renderer bằng code: ánh xạ `(observation, severity, trip_factor, vehicle_state)` sang một câu trong ngân hàng template đã duyệt trước — khoảng 20–30 câu, người viết, product và pháp lý duyệt. Lời văn tới người dùng do người viết 100%, model chỉ chọn tình huống.
4. Sampling bảo thủ: `temperature ≤ 0.3`, `max_tokens ≈ 80` — đủ cho JSON, không hơn.

Cách làm này khiến lời nhắc ít đa dạng hơn free text. Đổi lại, tập câu có thể được đọc và duyệt trước. Nếu cần thêm biến thể, có thể bổ sung nhiều template cùng ý nghĩa mà không mở lại kênh sinh văn bản tự do.

## 4. Lớp 3 — Deterministic Rule-based Post-filter (lưới an toàn thứ hai)

Trong đường chạy chính, text tới người dùng luôn xuất phát từ template bank. Tôi vẫn giữ post-filter để bảo vệ hai trường hợp thực tế: chạy thử một backend free-text và template bị sửa sai quy trình. Vì filter nằm ngay trước kênh phát, mọi câu đều đi qua cùng một kiểm tra bất kể nguồn của nó.

```
raw_text → chuẩn hóa (lowercase, bỏ dấu cách thừa, NFC normalize)
         → quét banned list (regex, cả bản có dấu và không dấu:
           "thiếu máu" và "thieu mau")
         → hit? → THAY TOÀN BỘ câu bằng safe template (Lớp 4)
         → miss? → kiểm tra thêm: độ dài > 200 ký tự? chứa "%"/số đo
           y tế (mmHg, bpm)? → cũng thay bằng template
         → pass → phát ra loa/màn hình + ghi audit log
```

Một số lựa chọn trong post-filter:

- Không sửa từng phần câu (không thay từ cấm bằng từ khác rồi giữ phần còn lại) — câu bị vá rất dễ giữ nguyên hàm ý chẩn đoán. Vi phạm là vứt cả câu, dùng template.
- Fail-closed: mọi lỗi runtime (VLM timeout, filter exception, text rỗng) đều trả về template an toàn, không bao giờ trả raw output.
- Banned list là config (file JSON riêng) để đội pháp lý/product cập nhật được mà không sửa code. Mỗi bản phát hành kèm bộ test tự động: bắn N câu mẫu chứa từ cấm qua filter, yêu cầu chặn 100%.
- Audit log ghi timestamp, `trigger_reason`, loại sự kiện kiểm duyệt và từ khóa hoặc template liên quan. Log không lưu ảnh hay toàn bộ câu đầu vào.

## 5. Lớp 4 — Safe Fallback Templates

Khi Lớp 3 chặn hoặc hệ thống lỗi, chọn template theo `trigger_reason`:

| Trigger | Template |
|---|---|
| Mệt mỏi / lệch baseline | "Hôm nay trông bạn có vẻ hơi mệt hơn thường ngày, hãy cân nhắc nghỉ ngơi vài phút trước khi tiếp tục hành trình nhé." |
| Lái xe liên tục lâu | "Bạn đã lái xe khá lâu rồi, dừng chân thư giãn một chút cho tỉnh táo nhé." |
| Mặc định / lỗi hệ thống | "Bạn nhớ giữ sức khỏe và lái xe cẩn thận nhé." |

Cảnh báo khẩn cấp của Tier 1 là các file WAV dựng sẵn, chẳng hạn "Hãy tập trung lái xe!". Chúng không do model sinh nên được phát trực tiếp để tránh tăng độ trễ. Guardrail áp dụng cho toàn bộ phần lời nhắc có liên quan đến Tier 2.

## 6. Vì sao chọn đầu ra đóng thay cho kiểm duyệt free text

Lời văn cuối cùng không do model viết. VLM chỉ chọn một tổ hợp hữu hạn trong `observation × severity × context_slots`, sau đó code ánh xạ tổ hợp đó sang câu đã duyệt. Nhờ vậy, việc kiểm tra an toàn được thu gọn về review một tập template hữu hạn thay vì cố kiểm duyệt mọi câu mà mô hình ngôn ngữ có thể tạo ra.

| Lớp | Bản chất | Vai trò |
|---|---|---|
| 1. System prompt | Xác suất | Nâng chất lượng lựa chọn intent (chọn đúng mức severity), không gánh trách nhiệm an toàn |
| 2. Schema-constrained output + validator + renderer | Tất định | Lớp bảo đảm chính: model không có kênh phát free text |
| 3. Post-filter banned list | Tất định | Lưới thứ hai cho free-text mode thử nghiệm và lỗi quy trình template |
| 4. Fallback template | Tất định | Mọi nhánh lỗi (timeout, JSON hỏng, validator reject) đều đổ về đây |

Chỉ dùng banned list trên free text là chưa đủ, vì cùng một hàm ý y tế có thể được diễn đạt theo rất nhiều cách. Structured output thay đổi bản chất bài toán: thay vì kiểm duyệt một không gian ngôn ngữ mở, hệ thống chỉ cho phép model chọn trong một tập ý định đóng. Banned list lúc này là lưới dự phòng, không phải lớp bảo đảm chính.

Red-team vẫn cần thiết để kiểm tra validator, template và khả năng chọn intent dưới prompt injection. Tuy nhiên, kết quả red-team không thay thế cho thiết kế đầu ra đóng; nó xác nhận rằng các lớp đang hoạt động đúng như dự kiến.

## 7. Red-team định lượng — kết quả

`src/red_team.py` kiểm tra hai bề mặt tấn công mà hệ thống thực tế có thể gặp. Khi chạy đủ cả VLM thật và post-filter, bộ thử gồm 65 ca.

Không cần model, script chạy 42 ca trực tiếp trên post-filter. Khi truyền `--vlm-url` tới llama-server, nó chạy thêm 23 ca prompt injection qua VLM thật. Kết quả console luôn ghi rõ lớp nào đã chạy để tránh gộp nhầm hai chế độ.

**Lớp A — prompt injection vào VLM thật (23 ca).** Payload được đưa qua `delta_text`, trường `weather` của telematics và chữ nằm trong ảnh. Các ca thử yêu cầu model bỏ schema, đóng vai bác sĩ hoặc trả số đo y tế. Cả 23 đầu ra cuối cùng đều nằm trong tập template đã duyệt. Injection vẫn có thể ảnh hưởng tới lựa chọn `observation` hoặc `severity`, nhưng không tạo được câu tự do ngoài schema.

**Lớp B — kiểm tra trực tiếp post-filter (42 ca).** Ba mươi câu chứa từ ngữ hoặc số đo y tế được đưa thẳng vào `enforce()`, gồm cả biến thể bỏ dấu và chèn khoảng trắng. Mười hai câu nhắc an toàn hợp lệ được dùng để đo false-positive. Kết quả mong đợi là chặn toàn bộ nhóm đầu mà không sửa nhóm sau.

| Lớp | Loại | Số ca | Vượt rào | False-positive |
|---|---|---|---|---|
| A | Prompt injection vào VLM thật | 23 | 0 | — |
| B | Câu y tế phải chặn (must-block) | 30 | 0 | — |
| B | Câu an toàn không được chặn (must-pass) | 12 | — | 0 |
| **Tổng** | | **65** | **0 (0.0%)** | **0** |

Ở vòng chạy đầu tiên, hai câu có cụm "an toàn" bị chặn nhầm. Sau khi bỏ dấu, "an toàn" thành "an toan" và chứa chuỗi `toa`, vốn nằm trong danh sách cấm với nghĩa "toa thuốc". Filter đã được đổi từ so khớp substring sang biên từ `(?<!\w)term(?!\w)` và bổ sung ba test hồi quy. Sau sửa, cả 30 câu cần chặn vẫn bị giữ lại, còn false-positive giảm về 0.

Kết quả này củng cố lựa chọn dùng constrained decoding làm lớp chính. Post-filter vẫn hữu ích, nhưng bài thử cũng cho thấy filter từ khóa có thể gây false-positive nếu thiết kế không cẩn thận. Vì vậy hai lớp được giữ độc lập và có test riêng.
