# Khối 2 — System Prompt & Medical Guardrails (Anti-Medical Diagnosis)

## 1. Mô hình mối đe dọa

VLM thấy "mặt tái, môi tím, mắt lờ đờ" sẽ có xu hướng tự suy ra chẩn đoán ("dấu hiệu thiếu máu", "tụt huyết áp"). Vi phạm này mang rủi ro pháp lý (thiết bị không phải thiết bị y tế được cấp phép) và rủi ro an toàn (chẩn đoán sai làm người dùng chủ quan hoặc hoảng loạn).

**Nguyên tắc: không tin bất kỳ lớp nào 100%. Phòng thủ theo chiều sâu (defense-in-depth), lớp cuối cùng là deterministic — không phụ thuộc xác suất của model.**

```
VLM input ──► [Lớp 1: System Prompt] ──► [Lớp 2: Constrained Decoding]
                                                     │
Loa/màn hình ◄── [Lớp 4: Safe Fallback] ◄── [Lớp 3: Rule-based Post-filter]
                        ▲ (chỉ khi Lớp 3 phát hiện vi phạm)
```

## 2. Lớp 1 — System Prompt (kiểm soát hành vi)

```text
You are a Caring Vehicle AI Companion for a driver. You speak Vietnamese.

## IDENTITY & HARD LIMITS (NON-NEGOTIABLE)
1. You are NOT a doctor, nurse, or medical device. You CANNOT and MUST NOT:
   - Name any disease, medical condition, or clinical state
     (NO: thiếu máu, đột quỵ, tụt huyết áp, suy nhược, sốt, viêm, bệnh...)
   - Use clinical/diagnostic verbs: chẩn đoán, triệu chứng, dấu hiệu bệnh lý
   - Suggest medication, treatment, or that the user "has" anything medical
2. You MAY ONLY:
   - Make gentle, caring observations about how the driver LOOKS today
     compared to their usual self ("trông bạn có vẻ hơi mệt hơn thường ngày")
   - Suggest rest, hydration, pulling over safely, or ending the trip early
   - Remind about traffic safety (helmet, focus, speed) tied to current context

## INPUT CONTEXT (provided each call)
- Trigger reason, baseline deltas (text), and vehicle telematics:
  vehicle_state (moving/stopped), speed_kmh, continuous_driving_min,
  ambient_temp_c, weather. Use them to pick severity and trip_factor:
  e.g. long_drive + hot_weather + tired face => "recommend_rest_now".

## OUTPUT CONTRACT
- Respond ONLY with a JSON object matching the provided schema
  (observation / severity / context_slots). No prose, no explanations.
- If unsure whether the driver looks unwell vs just lighting/angle,
  choose observation="looks_normal", severity="none".

## EXAMPLES
Input: pale face vs baseline, driving 125 min, 35°C sunny, moving
GOOD: {"observation":"looks_more_tired_than_usual","severity":"recommend_rest_now",
       "context_slots":{"trip_factor":"long_drive_hot_weather","vehicle_state":"moving"}}
BAD (NEVER): any free text, any medical wording, any field outside the schema.
```

(Ví dụ GOOD/BAD ở mức *lời văn* — "Hôm nay trông bạn có vẻ hơi mệt..." vs "Bạn có dấu hiệu thiếu máu" — chuyển xuống tài liệu template bank, vì lời văn giờ thuộc trách nhiệm template, không thuộc model.)

Điểm thiết kế đáng chú ý:

- **Cho phép rõ ràng (allowlist) quan trọng hơn cấm (denylist)**: prompt định nghĩa chính xác model *được nói gì* (quan sát bề ngoài + khuyến nghị nghỉ ngơi), không chỉ liệt kê điều cấm — thu hẹp không gian đầu ra ngay từ đầu.
- **Output contract 1 câu theo template** làm đầu ra dễ kiểm soát và dễ kiểm tra ở lớp sau.
- **Few-shot GOOD/BAD** hiệu quả hơn mô tả trừu tượng với model nhỏ 2B.

## 3. Lớp 2 — Constrained Decoding (kiểm soát tại lúc sinh token)

Prompt là xác suất — model nhỏ quantized vẫn có thể trượt. Lớp này **đóng hoàn toàn không gian đầu ra**: VLM không được sinh câu tự do tới người dùng. VLM chỉ được xuất **JSON theo schema enum cố định** — chọn *ý định*, không viết *lời văn*:

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

Cơ chế ép schema:

1. **Grammar constraint (GBNF — llama.cpp)**: grammar sinh tự động từ JSON schema trên, mọi giá trị là enum đóng — model **không có token nào** để viết chữ tự do. Đây là ràng buộc thực thi tại decoder, không phải quy ước trong prompt.
2. **Deterministic validator sau decode**: parse JSON, từ chối mọi key/value ngoài schema (kể cả khi grammar lỗi hoặc backend không hỗ trợ GBNF — validator là lưới độc lập).
3. **Renderer bằng code**: ánh xạ `(observation, severity, trip_factor, vehicle_state)` → 1 câu trong **ngân hàng template đã duyệt trước** (~20–30 câu, viết bởi người, duyệt bởi product/pháp lý). Lời văn tới người dùng do người viết 100%, model chỉ chọn tình huống.
4. **Cấu hình sampling bảo thủ**: `temperature ≤ 0.3`, `max_tokens ≈ 80` — đủ cho JSON, không hơn.

Chấp nhận đánh đổi: lời nhắc kém đa dạng hơn free text. Bù bằng ngân hàng template đủ lớn và biến thể ngẫu nhiên (chọn 1 trong 3 câu cùng nghĩa) — đa dạng nằm ở phía tất định, không phải phía model.

## 4. Lớp 3 — Deterministic Rule-based Post-filter (lưới an toàn thứ hai)

Với kiến trúc structured output ở Lớp 2, text tới người dùng luôn là template duyệt sẵn — về nguyên tắc không cần lọc nữa. Post-filter vẫn giữ vì 2 lý do: (1) hệ thống có thể được cấu hình chạy **free-text mode** khi thử nghiệm/so sánh chất lượng; (2) phòng thủ chiều sâu — bảo vệ cả trường hợp ai đó sửa template bank sai quy trình. Lọc **mọi** text trước khi ra loa, bất kể nguồn:

```
raw_text → chuẩn hóa (lowercase, bỏ dấu cách thừa, NFC normalize)
         → quét banned list (regex, cả bản có dấu và không dấu:
           "thiếu máu" và "thieu mau")
         → hit? → THAY TOÀN BỘ câu bằng safe template (Lớp 4)
         → miss? → kiểm tra thêm: độ dài > 200 ký tự? chứa "%"/số đo
           y tế (mmHg, bpm)? → cũng thay bằng template
         → pass → phát ra loa/màn hình + ghi audit log
```

Quy tắc quan trọng:

- **Không sửa từng phần câu** (không thay từ cấm bằng từ khác rồi giữ phần còn lại) — câu bị vá dễ giữ nguyên *hàm ý* chẩn đoán. Vi phạm = vứt cả câu, dùng template.
- **Fail-closed**: mọi lỗi runtime (VLM timeout, filter exception, text rỗng) đều trả về template an toàn, không bao giờ trả raw output.
- **Banned list là config** (file JSON riêng), đội pháp lý/product cập nhật được không cần sửa code; mỗi bản phát hành kèm bộ test tự động: chạy N câu đầu ra mẫu chứa từ cấm qua filter, yêu cầu 100% bị chặn.
- **Audit log**: ghi lại (timestamp, trigger_reason, raw đã bị chặn hay pass, template dùng) — không ghi ảnh — phục vụ chứng minh tuân thủ.

## 5. Lớp 4 — Safe Fallback Templates

Khi Lớp 3 chặn hoặc hệ thống lỗi, chọn template theo `trigger_reason`:

| Trigger | Template |
|---|---|
| Mệt mỏi / lệch baseline | "Hôm nay trông bạn có vẻ hơi mệt hơn thường ngày, hãy cân nhắc nghỉ ngơi vài phút trước khi tiếp tục hành trình nhé." |
| Lái xe liên tục lâu | "Bạn đã lái xe khá lâu rồi, dừng chân thư giãn một chút cho tỉnh táo nhé." |
| Mặc định / lỗi hệ thống | "Bạn nhớ giữ sức khỏe và lái xe cẩn thận nhé." |

**Phạm vi guardrail — làm rõ với kênh cảnh báo Tier 1**: cảnh báo khẩn cấp của Tier 1 (buzzer, TTS pre-recorded như "Hãy tập trung lái xe!") là tài sản tĩnh duyệt sẵn, không phải text do model sinh — không thể vi phạm y tế theo định nghĩa, nên không đi qua guardrail và không bị VLM làm chậm. Guardrail bao trọn kênh còn lại: **mọi text có nguồn gốc từ model**.

## 6. Vì sao tổ hợp này bảo đảm "không bao giờ vi phạm"

Luận điểm cốt lõi: **lời văn tới người dùng không bao giờ do model viết.** Model chỉ chọn một điểm trong không gian hữu hạn `observation × severity × context_slots` (~vài chục tổ hợp), mỗi tổ hợp ánh xạ tất định sang câu người viết đã duyệt. Không gian vi phạm y tế bị loại từ lúc thiết kế template bank, không phải bị "lọc" lúc runtime.

| Lớp | Bản chất | Vai trò |
|---|---|---|
| 1. System Prompt | Xác suất | Nâng chất lượng lựa chọn intent (chọn đúng mức severity), không gánh trách nhiệm an toàn |
| 2. Schema-constrained output + validator + renderer | **Tất định** | **Lớp bảo đảm chính**: model không có kênh phát free text |
| 3. Post-filter banned list | **Tất định** | Lưới thứ hai cho free-text mode thử nghiệm và lỗi quy trình template |
| 4. Fallback template | **Tất định** | Mọi nhánh lỗi (timeout, JSON hỏng, validator reject) đều đổ về đây |

So sánh với phương án chỉ dùng banned-list filter trên free text: filter từ khóa không bao giờ đóng được không gian diễn đạt vòng ("cơ thể bạn đang thiếu sắt" không chứa từ cấm nào) — nên free text **không thể** đạt cam kết "không bao giờ". Structured output đạt được vì đã đổi bài toán: từ *kiểm duyệt ngôn ngữ tự nhiên* (mở, không quyết định được) sang *chọn phần tử trong tập đóng* (kiểm chứng được bằng review từng template một lần duy nhất).

Red-team định kỳ vẫn chạy (bộ ảnh "trông ốm" + prompt injection thử ép model thoát schema) — mục tiêu kiểm chứng validator và đo chất lượng chọn intent, không phải vá lỗ hổng ngôn ngữ.
