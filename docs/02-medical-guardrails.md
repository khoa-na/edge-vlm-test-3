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

## OUTPUT CONTRACT
- Exactly ONE sentence, Vietnamese, warm and non-alarming tone.
- Template to follow: <quan sát nhẹ nhàng> + <khuyến nghị an toàn/nghỉ ngơi>.
- If you are unsure whether something crosses the medical line, DO NOT say it;
  fall back to: "Bạn nhớ giữ sức khỏe và lái xe cẩn thận nhé."

## EXAMPLES
GOOD: "Hôm nay trông bạn có vẻ hơi mệt hơn thường ngày, hãy cân nhắc dừng nghỉ vài phút trước khi đi tiếp nhé."
GOOD: "Bạn đã lái liên tục hơn một giờ dưới trời nắng nóng, tấp vào chỗ mát uống chút nước rồi hãy đi tiếp nhé."
BAD (NEVER): "Sắc mặt bạn tái, có thể bạn đang bị thiếu máu."
BAD (NEVER): "Môi bạn tím, đây là triệu chứng của vấn đề tim mạch."
```

Điểm thiết kế đáng chú ý:

- **Cho phép rõ ràng (allowlist) quan trọng hơn cấm (denylist)**: prompt định nghĩa chính xác model *được nói gì* (quan sát bề ngoài + khuyến nghị nghỉ ngơi), không chỉ liệt kê điều cấm — thu hẹp không gian đầu ra ngay từ đầu.
- **Output contract 1 câu theo template** làm đầu ra dễ kiểm soát và dễ kiểm tra ở lớp sau.
- **Few-shot GOOD/BAD** hiệu quả hơn mô tả trừu tượng với model nhỏ 2B.

## 3. Lớp 2 — Constrained Decoding (kiểm soát tại lúc sinh token)

Prompt là xác suất — model nhỏ quantized vẫn có thể trượt. Chặn thêm ở decode:

1. **Grammar constraint (GBNF — llama.cpp)**: ép đầu ra theo ngữ pháp
   `output ::= observation ", " recommendation "."` — độ dài giới hạn, không cho model tự do viết đoạn dài (rủi ro tăng theo độ dài).
2. **Logit bias / banned token sequences**: hạ logit của các token đầu chuỗi từ cấm (`bệnh`, `chẩn`, `triệu`, `đột quỵ`, `thiếu máu`, `huyết áp`, `suy`, `viêm`...) xuống −inf. Với tokenizer đa ngôn ngữ cần ban theo *chuỗi token* chứ không chỉ token đơn.
3. **Cấu hình sampling bảo thủ**: `temperature ≤ 0.3`, `max_tokens ≈ 60`. Nhiệt thấp làm model bám template; trần token ngắn cắt khả năng "diễn giải thêm".

## 4. Lớp 3 — Deterministic Rule-based Post-filter (lớp quyết định)

Lớp cuối, **thuần code, không xác suất** — đây là lớp bảo đảm "100%":

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

Template do người viết, đã được duyệt trước — nên đầu ra tới người dùng **luôn luôn** hoặc là câu VLM đã qua đủ 3 lớp, hoặc là câu duyệt sẵn. Không tồn tại đường nào khác ra loa.

## 6. Vì sao tổ hợp này bảo đảm "không bao giờ vi phạm"

| Lớp | Bản chất | Chặn được | Lỗ hổng còn lại |
|---|---|---|---|
| 1. System Prompt | Xác suất | ~95% trường hợp thường | Model nhỏ trượt khi input bất thường |
| 2. Constrained decoding | Bán tất định | Từ cấm ở mức token, độ dài | Cách diễn đạt vòng ("cơ thể bạn đang thiếu sắt") |
| 3. Post-filter | **Tất định** | Mọi chuỗi khớp banned list + heuristic | Từ cấm chưa có trong list |
| 4. Fallback + duyệt template | **Tất định** | Mọi trường hợp lỗi/nghi ngờ | — |

Lỗ hổng cuối (từ mới chưa vào list) xử lý bằng quy trình: red-team định kỳ đầu ra VLM với bộ ảnh "trông ốm", bổ sung list, và nhờ Lớp 2 giới hạn output theo template nên không gian diễn đạt vòng đã rất hẹp. Trách nhiệm pháp lý chốt tại: **mọi câu tới người dùng hoặc qua filter tất định, hoặc là template duyệt sẵn.**
