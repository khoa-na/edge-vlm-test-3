"""Medical guardrails: structured-output renderer + deterministic post-filter.

Hai cơ chế (xem docs/02-medical-guardrails.md):
1. StructuredRenderer — đường chính: VLM chỉ trả JSON enum, lời văn lấy từ
   template bank duyệt sẵn. Model không có kênh phát free text.
2. enforce_medical_guardrails — lưới thứ hai: lọc banned list trên MỌI text
   trước khi ra loa (phòng free-text mode thử nghiệm / lỗi quy trình template).

Fail-closed: mọi nhánh lỗi trả về fallback template, không bao giờ trả raw text.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_PATH = Path(__file__).parent / "config" / "guardrails_config.json"

# Không gian đầu ra đóng của VLM — validator từ chối mọi giá trị ngoài các enum này
ALLOWED_OBSERVATIONS = {
    "looks_more_tired_than_usual", "looks_normal",
    "eyes_heavy", "signs_of_long_trip_fatigue",
}
ALLOWED_SEVERITIES = {"none", "gentle", "recommend_rest_now"}
ALLOWED_TRIP_FACTORS = {"none", "long_drive", "hot_weather", "long_drive_hot_weather"}
ALLOWED_VEHICLE_STATES = {"moving", "stopped"}


def _load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_marks.replace("đ", "d").replace("Đ", "D")


class MedicalGuardrails:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or _load_config()
        self.fallbacks = self.config["fallback_templates"]
        self.template_bank = self.config["template_bank"]
        # Banned list quét cả bản có dấu và không dấu
        terms = self.config["banned_medical_terms"]
        self._banned = [(t.lower(), _strip_diacritics(t.lower())) for t in terms]
        self._banned_patterns = [p.lower() for p in self.config["banned_patterns"]]
        self.max_chars = self.config["max_output_chars"]
        self.audit_log: list = []

    # ------------------------------------------------------------------
    # Đường chính: structured output -> template duyệt sẵn
    # ------------------------------------------------------------------
    def validate_and_render(self, vlm_json: Any, trigger_reason: str) -> str:
        """Validate JSON của VLM theo schema enum, render từ template bank.

        Mọi vi phạm schema (key lạ, value ngoài enum, không parse được)
        đều fail-closed về fallback template.
        """
        try:
            if isinstance(vlm_json, str):
                vlm_json = json.loads(vlm_json)
            obs = vlm_json["observation"]
            sev = vlm_json["severity"]
            slots = vlm_json["context_slots"]
            trip = slots["trip_factor"]
            veh = slots["vehicle_state"]
            extra_keys = set(vlm_json) - {"observation", "severity", "context_slots"}
            extra_slots = set(slots) - {"trip_factor", "vehicle_state"}
            if extra_keys or extra_slots:
                raise ValueError(f"unexpected keys: {extra_keys | extra_slots}")
            if (obs not in ALLOWED_OBSERVATIONS or sev not in ALLOWED_SEVERITIES
                    or trip not in ALLOWED_TRIP_FACTORS
                    or veh not in ALLOWED_VEHICLE_STATES):
                raise ValueError("enum violation")
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            self._audit("schema_reject", trigger_reason, detail=str(e))
            return self._fallback(trigger_reason)

        key = f"{obs}|{sev}|{trip}|{veh}"
        text = self.template_bank.get(key)
        if text is None:
            # Tổ hợp hợp lệ nhưng chưa có template — fail-closed
            self._audit("template_missing", trigger_reason, detail=key)
            return self._fallback(trigger_reason)
        self._audit("rendered", trigger_reason, detail=key)
        return text  # "" nghĩa là looks_normal: không nhắc gì

    # ------------------------------------------------------------------
    # Lưới thứ hai: post-filter tất định trên free text (TODO 3 của đề)
    # ------------------------------------------------------------------
    def enforce(self, raw_text: str, trigger_reason: str = "default") -> str:
        """Quét banned list; vi phạm = thay TOÀN BỘ câu bằng fallback template.

        Không vá từng phần câu — câu bị vá dễ giữ nguyên hàm ý chẩn đoán.
        """
        try:
            if not raw_text or not raw_text.strip():
                self._audit("empty_output", trigger_reason)
                return self._fallback(trigger_reason)

            normalized = unicodedata.normalize("NFC", raw_text).lower()
            stripped = _strip_diacritics(normalized)

            for term, term_no_marks in self._banned:
                if term in normalized or term_no_marks in stripped:
                    self._audit("banned_term", trigger_reason, detail=term)
                    return self._fallback(trigger_reason)

            for pat in self._banned_patterns:
                if pat in stripped:
                    self._audit("banned_pattern", trigger_reason, detail=pat)
                    return self._fallback(trigger_reason)

            if len(raw_text) > self.max_chars:
                self._audit("too_long", trigger_reason, detail=len(raw_text))
                return self._fallback(trigger_reason)

            # Số đo kèm đơn vị y tế lọt qua pattern (vd "120/80")
            if re.search(r"\d+\s*/\s*\d+", raw_text):
                self._audit("numeric_reading", trigger_reason)
                return self._fallback(trigger_reason)

            self._audit("passed", trigger_reason)
            return raw_text
        except Exception as e:  # fail-closed cả khi chính filter lỗi
            self._audit("filter_error", trigger_reason, detail=repr(e))
            return self._fallback(trigger_reason)

    # ------------------------------------------------------------------
    def _fallback(self, trigger_reason: str) -> str:
        if "fatigue" in trigger_reason or "baseline" in trigger_reason:
            return self.fallbacks["fatigue"]
        if "driving" in trigger_reason or "long" in trigger_reason:
            return self.fallbacks["long_drive"]
        return self.fallbacks["default"]

    def _audit(self, event: str, trigger_reason: str, detail: Any = None) -> None:
        # Audit log phục vụ chứng minh tuân thủ — không bao giờ chứa ảnh
        self.audit_log.append(
            {"event": event, "trigger": trigger_reason, "detail": detail}
        )
