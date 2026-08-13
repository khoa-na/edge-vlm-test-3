"""Tier 2 — Edge VLM backend, cắm được (mock mặc định, llama.cpp khi có model).

Cả hai backend trả JSON theo schema enum trong guardrails.py — không backend
nào được trả free text tới người dùng.
"""

import json
import re
from typing import Any, Dict, Optional

import numpy as np

SYSTEM_PROMPT = """You are a Caring Vehicle AI Companion assessing a driver's fatigue.

## IDENTITY & HARD LIMITS (NON-NEGOTIABLE)
1. You are NOT a doctor, nurse, or medical device. NEVER name diseases,
   medical conditions, or use diagnostic language.
2. You only assess how the driver LOOKS (fatigue/alertness) and whether
   they should rest. Safety reminders only — never medical claims.

## INPUT CONTEXT
Trigger reason, MEASURED baseline deltas, vehicle telematics
(speed_kmh, continuous_driving_min, ambient_temp_c, weather).

## DECISION RULE (apply in order)
1. Eyes look closed, half-closed, heavy, or the head droops in the image
   -> observation="eyes_heavy".
2. Measured deltas indicate fatigue AND the face does not contradict them
   (tired, droopy, pale-looking, looking down) -> "looks_more_tired_than_usual".
3. No deltas, but continuous_driving_min > 60 and the face shows any tiredness
   -> "signs_of_long_trip_fatigue".
4. Face clearly alert AND no meaningful deltas -> "looks_normal".
Severity: "recommend_rest_now" when rule 1 applies or (rule 2/3 with long
drive or hot weather); "gentle" for milder cases; "none" only with rule 4.

## OUTPUT CONTRACT
Respond ONLY with JSON: {"observation": ..., "severity": ...,
"context_slots": {"trip_factor": ..., "vehicle_state": ...}}
using only the allowed enum values. No prose.
"""


def build_user_prompt(trigger_reason: str, delta_text: str,
                      telematics: Dict[str, Any]) -> str:
    return (
        f"Trigger: {trigger_reason}\n"
        f"Baseline deltas (MEASURED on-device, vs this driver's own 7-day "
        f"baseline): {delta_text or 'none'}\n"
        f"Telematics: {json.dumps(telematics, ensure_ascii=False)}\n"
        "Assess the driver in the attached cabin frame. The deltas above are "
        "measured evidence, not guesses — weigh them together with the image. "
        "Choose looks_normal ONLY if the image clearly contradicts the "
        "measurements. If deltas indicate fatigue and the face is consistent "
        "with it, pick the matching observation."
    )


class MockVLMBackend:
    """Suy luận giả lập bằng rule trên context — dùng cho demo/test không có model.

    Trả đúng schema enum như model thật với GBNF grammar sẽ trả.
    """

    def generate(self, frame: np.ndarray, trigger_reason: str,
                 delta_text: str, telematics: Dict[str, Any]) -> Dict[str, Any]:
        driving_min = telematics.get("continuous_driving_min", 0)
        temp = telematics.get("ambient_temp_c", 25)
        speed = telematics.get("speed_kmh", 0)
        vehicle_state = "moving" if speed > 3 else "stopped"

        long_drive = driving_min > 60
        hot = temp >= 33
        if long_drive and hot:
            trip_factor = "long_drive_hot_weather"
        elif long_drive:
            trip_factor = "long_drive"
        elif hot:
            trip_factor = "hot_weather"
        else:
            trip_factor = "none"

        if "baseline" in trigger_reason or delta_text:
            observation = "looks_more_tired_than_usual"
            severity = "recommend_rest_now" if (long_drive or hot) else "gentle"
        elif "perclos" in trigger_reason or "yawning" in trigger_reason:
            observation = "eyes_heavy"
            severity = "recommend_rest_now" if long_drive else "gentle"
        elif long_drive:
            observation = "signs_of_long_trip_fatigue"
            severity = "recommend_rest_now"
        else:
            observation = "looks_normal"
            severity = "none"
            trip_factor = "none"

        return {
            "observation": observation,
            "severity": severity,
            "context_slots": {
                "trip_factor": trip_factor,
                "vehicle_state": vehicle_state,
            },
        }


# GBNF grammar: enum ĐÓNG ở mức decoder — model không có token nào để viết
# chữ tự do (docs/02 Lớp 2). Validator trong guardrails.py vẫn là lưới độc lập.
OUTPUT_GRAMMAR = r'''
root ::= "{" ws "\"observation\"" ws ":" ws obs "," ws "\"severity\"" ws ":" ws sev "," ws "\"context_slots\"" ws ":" ws slots ws "}"
slots ::= "{" ws "\"trip_factor\"" ws ":" ws trip "," ws "\"vehicle_state\"" ws ":" ws veh ws "}"
obs ::= "\"looks_more_tired_than_usual\"" | "\"looks_normal\"" | "\"eyes_heavy\"" | "\"signs_of_long_trip_fatigue\""
sev ::= "\"none\"" | "\"gentle\"" | "\"recommend_rest_now\""
trip ::= "\"none\"" | "\"long_drive\"" | "\"hot_weather\"" | "\"long_drive_hot_weather\""
veh ::= "\"moving\"" | "\"stopped\""
ws ::= [ \t\n]*
'''


# JSON schema enum đóng — llama-server tự chuyển thành grammar ép tại decoder
# (tương đương GBNF, docs/02 Lớp 2)
OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"enum": ["looks_more_tired_than_usual", "looks_normal",
                                 "eyes_heavy", "signs_of_long_trip_fatigue"]},
        "severity": {"enum": ["none", "gentle", "recommend_rest_now"]},
        "context_slots": {
            "type": "object",
            "properties": {
                "trip_factor": {"enum": ["none", "long_drive", "hot_weather",
                                         "long_drive_hot_weather"]},
                "vehicle_state": {"enum": ["moving", "stopped"]},
            },
            "required": ["trip_factor", "vehicle_state"],
            "additionalProperties": False,
        },
    },
    "required": ["observation", "severity", "context_slots"],
    "additionalProperties": False,
}


class LlamaServerVLMBackend:
    """Backend thật qua llama-server (llama.cpp gốc, API OpenAI-compatible).

    Ưu điểm so với binding llama-cpp-python: hỗ trợ model mới (Qwen3.5,
    Gemma 4) từ day-1, không chờ binding cập nhật chat handler; đúng cách
    deploy edge thực tế (server process riêng, pipeline gọi HTTP localhost).

    Khởi động server:
      llama-server -m qwen3.5-2b-q4_k_m.gguf --mmproj mmproj-f16.gguf --port 8080
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 timeout_sec: float = 60.0):
        import urllib.request
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._timeout = timeout_sec
        # fail-fast nếu server chưa chạy -> pipeline fallback mock
        health = base_url.rstrip("/") + "/health"
        with urllib.request.urlopen(health, timeout=5) as r:
            if r.status != 200:
                raise ValueError(f"llama-server not healthy: {r.status}")

    def generate(self, frame: np.ndarray, trigger_reason: str,
                 delta_text: str, telematics: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import urllib.request

        import cv2
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        image_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text",
                     "text": build_user_prompt(trigger_reason, delta_text, telematics)},
                ]},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "driver_state",
                                "schema": OUTPUT_JSON_SCHEMA},
            },
            "temperature": 0.3,
            "max_tokens": 512,  # đủ chỗ cho thinking + JSON (model hybrid reasoning)
        }
        req = urllib.request.Request(
            self._url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            out = json.load(r)
        content = out["choices"][0]["message"]["content"] or ""
        # Model hybrid reasoning (Qwen3.5) có thể kèm thinking text quanh JSON;
        # cắt đúng object JSON — validator của guardrails vẫn là lưới cuối
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in VLM output: {content[:120]!r}")
        return json.loads(content[start:end + 1])


class LlamaCppVLMBackend:
    """Backend thật: model VLM quantized GGUF qua llama-cpp-python.

    Dùng: LlamaCppVLMBackend("qwen2-vl-2b-q4.gguf", mmproj="mmproj.gguf")
    mmproj (projector đa phương thức) BẮT BUỘC — không có thì model không
    nhận ảnh được; thiếu sẽ raise để pipeline fallback mock thay vì chạy mù.
    """

    def __init__(self, model_path: str, mmproj: Optional[str] = None,
                 n_ctx: int = 2048):
        from llama_cpp import Llama  # ImportError nếu chưa cài
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        if not mmproj:
            raise ValueError("mmproj is required for image input")
        handler = Qwen25VLChatHandler(clip_model_path=mmproj)
        self._llm = Llama(model_path=model_path, chat_handler=handler,
                          n_ctx=n_ctx, verbose=False)
        try:
            from llama_cpp import LlamaGrammar
            self._grammar = LlamaGrammar.from_string(OUTPUT_GRAMMAR)
        except Exception:
            self._grammar = None  # backend cũ: dựa vào json_object + validator

    def generate(self, frame: np.ndarray, trigger_reason: str,
                 delta_text: str, telematics: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import cv2
        # pipeline làm việc bằng RGB; imencode kỳ vọng BGR
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        image_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        kwargs = ({"grammar": self._grammar} if self._grammar
                  else {"response_format": {"type": "json_object"}})
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text",
                     "text": build_user_prompt(trigger_reason, delta_text, telematics)},
                ]},
            ],
            temperature=0.3, max_tokens=80, **kwargs,
        )
        return json.loads(out["choices"][0]["message"]["content"])
