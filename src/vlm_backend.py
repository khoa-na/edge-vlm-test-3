"""Tier 2 — Edge VLM backend, cắm được (mock mặc định, llama.cpp khi có model).

Cả hai backend trả JSON theo schema enum trong guardrails.py — không backend
nào được trả free text tới người dùng.
"""

import json
from typing import Any, Dict, Optional

import numpy as np

SYSTEM_PROMPT = """You are a Caring Vehicle AI Companion for a driver.

## IDENTITY & HARD LIMITS (NON-NEGOTIABLE)
1. You are NOT a doctor, nurse, or medical device. NEVER name diseases,
   medical conditions, or use diagnostic language.
2. You only assess how the driver LOOKS compared to their usual baseline
   and whether they should rest.

## INPUT CONTEXT
Trigger reason, baseline deltas (text), vehicle telematics:
vehicle_state, speed_kmh, continuous_driving_min, ambient_temp_c, weather.

## OUTPUT CONTRACT
Respond ONLY with JSON: {"observation": ..., "severity": ...,
"context_slots": {"trip_factor": ..., "vehicle_state": ...}}
using only the allowed enum values. No prose. If unsure whether the driver
looks unwell vs just lighting/angle: observation="looks_normal", severity="none".
"""


def build_user_prompt(trigger_reason: str, delta_text: str,
                      telematics: Dict[str, Any]) -> str:
    return (
        f"Trigger: {trigger_reason}\n"
        f"Baseline deltas: {delta_text or 'none'}\n"
        f"Telematics: {json.dumps(telematics, ensure_ascii=False)}\n"
        "Assess the driver in the attached cabin frame."
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


class LlamaCppVLMBackend:
    """Backend thật: model VLM quantized GGUF qua llama-cpp-python.

    Dùng: LlamaCppVLMBackend("qwen2-vl-2b-q4.gguf", mmproj="mmproj.gguf")
    GBNF grammar sinh từ schema enum ép model chỉ trả JSON hợp lệ.
    """

    def __init__(self, model_path: str, mmproj: Optional[str] = None,
                 n_ctx: int = 2048):
        from llama_cpp import Llama  # ImportError nếu chưa cài
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        handler = Qwen25VLChatHandler(clip_model_path=mmproj) if mmproj else None
        self._llm = Llama(model_path=model_path, chat_handler=handler,
                          n_ctx=n_ctx, verbose=False)

    def generate(self, frame: np.ndarray, trigger_reason: str,
                 delta_text: str, telematics: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import cv2
        ok, buf = cv2.imencode(".jpg", frame)
        image_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text",
                     "text": build_user_prompt(trigger_reason, delta_text, telematics)},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0.3, max_tokens=80,
        )
        return json.loads(out["choices"][0]["message"]["content"])
