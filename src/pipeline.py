"""Module Safety & Driver Health Monitor — pipeline cascade 2 tầng.

Hoàn thiện skeleton của đề bài (TODO 1-4). Kiến trúc: docs/01-two-tier-cascade.md.
Chạy demo: python -m src.demo
"""

import time
from typing import Any, Dict, Optional

import numpy as np

try:  # chạy được cả dạng package (python -m src.demo) lẫn script trong src/
    from .guardrails import MedicalGuardrails
    from .health_baseline import HealthBaseline
    from .tier1 import MediaPipeLandmarkBackend, MockLandmarkBackend, Tier1Analyzer
    from .vlm_backend import LlamaCppVLMBackend, MockVLMBackend
except ImportError:
    from guardrails import MedicalGuardrails
    from health_baseline import HealthBaseline
    from tier1 import MediaPipeLandmarkBackend, MockLandmarkBackend, Tier1Analyzer
    from vlm_backend import LlamaCppVLMBackend, MockVLMBackend

VLM_COOLDOWN_SEC = 180        # khóa trigger cùng loại 3 phút, tránh spam
PERIODIC_VLM_INTERVAL_SEC = 300
LONG_DRIVE_TRIGGER_MIN = 60

# Cảnh báo Tier 1 — tài sản tĩnh duyệt sẵn, không phải text model sinh,
# nên không đi qua guardrail và không bị VLM làm chậm (SLA < 300ms)
IMMEDIATE_ALERTS = {
    "T0_eyes_closed": "CẢNH BÁO: Báo động! Hãy tập trung lái xe!",
    "T1_phone_or_head_down": "CẢNH BÁO: Vui lòng bỏ điện thoại xuống và nhìn đường!",
    "T2_perclos_fatigue": "Bạn có vẻ buồn ngủ, chú ý tập trung nhé.",
    "T3_frequent_yawning": "Bạn có vẻ buồn ngủ, chú ý tập trung nhé.",
    "T4_repeated_head_turns": "Chú ý quan sát phía trước nhé.",
}


class SafetyAndHealthMonitorPipeline:
    def __init__(self, edge_vlm_path: str, device: str = "cuda",
                 profile_id: int = 1, db_path: str = ":memory:",
                 tier1_backend=None):
        self.device = device
        self.profile_id = profile_id
        self.baseline = HealthBaseline(db_path)
        self.guardrails = MedicalGuardrails()
        self.init_cascade_models(edge_vlm_path, tier1_backend)
        self._last_vlm_ts: Dict[str, float] = {}
        self._last_periodic_ts = 0.0

    # ------------------------------------------------------------------
    def init_cascade_models(self, vlm_path: str, tier1_backend=None):
        """TODO 1: Init Tier 1 (Face Landmark EAR, Head Pose) và Tier 2 (VLM).

        Cả hai tầng đều cắm backend: dùng model thật nếu môi trường có,
        fallback mock để demo/test chạy được ở mọi nơi.
        """
        print(f"Loading Tier 1 (CV) & Tier 2 (Edge VLM) on {self.device}...")
        if tier1_backend is None:
            try:
                tier1_backend = MediaPipeLandmarkBackend()
                print("Tier 1: MediaPipe FaceLandmarker (real)")
            except ImportError:
                tier1_backend = MockLandmarkBackend()
                print("Tier 1: mock landmark backend (mediapipe not installed)")
        self.tier1 = Tier1Analyzer(backend=tier1_backend)

        try:
            self.vlm = LlamaCppVLMBackend(vlm_path)
            print(f"Tier 2: llama.cpp VLM (real) — {vlm_path}")
        except (ImportError, FileNotFoundError, ValueError):
            self.vlm = MockVLMBackend()
            print("Tier 2: mock VLM backend (no model available)")

    # ------------------------------------------------------------------
    def tier1_fast_stream_check(self, frame: np.ndarray,
                                now: Optional[float] = None) -> Dict[str, Any]:
        """TODO 2: [Tier 1 - 10 FPS] EAR, head pose, phone + temporal state."""
        return self.tier1.analyze(frame, now=now)

    # ------------------------------------------------------------------
    def enforce_medical_guardrails(self, raw_vlm_text: str,
                                   trigger_reason: str = "default") -> str:
        """TODO 3: lọc tất định — vi phạm là thay TOÀN BỘ câu bằng template."""
        return self.guardrails.enforce(raw_vlm_text, trigger_reason)

    # ------------------------------------------------------------------
    def tier2_run_vlm_context_analysis(self, frame: np.ndarray,
                                       trigger_reason: str,
                                       telematics: Dict[str, Any],
                                       delta_text: str = "") -> Optional[str]:
        """TODO 4: [Tier 2 - event-driven] VLM structured output -> template bank.

        VLM chỉ trả JSON enum (không free text); renderer ánh xạ sang câu
        duyệt sẵn; kết quả vẫn qua enforce_medical_guardrails làm lưới thứ hai.
        """
        print(f"Running Tier 2 VLM due to trigger: {trigger_reason}")
        try:
            vlm_json = self.vlm.generate(frame, trigger_reason, delta_text, telematics)
        except Exception:
            vlm_json = None  # fail-closed: validator sẽ đổ về fallback
        rendered = self.guardrails.validate_and_render(vlm_json, trigger_reason)
        if not rendered:  # looks_normal — không nhắc gì
            return None
        return self.enforce_medical_guardrails(rendered, trigger_reason)

    # ------------------------------------------------------------------
    def process_stream_frame(self, frame: np.ndarray,
                             telematics: Dict[str, Any],
                             now: Optional[float] = None) -> Optional[str]:
        """Hàm chính xử lý từng frame theo kiến trúc Cascade."""
        now = time.monotonic() if now is None else now

        # 1. Tier 1 fast check (< 50ms)
        t1 = self.tier1_fast_stream_check(frame, now=now)

        # 2. Khẩn cấp T0/T1: phát ngay câu tĩnh duyệt sẵn, không đụng VLM
        if t1["immediate_alert"] in ("T0_eyes_closed", "T1_phone_or_head_down"):
            return IMMEDIATE_ALERTS[t1["immediate_alert"]]

        # 3. T2-T4: nhắc nhẹ tức thời + VLM nâng cấp lời nhắc sau (cooldown)
        if t1["trigger_vlm_needed"]:
            reason = t1["trigger_reason"]
            instant = IMMEDIATE_ALERTS.get(reason)
            if self._cooldown_ok(reason, now):
                vlm_text = self.tier2_run_vlm_context_analysis(frame, reason, telematics)
                return vlm_text or instant
            return instant

        # 4. T5: lái liên tục > 60 phút (telematics thuần)
        if telematics.get("continuous_driving_min", 0) > LONG_DRIVE_TRIGGER_MIN:
            if self._cooldown_ok("T5_long_driving", now):
                return self.tier2_run_vlm_context_analysis(
                    frame, "T5_long_driving", telematics)

        # 5. T6/T7: định kỳ 5 phút — trích feature bằng CV, so baseline (Khối 3)
        if now - self._last_periodic_ts >= PERIODIC_VLM_INTERVAL_SEC:
            self._last_periodic_ts = now
            return self._periodic_health_check(frame, t1, telematics, now)

        return None

    # ------------------------------------------------------------------
    def _periodic_health_check(self, frame: np.ndarray, t1: Dict[str, Any],
                               telematics: Dict[str, Any],
                               now: float) -> Optional[str]:
        features = self._extract_health_features(frame, t1)
        light_bucket = telematics.get("light_bucket", 1)
        self.baseline.add_sample(self.profile_id, int(time.time()),
                                 light_bucket, features)
        verdict = self.baseline.check_anomaly(self.profile_id, light_bucket, features)
        if verdict["is_anomaly"] and self._cooldown_ok("T7_baseline_anomaly", now):
            return self.tier2_run_vlm_context_analysis(
                frame, "T7_baseline_anomaly", telematics,
                delta_text=verdict["delta_text"])
        return None

    def _extract_health_features(self, frame: np.ndarray,
                                 t1: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Trích feature sức khỏe từ landmark + thống kê màu (không cần VLM).

        Demo dùng chỉ số hành vi có sẵn từ Tier 1; các feature màu (quầng thâm,
        tái nhợt) cần crop ROI theo landmark + chuyển Lab — trả None khi backend
        không cung cấp (mock), đúng quy ước NULL = không có dữ liệu.
        """
        raw = t1.get("raw")
        return {
            "eye_openness": raw.ear if raw and raw.face_found else None,
            "perclos": t1.get("perclos"),
            "yawn_rate": float(t1.get("yawn_count_10min", 0)),
            "blink_rate": None,
            "eye_darkness": getattr(raw, "eye_darkness", None),
            "eye_puffiness": getattr(raw, "eye_puffiness", None),
            "skin_paleness": getattr(raw, "skin_paleness", None),
            "lip_color_index": getattr(raw, "lip_color_index", None),
        }

    def _cooldown_ok(self, reason: str, now: float) -> bool:
        last = self._last_vlm_ts.get(reason)
        if last is not None and now - last < VLM_COOLDOWN_SEC:
            return False
        self._last_vlm_ts[reason] = now
        return True


if __name__ == "__main__":
    monitor = SafetyAndHealthMonitorPipeline(edge_vlm_path="quantized_vlm.gguf")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    telematics_data = {"continuous_driving_min": 65, "speed_kmh": 35}
    result = monitor.process_stream_frame(dummy_frame, telematics_data)
    print(result)
