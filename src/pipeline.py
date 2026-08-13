"""Module Safety & Driver Health Monitor — pipeline cascade 2 tầng.

Hoàn thiện skeleton của đề bài (TODO 1-4). Kiến trúc: docs/01-two-tier-cascade.md.
Chạy demo: python -m src.demo
"""

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

try:  # chạy được cả dạng package (python -m src.demo) lẫn script trong src/
    from .guardrails import MedicalGuardrails
    from .health_baseline import HealthBaseline
    from .tier1 import (PERCLOS_TRIGGER, PERCLOS_TRIGGER_FATIGUED,
                        MediaPipeLandmarkBackend, MockLandmarkBackend,
                        Tier1Analyzer)
    from .vlm_backend import LlamaCppVLMBackend, MockVLMBackend
except ImportError:
    from guardrails import MedicalGuardrails
    from health_baseline import HealthBaseline
    from tier1 import (PERCLOS_TRIGGER, PERCLOS_TRIGGER_FATIGUED,
                       MediaPipeLandmarkBackend, MockLandmarkBackend,
                       Tier1Analyzer)
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

# Pre-ride: câu nhắc tĩnh duyệt sẵn (không phải text model sinh)
PRE_RIDE_REMINDERS = {
    "helmet_strap": "Bạn chưa cài quai mũ bảo hiểm, cài chắc chắn rồi hãy đi nhé.",
    "mask": "Trời nắng bụi đấy, bạn nên đeo khẩu trang trước khi đi nhé.",
    "sunglasses": "Trời nắng bụi, bạn nên đeo kính bảo vệ mắt nhé.",
}
PRE_RIDE_CONF = 0.6


class MockObjectDetector:
    """Detector giả cho pre-ride check (helmet/mask/kính). Production: YOLO INT8
    cùng model với phone detection, bật đủ class khi xe chưa lăn bánh.
    Trả dict class -> confidence "đang đeo/đã cài".
    """

    def __init__(self, scenario: Optional[Dict[str, float]] = None):
        self.scenario = scenario or {}

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        return {"helmet_strap": 0.9, "mask": 0.9, "sunglasses": 0.9,
                **self.scenario}


def delivery_channel(telematics: Dict[str, Any]) -> str:
    """Chính sách phát theo trạng thái xe (docs/01 §6c): đang chạy nhanh ->
    âm thanh ngắn không màn hình; chậm/kẹt xe -> âm thanh đầy đủ;
    dừng đỗ -> âm thanh + màn hình.
    """
    speed = telematics.get("speed_kmh", 0)
    if speed <= 3:
        return "audio_and_screen"
    return "audio_short" if speed >= 30 else "audio_full"


class SafetyAndHealthMonitorPipeline:
    def __init__(self, edge_vlm_path: str, device: str = "cuda",
                 profile_id: int = 1, db_path: str = ":memory:",
                 tier1_backend=None, object_detector=None, vlm_mmproj=None):
        self.device = device
        self.profile_id = profile_id
        self.baseline = HealthBaseline(db_path)
        self.guardrails = MedicalGuardrails()
        self.object_detector = object_detector or MockObjectDetector()
        self.init_cascade_models(edge_vlm_path, tier1_backend, vlm_mmproj)
        self._last_vlm_ts: Dict[str, float] = {}
        self._last_periodic_ts = 0.0
        # VLM chạy async 1 slot: T2-T4 phát cảnh báo tĩnh ngay, kết quả VLM
        # trả ở frame sau — VLM không bao giờ chặn vòng lặp frame (NC-01)
        self._vlm_lock = threading.Lock()
        self._vlm_ready: Optional[str] = None
        self._vlm_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def init_cascade_models(self, vlm_path: str, tier1_backend=None,
                            vlm_mmproj=None):
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
            self.vlm = LlamaCppVLMBackend(vlm_path, mmproj=vlm_mmproj)
            print(f"Tier 2: llama.cpp VLM (real) — {vlm_path}")
        except (ImportError, FileNotFoundError, ValueError):
            self.vlm = MockVLMBackend()
            print("Tier 2: mock VLM backend (no model available)")

    # ------------------------------------------------------------------
    def tier1_fast_stream_check(self, frame: np.ndarray,
                                now: Optional[float] = None,
                                telematics: Optional[Dict[str, Any]] = None,
                                ) -> Dict[str, Any]:
        """TODO 2: [Tier 1 - 10 FPS] EAR, head pose, phone + temporal state.

        Telematics điều biến ngưỡng (docs/01 §6a): lái đêm hoặc quá 90 phút
        liên tục thì hạ ngưỡng PERCLOS 25% -> 20% (rủi ro tích lũy cao hơn).
        """
        telematics = telematics or {}
        fatigued_context = (
            telematics.get("continuous_driving_min", 0) > 90
            or telematics.get("time_of_day") == "night")
        thr = PERCLOS_TRIGGER_FATIGUED if fatigued_context else PERCLOS_TRIGGER
        return self.tier1.analyze(frame, now=now, perclos_threshold=thr)

    # ------------------------------------------------------------------
    def enforce_medical_guardrails(self, raw_vlm_text: str,
                                   trigger_reason: str = "default") -> str:
        """TODO 3: lọc tất định — vi phạm là thay TOÀN BỘ câu bằng template."""
        return self.guardrails.enforce(raw_vlm_text, trigger_reason)

    # ------------------------------------------------------------------
    def tier2_run_vlm_context_analysis(self, frame: np.ndarray,
                                       trigger_reason: str,
                                       telematics: Dict[str, Any],
                                       delta_text: str = "",
                                       max_severity: Optional[str] = None,
                                       ) -> Optional[str]:
        """TODO 4: [Tier 2 - event-driven] VLM structured output -> template bank.

        VLM chỉ trả JSON enum (không free text); renderer ánh xạ sang câu
        duyệt sẵn; kết quả vẫn qua enforce_medical_guardrails làm lưới thứ hai.
        max_severity="gentle" khi baseline còn provisional (docs/03).
        """
        print(f"Running Tier 2 VLM due to trigger: {trigger_reason}")
        try:
            vlm_json = self.vlm.generate(frame, trigger_reason, delta_text, telematics)
        except Exception:
            vlm_json = None  # fail-closed: validator sẽ đổ về fallback
        rendered = self.guardrails.validate_and_render(
            vlm_json, trigger_reason, max_severity=max_severity)
        if not rendered:  # looks_normal — không nhắc gì
            return None
        return self.enforce_medical_guardrails(rendered, trigger_reason)

    def _start_async_vlm(self, frame: np.ndarray, reason: str,
                         telematics: Dict[str, Any], delta_text: str = "",
                         max_severity: Optional[str] = None) -> None:
        """1 slot: job đang chạy thì bỏ trigger mới (không xếp hàng dài)."""
        if self._vlm_thread is not None and self._vlm_thread.is_alive():
            return

        frame_copy = frame.copy()
        telem_copy = dict(telematics)

        def worker():
            text = self.tier2_run_vlm_context_analysis(
                frame_copy, reason, telem_copy, delta_text, max_severity)
            with self._vlm_lock:
                self._vlm_ready = text

        self._vlm_thread = threading.Thread(target=worker, daemon=True)
        self._vlm_thread.start()

    def _pop_vlm_result(self) -> Optional[str]:
        with self._vlm_lock:
            result, self._vlm_ready = self._vlm_ready, None
        return result

    # ------------------------------------------------------------------
    def process_stream_frame(self, frame: np.ndarray,
                             telematics: Dict[str, Any],
                             now: Optional[float] = None) -> Optional[str]:
        """Hàm chính xử lý từng frame theo kiến trúc Cascade."""
        now = time.monotonic() if now is None else now

        # 1. Tier 1 fast check (< 50ms), ngưỡng điều biến theo telematics
        t1 = self.tier1_fast_stream_check(frame, now=now, telematics=telematics)

        # 2. Khẩn cấp T0/T1: phát ngay câu tĩnh duyệt sẵn, không đụng VLM
        if t1["immediate_alert"] in ("T0_eyes_closed", "T1_phone_or_head_down"):
            return IMMEDIATE_ALERTS[t1["immediate_alert"]]

        # 3. Kết quả VLM async từ frame trước (nếu có) — trả trước khi xét
        # trigger mới, để lời nhắc ngữ cảnh không bị nuốt
        pending = self._pop_vlm_result()
        if pending:
            return pending

        # 4. T2-T4: nhắc nhẹ tức thời NGAY; VLM chạy nền, kết quả frame sau
        if t1["trigger_vlm_needed"]:
            reason = t1["trigger_reason"]
            if self._cooldown_ok(reason, now):
                self._start_async_vlm(frame, reason, telematics)
            return IMMEDIATE_ALERTS.get(reason)

        # 5. T5: lái liên tục > 60 phút (telematics thuần; không khẩn cấp nên
        # gọi đồng bộ được, nhưng vẫn không nằm trên safety path)
        if telematics.get("continuous_driving_min", 0) > LONG_DRIVE_TRIGGER_MIN:
            if self._cooldown_ok("T5_long_driving", now):
                return self.tier2_run_vlm_context_analysis(
                    frame, "T5_long_driving", telematics)

        # 6. T6/T7: định kỳ 5 phút — trích feature bằng CV, so baseline (Khối 3)
        if now - self._last_periodic_ts >= PERIODIC_VLM_INTERVAL_SEC:
            self._last_periodic_ts = now
            return self._periodic_health_check(frame, t1, telematics, now)

        return None

    # ------------------------------------------------------------------
    def pre_ride_check(self, frames: List[np.ndarray],
                       telematics: Dict[str, Any]) -> List[str]:
        """Kiểm tra trước khi chạy (đề bài Mục 1): quai mũ luôn kiểm tra;
        khẩu trang/kính chỉ nhắc khi trời nắng bụi. Lấy đa số phiếu trên
        nhiều frame (docs/01 §2.3) — không kết luận từ 1 frame mờ.
        """
        votes: Dict[str, int] = {"helmet_strap": 0, "mask": 0, "sunglasses": 0}
        for frame in frames:
            det = self.object_detector.detect(frame)
            for cls in votes:
                if det.get(cls, 0.0) < PRE_RIDE_CONF:  # thiếu/chưa cài
                    votes[cls] += 1
        majority = len(frames) / 2
        reminders = []
        if votes["helmet_strap"] > majority:
            reminders.append(PRE_RIDE_REMINDERS["helmet_strap"])
        if telematics.get("weather") == "sunny_dusty":
            for cls in ("mask", "sunglasses"):
                if votes[cls] > majority:
                    reminders.append(PRE_RIDE_REMINDERS[cls])
        return reminders

    def end_trip(self) -> None:
        """Gọi cuối mỗi chuyến: tổng hợp phiên đo hôm nay vào health_daily
        (baseline 7 ngày chỉ sống được nếu bước này chạy) + dọn dữ liệu cũ.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.baseline.aggregate_day(self.profile_id, today)
        self.baseline.prune_old(int(time.time()))

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
            # Baseline provisional (< 7 ngày dữ liệu): chỉ cho nhắc mức nhẹ nhất
            cap = "gentle" if verdict["is_provisional"] else None
            return self.tier2_run_vlm_context_analysis(
                frame, "T7_baseline_anomaly", telematics,
                delta_text=verdict["delta_text"], max_severity=cap)
        return None

    def _extract_health_features(self, frame: np.ndarray,
                                 t1: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Trích feature sức khỏe từ landmark + thống kê màu Lab (không cần VLM).

        MediaPipe backend đo trực tiếp các feature màu (quầng thâm, tái nhợt,
        sắc môi — xem tier1._health_features); mock backend trả None cho các
        feature đó, đúng quy ước NULL = không có dữ liệu (docs/03).
        """
        raw = t1.get("raw")
        return {
            "eye_openness": raw.ear if raw and raw.face_found else None,
            "perclos": t1.get("perclos"),
            "yawn_rate": float(t1.get("yawn_count_10min", 0)),
            "blink_rate": t1.get("blink_rate"),
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
