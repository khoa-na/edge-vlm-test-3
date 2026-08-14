"""Unit tests: trigger logic, guardrails, baseline anomaly.

Chạy: python -m pytest tests/ -q  (từ thư mục gốc repo)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails import MedicalGuardrails
from health_baseline import HealthBaseline
from pipeline import (MockObjectDetector, SafetyAndHealthMonitorPipeline,
                      delivery_channel)
from tier1 import MockLandmarkBackend, Tier1Analyzer

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# Tier 1 — temporal triggers
# ----------------------------------------------------------------------
def _run_frames(analyzer, backend, metrics, seconds, fps=10, t0=0.0):
    backend.set_scenario(**metrics)
    out = None
    for i in range(int(seconds * fps)):
        out = analyzer.analyze(FRAME, now=t0 + i / fps)
    return out


def test_eyes_closed_over_1_5s_is_emergency():
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    out = _run_frames(analyzer, backend, dict(ear=0.10), seconds=2.0)
    assert out["immediate_alert"] == "T0_eyes_closed"


def test_normal_blink_no_alert():
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    out = _run_frames(analyzer, backend, dict(ear=0.10), seconds=0.3)  # chớp mắt
    assert out["immediate_alert"] is None


def test_phone_needs_3_consecutive_frames():
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    backend.set_scenario(ear=0.3, phone_conf=0.9)
    r1 = analyzer.analyze(FRAME, now=0.0)
    r2 = analyzer.analyze(FRAME, now=0.1)
    r3 = analyzer.analyze(FRAME, now=0.2)
    assert not r1["using_phone"] and not r2["using_phone"]
    assert r3["using_phone"] and r3["immediate_alert"] == "T1_phone_or_head_down"


def test_perclos_needs_30s_of_data():
    """PERCLOS không được kết luận khi chưa gom đủ 30s dữ liệu (NC-05)."""
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    # 10s nhắm mắt 40% duty cycle — PERCLOS thô ~0.4 nhưng cửa sổ < 30s
    out = None
    for i in range(100):
        backend.set_scenario(ear=0.10 if i % 5 < 2 else 0.30)
        out = analyzer.analyze(FRAME, now=i / 10)
    assert out["perclos"] == 0.0
    # thêm 25s nữa (tổng 35s) — giờ mới được kết luận
    for i in range(100, 350):
        backend.set_scenario(ear=0.10 if i % 5 < 2 else 0.30)
        out = analyzer.analyze(FRAME, now=i / 10)
    assert out["perclos"] > 0.25


def test_face_loss_gap_resets_eyes_timer():
    """Mất mặt >0.5s không được cộng dồn vào thời gian nhắm mắt (NC-06)."""
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    _run_frames(analyzer, backend, dict(ear=0.10), seconds=1.0)       # nhắm 1s
    backend.set_scenario(face_found=False)
    for i in range(10):                                               # mất mặt 1s
        analyzer.analyze(FRAME, now=1.0 + i / 10)
    out = _run_frames(analyzer, backend, dict(ear=0.10), 0.6, t0=2.0)  # nhắm 0.6s
    # 1.0 + 0.6 > 1.5 nhưng có gap ở giữa -> timer đã reset, không báo động giả
    assert out["immediate_alert"] is None


def test_perclos_threshold_modulated_by_telematics():
    """Lái quá 90 phút -> ngưỡng PERCLOS hạ 0.25 -> 0.20 (docs/01 §6a)."""
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    p.tier1.backend.set_scenario(ear=0.30)
    out = None
    # PERCLOS ~22%: dưới ngưỡng thường, trên ngưỡng mệt mỏi
    for i in range(400):
        p.tier1.backend.set_scenario(ear=0.10 if i % 9 < 2 else 0.30)
        out = p.tier1_fast_stream_check(
            FRAME, now=i / 10, telematics={"continuous_driving_min": 120})
    assert out["trigger_reason"] == "T2_perclos_fatigue"


def test_yawning_triggers_vlm():
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    t = 0.0
    for _ in range(3):  # 3 lần ngáp: MAR cao 2.5s rồi đóng miệng
        _run_frames(analyzer, backend, dict(ear=0.3, mar=0.75), 2.5, t0=t)
        t += 2.5
        out = _run_frames(analyzer, backend, dict(ear=0.3, mar=0.2), 1.0, t0=t)
        t += 1.0
    assert out["trigger_vlm_needed"]
    assert out["trigger_reason"] == "T3_frequent_yawning"


# ----------------------------------------------------------------------
# Guardrails
# ----------------------------------------------------------------------
@pytest.fixture
def guard():
    return MedicalGuardrails()


def test_banned_term_replaced_entirely(guard):
    out = guard.enforce("Bạn có dấu hiệu thiếu máu.", "T7_baseline_anomaly")
    assert "thiếu máu" not in out
    assert out == guard.fallbacks["fatigue"]


def test_banned_term_without_diacritics_caught(guard):
    out = guard.enforce("Huyet ap cua ban hoi thap.", "default")
    assert out == guard.fallbacks["default"]


def test_all_banned_terms_blocked(guard):
    for term in guard.config["banned_medical_terms"]:
        out = guard.enforce(f"Bạn đang bị {term} đấy.", "default")
        assert term not in out, f"banned term '{term}' leaked"


def test_safe_text_passes(guard):
    text = "Hôm nay trông bạn có vẻ hơi mệt, nghỉ chút nhé."
    assert guard.enforce(text, "default") == text


def test_empty_output_fail_closed(guard):
    assert guard.enforce("", "default") == guard.fallbacks["default"]


def test_numeric_reading_blocked(guard):
    out = guard.enforce("Chỉ số của bạn là 120/80.", "default")
    assert out == guard.fallbacks["default"]


def test_schema_validator_rejects_free_text(guard):
    out = guard.validate_and_render("Bạn bị thiếu máu.", "T7_baseline_anomaly")
    assert out == guard.fallbacks["fatigue"]


def test_schema_validator_rejects_unknown_enum(guard):
    bad = {"observation": "has_anemia", "severity": "gentle",
           "context_slots": {"trip_factor": "none", "vehicle_state": "moving"}}
    out = guard.validate_and_render(bad, "T7_baseline_anomaly")
    assert out == guard.fallbacks["fatigue"]


def test_schema_validator_renders_valid_intent(guard):
    ok = {"observation": "looks_more_tired_than_usual",
          "severity": "recommend_rest_now",
          "context_slots": {"trip_factor": "long_drive_hot_weather",
                            "vehicle_state": "moving"}}
    out = guard.validate_and_render(ok, "T7_baseline_anomaly")
    assert "nắng nóng" in out  # template duyệt sẵn, không phải text model


# ----------------------------------------------------------------------
# Health baseline
# ----------------------------------------------------------------------
def _day_start_ts(day: str) -> int:
    from datetime import datetime, timezone
    return int(datetime.strptime(day, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def _seed_baseline(hb, days=7, features=None):
    for d in range(days):
        day = f"2026-08-{d + 1:02d}"
        ts0 = _day_start_ts(day)
        feats = features(d) if features else {"eye_openness": 0.30, "perclos": 0.05}
        for s in range(5):
            hb.add_sample(1, ts0 + s * 300, 1, feats)
        hb.aggregate_day(1, day)


def test_no_anomaly_when_normal():
    hb = HealthBaseline()
    _seed_baseline(hb)
    v = hb.check_anomaly(1, 1, {"eye_openness": 0.30, "perclos": 0.05})
    assert not v["is_anomaly"]


def test_two_features_bad_direction_is_anomaly():
    hb = HealthBaseline()
    # nhiễu nhỏ theo ngày để std > 0
    _seed_baseline(hb, days=7, features=lambda d: {
        "eye_openness": 0.30 + 0.005 * (d % 3),
        "perclos": 0.05 + 0.002 * (d % 3)})
    v = hb.check_anomaly(1, 1, {"eye_openness": 0.15, "perclos": 0.40})
    assert v["is_anomaly"]
    assert "eye_openness" in v["anomalous_features"]


def test_single_strong_feature_needs_persistence():
    hb = HealthBaseline()
    _seed_baseline(hb, days=7,
                   features=lambda d: {"eye_openness": 0.30 + 0.005 * (d % 3)})
    v1 = hb.check_anomaly(1, 1, {"eye_openness": 0.10})  # phiên 1: chưa cờ
    assert not v1["is_anomaly"]
    v2 = hb.check_anomaly(1, 1, {"eye_openness": 0.10})  # phiên 2 liên tiếp: cờ
    assert v2["is_anomaly"]


def test_cold_start_needs_3_days():
    hb = HealthBaseline()
    _seed_baseline(hb, days=2)
    v = hb.check_anomaly(1, 1, {"eye_openness": 0.05})
    assert not v["is_anomaly"]  # chưa đủ dữ liệu — không kết luận


def test_null_feature_skipped():
    hb = HealthBaseline()
    _seed_baseline(hb)
    v = hb.check_anomaly(1, 1, {"eye_openness": None, "perclos": 0.05})
    assert not v["is_anomaly"]


def test_provisional_baseline_caps_severity(guard):
    """Baseline provisional -> hạ recommend_rest_now xuống gentle (docs/03)."""
    strong = {"observation": "looks_more_tired_than_usual",
              "severity": "recommend_rest_now",
              "context_slots": {"trip_factor": "none",
                                "vehicle_state": "stopped"}}
    out = guard.validate_and_render(strong, "T7_baseline_anomaly",
                                    max_severity="gentle")
    assert out == guard.template_bank[
        "looks_more_tired_than_usual|gentle|none|stopped"]


# ----------------------------------------------------------------------
# Pipeline end-to-end (mock backends)
# ----------------------------------------------------------------------
def test_emergency_alert_bypasses_vlm():
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf", tier1_backend=MockLandmarkBackend())
    p.tier1.backend.set_scenario(ear=0.10)
    out = None
    for i in range(25):
        out = p.process_stream_frame(FRAME, {"speed_kmh": 40}, now=i / 10)
    assert out is not None and "tập trung" in out


def test_long_driving_gets_guarded_vlm_response():
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf", tier1_backend=MockLandmarkBackend())
    p.tier1.backend.set_scenario(ear=0.30)
    out = p.process_stream_frame(
        FRAME, {"continuous_driving_min": 125, "speed_kmh": 52,
                "ambient_temp_c": 35}, now=0.0)
    assert out is not None
    for term in p.guardrails.config["banned_medical_terms"]:
        assert term not in out


def test_vlm_cooldown_prevents_spam():
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf", tier1_backend=MockLandmarkBackend())
    p.tier1.backend.set_scenario(ear=0.30)
    telem = {"continuous_driving_min": 91, "speed_kmh": 40}
    out1 = p.process_stream_frame(FRAME, telem, now=0.0)
    out2 = p.process_stream_frame(FRAME, telem, now=1.0)  # trong cooldown
    assert out1 is not None and out2 is None


def test_t2_t4_instant_alert_never_waits_for_vlm():
    """T2-T4: frame trigger phải trả cảnh báo tĩnh NGAY, VLM trả frame sau (NC-01)."""
    import time as time_mod

    class SlowVLM:
        def generate(self, frame, reason, delta, telem):
            time_mod.sleep(0.3)  # VLM chậm 300ms — không được chặn frame loop
            return {"observation": "eyes_heavy", "severity": "gentle",
                    "context_slots": {"trip_factor": "none",
                                      "vehicle_state": "moving"}}

    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    p.vlm = SlowVLM()
    # tạo 3 lần ngáp -> T3 trigger
    t = 0.0
    for _ in range(3):
        p.tier1.backend.set_scenario(ear=0.3, mar=0.75)
        for i in range(25):
            p.tier1.analyze(FRAME, now=t + i / 10)
        t += 2.5
        p.tier1.backend.set_scenario(ear=0.3, mar=0.1)
        for i in range(10):
            p.tier1.analyze(FRAME, now=t + i / 10)
        t += 1.0

    start = time_mod.monotonic()
    out = p.process_stream_frame(FRAME, {"speed_kmh": 40}, now=t)
    elapsed = time_mod.monotonic() - start
    assert out is not None and "ngáp" in out   # cảnh báo tĩnh tức thời (T3)
    assert elapsed < 0.15                          # không chờ VLM 300ms
    # kết quả VLM (template eyes_heavy gentle) nổi lên ở frame sau —
    # trigger còn active nên frame chưa có kết quả vẫn trả cảnh báo tĩnh
    p.tier1.backend.set_scenario(ear=0.3, mar=0.1)
    deadline = time_mod.monotonic() + 2.0
    outputs = []
    while time_mod.monotonic() < deadline:
        time_mod.sleep(0.05)
        t += 0.1
        out = p.process_stream_frame(FRAME, {"speed_kmh": 40}, now=t)
        if out:
            outputs.append(out)
        if any("mỏi" in o for o in outputs):
            break
    assert any("mỏi" in o for o in outputs)


def test_pre_ride_check_helmet_and_weather_gate():
    """Pre-ride: quai mũ luôn nhắc khi thiếu; khẩu trang chỉ khi nắng bụi (NC-02)."""
    det = MockObjectDetector(scenario={"helmet_strap": 0.2, "mask": 0.2})
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend(),
                                       object_detector=det)
    frames = [FRAME] * 10
    normal = p.pre_ride_check(frames, {"weather": "normal"})
    assert any("mũ bảo hiểm" in r for r in normal)
    assert not any("khẩu trang" in r for r in normal)  # trời thường: không nhắc
    dusty = p.pre_ride_check(frames, {"weather": "sunny_dusty"})
    assert any("khẩu trang" in r for r in dusty)


def test_end_trip_populates_daily_baseline():
    """end_trip phải tổng hợp phiên đo hôm nay vào health_daily (NC-08)."""
    from datetime import datetime, timezone
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    import time as time_mod
    for s in range(5):
        p.baseline.add_sample(p.profile_id, int(time_mod.time()) - s * 300, 1,
                              {"eye_openness": 0.3})
    p.end_trip()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = p.baseline.db.execute(
        "SELECT COUNT(*) FROM health_daily WHERE day=?", (today,)).fetchone()
    assert rows[0] > 0


def test_delivery_channel_policy():
    assert delivery_channel({"speed_kmh": 60}) == "audio_short"
    assert delivery_channel({"speed_kmh": 10}) == "audio_full"
    assert delivery_channel({"speed_kmh": 0}) == "audio_and_screen"


# ----------------------------------------------------------------------
# Object detector (YOLO26n) + TTS audio alerts — giai đoạn 1 roadmap
# ----------------------------------------------------------------------
def test_phone_detector_stride_caches_confidence():
    """YOLO chỉ chạy mỗi N frame; frame giữa dùng lại conf gần nhất."""
    from object_detector import Yolo26PhoneDetector

    det = Yolo26PhoneDetector.__new__(Yolo26PhoneDetector)  # bỏ qua load model
    det.stride, det._frame_idx, det._last_conf = 3, 0, 0.0
    det.imgsz, det.conf, det.device = 384, 0.25, "cpu"
    calls = []

    class FakeBoxes(list):
        @property
        def conf(self):
            import numpy as _np
            return _np.array([0.8])

    class FakeResult:
        boxes = FakeBoxes([1])

    class FakeModel:
        def predict(self, *a, **k):
            calls.append(1)
            return [FakeResult()]

    det.model = FakeModel()
    confs = [det(FRAME) for _ in range(6)]
    assert len(calls) == 2                      # chạy frame 0 và 3
    assert confs == [0.8] * 6                   # frame giữa dùng cache


def test_phone_detector_fallback_none_on_missing_weights():
    from object_detector import try_create_phone_detector
    assert try_create_phone_detector("nonexistent/path.pt") is None


def test_pipeline_mock_backend_never_gets_auto_yolo():
    """Kịch bản mock điều khiển phone qua phone_conf backend — YOLO không
    được auto-cắm đè lên (chỉ bật khi Tier 1 là backend thật)."""
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    assert p.tier1.phone_detector is None


def test_tts_manifest_covers_every_approved_sentence():
    """Mọi câu trong tập đóng duyệt sẵn phải có WAV pre-render — thêm câu
    mới mà quên chạy src.tts_prerender là test này đỏ."""
    import json
    from pipeline import IMMEDIATE_ALERTS, PRE_RIDE_REMINDERS
    from guardrails import CONFIG_PATH

    tts_dir = Path(__file__).parent.parent / "assets" / "tts"
    manifest = json.loads((tts_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    sentences = (set(IMMEDIATE_ALERTS.values()) | set(PRE_RIDE_REMINDERS.values())
                 | set(cfg["fallback_templates"].values())
                 | {t for t in cfg["template_bank"].values() if t})
    for text in sentences:
        assert text in manifest, f"thiếu TTS cho: {text}"
        assert (tts_dir / manifest[text]).exists()


def test_speaker_rejects_free_text():
    """AlertSpeaker chỉ phát câu thuộc manifest — text lạ (free text) bị bỏ,
    không synthesize runtime (docs/02)."""
    import audio_alerts

    sp = audio_alerts.AlertSpeaker.__new__(audio_alerts.AlertSpeaker)
    sp.manifest = {"câu duyệt sẵn": "x.wav"}
    sp._thread = None
    sp.tts_dir = Path("/nonexistent")
    assert sp.play("text model tự bịa ra") is False
    assert sp.play(None) is False
    assert sp.play("") is False


def test_pose_calibration_neutralizes_camera_angle():
    """Camera lệch -14° (webcam thấp/cao hơn tầm mắt): sau calibration,
    ngồi bình thường không nổ head-down; cúi thêm thật (-40 raw) vẫn nổ."""
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend, pose_calibration_sec=5.0)
    # 6s đầu ngồi bình thường với camera lệch — gom mẫu + chốt offset
    out = _run_frames(analyzer, backend, dict(pitch_deg=-14.0), seconds=6.0)
    assert analyzer.pose_calibrated
    assert out["head_tilted_down"] is False
    assert abs(out["raw"].pitch_deg) < 1.0     # -14 raw đã về ~0
    # Giữ nguyên tư thế lệch thêm 8s nữa — vẫn không báo oan
    out = _run_frames(analyzer, backend, dict(pitch_deg=-14.0), seconds=8.0,
                      t0=6.0)
    assert out["head_tilted_down"] is False
    # Cúi thật: -40 raw = -26 sau hiệu chỉnh, vượt ngưỡng -25
    out = _run_frames(analyzer, backend, dict(pitch_deg=-40.0), seconds=2.0,
                      t0=14.0)
    assert out["head_tilted_down"] is True


def test_pose_calibration_disabled_by_default():
    """pose_calibration_sec=0 (mặc định): đo thô như cũ, eval FL3D không đổi."""
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    out = _run_frames(analyzer, backend, dict(pitch_deg=-30.0), seconds=2.0)
    assert out["head_tilted_down"] is True
    assert out["raw"].pitch_deg == -30.0


def test_ear_gated_when_head_turned():
    """Quay đầu >35°: EAR không đáng tin — không đếm nhắm mắt/PERCLOS,
    không nổ T0/T2 oan; bộ đếm quay đầu vẫn chạy bình thường."""
    backend = MockLandmarkBackend()
    analyzer = Tier1Analyzer(backend=backend)
    # 40s "mắt nhắm" nhưng đầu đang quay 60° — EAR là artifact phối cảnh
    out = _run_frames(analyzer, backend, dict(ear=0.05, yaw_deg=60.0),
                      seconds=40.0)
    assert out["immediate_alert"] is None
    assert out["eyes_closed_duration_sec"] == 0.0
    assert out["perclos"] == 0.0
    # Về chính diện mắt nhắm thật -> vẫn nổ T0 như thường
    out = _run_frames(analyzer, backend, dict(ear=0.05, yaw_deg=0.0),
                      seconds=2.0, t0=40.0)
    assert out["immediate_alert"] == "T0_eyes_closed"


def test_static_reminder_respects_cooldown():
    """PERCLOS là trạng thái kéo dài — câu nhắc tĩnh chỉ phát 1 lần mỗi
    cooldown, không lặp mỗi frame."""
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    p.tier1.backend.set_scenario(ear=0.10)  # nhắm hờ liên tục -> PERCLOS cao
    # chạy 35s cho đủ cửa sổ PERCLOS; T0 nổ trước (nhắm liên tục) nên
    # dùng kịch bản chớp: 2 frame nhắm 1 frame mở
    alerts = []
    for i in range(400):
        ear = 0.10 if i % 3 else 0.30
        p.tier1.backend.set_scenario(ear=ear)
        out = p.process_stream_frame(FRAME, {"speed_kmh": 40}, now=i / 10)
        if out:
            alerts.append((i / 10, out))
    perclos_alerts = [a for _, a in alerts if "buồn ngủ" in a]
    assert len(perclos_alerts) <= 1  # 40s < cooldown 180s -> tối đa 1 lần


def test_t2_cooldown_does_not_mask_t4():
    """T2 PERCLOS là trạng thái kéo dài nhiều phút — khi T2 đang cooldown,
    T4 (quay đầu 3 lần/30s) vẫn phải nổ, không bị T2 che mất."""
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    alerts = []
    # 0..35s: chớp mắt lim dim -> PERCLOS cao, T2 nổ 1 lần rồi vào cooldown
    for i in range(350):
        ear = 0.10 if i % 3 else 0.30
        p.tier1.backend.set_scenario(ear=ear, yaw_deg=0.0)
        out = p.process_stream_frame(FRAME, {"speed_kmh": 40}, now=i / 10)
        if out:
            alerts.append(out)
    assert any("buồn ngủ" in a for a in alerts)  # T2 đã nổ
    # 35..41s: quay đầu 3 lần (edge-triggered qua ngưỡng 45°); PERCLOS trong
    # cửa sổ 60s vẫn cao nên T2 vẫn "đúng" nhưng đang cooldown
    alerts.clear()
    for i in range(60):
        yaw = 60.0 if (i // 10) % 2 == 0 else 0.0
        p.tier1.backend.set_scenario(ear=0.30, yaw_deg=yaw)
        out = p.process_stream_frame(FRAME, {"speed_kmh": 40}, now=35 + i / 10)
        if out:
            alerts.append(out)
    assert any("quan sát phía trước" in a for a in alerts), alerts


def test_t5_long_driving_reminds_even_when_face_looks_normal():
    """T5 là trigger telematics (nhắc nghỉ theo luật >60'): VLM thấy mặt
    tươi tỉnh (looks_normal) vẫn phải nhắc nghỉ mức nhẹ, không im lặng."""
    p = SafetyAndHealthMonitorPipeline(edge_vlm_path="none.gguf",
                                       tier1_backend=MockLandmarkBackend())
    p.vlm.generate = lambda *a, **k: {"observation": "looks_normal",
                                      "severity": "none"}
    out = p.tier2_run_vlm_context_analysis(
        FRAME, "T5_long_driving",
        {"continuous_driving_min": 65, "speed_kmh": 45, "ambient_temp_c": 30})
    assert out is not None and "nghỉ" in out, out


def test_redteam_banned_terms_all_blocked():
    """Lớp B red-team: mọi câu chứa hàm ý y tế phải bị thay bằng fallback."""
    from red_team import MUST_BLOCK
    from guardrails import MedicalGuardrails
    g = MedicalGuardrails()
    fallbacks = set(g.fallbacks.values())
    for atk in MUST_BLOCK:
        out = g.enforce(atk, "T7_baseline_anomaly")
        assert out in fallbacks, f"lọt lưới: {atk!r} -> {out!r}"


def test_redteam_benign_reminders_not_false_blocked():
    """Câu nhắc an toàn lành tính không được chặn nhầm — 'an toàn' (bỏ dấu
    'an toan') từng dính từ cấm 'toa' do so khớp substring; nay khớp biên từ."""
    from red_team import MUST_PASS
    from guardrails import MedicalGuardrails
    g = MedicalGuardrails()
    for benign in MUST_PASS:
        out = g.enforce(benign, "T2_perclos_fatigue")
        assert out == benign, f"chặn nhầm: {benign!r} -> {out!r}"


def test_banned_term_word_boundary_not_substring():
    """Từ cấm ngắn khớp cả từ, không khớp bên trong từ khác."""
    from guardrails import MedicalGuardrails
    g = MedicalGuardrails()
    fb = set(g.fallbacks.values())
    # 'toa' (toa thuốc) là từ cấm -> chặn khi đứng riêng
    assert g.enforce("Bác sĩ đưa toa thuốc cho bạn.", "x") in fb
    # nhưng KHÔNG chặn 'an toàn' / 'toàn bộ' chứa chuỗi con 'toa'
    assert g.enforce("Lái xe an toàn nhé.", "x") == "Lái xe an toàn nhé."
    assert g.enforce("Chú ý toàn bộ mặt đường.", "x") == "Chú ý toàn bộ mặt đường."
