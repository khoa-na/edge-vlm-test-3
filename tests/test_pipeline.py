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
from pipeline import SafetyAndHealthMonitorPipeline
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
    telem = {"continuous_driving_min": 90, "speed_kmh": 40}
    out1 = p.process_stream_frame(FRAME, telem, now=0.0)
    out2 = p.process_stream_frame(FRAME, telem, now=1.0)  # trong cooldown
    assert out1 is not None and out2 is None
