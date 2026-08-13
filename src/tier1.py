"""Tier 1 — Lightweight CV, chạy mọi frame (xem docs/01-two-tier-cascade.md).

Backend landmark cắm được: MediaPipe Face Mesh nếu cài (chạy thật với webcam),
MockLandmarkBackend cho demo/test không cần camera. Cả hai trả cùng dict
metric thô; Tier1Analyzer lo phần temporal state machine (debounce, PERCLOS,
đếm ngáp/quay đầu) — phần này chạy thật trong mọi chế độ.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# Ngưỡng — một chỗ duy nhất, hai tài liệu 01/03 cùng tham chiếu
EAR_CLOSED_THRESHOLD = 0.20
EYES_CLOSED_ALERT_SEC = 1.5
MAR_YAWN_THRESHOLD = 0.60
YAWN_MIN_DURATION_SEC = 2.0
YAWN_WINDOW_SEC = 600.0
YAWN_TRIGGER_COUNT = 3
PITCH_DOWN_THRESHOLD_DEG = -25.0
PITCH_DOWN_ALERT_SEC = 1.5
YAW_TURN_THRESHOLD_DEG = 45.0
YAW_WINDOW_SEC = 30.0
YAW_TRIGGER_COUNT = 3
PERCLOS_WINDOW_SEC = 60.0
PERCLOS_TRIGGER = 0.25
PHONE_CONF_THRESHOLD = 0.5
PHONE_CONFIRM_FRAMES = 3
PHONE_RELEASE_FRAMES = 10


@dataclass
class RawMetrics:
    """Metric thô 1 frame — hợp đồng chung cho mọi backend."""
    face_found: bool = True
    ear: float = 0.3            # trung bình 2 mắt
    mar: float = 0.2
    pitch_deg: float = 0.0      # âm = cúi xuống
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    phone_conf: float = 0.0
    landmark_conf: float = 1.0


class MockLandmarkBackend:
    """Backend giả cho demo/test: nhận metrics kịch bản qua set_scenario()."""

    def __init__(self):
        self._current = RawMetrics()

    def set_scenario(self, **kwargs):
        self._current = RawMetrics(**kwargs)

    def extract(self, frame: np.ndarray) -> RawMetrics:
        return self._current


class MediaPipeLandmarkBackend:
    """Backend thật: MediaPipe Face Mesh (468 landmark) + solvePnP head pose."""

    # Chỉ số landmark MediaPipe cho EAR/MAR (chuẩn Face Mesh)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 81, 311, 291, 402, 178]

    def __init__(self):
        import mediapipe as mp  # ImportError nếu chưa cài -> pipeline tự fallback mock
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    @staticmethod
    def _aspect_ratio(pts: np.ndarray) -> float:
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def extract(self, frame: np.ndarray) -> RawMetrics:
        res = self._mesh.process(frame)
        if not res.multi_face_landmarks:
            return RawMetrics(face_found=False, landmark_conf=0.0)
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in lm])

        ear_l = self._aspect_ratio(pts[self.LEFT_EYE])
        ear_r = self._aspect_ratio(pts[self.RIGHT_EYE])
        mar = self._aspect_ratio(pts[self.MOUTH])

        # Head pose xấp xỉ nhanh từ hình học landmark (đủ cho ngưỡng 25°/45°;
        # bản production dùng solvePnP với camera matrix hiệu chỉnh)
        nose, chin = pts[1], pts[152]
        left_face, right_face = pts[234], pts[454]
        face_w = np.linalg.norm(right_face - left_face) + 1e-6
        yaw = float(np.degrees(np.arcsin(np.clip(
            ((nose[0] - left_face[0]) / face_w - 0.5) * 2, -1, 1))))
        face_h = np.linalg.norm(chin - pts[10]) + 1e-6
        pitch = float(-np.degrees(np.arcsin(np.clip(
            ((nose[1] - pts[10][1]) / face_h - 0.45) * 2, -1, 1))))

        return RawMetrics(face_found=True, ear=(ear_l + ear_r) / 2, mar=mar,
                          pitch_deg=pitch, yaw_deg=yaw)


@dataclass
class Tier1Analyzer:
    """Temporal state machine trên metric thô — phần lõi chống nhiễu."""

    backend: Any = field(default_factory=MockLandmarkBackend)

    def __post_init__(self):
        self._eyes_closed_since: Optional[float] = None
        self._pitch_down_since: Optional[float] = None
        self._yawn_started: Optional[float] = None
        self._yawn_times: deque = deque()
        self._yaw_turn_times: deque = deque()
        self._yaw_in_turn = False
        self._ear_history: deque = deque()      # (ts, closed: bool) cho PERCLOS
        self._phone_streak = 0
        self._phone_miss_streak = 0
        self._phone_active = False

    def analyze(self, frame: np.ndarray, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.monotonic() if now is None else now
        m = self.backend.extract(frame)

        result = {
            "face_found": m.face_found,
            "eyes_closed_duration_sec": 0.0,
            "head_tilted_down": False,
            "using_phone": False,
            "perclos": 0.0,
            "yawn_count_10min": 0,
            "head_turn_count_30s": 0,
            "raw": m,
            "trigger_vlm_needed": False,
            "trigger_reason": None,
            "immediate_alert": None,
        }
        if not m.face_found:
            return result

        # --- Mắt nhắm liên tục + PERCLOS ---
        closed = m.ear < EAR_CLOSED_THRESHOLD
        if closed and self._eyes_closed_since is None:
            self._eyes_closed_since = now
        elif not closed:
            self._eyes_closed_since = None
        if self._eyes_closed_since is not None:
            result["eyes_closed_duration_sec"] = now - self._eyes_closed_since

        self._ear_history.append((now, closed))
        while self._ear_history and self._ear_history[0][0] < now - PERCLOS_WINDOW_SEC:
            self._ear_history.popleft()
        if len(self._ear_history) >= 10:
            result["perclos"] = sum(c for _, c in self._ear_history) / len(self._ear_history)

        # --- Cúi đầu (nhìn điện thoại / gật gù) ---
        if m.pitch_deg < PITCH_DOWN_THRESHOLD_DEG:
            self._pitch_down_since = self._pitch_down_since or now
            if now - self._pitch_down_since >= PITCH_DOWN_ALERT_SEC:
                result["head_tilted_down"] = True
        else:
            self._pitch_down_since = None

        # --- Phone: confidence + debounce 3 frame vào / 10 frame ra ---
        if m.phone_conf > PHONE_CONF_THRESHOLD:
            self._phone_streak += 1
            self._phone_miss_streak = 0
            if self._phone_streak >= PHONE_CONFIRM_FRAMES:
                self._phone_active = True
        else:
            self._phone_miss_streak += 1
            self._phone_streak = 0
            if self._phone_miss_streak >= PHONE_RELEASE_FRAMES:
                self._phone_active = False
        result["using_phone"] = self._phone_active

        # --- Ngáp: MAR cao kéo dài >= 2s = 1 lần ---
        if m.mar > MAR_YAWN_THRESHOLD:
            self._yawn_started = self._yawn_started or now
        else:
            if self._yawn_started and now - self._yawn_started >= YAWN_MIN_DURATION_SEC:
                self._yawn_times.append(now)
            self._yawn_started = None
        while self._yawn_times and self._yawn_times[0] < now - YAWN_WINDOW_SEC:
            self._yawn_times.popleft()
        result["yawn_count_10min"] = len(self._yawn_times)

        # --- Quay đầu: đếm lần vượt ngưỡng yaw (edge-triggered) ---
        turning = abs(m.yaw_deg) > YAW_TURN_THRESHOLD_DEG
        if turning and not self._yaw_in_turn:
            self._yaw_turn_times.append(now)
        self._yaw_in_turn = turning
        while self._yaw_turn_times and self._yaw_turn_times[0] < now - YAW_WINDOW_SEC:
            self._yaw_turn_times.popleft()
        result["head_turn_count_30s"] = len(self._yaw_turn_times)

        # --- Xếp loại: khẩn cấp (T0/T1) vs trigger VLM (T2-T4) ---
        if result["eyes_closed_duration_sec"] > EYES_CLOSED_ALERT_SEC:
            result["immediate_alert"] = "T0_eyes_closed"
        elif result["using_phone"] or result["head_tilted_down"]:
            result["immediate_alert"] = "T1_phone_or_head_down"
        elif result["perclos"] > PERCLOS_TRIGGER:
            result["trigger_vlm_needed"] = True
            result["trigger_reason"] = "T2_perclos_fatigue"
        elif result["yawn_count_10min"] >= YAWN_TRIGGER_COUNT:
            result["trigger_vlm_needed"] = True
            result["trigger_reason"] = "T3_frequent_yawning"
        elif result["head_turn_count_30s"] >= YAW_TRIGGER_COUNT:
            result["trigger_vlm_needed"] = True
            result["trigger_reason"] = "T4_repeated_head_turns"

        return result
