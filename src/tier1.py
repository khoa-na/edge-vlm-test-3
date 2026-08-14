"""Tier 1 — Lightweight CV, chạy mọi frame (xem docs/01-two-tier-cascade.md).

Backend landmark cắm được: MediaPipe Face Mesh nếu cài (chạy thật với webcam),
MockLandmarkBackend cho demo/test không cần camera. Cả hai trả cùng dict
metric thô; Tier1Analyzer lo phần temporal state machine (debounce, PERCLOS,
đếm ngáp/quay đầu) — phần này chạy thật trong mọi chế độ.
"""

import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np

# Ngưỡng — một chỗ duy nhất, hai tài liệu 01/03 cùng tham chiếu
EAR_CLOSED_THRESHOLD = 0.20
EYES_CLOSED_ALERT_SEC = 1.5
MAR_YAWN_THRESHOLD = 0.35  # hiệu chỉnh trên FL3D: yawning MAR~0.5, nói chuyện <0.2
YAWN_MIN_DURATION_SEC = 2.0
YAWN_WINDOW_SEC = 600.0
YAWN_TRIGGER_COUNT = 3
PITCH_DOWN_THRESHOLD_DEG = -25.0
PITCH_DOWN_ALERT_SEC = 1.5
YAW_TURN_THRESHOLD_DEG = 45.0
YAW_WINDOW_SEC = 30.0
YAW_TRIGGER_COUNT = 3
PERCLOS_WINDOW_SEC = 60.0
PERCLOS_MIN_WINDOW_SEC = 30.0   # chưa đủ 30s dữ liệu thì không kết luận PERCLOS
PERCLOS_TRIGGER = 0.25
PERCLOS_TRIGGER_FATIGUED = 0.20  # ngưỡng nhạy hơn khi lái đêm / lái quá lâu
FACE_GAP_RESET_SEC = 0.5         # mất mặt quá lâu thì reset các bộ đếm thời gian
PHONE_CONF_THRESHOLD = 0.5
PHONE_CONFIRM_FRAMES = 3
PHONE_RELEASE_FRAMES = 10
EAR_VALID_YAW_DEG = 45.0  # quá góc này landmark mắt/miệng không đáng tin


@dataclass
class RawMetrics:
    """Metric thô 1 frame — hợp đồng chung cho mọi backend."""
    face_found: bool = True
    ear: float = 0.3            # trung bình 2 mắt
    mar: float = 0.2
    pitch_deg: float = 0.0      # âm = cúi xuống
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    phone_conf: float = 0.0     # backend không có phone detector thì để 0
    landmark_conf: float = 1.0
    # Feature sức khỏe cho baseline (Khối 3) — None = backend không đo được
    eye_darkness: Optional[float] = None    # L hốc mắt / L má
    eye_puffiness: Optional[float] = None   # bề rộng mí dưới / interocular
    skin_paleness: Optional[float] = None   # a má / a trán
    lip_color_index: Optional[float] = None # a môi / a má


class MockLandmarkBackend:
    """Backend giả cho demo/test: nhận metrics kịch bản qua set_scenario()."""

    def __init__(self):
        self._current = RawMetrics()

    def set_scenario(self, **kwargs):
        self._current = RawMetrics(**kwargs)

    def extract(self, frame: np.ndarray) -> RawMetrics:
        return self._current


class MediaPipeLandmarkBackend:
    """Backend thật: MediaPipe FaceLandmarker (Tasks API, 478 landmark).

    Cần model asset (tải 1 lần, ~3.7MB):
    curl -sL -o models/face_landmarker.task --create-dirs \\
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    """

    # Chỉ số landmark MediaPipe cho EAR/MAR (chuẩn Face Mesh)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 81, 311, 291, 402, 178]

    DEFAULT_MODEL = "models/face_landmarker.task"

    def __init__(self, model_path: Optional[str] = None):
        import mediapipe as mp  # ImportError nếu chưa cài -> pipeline tự fallback mock
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (FaceLandmarker,
                                                   FaceLandmarkerOptions)
        from pathlib import Path
        path = model_path or self.DEFAULT_MODEL
        if not Path(path).exists():
            # thử tương đối với gốc repo (khi chạy từ src/)
            alt = Path(__file__).parent.parent / path
            if alt.exists():
                path = str(alt)
            else:
                raise ImportError(f"face_landmarker model not found: {path}")
        self._mp = mp
        self._mesh = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        ))

    @staticmethod
    def _aspect_ratio(pts: np.ndarray) -> float:
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def extract(self, frame: np.ndarray) -> RawMetrics:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(frame))
        res = self._mesh.detect(mp_image)
        if not res.face_landmarks:
            return RawMetrics(face_found=False, landmark_conf=0.0)
        lm = res.face_landmarks[0]
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

        health = self._health_features(frame, pts)
        return RawMetrics(face_found=True, ear=(ear_l + ear_r) / 2, mar=mar,
                          pitch_deg=pitch, yaw_deg=yaw, **health)

    # Landmark mốc cho feature sức khỏe (Khối 3)
    UNDER_EYE = [145, 374]   # mí dưới trái/phải
    CHEEK = [205, 425]
    FOREHEAD = [10]
    LIPS = [13, 14]
    INTEROCULAR = (33, 263)

    def _health_features(self, frame: np.ndarray, pts: np.ndarray) -> dict:
        """Trích feature màu/hình học cho health baseline — thống kê Lab trên
        patch nhỏ quanh landmark, mọi giá trị là TỈ LỆ giữa 2 vùng cùng mặt
        (tự chuẩn hóa ánh sáng, xem docs/03). Lỗi nào -> None (NULL trong DB).
        """
        try:
            import cv2
            h, w = frame.shape[:2]

            def patch_lab(idx_list, r=6):
                vals = []
                for i in idx_list:
                    x, y = int(pts[i][0]), int(pts[i][1])
                    if not (r <= x < w - r and r <= y < h - r):
                        return None
                    patch = frame[y - r:y + r, x - r:x + r]
                    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2Lab)
                    vals.append(lab.reshape(-1, 3).mean(axis=0))
                return np.mean(vals, axis=0)  # (L, a, b)

            eye, cheek = patch_lab(self.UNDER_EYE), patch_lab(self.CHEEK)
            forehead, lips = patch_lab(self.FOREHEAD), patch_lab(self.LIPS)
            if any(v is None for v in (eye, cheek, forehead, lips)):
                return {}
            eps = 1e-6
            inter = np.linalg.norm(pts[self.INTEROCULAR[0]]
                                   - pts[self.INTEROCULAR[1]]) + eps
            # bề rộng vùng mí dưới (mí -> gò má) chuẩn hóa theo interocular
            puff = np.mean([np.linalg.norm(pts[145] - pts[205]),
                            np.linalg.norm(pts[374] - pts[425])]) / inter
            return {
                "eye_darkness": float(eye[0] / (cheek[0] + eps)),
                "eye_puffiness": float(puff),
                "skin_paleness": float(cheek[1] / (forehead[1] + eps)),
                "lip_color_index": float(lips[1] / (cheek[1] + eps)),
            }
        except Exception:
            return {}


@dataclass
class Tier1Analyzer:
    """Temporal state machine trên metric thô — phần lõi chống nhiễu.

    phone_detector: callable(frame) -> confidence; prototype cắm YOLO26n `.pt`,
    production có thể export INT8 theo NPU đích;
    None thì dùng phone_conf từ backend (mock cung cấp, MediaPipe không có).
    """

    backend: Any = field(default_factory=MockLandmarkBackend)
    phone_detector: Optional[Any] = None
    # Hiệu chỉnh tư thế trung tính theo từng người/lần gắn camera: gom
    # pitch/yaw trong N giây đầu có mặt (người ngồi bình thường), lấy median
    # làm gốc 0 rồi trừ khỏi mọi phép đo sau đó. 0 = tắt. Camera lệch tầm
    # mắt ±15° là chuyện thường (webcam laptop, dashboard mount) — không
    # hiệu chỉnh thì ngưỡng cúi đầu -25° bị ăn mòn gần hết margin.
    pose_calibration_sec: float = 0.0

    def __post_init__(self):
        self._calib_start: Optional[float] = None
        self._calib_samples: list = []
        self._pitch_offset = 0.0
        self._yaw_offset = 0.0
        self.pose_calibrated = self.pose_calibration_sec <= 0
        self._eyes_closed_since: Optional[float] = None
        self._pitch_down_since: Optional[float] = None
        self._yawn_started: Optional[float] = None
        self._yawn_times: deque = deque()
        self._yaw_turn_times: deque = deque()
        self._yaw_in_turn = False
        self._ear_history: deque = deque()      # (ts, closed: bool) cho PERCLOS
        self._blink_times: deque = deque()      # ts các lần chớp (closed->open)
        self._prev_closed = False
        self._phone_streak = 0
        self._phone_miss_streak = 0
        self._phone_active = False
        self._last_face_ts: Optional[float] = None
        self._first_face_ts: Optional[float] = None

    def analyze(self, frame: np.ndarray, now: Optional[float] = None,
                perclos_threshold: float = PERCLOS_TRIGGER) -> Dict[str, Any]:
        now = time.monotonic() if now is None else now
        m = self.backend.extract(frame)

        result = {
            "face_found": m.face_found,
            "eyes_closed_duration_sec": 0.0,
            "head_tilted_down": False,
            "using_phone": False,
            "perclos": 0.0,
            "perclos_valid": False,
            "yawn_count_10min": 0,
            "yawn_rate_valid": False,
            "head_turn_count_30s": 0,
            "blink_rate": None,
            "raw": m,
            "trigger_vlm_needed": False,
            "trigger_reason": None,
            "trigger_reasons": [],
            "immediate_alert": None,
        }

        # Phone detector nhìn toàn frame và không phụ thuộc face landmark.
        # Chạy trước nhánh no-face để vẫn bắt được điện thoại khi khuôn mặt bị
        # che hoặc MediaPipe mất dấu.
        phone_conf = (self.phone_detector(frame) if self.phone_detector
                      else m.phone_conf)
        if phone_conf > PHONE_CONF_THRESHOLD:
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

        if not m.face_found:
            if result["using_phone"]:
                result["immediate_alert"] = "T1_phone"
            self.last_result = result
            return result

        if self._first_face_ts is None:
            self._first_face_ts = now

        # Mất mặt quá FACE_GAP_RESET_SEC (che khuất, quay hẳn đi) thì reset
        # các bộ đếm thời gian — tránh cộng dồn khoảng trống thành báo động giả
        if (self._last_face_ts is not None
                and now - self._last_face_ts > FACE_GAP_RESET_SEC):
            self._eyes_closed_since = None
            self._pitch_down_since = None
            self._yawn_started = None
        self._last_face_ts = now

        # --- Calibration tư thế trung tính (N giây đầu có mặt) ---
        if not self.pose_calibrated:
            self._calib_start = self._calib_start or now
            self._calib_samples.append((m.pitch_deg, m.yaw_deg))
            if now - self._calib_start >= self.pose_calibration_sec:
                self._pitch_offset = float(
                    np.median([p for p, _ in self._calib_samples]))
                self._yaw_offset = float(
                    np.median([y for _, y in self._calib_samples]))
                self._calib_samples.clear()
                self.pose_calibrated = True
        # Bản sao đã trừ offset: threshold, overlay, health feature cùng nhìn
        # một hệ quy chiếu (đang calibration thì offset còn 0). Không mutate
        # in-place — backend có thể dùng lại cùng object giữa các frame
        if self._pitch_offset or self._yaw_offset:
            m = replace(m, pitch_deg=m.pitch_deg - self._pitch_offset,
                        yaw_deg=m.yaw_deg - self._yaw_offset)
            result["raw"] = m

        # --- Mắt nhắm liên tục + PERCLOS + blink rate ---
        # EAR/MAR chỉ đáng tin khi mặt gần chính diện: quay đầu quá
        # EAR_VALID_YAW_DEG thì landmark mắt/miệng bị "dẹt" theo phối cảnh,
        # mắt mở cũng đo như nhắm — frame đó coi là KHÔNG có dữ liệu mắt
        # (không đếm vào closed/PERCLOS/blink/yawn), tránh báo buồn ngủ oan
        # khi tài xế chỉ đang quay đầu (đường quay đầu đã có bộ đếm yaw riêng)
        frontal = abs(m.yaw_deg) <= EAR_VALID_YAW_DEG
        if not frontal:
            self._eyes_closed_since = None
            self._prev_closed = False
            self._yawn_started = None
        closed = frontal and m.ear < EAR_CLOSED_THRESHOLD
        if self._prev_closed and not closed:  # closed->open = 1 lần chớp
            self._blink_times.append(now)
        self._prev_closed = closed
        while self._blink_times and self._blink_times[0] < now - 60.0:
            self._blink_times.popleft()
        if closed and self._eyes_closed_since is None:
            self._eyes_closed_since = now
        elif not closed:
            self._eyes_closed_since = None
        if self._eyes_closed_since is not None:
            result["eyes_closed_duration_sec"] = now - self._eyes_closed_since

        if frontal:
            self._ear_history.append((now, closed))
        while self._ear_history and self._ear_history[0][0] < now - PERCLOS_WINDOW_SEC:
            self._ear_history.popleft()
        window_span = now - self._ear_history[0][0] if self._ear_history else 0.0
        # Chưa gom đủ 30s dữ liệu thì không kết luận PERCLOS (tránh trigger sớm
        # từ vài giây đầu); blink rate cũng cần đủ 60s cửa sổ mới có nghĩa
        if window_span >= PERCLOS_MIN_WINDOW_SEC:
            result["perclos"] = sum(c for _, c in self._ear_history) / len(self._ear_history)
            result["perclos_valid"] = True
        if window_span >= 60.0 - 1.0:
            result["blink_rate"] = float(len(self._blink_times))

        # --- Cúi đầu (nhìn điện thoại / gật gù) ---
        if m.pitch_deg < PITCH_DOWN_THRESHOLD_DEG:
            self._pitch_down_since = self._pitch_down_since or now
            if now - self._pitch_down_since >= PITCH_DOWN_ALERT_SEC:
                result["head_tilted_down"] = True
        else:
            self._pitch_down_since = None

        # --- Ngáp: MAR cao kéo dài >= 2s = 1 lần (chỉ khi mặt chính diện) ---
        if frontal and m.mar > MAR_YAWN_THRESHOLD:
            self._yawn_started = self._yawn_started or now
        else:
            if self._yawn_started and now - self._yawn_started >= YAWN_MIN_DURATION_SEC:
                self._yawn_times.append(now)
            self._yawn_started = None
        while self._yawn_times and self._yawn_times[0] < now - YAWN_WINDOW_SEC:
            self._yawn_times.popleft()
        result["yawn_count_10min"] = len(self._yawn_times)
        result["yawn_rate_valid"] = (
            self._first_face_ts is not None
            and now - self._first_face_ts >= YAWN_WINDOW_SEC
        )

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
        elif result["using_phone"]:
            result["immediate_alert"] = "T1_phone"
        elif result["head_tilted_down"]:
            result["immediate_alert"] = "T1_head_down"
        else:
            # T2-T4 có thể ĐỒNG THỜI đúng (PERCLOS là trạng thái kéo dài
            # nhiều phút, dễ che T3/T4 nếu chỉ trả 1 reason): trả đủ danh
            # sách theo ưu tiên, pipeline chọn reason đầu tiên chưa cooldown
            reasons = []
            if result["perclos"] > perclos_threshold:
                reasons.append("T2_perclos_fatigue")
            if result["yawn_count_10min"] >= YAWN_TRIGGER_COUNT:
                reasons.append("T3_frequent_yawning")
            if result["head_turn_count_30s"] >= YAW_TRIGGER_COUNT:
                reasons.append("T4_repeated_head_turns")
            if reasons:
                result["trigger_vlm_needed"] = True
                result["trigger_reason"] = reasons[0]
            result["trigger_reasons"] = reasons

        self.last_result = result  # cho overlay/debug đọc, khỏi extract lại
        return result
