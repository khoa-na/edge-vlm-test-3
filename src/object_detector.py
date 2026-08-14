"""Object detector Tier 1 — YOLO26n cho phone detection (và pre-ride tùy chọn).

YOLO26n: NMS-free/DFL-free, tối ưu CPU edge (docs/01 §2). Pretrained COCO có
sẵn class `cell phone` nên phone detection chạy thật không cần train; pre-ride
(helmet_strap/mask/sunglasses) cần weights finetune riêng — truyền qua
`preride_weights`, không có thì pipeline giữ MockObjectDetector.

Cắm vào Tier1Analyzer qua giao diện callable(frame) -> confidence.
"""

from typing import Any, Dict, Optional

import numpy as np

COCO_CELL_PHONE = 67

# YOLO chạy mỗi N frame để giữ budget Tier 1 <50ms/frame (landmark đã chiếm
# ~15-25ms); giữa các lần chạy dùng lại confidence gần nhất. Debounce trong
# Tier1Analyzer (3 frame vào / 10 frame ra) đã chống nhiễu nên stride nhỏ
# không làm trễ cảnh báo quá 1 giây ở 10 FPS.
DEFAULT_STRIDE = 3
DEFAULT_IMGSZ = 384
DEFAULT_CONF = 0.25


class Yolo26PhoneDetector:
    """callable(frame BGR) -> confidence `cell phone` cao nhất trong frame."""

    DEFAULT_MODEL = "models/yolo26n.pt"

    def __init__(self, model_path: Optional[str] = None,
                 stride: int = DEFAULT_STRIDE, imgsz: int = DEFAULT_IMGSZ,
                 conf: float = DEFAULT_CONF, device: str = "cpu"):
        from ultralytics import YOLO  # ImportError -> caller fallback

        self.model = YOLO(model_path or self.DEFAULT_MODEL)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.stride = max(1, stride)
        self._frame_idx = 0
        self._last_conf = 0.0

    def __call__(self, frame: np.ndarray) -> float:
        self._frame_idx += 1
        if (self._frame_idx - 1) % self.stride:
            return self._last_conf
        # pipeline đưa frame RGB (quy ước MediaPipe); ultralytics coi ndarray
        # là BGR nên đảo kênh trước khi predict
        results = self.model.predict(frame[..., ::-1], imgsz=self.imgsz,
                                     conf=self.conf,
                                     classes=[COCO_CELL_PHONE],
                                     device=self.device, verbose=False)
        boxes = results[0].boxes
        self._last_conf = float(boxes.conf.max()) if len(boxes) else 0.0
        return self._last_conf


class Yolo26PreRideDetector:
    """Pre-ride check với weights finetune (helmet_strap/mask/sunglasses).

    Model phải có tên class trùng key mà pipeline.pre_ride_check dùng.
    Trả dict class -> confidence "đang đeo/đã cài"; class model không biết
    thì KHÔNG trả key đó (pipeline coi thiếu key là vi phạm, nên chỉ trả
    những class model thật sự phân biệt được).
    """

    def __init__(self, weights: str, imgsz: int = DEFAULT_IMGSZ,
                 conf: float = 0.1, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        out: Dict[str, float] = {}
        for cls_id, c in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            name = names[int(cls_id)]
            out[name] = max(out.get(name, 0.0), float(c))
        return out


def try_create_phone_detector(model_path: Optional[str] = None,
                              **kwargs: Any) -> Optional[Yolo26PhoneDetector]:
    """None nếu thiếu ultralytics/weights — pipeline fallback phone_conf từ
    backend (mock), giữ nguyên tính fail-safe của repo."""
    try:
        return Yolo26PhoneDetector(model_path, **kwargs)
    except Exception:
        return None
