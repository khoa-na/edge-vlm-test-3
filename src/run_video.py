"""Chạy pipeline trên video file hoặc thư mục frame (dataset DMS công khai).

Không cần webcam — dùng để demo model thật + đối chiếu nhãn dataset.

Cần: pip install mediapipe opencv-python
Chạy:
  python -m src.run_video --input drowsy_clip.mp4
  python -m src.run_video --input path/to/frames_dir --fps 10
  python -m src.run_video --input clip.mp4 --vlm qwen2-vl-2b-q4.gguf
"""

import argparse
from pathlib import Path

try:
    from .pipeline import SafetyAndHealthMonitorPipeline
except ImportError:
    from pipeline import SafetyAndHealthMonitorPipeline

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_frames(input_path: str, fps_hint: float):
    """Yield (timestamp_sec, frame_rgb) từ video file hoặc thư mục ảnh."""
    import cv2

    p = Path(input_path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        if not files:
            raise SystemExit(f"Không có ảnh trong {p}")
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            yield i / fps_hint, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f.name
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise SystemExit(f"Không mở được {p}")
        fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"frame{i}"
            i += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="file video hoặc thư mục frame")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="FPS giả định khi input là thư mục ảnh")
    ap.add_argument("--vlm", default="quantized_vlm.gguf")
    ap.add_argument("--vlm-url", default=None,
                    help="URL llama-server (vd http://127.0.0.1:8090) — Tier 2 thật")
    ap.add_argument("--driving-min", type=float, default=30,
                    help="continuous_driving_min giả lập cho telematics")
    args = ap.parse_args()

    monitor = SafetyAndHealthMonitorPipeline(edge_vlm_path=args.vlm,
                                             vlm_server_url=args.vlm_url)
    telematics = {"continuous_driving_min": args.driving_min, "speed_kmh": 45,
                  "ambient_temp_c": 30, "weather": "normal"}

    n_frames, n_alerts, last = 0, 0, None
    for ts, frame, name in iter_frames(args.input, args.fps):
        alert = monitor.process_stream_frame(frame, telematics, now=ts)
        n_frames += 1
        if alert and alert != last:
            n_alerts += 1
            print(f"[t={ts:7.2f}s | {name}] 🔊 {alert}")
        last = alert

    print(f"\nXử lý {n_frames} frame, {n_alerts} cảnh báo.")


if __name__ == "__main__":
    main()
