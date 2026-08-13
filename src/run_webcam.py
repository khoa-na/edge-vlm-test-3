"""Chạy thật với webcam — Tier 1 MediaPipe real-time, Tier 2 mock hoặc GGUF.

Cần: pip install mediapipe opencv-python
Chạy: python -m src.run_webcam [--vlm path/to/model.gguf]
Thoát: phím q.
"""

import argparse
import time

try:
    from .pipeline import SafetyAndHealthMonitorPipeline, delivery_channel
except ImportError:
    from pipeline import SafetyAndHealthMonitorPipeline, delivery_channel


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="quantized_vlm.gguf",
                    help="đường dẫn model GGUF (không có thì dùng mock)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    monitor = SafetyAndHealthMonitorPipeline(edge_vlm_path=args.vlm)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Không mở được camera")

    telematics = {"continuous_driving_min": 0, "speed_kmh": 40,
                  "ambient_temp_c": 30, "weather": "normal"}
    trip_start = time.monotonic()
    last_alert = ""

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        telematics["continuous_driving_min"] = (time.monotonic() - trip_start) / 60

        alert = monitor.process_stream_frame(rgb, telematics)
        if alert:
            last_alert = alert
            # Chính sách kênh phát theo trạng thái xe (docs/01 §6c):
            # production đang chạy nhanh chỉ phát audio; overlay ở đây
            # là công cụ dev để quan sát
            channel = delivery_channel(telematics)
            print(f"🔊 [{channel}] {alert}")

        cv2.putText(frame_bgr, last_alert[:60], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Driver Safety Monitor", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
