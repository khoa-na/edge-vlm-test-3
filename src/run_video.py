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

import numpy as np

try:
    from .audio_alerts import (load_manifest, mux_alerts_into_video,
                               try_create_speaker)
    from .pipeline import SafetyAndHealthMonitorPipeline
except ImportError:
    from audio_alerts import (load_manifest, mux_alerts_into_video,
                              try_create_speaker)
    from pipeline import SafetyAndHealthMonitorPipeline

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Cùng một câu cảnh báo chỉ nhắc lại sau khi bản trước phát xong + ngần này
# giây — chống chuỗi T0 lim dim xếp chồng chục bản giống hệt
REPEAT_GAP = 2.0


def iter_frames(input_path: str, fps_hint: float):
    """Yield (timestamp_sec, frame_rgb) từ video file hoặc thư mục ảnh."""
    import cv2

    import re

    def numeric_key(f):
        m = re.search(r"(\d+)", f.stem)
        return (int(m.group(1)) if m else 0, f.name)

    p = Path(input_path)
    if p.is_dir():
        # sort theo SỐ trong tên file — sort chữ cái làm frame12 đứng sau frame1052
        files = sorted((f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS),
                       key=numeric_key)
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
        # Lấy mẫu xuống fps_hint (spec pipeline 5-10 FPS): nguồn 30/60fps mà
        # đưa hết frame vào thì writer (ghi ở fps_hint) kéo dài video gấp
        # 3-6 lần thời gian thật, audio mux lệch hết; timestamp vẫn theo
        # thời gian thật của nguồn
        stride = max(1, round(fps / fps_hint))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
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
    ap.add_argument("--output", default=None,
                    help="đường dẫn mp4 xuất video annotate (EAR/MAR/pose + cảnh báo)")
    ap.add_argument("--driving-min", type=float, default=30,
                    help="continuous_driving_min giả lập cho telematics")
    ap.add_argument("--audio", action="store_true",
                    help="phát cảnh báo TTS tiếng Việt khi chạy (cần assets/tts/)")
    ap.add_argument("--audio-mux", action="store_true",
                    help="ghi giọng cảnh báo TTS vào video --output (cần ffmpeg)")
    ap.add_argument("--calibrate", type=float, default=5.0, metavar="SEC",
                    help="hiệu chỉnh tư thế trung tính theo N giây đầu clip "
                         "(người ngồi bình thường); 0 = tắt")
    args = ap.parse_args()
    if args.audio_mux and not args.output:
        ap.error("--audio-mux cần --output")

    import cv2

    speaker = try_create_speaker() if args.audio else None
    if args.audio and speaker is None:
        print("Audio không khả dụng (thiếu assets/tts hoặc sounddevice) — chạy tiếp không tiếng")
    tts_manifest = load_manifest() if args.audio_mux else {}

    monitor = SafetyAndHealthMonitorPipeline(
        edge_vlm_path=args.vlm, vlm_server_url=args.vlm_url,
        pose_calibration_sec=args.calibrate)
    telematics = {"continuous_driving_min": args.driving_min, "speed_kmh": 45,
                  "ambient_temp_c": 30, "weather": "normal"}

    writer = None
    n_frames, n_alerts, last = 0, 0, None
    alert_banner, banner_until = "", 0.0
    audio_events = []  # (timestamp_sec, wav_path) cho --audio-mux

    for ts, frame, name in iter_frames(args.input, args.fps):
        # driving_min cộng dồn theo thời gian clip — hành trình dài dần như
        # thật, T5 (>60') nổ giữa clip thay vì ngay frame đầu khi khai 65'
        telematics["continuous_driving_min"] = args.driving_min + ts / 60
        alert = monitor.process_stream_frame(frame, telematics, now=ts)
        n_frames += 1
        if alert and alert != last:
            n_alerts += 1
            print(f"[t={ts:7.2f}s | {name}] 🔊 {alert}")
            if speaker:
                speaker.play(alert)
            wav = tts_manifest.get(alert.strip())
            if wav is not None:
                dur = wav.stat().st_size / (22050 * 2)
                # 1 slot như AlertSpeaker: câu trước chưa đọc xong thì câu
                # thường không chèn đè (ước lượng theo kích thước WAV 22kHz);
                # riêng câu KHẨN CẤP (T0/T1, prefix "CẢNH BÁO") không được
                # phép rơi im lặng — xếp hàng phát ngay khi loa rảnh
                last_end = (audio_events[-1][0]
                            + audio_events[-1][1].stat().st_size / (22050 * 2)
                            if audio_events else 0.0)
                # Dedupe câu GIỐNG HỆT: T0 khi lim dim nhấp nháy nổ hàng chục
                # lần trong vài giây (mắt mở/nhắm xen kẽ), nếu cứ xếp hàng thì
                # cùng một câu "Báo động!" đè nhau 6-7 bản liền. Chỉ nhắc lại
                # sau khi bản trước phát xong + REPEAT_GAP — vẫn cảnh báo tuần
                # hoàn khi mắt nhắm kéo dài, nhưng không chói.
                same = [t for t, w in audio_events if w == wav]
                if same and ts < same[-1] + dur + REPEAT_GAP:
                    pass  # câu này vừa nhắc, chưa tới lúc lặp
                elif ts >= last_end:
                    audio_events.append((ts, wav))
                elif alert.startswith("CẢNH BÁO"):
                    audio_events.append((last_end, wav))
        if alert:
            alert_banner, banner_until = alert, ts + 2.0  # giữ banner 2s
        last = alert

        if args.output:
            # Canvas cố định: nguồn (vd FL3D) có thể là crop mặt với kích
            # thước MỖI FRAME MỖI KHÁC — VideoWriter lặng lẽ bỏ frame sai
            # size, nên letterbox tất cả về một khung
            CW, CH = 640, 480
            src = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            sh, sw = src.shape[:2]
            scale = min(CW / sw, CH / sh)
            nw, nh = int(sw * scale), int(sh * scale)
            resized = cv2.resize(src, (nw, nh))
            bgr = np.zeros((CH, CW, 3), dtype=np.uint8)
            x0, y0 = (CW - nw) // 2, (CH - nh) // 2
            bgr[y0:y0 + nh, x0:x0 + nw] = resized
            if writer is None:
                writer = cv2.VideoWriter(args.output,
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (CW, CH))
            # overlay chỉ số Tier 1 — đọc từ lần analyze cuối, không extract lại
            t1_res = getattr(monitor.tier1, "last_result", {})
            m = t1_res.get("raw")
            if m is not None and m.face_found:
                lines = [
                    f"t={ts:6.1f}s EAR={m.ear:.2f} MAR={m.mar:.2f} "
                    f"perclos={t1_res.get('perclos', 0):.2f}",
                    f"pitch={m.pitch_deg:+.0f} yaw={m.yaw_deg:+.0f} "
                    f"closed={t1_res.get('eyes_closed_duration_sec', 0):.1f}s",
                ]
            else:
                lines = [f"t={ts:6.1f}s (no face)"]
            for i, txt in enumerate(lines):
                cv2.putText(bgr, txt, (8, 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(bgr, txt, (8, 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1)
            if ts < banner_until and alert_banner:
                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
                cv2.rectangle(bgr, (0, h - 26), (w, h), (0, 0, 200), -1)
                cv2.putText(bgr, alert_banner[:70], (6, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            writer.write(bgr)

    if writer is not None:
        writer.release()
        if args.audio_mux and mux_alerts_into_video(args.output, audio_events):
            print(f"Đã ghi {len(audio_events)} câu cảnh báo TTS vào audio track")
        print(f"Video annotate: {args.output}")
    print(f"\nXử lý {n_frames} frame, {n_alerts} cảnh báo.")


if __name__ == "__main__":
    main()
