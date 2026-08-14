"""Phát cảnh báo TTS pre-render (assets/tts/) qua sounddevice — non-blocking.

Chỉ phát file WAV có sẵn trong manifest (tập câu đóng duyệt sẵn); text lạ
không có audio -> bỏ qua, KHÔNG synthesize runtime — đúng nguyên tắc
"không tồn tại kênh free text" (docs/02). Phát trên thread riêng nên không
bao giờ chặn vòng lặp frame.
"""

import json
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

TTS_DIR = Path(__file__).parent.parent / "assets" / "tts"


def load_manifest(tts_dir: Optional[Path] = None) -> dict:
    """text -> đường dẫn WAV tuyệt đối. Không cần sounddevice — dùng được
    cho cả mux audio vào video (run_video --audio-mux) lẫn player."""
    tts_dir = Path(tts_dir) if tts_dir else TTS_DIR
    manifest_path = tts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run: python -m src.tts_prerender")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {text: tts_dir / fname for text, fname in raw.items()}


class AlertSpeaker:
    """play(text) -> True nếu có audio và đã bắt đầu phát.

    Câu thường đang phát thì câu thường mới bị bỏ (1 slot). Cảnh báo khẩn cấp
    T0/T1 có prefix ``CẢNH BÁO:`` sẽ preempt câu thường để không bao giờ bị
    nuốt chỉ vì loa đang bận.
    """

    def __init__(self, tts_dir: Optional[Path] = None):
        self.manifest = load_manifest(tts_dir)
        import sounddevice  # ImportError -> caller quyết định tắt audio
        self._sd = sounddevice
        self._thread: Optional[threading.Thread] = None
        self._play_lock = threading.Lock()

    def play(self, text: Optional[str]) -> bool:
        if not text:
            return False
        path = self.manifest.get(text.strip())
        if path is None:
            return False

        critical = text.strip().startswith("CẢNH BÁO:")
        with self._play_lock:
            if self._thread is not None and self._thread.is_alive():
                if not critical:
                    return False
                # sounddevice.stop() giải phóng lệnh play(blocking=True) đang
                # chạy trên worker cũ; chờ rất ngắn để tránh hai stream đè nhau.
                try:
                    self._sd.stop()
                    self._thread.join(timeout=0.1)
                except Exception:
                    # Cảnh báo khẩn vẫn được thử phát bằng worker mới.
                    pass

            def worker():
                with wave.open(str(path), "rb") as wf:
                    rate = wf.getframerate()
                    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                self._sd.play(data, rate, blocking=True)

            self._thread = threading.Thread(target=worker, daemon=True)
            self._thread.start()
        return True

    def wait(self) -> None:
        """Chờ câu đang phát xong (dùng cuối demo, không dùng trong stream)."""
        if self._thread is not None:
            self._thread.join()


def try_create_speaker() -> Optional[AlertSpeaker]:
    """None nếu thiếu manifest/sounddevice — caller chạy tiếp không audio."""
    try:
        return AlertSpeaker()
    except Exception:
        return None


def mux_alerts_into_video(video_path: str, events: list) -> bool:
    """Ghi audio track vào video annotate: mỗi (timestamp_sec, wav_path)
    chèn đúng thời điểm bằng ffmpeg adelay + amix, video giữ nguyên (-c:v
    copy). Ghi ra file tạm rồi thay thế — lỗi ffmpeg thì video gốc còn
    nguyên. Dùng cho demo: clip có giọng cảnh báo baked-in, không phụ
    thuộc loa/recorder lúc quay.
    """
    import os
    import subprocess

    if not events:
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path]
    filters, amix_in = [], []
    for i, (ts, wav) in enumerate(events, start=1):
        cmd += ["-i", str(wav)]
        delay_ms = max(0, int(ts * 1000))
        filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
        amix_in.append(f"[a{i}]")
    filters.append(
        f"{''.join(amix_in)}amix=inputs={len(events)}:normalize=0[aout]")
    tmp_path = f"{video_path}.mux.mp4"
    cmd += ["-filter_complex", ";".join(filters), "-map", "0:v",
            "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", tmp_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp_path, video_path)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        Path(tmp_path).unlink(missing_ok=True)
        err = getattr(e, "stderr", b"") or b""
        print(f"Mux audio thất bại ({err.decode(errors='replace').strip() or e}) "
              "— video giữ nguyên không tiếng")
        return False
