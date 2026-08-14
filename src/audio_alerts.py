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


class AlertSpeaker:
    """play(text) -> True nếu có audio và đã bắt đầu phát.

    Câu đang phát thì câu mới cùng lúc bị bỏ (1 slot) — cảnh báo an toàn
    không xếp hàng đọc dồn; trạng thái mới nhất luôn hiển thị trên overlay.
    """

    def __init__(self, tts_dir: Optional[Path] = None):
        self.tts_dir = Path(tts_dir) if tts_dir else TTS_DIR
        manifest_path = self.tts_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found — run: python -m src.tts_prerender")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import sounddevice  # ImportError -> caller quyết định tắt audio
        self._sd = sounddevice
        self._thread: Optional[threading.Thread] = None

    def play(self, text: Optional[str]) -> bool:
        if not text:
            return False
        fname = self.manifest.get(text.strip())
        if fname is None:
            return False
        if self._thread is not None and self._thread.is_alive():
            return False
        path = self.tts_dir / fname

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
