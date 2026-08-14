"""Pre-render TTS tiếng Việt cho toàn bộ template bank -> assets/tts/*.wav.

Chạy 1 lần lúc build (cần models/tts/vi_VN-vais1000-medium.onnx — Piper,
offline hoàn toàn). Runtime chỉ phát file WAV: 0ms synthesize, giữ SLA
cảnh báo <300ms. Khả thi vì mọi câu ra loa đều thuộc tập ĐÓNG duyệt sẵn
(immediate alerts + pre-ride + template bank + fallback — docs/02): không
tồn tại free text nên không cần TTS động.

Chạy: python -m src.tts_prerender
"""

import hashlib
import json
import sys
import wave
from pathlib import Path

VOICE_PATH = Path("models/tts/vi_VN-vais1000-medium.onnx")
OUT_DIR = Path("assets/tts")
MANIFEST = OUT_DIR / "manifest.json"


def collect_sentences() -> list:
    try:
        from .pipeline import IMMEDIATE_ALERTS, PRE_RIDE_REMINDERS
        from .guardrails import CONFIG_PATH
    except ImportError:
        from pipeline import IMMEDIATE_ALERTS, PRE_RIDE_REMINDERS
        from guardrails import CONFIG_PATH

    cfg = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    sentences = set(IMMEDIATE_ALERTS.values()) | set(PRE_RIDE_REMINDERS.values())
    sentences |= set(cfg["fallback_templates"].values())
    sentences |= {t for t in cfg["template_bank"].values() if t}
    return sorted(sentences)


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    from piper import PiperVoice

    if not VOICE_PATH.exists():
        print(f"Missing voice model: {VOICE_PATH}\n"
              "curl -sL -o models/tts/vi_VN-vais1000-medium.onnx --create-dirs \\\n"
              "  https://huggingface.co/rhasspy/piper-voices/resolve/main/"
              "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx (+.json)")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(VOICE_PATH))
    sentences = collect_sentences()
    manifest = {}
    for text in sentences:
        fname = f"{text_key(text)}.wav"
        path = OUT_DIR / fname
        if not path.exists():
            with wave.open(str(path), "wb") as wf:
                voice.synthesize_wav(text, wf)
        manifest[text] = fname
        print(f"  {fname}  {text[:60]}")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Rendered {len(manifest)} sentences -> {OUT_DIR}/ (manifest.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
