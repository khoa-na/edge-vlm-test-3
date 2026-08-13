"""Đánh giá Tier 1 trên dataset FL3D (frame-level nhãn alert/microsleep/yawning).

Dataset: kagglehub matjazmuc/frame-level-driver-drowsiness-detection-fl3d
Chạy:  python -m src.eval_fl3d [--limit-seq 10] [--fps 25]

Hai mức đánh giá:
1. Frame-level: EAR/MAR tức thời so với nhãn từng frame (đo chất lượng chỉ số).
2. Pipeline-level: replay chuỗi frame qua process_stream_frame — đếm cảnh báo
   T0 (nhắm mắt >1.5s) trên các đoạn microsleep thật.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path.home() / (
    ".cache/kagglehub/datasets/matjazmuc/"
    "frame-level-driver-drowsiness-detection-fl3d/versions/1/classification_frames")

try:
    from .tier1 import (EAR_CLOSED_THRESHOLD, MAR_YAWN_THRESHOLD,
                        MediaPipeLandmarkBackend, Tier1Analyzer)
except ImportError:
    from tier1 import (EAR_CLOSED_THRESHOLD, MAR_YAWN_THRESHOLD,
                       MediaPipeLandmarkBackend, Tier1Analyzer)


def frame_no(path: str) -> int:
    m = re.search(r"frame(\d+)", path)
    return int(m.group(1)) if m else 0


def load_sequences(root: Path):
    ann = json.load(open(root / "annotations_all.json"))
    seqs = defaultdict(list)
    for rel, meta in ann.items():
        rel = rel.replace("./classification_frames/", "")
        seqs[rel.split("/")[0]].append((frame_no(rel), root / rel,
                                        meta["driver_state"]))
    for k in seqs:
        seqs[k].sort()
    return seqs


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit-seq", type=int, default=10,
                    help="số sequence đánh giá (0 = tất cả, chậm)")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="FPS gốc của video nguồn (NITYMED ~25)")
    args = ap.parse_args()

    root = Path(args.root)
    seqs = load_sequences(root)
    names = sorted(seqs)
    # ưu tiên sequence có microsleep/yawning để pipeline có gì mà bắt
    names.sort(key=lambda n: -sum(1 for _, _, s in seqs[n] if s != "alert"))
    if args.limit_seq:
        names = names[: args.limit_seq]

    backend = MediaPipeLandmarkBackend()
    confusion = Counter()
    t0_alerts, microsleep_episodes = 0, 0
    n_frames = 0

    for name in names:
        analyzer = Tier1Analyzer(backend=backend)  # reset state mỗi sequence
        prev_state = "alert"
        for i, (_, path, label) in enumerate(seqs[name]):
            img = cv2.imread(str(path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = analyzer.analyze(rgb, now=i / args.fps)
            n_frames += 1

            raw = out["raw"]
            # MAR xét trước EAR: ngáp thường kèm nheo/nhắm mắt,
            # nếu xét EAR trước thì frame ngáp bị gán nhầm microsleep
            if not raw.face_found:
                pred = "no_face"
            elif raw.mar > MAR_YAWN_THRESHOLD:
                pred = "yawning"
            elif raw.ear < EAR_CLOSED_THRESHOLD:
                pred = "microsleep"
            else:
                pred = "alert"
            confusion[(label, pred)] += 1

            if label == "microsleep" and prev_state != "microsleep":
                microsleep_episodes += 1
            if out["immediate_alert"] == "T0_eyes_closed" and label == "microsleep":
                t0_alerts += 1
            prev_state = label
        print(f"  done {name} ({len(seqs[name])} frames)")

    labels = ["alert", "microsleep", "yawning"]
    preds = labels + ["no_face"]
    print(f"\n=== Frame-level confusion ({n_frames} frames, "
          f"{len(names)} sequences) ===")
    print(f"{'label \\ pred':>14} " + " ".join(f"{p:>11}" for p in preds))
    for lb in labels:
        row = [confusion[(lb, p)] for p in preds]
        total = sum(row) or 1
        cells = " ".join(f"{c:>6}({c / total:4.0%})" for c in row)
        print(f"{lb:>14} {cells}")

    correct = sum(confusion[(l, l)] for l in labels)
    total = sum(confusion.values()) or 1
    print(f"\nFrame-level accuracy: {correct / total:.1%}")
    print(f"Pipeline: {t0_alerts} frame phát cảnh báo T0 trong "
          f"{microsleep_episodes} đoạn microsleep")


if __name__ == "__main__":
    main()
