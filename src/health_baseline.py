"""Khối 3 — Privacy-preserving health baseline (xem docs/03-health-baseline.md).

Chỉ lưu feature vô hướng, không ảnh/embedding. Baseline = rolling mean/std
đúng nghĩa trên 7 ngày hợp lệ gần nhất, tổng hợp theo ngày (median các phiên),
ngày anomaly bị loại khỏi cửa sổ.
"""

import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

FEATURES = [
    "eye_darkness", "eye_openness", "eye_puffiness",
    "skin_paleness", "lip_color_index",
    "blink_rate", "perclos", "yawn_rate",
]

# Hướng xấu: +1 = z dương là xấu, -1 = z âm là xấu (dùng chung với trigger T7)
BAD_DIRECTION = {
    "eye_darkness": -1, "eye_openness": -1, "eye_puffiness": +1,
    "skin_paleness": -1, "lip_color_index": -1,
    "blink_rate": +1, "perclos": +1, "yawn_rate": +1,
}

Z_CONSENSUS = 2.0       # >= 2 feature cùng hướng xấu |z| > 2
Z_STRONG = 3.0          # hoặc 1 feature |z| > 3 lặp >= 2 phiên liên tiếp
MIN_DAYS = 3
FULL_DAYS = 7
RETENTION_DAYS = 14

# Độ nhiễu tối thiểu theo đơn vị của từng feature. Nếu 7 giá trị daily giống
# hệt nhau, std thống kê bằng 0; dùng epsilon số học cực nhỏ sẽ biến một sai
# khác không đáng kể thành hàng trăm sigma. Các floor này là guard thực dụng
# cho prototype và cần calibration lại theo camera/sensor production.
STD_FLOOR = {
    "eye_darkness": 0.02,
    "eye_openness": 0.01,
    "eye_puffiness": 0.01,
    "skin_paleness": 0.02,
    "lip_color_index": 0.03,
    "blink_rate": 2.0,
    "perclos": 0.02,
    "yawn_rate": 0.5,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS health_samples (
    sample_id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    light_bucket INTEGER NOT NULL,
    eye_darkness REAL, eye_openness REAL, eye_puffiness REAL,
    skin_paleness REAL, lip_color_index REAL,
    blink_rate REAL, perclos REAL, yawn_rate REAL
);
CREATE TABLE IF NOT EXISTS health_daily (
    profile_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    light_bucket INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    day_value REAL NOT NULL,
    session_count INTEGER NOT NULL,
    is_anomalous INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (profile_id, day, light_bucket, feature_name)
);
CREATE TABLE IF NOT EXISTS health_anomaly_days (
    profile_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    PRIMARY KEY (profile_id, day)
);
"""


class HealthBaseline:
    def __init__(self, db_path: str = ":memory:"):
        if db_path != ":memory:":
            db_path = str(Path(db_path).expanduser())
            Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript(_SCHEMA)
        # Trạng thái persistence cho rule "1 feature mạnh lặp 2 phiên liên tiếp"
        self._prev_strong: Dict[str, set] = {}

    # ------------------------------------------------------------------
    def add_sample(self, profile_id: int, ts: int, light_bucket: int,
                   features: Dict[str, Optional[float]]) -> None:
        """Ghi 1 phiên đo. Feature bị che ROI truyền None (ghi NULL)."""
        cols = ", ".join(FEATURES)
        ph = ", ".join("?" * len(FEATURES))
        self.db.execute(
            f"INSERT INTO health_samples (profile_id, ts, light_bucket, {cols}) "
            f"VALUES (?, ?, ?, {ph})",
            [profile_id, ts, light_bucket] + [features.get(f) for f in FEATURES],
        )
        self.db.commit()

    def prune_old(self, now_ts: int) -> None:
        cutoff = now_ts - RETENTION_DAYS * 86400
        cutoff_day = datetime.fromtimestamp(cutoff, timezone.utc).strftime("%Y-%m-%d")
        self.db.execute("DELETE FROM health_samples WHERE ts < ?", (cutoff,))
        self.db.execute("DELETE FROM health_daily WHERE day < ?", (cutoff_day,))
        self.db.execute("DELETE FROM health_anomaly_days WHERE day < ?",
                        (cutoff_day,))
        self.db.commit()

    # ------------------------------------------------------------------
    def aggregate_day(self, profile_id: int, day: str) -> None:
        """Cuối chuyến: mỗi (ngày, bucket, feature) = median các phiên trong ngày."""
        start = int(datetime.strptime(day, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp())
        rows = self.db.execute(
            "SELECT light_bucket, " + ", ".join(FEATURES) +
            " FROM health_samples WHERE profile_id=? AND ts>=? AND ts<?",
            (profile_id, start, start + 86400),
        ).fetchall()
        by_bucket: Dict[int, List] = {}
        for row in rows:
            by_bucket.setdefault(row[0], []).append(row[1:])
        for bucket, samples in by_bucket.items():
            for i, feat in enumerate(FEATURES):
                vals = [s[i] for s in samples if s[i] is not None]
                if not vals:
                    continue
                self.db.execute(
                    "INSERT OR REPLACE INTO health_daily "
                    "(profile_id, day, light_bucket, feature_name, day_value, "
                    " session_count, is_anomalous) VALUES (?,?,?,?,?,?,"
                    " MAX(COALESCE((SELECT is_anomalous FROM health_daily WHERE "
                    "  profile_id=? AND day=? AND light_bucket=? AND feature_name=?), 0),"
                    " COALESCE((SELECT 1 FROM health_anomaly_days WHERE "
                    "  profile_id=? AND day=?), 0)))",
                    (profile_id, day, bucket, feat, statistics.median(vals),
                     len(vals), profile_id, day, bucket, feat,
                     profile_id, day),
                )
        self.db.commit()

    def mark_day_anomalous(self, profile_id: int, day: str) -> None:
        """Ghi bền cờ ngày anomaly, kể cả khi daily chưa được aggregate.

        Nhờ bảng sự kiện riêng, trigger giữa chuyến không bị mất khi tiến trình
        kết thúc trước bước tổng hợp cuối ngày.
        """
        self.db.execute(
            "INSERT OR IGNORE INTO health_anomaly_days (profile_id, day) "
            "VALUES (?,?)", (profile_id, day),
        )
        self.db.execute(
            "UPDATE health_daily SET is_anomalous=1 WHERE profile_id=? AND day=?",
            (profile_id, day),
        )
        self.db.commit()

    # ------------------------------------------------------------------
    def get_baseline(self, profile_id: int, light_bucket: int,
                     feature: str) -> Optional[Dict]:
        """mean/std trên tối đa 7 ngày hợp lệ gần nhất. None nếu < MIN_DAYS ngày."""
        rows = self.db.execute(
            "SELECT day_value FROM health_daily WHERE profile_id=? AND "
            "light_bucket=? AND feature_name=? AND is_anomalous=0 "
            "ORDER BY day DESC LIMIT ?",
            (profile_id, light_bucket, feature, FULL_DAYS),
        ).fetchall()
        vals = [r[0] for r in rows]
        if len(vals) < MIN_DAYS:
            return None
        return {
            "mean_7d": statistics.mean(vals),
            "std_7d": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "day_count": len(vals),
            "is_provisional": len(vals) < FULL_DAYS,
        }

    def check_anomaly(self, profile_id: int, light_bucket: int,
                      features: Dict[str, Optional[float]]) -> Dict:
        """Chạy mỗi phiên đo mới. Trả cờ anomaly + delta text cho VLM (T7).

        Điều kiện HOẶC:
        - >= 2 feature cùng hướng xấu |z| > 2 (đồng thuận), hoặc
        - 1 feature hướng xấu |z| > 3 lặp lại >= 2 phiên liên tiếp (persistence).
        """
        consensus, strong_now, deltas, provisional = [], set(), [], False
        for feat, value in features.items():
            if value is None or feat not in BAD_DIRECTION:
                continue
            base = self.get_baseline(profile_id, light_bucket, feat)
            if base is None:
                continue
            provisional = provisional or base["is_provisional"]
            scale = max(base["std_7d"], STD_FLOOR[feat])
            z = (value - base["mean_7d"]) / scale
            if z * BAD_DIRECTION[feat] > Z_CONSENSUS:
                consensus.append(feat)
                deltas.append(f"{feat} lệch {abs(z):.1f} sigma theo hướng xấu")
            if z * BAD_DIRECTION[feat] > Z_STRONG:
                strong_now.add(feat)

        key = f"{profile_id}:{light_bucket}"
        prev_strong = self._prev_strong.get(key, set())
        persistent_strong = strong_now & prev_strong
        self._prev_strong[key] = strong_now

        is_anomaly = len(consensus) >= 2 or bool(persistent_strong)
        return {
            "is_anomaly": is_anomaly,
            "is_provisional": provisional,
            "anomalous_features": sorted(set(consensus) | persistent_strong),
            "delta_text": "; ".join(deltas) if deltas else "",
        }
