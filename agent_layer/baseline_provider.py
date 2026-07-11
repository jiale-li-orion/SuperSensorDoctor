"""Personal baseline provider — DB-grounded per-resident vital sign baselines.

Queries sensing_windows from the past 3 weeks to compute mean, std, and z_score
relative to the event's timestamp (not system clock). This enables paper-defined
detection like "02:14 RR=24, past-3-week same-period baseline=19±2, z=2.5".
"""

from datetime import datetime, timedelta
from typing import Optional
from storage.db import get_db

DEFAULT_STD = {"hr": 3.0, "rr": 2.0, "temp": 0.3, "body_temp": 0.3,
               "heart_rate": 3.0, "respiration_rate": 2.0}
COL_MAP = {"heart_rate": "hr", "respiration_rate": "rr", "body_temp": "body_temp",
           "hr": "hr", "rr": "rr", "temp": "body_temp"}


class BaselineProvider:
    """Compute personal baseline from historical sensing_windows data."""

    def __init__(self, days: int = 21, min_points: int = 5):
        self.days = days
        self.min_points = min_points

    def compute(self, resident_id: str, metric: str,
                at_timestamp: datetime) -> Optional[dict]:
        """Compute baseline for a metric at a given timestamp.
        
        Args:
            resident_id: Resident identifier
            metric: One of "hr", "rr", "temp", "heart_rate", "respiration_rate", "body_temp"
            at_timestamp: The event timestamp — baseline queries data BEFORE this time
        
        Returns:
            {"mean": float, "std": float, "count": int, "z_score_2sigma": float} or None
        """
        col = COL_MAP.get(metric)
        if col is None:
            return None

        # Whitelist guard: only allow known column names in the SQL query
        _ALLOWED_COLS = {"hr", "rr", "body_temp"}
        if col not in _ALLOWED_COLS:
            return None

        threshold = (at_timestamp - timedelta(days=self.days)).isoformat()
        ts_end = at_timestamp.isoformat()

        with get_db() as conn:
            rows = conn.execute(f"""
                SELECT {col} AS value FROM sensing_windows
                WHERE resident_id=? AND timestamp >= ? AND timestamp < ?
                  AND {col} IS NOT NULL
                ORDER BY timestamp
            """, (resident_id, threshold, ts_end)).fetchall()

        values = [r["value"] for r in rows if r["value"] is not None]
        if len(values) < self.min_points:
            return None  # too few data points for a meaningful baseline

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance ** 0.5

        # Floor std to avoid division-by-zero in z_score
        default_std = DEFAULT_STD.get(metric, 1.0)
        std = max(std, 0.5 * default_std)

        return {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "count": n,
            "z_score_2sigma": round(2.0 * std, 2),
        }

    def compute_metric(self, resident_id: str, metric: str,
                       value: float, at_timestamp: datetime) -> Optional[dict]:
        """Compute baseline AND z_score for a specific value.
        
        Returns baseline dict with added "z_score" and "value" fields, or None.
        """
        baseline = self.compute(resident_id, metric, at_timestamp)
        if baseline is None:
            return None
        z = (value - baseline["mean"]) / baseline["std"]
        baseline["z_score"] = round(z, 2)
        baseline["value"] = value
        return baseline
