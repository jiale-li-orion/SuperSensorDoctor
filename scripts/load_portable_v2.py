"""Load portable_v2 real per-modality data into SuperSenseDoctor pipeline."""

import asyncio
from pathlib import Path
from typing import Optional, Callable

import pandas as pd


async def load_portable_v2_csv(
    sensor_hub: "SensorHub",
    progress_callback: Optional[Callable[[int], None]] = None,
    limit: Optional[int] = None,
) -> int:
    """Read fusion_debug_transfer.csv and feed each row into sensor_hub.compose().

    Parameters
    ----------
    sensor_hub : SensorHub
        The data pipeline hub (from sensing_simulator.sensor_hub).
    progress_callback : callable, optional
        Called with the current row count after each row (for web UI).
    limit : int, optional
        Only process this many rows (for testing / preview).

    Returns
    -------
    int
        Total number of windows composed.
    """
    # ── Resolve CSV path relative to project root (parent of scripts/) ──
    project_root = Path(__file__).resolve().parent.parent
    csv_path = (
        project_root
        / "portable_v2"
        / "portable_v2"
        / "new_run_20260709_hr_ml_rr_simple"
        / "fusion_debug_transfer.csv"
    )

    # ── Read CSV ──
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    count = 0
    for idx, row in df.iterrows():
        # Parse timestamp
        ts = pd.to_datetime(row["timestamp"]).to_pydatetime()

        # Build data dict, handling NaN → None for every value
        def _v(val, target_type=None):
            """Return val if not NaN; None otherwise. Optionally cast to type."""
            if pd.isna(val):
                return None
            if target_type is not None:
                return target_type(val)
            return val

        data = {
            "heart_rate": _v(row.get("HR_fused_transfer"), float),
            "respiration_rate": _v(row.get("RR_fused_transfer"), float),
            "rr_wifi": _v(row.get("RR_wifi"), float),
            "rr_mm": _v(row.get("RR_mm"), float),
            "hr_wifi": _v(row.get("HR_wifi"), float),
            "hr_mm": _v(row.get("HR_mm"), float),
            "rr_conf": _v(row.get("RR_confidence"), float),
            "hr_conf": _v(row.get("HR_confidence"), float),
            "quality_event": (
                int(row["quality_event"])
                if not pd.isna(row.get("quality_event"))
                else None
            ),
            "rr_source": _v(row.get("RR_source"), str),
            "hr_source": _v(row.get("HR_source"), str),
            "rr_truth": _v(row.get("RR_truth"), float),
            "hr_truth": _v(row.get("HR_truth"), float),
            "nlos_flag": (
                bool(row["nlos_flag"])
                if not pd.isna(row.get("nlos_flag"))
                else None
            ),
            "source": "portable_v2",
        }

        window_id = f"pv2_{row['fusion_epoch_s']}"

        await sensor_hub.compose(window_id=window_id, timestamp=ts, data=data)

        count += 1
        if progress_callback:
            progress_callback(count)

        # Yield control every 100 rows to keep the event loop responsive
        if count % 100 == 0:
            await asyncio.sleep(0)

    return count


if __name__ == "__main__":
    import asyncio
    from sensing_simulator.sensor_hub import SensorHub
    from agent_layer.nurse_agent import NurseAgent

    async def main():
        sensor_hub = SensorHub(NurseAgent())
        count = await load_portable_v2_csv(sensor_hub)
        print(f"Loaded {count} portable_v2 windows")

    asyncio.run(main())
