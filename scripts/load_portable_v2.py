"""Load portable_v2 real per-modality data into SuperSenseDoctor pipeline."""

import asyncio
from pathlib import Path
from typing import Optional, Callable

import pandas as pd


def _is_missing(value) -> bool:
    return value is None or pd.isna(value)


def _optional_float(value):
    if _is_missing(value):
        return None
    return float(value)


def _optional_str(value):
    if _is_missing(value):
        return None
    return str(value)


def _parse_bool_flag(value, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_flag(value, default: int = 0) -> int:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "on"}:
            return 1
        if text in {"false", "no", "n", "off", ""}:
            return 0
    return int(float(value))


async def load_portable_v2_csv(
    sensor_hub: "SensorHub",
    progress_callback: Optional[Callable[[int], None]] = None,
    limit: Optional[int] = None,
    evaluate: bool = False,
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

        data = {
            "heart_rate": _optional_float(row.get("HR_fused_transfer")),
            "respiration_rate": _optional_float(row.get("RR_fused_transfer")),
            "rr_wifi": _optional_float(row.get("RR_wifi")),
            "rr_mm": _optional_float(row.get("RR_mm")),
            "hr_wifi": _optional_float(row.get("HR_wifi")),
            "hr_mm": _optional_float(row.get("HR_mm")),
            "rr_conf": _optional_float(row.get("RR_confidence")),
            "hr_conf": _optional_float(row.get("HR_confidence")),
            "quality_event": _parse_int_flag(row.get("quality_event")),
            "rr_source": _optional_str(row.get("RR_source")),
            "hr_source": _optional_str(row.get("HR_source")),
            "rr_truth": _optional_float(row.get("RR_truth")),
            "hr_truth": _optional_float(row.get("HR_truth")),
            "nlos_flag": _parse_bool_flag(row.get("nlos_flag")),
            "source": "portable_v2",
        }

        window_id = f"pv2_{row['fusion_epoch_s']}"

        await sensor_hub.compose(window_id=window_id, timestamp=ts, data=data, evaluate=evaluate)

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
    from agent_layer.event_bus import EventBus

    async def main():
        bus = EventBus()
        sensor_hub = SensorHub(NurseAgent(event_bus=bus))
        count = await load_portable_v2_csv(sensor_hub)
        print(f"Loaded {count} portable_v2 windows")

    asyncio.run(main())
