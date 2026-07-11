import pandas as pd
import pytest
from datetime import datetime

from scripts import load_portable_v2


def test_portable_v2_flag_parsers_handle_missing_and_strings():
    assert load_portable_v2._parse_bool_flag(None) is False
    assert load_portable_v2._parse_bool_flag(float("nan")) is False
    assert load_portable_v2._parse_bool_flag("False") is False
    assert load_portable_v2._parse_bool_flag("true") is True
    assert load_portable_v2._parse_int_flag(None) == 0
    assert load_portable_v2._parse_int_flag(float("nan")) == 0
    assert load_portable_v2._parse_int_flag("False") == 0
    assert load_portable_v2._parse_int_flag("1") == 1
    assert load_portable_v2._optional_float(0.0) == 0.0


@pytest.mark.asyncio
async def test_portable_v2_loader_defaults_to_ingest(monkeypatch):
    row = {
        "timestamp": "2026-07-09 16:53:27+08:00",
        "HR_fused_transfer": 63.0,
        "RR_fused_transfer": 19.0,
        "RR_wifi": 18.0,
        "RR_mm": 20.0,
        "HR_wifi": 60.0,
        "HR_mm": 70.0,
        "RR_confidence": 0.0,
        "HR_confidence": 0.65,
        "quality_event": float("nan"),
        "nlos_flag": "False",
        "RR_source": "fused_consistent",
        "HR_source": "mmwave_main",
        "RR_truth": 20.0,
        "HR_truth": 73.0,
        "fusion_epoch_s": 1783587207.0,
    }
    monkeypatch.setattr(load_portable_v2.pd, "read_csv", lambda path: pd.DataFrame([row]))

    calls = []

    class Hub:
        async def compose(self, window_id, timestamp, data, evaluate=True):
            calls.append((window_id, timestamp, data, evaluate))

    count = await load_portable_v2.load_portable_v2_csv(Hub())

    assert count == 1
    assert calls[0][0] == "pv2_1783587207.0"
    assert calls[0][3] is False
    assert calls[0][2]["nlos_flag"] is False
    assert calls[0][2]["quality_event"] == 0
    assert calls[0][2]["rr_conf"] == 0.0
