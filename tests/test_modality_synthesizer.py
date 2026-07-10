"""Tests for modality_synthesizer"""
import json
import pytest
from datetime import datetime
from agent_layer.state_objects import StateObject
from agent_layer.modality_synthesizer import (
    synthesize_modalities, parse_modalities, get_modality_estimate,
)


class TestSynthesizeModalities:
    def test_synthesize_from_fused_values(self):
        """Should produce per-modality estimates from fused values"""
        state = StateObject(
            window_id="w1", timestamp=datetime.now(),
            heart_rate=72.0, respiration_rate=16.0, body_temp=36.5,
            wifi_confidence=0.9, mmwave_confidence=0.8, thermal_confidence=0.95,
            nlos_flag=False,
        )
        result = json.loads(synthesize_modalities(state))
        assert "wifi" in result
        assert "mmwave" in result
        assert "thermal" in result
        # Fused values are the same across modalities
        assert result["wifi"]["hr"] == 72.0
        assert result["wifi"]["rr"] == 16.0
        assert result["mmwave"]["hr"] == 72.0
        assert result["thermal"]["temp"] == 36.5
        # Confidence should match
        assert result["wifi"]["confidence"] == 0.9
        assert result["mmwave"]["nlos_affected"] is False

    def test_nlos_halves_mmwave_confidence(self):
        """NLOS flag should halve mmWave confidence"""
        state = StateObject(
            window_id="w2", timestamp=datetime.now(),
            heart_rate=72.0, respiration_rate=16.0,
            wifi_confidence=0.9, mmwave_confidence=0.8,
            nlos_flag=True,
        )
        result = json.loads(synthesize_modalities(state))
        assert result["mmwave"]["nlos_affected"] is True
        # mmWave confidence halved: 0.8 * 0.5 = 0.4
        assert result["mmwave"]["confidence"] == 0.4

    def test_passthrough_real_modalities(self):
        """When modalities_json is embedded in missing_modalities, pass through"""
        real_data = json.dumps({
            "wifi": {"hr": 71.0, "rr": 15.0, "confidence": 0.85, "nlos_affected": False},
            "mmwave": {"hr": 73.0, "rr": 16.5, "confidence": 0.72, "nlos_affected": False},
            "thermal": {"temp": 36.4, "confidence": 0.9, "nlos_affected": False},
        })
        state = StateObject(
            window_id="w3", timestamp=datetime.now(),
            heart_rate=72.0,  # fused value would be 72 but real data overrides
            missing_modalities=real_data,
        )
        result = json.loads(synthesize_modalities(state))
        # Should use the real data, not fused values
        assert result["wifi"]["hr"] == 71.0
        assert result["mmwave"]["hr"] == 73.0
        assert result["wifi"]["rr"] == 15.0

    def test_empty_state_partial_data(self):
        """Should handle None values gracefully"""
        state = StateObject(window_id="w4", timestamp=datetime.now())
        result = json.loads(synthesize_modalities(state))
        assert result["wifi"]["hr"] is None
        assert result["mmwave"]["hr"] is None
        assert result["thermal"]["temp"] is None


class TestParseModalities:
    def test_parse_valid_json(self):
        data = '{"wifi": {"hr": 72.0}}'
        result = parse_modalities(data)
        assert result["wifi"]["hr"] == 72.0

    def test_parse_invalid(self):
        assert parse_modalities(None) == {}
        assert parse_modalities("") == {}
        assert parse_modalities("not json") == {}


class TestGetModalityEstimate:
    def test_get_hr(self):
        mods = {"wifi": {"hr": 72.0, "rr": 16.0}}
        assert get_modality_estimate(mods, "wifi", "hr") == 72.0
        assert get_modality_estimate(mods, "wifi", "rr") == 16.0
        assert get_modality_estimate(mods, "mmwave", "hr") is None
