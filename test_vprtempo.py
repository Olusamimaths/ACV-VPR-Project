"""
Unit tests for VPRTempo integration modules.

Run with: pytest test_vprtempo.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


class TestVPRTempoConfig:
    """Tests for VPRTempoConfig."""

    def test_config_creation(self):
        """Test basic config creation."""
        from live_vpr.vprtempo_config import VPRTempoConfig

        config = VPRTempoConfig(
            model="VPRTempo",
            use_temporal=True,
            temporal_window_size=5,
        )

        assert config.model == "VPRTempo"
        assert config.use_temporal is True
        assert config.temporal_window_size == 5

    def test_config_to_dict(self):
        """Test config serialization."""
        from live_vpr.vprtempo_config import VPRTempoConfig

        config = VPRTempoConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "use_temporal" in config_dict

    def test_config_from_dict(self):
        """Test config deserialization."""
        from live_vpr.vprtempo_config import VPRTempoConfig

        config_dict = {
            "model": "VPRTempoQuant",
            "use_temporal": True,
            "temporal_window_size": 10,
        }
        config = VPRTempoConfig.from_dict(config_dict)

        assert config.model == "VPRTempoQuant"
        assert config.use_temporal is True
        assert config.temporal_window_size == 10

    def test_presets(self):
        """Test preset configurations."""
        from live_vpr.vprtempo_config import VPRTEMPO_FAST, VPRTEMPO_ACCURATE, VPRTEMPO_BALANCED

        assert VPRTEMPO_FAST.model == "VPRTempoQuant"
        assert VPRTEMPO_FAST.use_temporal is False

        assert VPRTEMPO_ACCURATE.model == "VPRTempo"
        assert VPRTEMPO_ACCURATE.use_temporal is True

        assert VPRTEMPO_BALANCED.model == "VPRTempo"
        assert VPRTEMPO_BALANCED.use_temporal is False


class TestTemporalBuffer:
    """Tests for TemporalVPRTempoBuffer."""

    def test_buffer_initialization(self):
        """Test buffer creation."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer

        buffer = TemporalVPRTempoBuffer(window_size=5)
        assert buffer.window_size == 5
        assert not buffer.is_ready()
        assert buffer.buffer_size() == 0

    def test_add_descriptor(self):
        """Test adding descriptors to buffer."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer

        buffer = TemporalVPRTempoBuffer(window_size=3)
        descriptor = np.random.randn(256).astype(np.float32)

        buffer.add_descriptor(descriptor)
        assert buffer.buffer_size() == 1
        assert not buffer.is_ready()

        buffer.add_descriptor(descriptor)
        buffer.add_descriptor(descriptor)
        assert buffer.buffer_size() == 3
        assert buffer.is_ready()

    def test_buffer_overflow(self):
        """Test that buffer maintains fixed size."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer

        buffer = TemporalVPRTempoBuffer(window_size=2)
        desc1 = np.ones(256).astype(np.float32)
        desc2 = np.ones(256).astype(np.float32) * 2
        desc3 = np.ones(256).astype(np.float32) * 3

        buffer.add_descriptor(desc1)
        buffer.add_descriptor(desc2)
        buffer.add_descriptor(desc3)

        # Buffer should only have last 2
        assert buffer.buffer_size() == 2
        agg = buffer.get_aggregated()
        assert agg is not None

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer

        descriptors = [
            np.ones(256).astype(np.float32),
            np.ones(256).astype(np.float32) * 2,
        ]

        # Mean aggregation
        buffer_mean = TemporalVPRTempoBuffer(
            window_size=2, aggregation_method="mean"
        )
        buffer_mean.add_descriptor(descriptors[0])
        buffer_mean.add_descriptor(descriptors[1])
        agg_mean = buffer_mean.get_aggregated()
        assert np.allclose(agg_mean, 1.5)

        # Max aggregation
        buffer_max = TemporalVPRTempoBuffer(
            window_size=2, aggregation_method="max"
        )
        buffer_max.add_descriptor(descriptors[0])
        buffer_max.add_descriptor(descriptors[1])
        agg_max = buffer_max.get_aggregated()
        assert np.allclose(agg_max, 2.0)

    def test_reset(self):
        """Test buffer reset."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer

        buffer = TemporalVPRTempoBuffer(window_size=3)
        descriptor = np.random.randn(256).astype(np.float32)

        buffer.add_descriptor(descriptor)
        assert buffer.buffer_size() == 1

        buffer.reset()
        assert buffer.buffer_size() == 0
        assert not buffer.is_ready()

    def test_invalid_descriptor_shape(self):
        """Test error on invalid descriptor shape."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer

        buffer = TemporalVPRTempoBuffer(window_size=3)
        invalid_descriptor = np.random.randn(256, 2)

        with pytest.raises(ValueError):
            buffer.add_descriptor(invalid_descriptor)


class TestTemporalLocalizer:
    """Tests for TemporalVPRTempoLocalizer."""

    def test_localizer_initialization(self):
        """Test temporal localizer creation."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoLocalizer

        reference_descriptors = np.random.randn(100, 256).astype(np.float32)
        reference_descriptors /= np.linalg.norm(
            reference_descriptors, axis=1, keepdims=True
        )

        localizer = TemporalVPRTempoLocalizer(reference_descriptors, window_size=5)
        assert localizer.reference_descriptors.shape == (100, 256)
        assert localizer.buffer.window_size == 5

    def test_localize(self):
        """Test localization with temporal buffer."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoLocalizer

        # Create random normalized descriptors
        np.random.seed(42)
        reference_descriptors = np.random.randn(10, 256).astype(np.float32)
        reference_descriptors /= np.linalg.norm(
            reference_descriptors, axis=1, keepdims=True
        )

        localizer = TemporalVPRTempoLocalizer(reference_descriptors, window_size=3)

        # Create a query descriptor similar to reference[0]
        query = reference_descriptors[0].copy()
        query /= np.linalg.norm(query) + 1e-8

        # First frame (no temporal)
        idx1, score1, desc1 = localizer.localize(query)
        result1 = localizer.get_last_result_info()
        assert not result1["is_temporal"]
        assert result1["buffer_size"] == 1

        # Second frame (no temporal yet)
        idx2, score2, desc2 = localizer.localize(query)
        result2 = localizer.get_last_result_info()
        assert not result2["is_temporal"]
        assert result2["buffer_size"] == 2

        # Third frame (temporal ready)
        idx3, score3, desc3 = localizer.localize(query)
        result3 = localizer.get_last_result_info()
        assert result3["is_temporal"]
        assert result3["buffer_size"] == 3

    def test_reset_temporal_localizer(self):
        """Test resetting temporal localizer."""
        from live_vpr.temporal_vprtempo import TemporalVPRTempoLocalizer

        reference_descriptors = np.random.randn(10, 256).astype(np.float32)
        reference_descriptors /= np.linalg.norm(
            reference_descriptors, axis=1, keepdims=True
        )

        localizer = TemporalVPRTempoLocalizer(reference_descriptors, window_size=3)
        query = reference_descriptors[0].copy()

        localizer.localize(query)
        localizer.localize(query)
        assert localizer.buffer.buffer_size() == 2

        localizer.reset()
        assert localizer.buffer.buffer_size() == 0
        assert localizer.get_last_result_info() is None


class TestVPRTempoUtils:
    """Tests for VPRTempo utility functions."""

    def test_get_preset_config(self):
        """Test getting preset configurations."""
        from live_vpr.vprtempo_utils import get_preset_config

        config_fast = get_preset_config("fast")
        assert config_fast.model == "VPRTempoQuant"

        config_accurate = get_preset_config("accurate")
        assert config_accurate.model == "VPRTempo"
        assert config_accurate.use_temporal is True

        config_balanced = get_preset_config("balanced")
        assert config_balanced.model == "VPRTempo"
        assert config_balanced.use_temporal is False

    def test_invalid_preset(self):
        """Test error on invalid preset."""
        from live_vpr.vprtempo_utils import get_preset_config

        with pytest.raises(ValueError):
            get_preset_config("nonexistent")

    def test_add_vprtempo_args(self):
        """Test CLI argument addition."""
        import argparse
        from live_vpr.vprtempo_utils import add_vprtempo_args

        parser = argparse.ArgumentParser()
        add_vprtempo_args(parser)

        args = parser.parse_args([
            "--vprtempo-model", "VPRTempoQuant",
            "--vprtempo-temporal",
            "--vprtempo-window-size", "10",
        ])

        assert args.vprtempo_model == "VPRTempoQuant"
        assert args.vprtempo_temporal is True
        assert args.vprtempo_window_size == 10

    def test_build_config_from_args(self):
        """Test building config from parsed args."""
        import argparse
        from live_vpr.vprtempo_utils import add_vprtempo_args, build_vprtempo_config_from_args

        parser = argparse.ArgumentParser()
        add_vprtempo_args(parser)

        args = parser.parse_args([
            "--vprtempo-model", "VPRTempoQuant",
            "--vprtempo-temporal",
        ])

        config = build_vprtempo_config_from_args(args)
        assert config.model == "VPRTempoQuant"
        assert config.use_temporal is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
