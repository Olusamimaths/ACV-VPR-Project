"""
VPRTempo configuration and defaults.

Centralized configuration for VPRTempo descriptor options in the live pipeline.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class VPRTempoConfig:
    """Configuration for VPRTempo feature extraction."""

    # Model variant
    model: Literal["VPRTempo", "VPRTempoQuant"] = "VPRTempo"

    # Temporal aggregation
    use_temporal: bool = False
    temporal_window_size: int = 5
    temporal_aggregation: Literal["mean", "max", "weighted_mean"] = "weighted_mean"

    # Inference settings
    batch_size: int = 8
    descriptor_dim: int = 256

    # Device
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"

    # Model loading
    pretrained: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "use_temporal": self.use_temporal,
            "temporal_window_size": self.temporal_window_size,
            "temporal_aggregation": self.temporal_aggregation,
            "batch_size": self.batch_size,
            "descriptor_dim": self.descriptor_dim,
            "device": self.device,
            "pretrained": self.pretrained,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VPRTempoConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# Preset configurations
VPRTEMPO_FAST = VPRTempoConfig(
    model="VPRTempoQuant",
    use_temporal=False,
    batch_size=8,
)

VPRTEMPO_ACCURATE = VPRTempoConfig(
    model="VPRTempo",
    use_temporal=True,
    temporal_window_size=5,
    batch_size=4,
)

VPRTEMPO_BALANCED = VPRTempoConfig(
    model="VPRTempo",
    use_temporal=False,
    batch_size=8,
)
