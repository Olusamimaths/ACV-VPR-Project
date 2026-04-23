"""
VPRTempo utilities and helpers for the live pipeline.

Provides convenience functions for VPRTempo integration.
"""

import argparse
from typing import Optional

from .vprtempo_config import VPRTempoConfig, VPRTEMPO_FAST, VPRTEMPO_ACCURATE, VPRTEMPO_BALANCED


def add_vprtempo_args(parser: argparse.ArgumentParser) -> None:
    """
    Add VPRTempo-specific arguments to an argument parser.

    Use this to easily add VPRTempo options to CLI tools.

    Args:
        parser: ArgumentParser to add arguments to
    """
    vprtempo_group = parser.add_argument_group("VPRTempo Options")

    vprtempo_group.add_argument(
        "--vprtempo-model",
        type=str,
        choices=["VPRTempo", "VPRTempoQuant"],
        default="VPRTempo",
        help="VPRTempo model variant (default: VPRTempo)",
    )

    vprtempo_group.add_argument(
        "--vprtempo-temporal",
        action="store_true",
        help="Enable temporal aggregation for VPRTempo",
    )

    vprtempo_group.add_argument(
        "--vprtempo-window-size",
        type=int,
        default=5,
        help="Temporal window size for aggregation (default: 5)",
    )

    vprtempo_group.add_argument(
        "--vprtempo-aggregation",
        type=str,
        choices=["mean", "max", "weighted_mean"],
        default="weighted_mean",
        help="Temporal aggregation method (default: weighted_mean)",
    )

    vprtempo_group.add_argument(
        "--vprtempo-preset",
        type=str,
        choices=["fast", "balanced", "accurate"],
        help="Use preset configuration (overrides other VPRTempo options)",
    )


def build_vprtempo_config_from_args(args: argparse.Namespace) -> VPRTempoConfig:
    """
    Build VPRTempoConfig from parsed arguments.

    Args:
        args: Parsed arguments (must include VPRTempo options from add_vprtempo_args)

    Returns:
        VPRTempoConfig instance
    """
    # Use preset if provided
    if hasattr(args, "vprtempo_preset") and args.vprtempo_preset:
        if args.vprtempo_preset == "fast":
            return VPRTEMPO_FAST
        elif args.vprtempo_preset == "accurate":
            return VPRTEMPO_ACCURATE
        elif args.vprtempo_preset == "balanced":
            return VPRTEMPO_BALANCED

    # Build from individual arguments
    config = VPRTempoConfig(
        model=getattr(args, "vprtempo_model", "VPRTempo"),
        use_temporal=getattr(args, "vprtempo_temporal", False),
        temporal_window_size=getattr(args, "vprtempo_window_size", 5),
        temporal_aggregation=getattr(args, "vprtempo_aggregation", "weighted_mean"),
    )

    return config


def get_preset_config(preset_name: str) -> VPRTempoConfig:
    """
    Get a preset VPRTempo configuration.

    Args:
        preset_name: "fast", "balanced", or "accurate"

    Returns:
        VPRTempoConfig instance

    Raises:
        ValueError: If preset_name is not recognized
    """
    presets = {
        "fast": VPRTEMPO_FAST,
        "balanced": VPRTEMPO_BALANCED,
        "accurate": VPRTEMPO_ACCURATE,
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {', '.join(presets.keys())}"
        )

    return presets[preset_name]


def print_vprtempo_info() -> None:
    """Print information about VPRTempo and available presets."""
    print("\n" + "=" * 70)
    print("VPRTempo Integration")
    print("=" * 70)
    print("\nSupported descriptors:")
    print("  - VPRTempo: Full precision (fp32)")
    print("  - VPRTempoQuant: Quantized (int8)")
    print("\nAvailable presets:")
    print(f"  FAST:      {VPRTEMPO_FAST.to_dict()}")
    print(f"  BALANCED:  {VPRTEMPO_BALANCED.to_dict()}")
    print(f"  ACCURATE:  {VPRTEMPO_ACCURATE.to_dict()}")
    print("\nFeatures:")
    print("  - Spiking Neural Network (SNN) architecture")
    print("  - Temporal spike encoding")
    print("  - Optional temporal aggregation across frames")
    print("=" * 70 + "\n")
