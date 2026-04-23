#!/usr/bin/env python3
"""
VPRTempo Configuration Presets Demo

Shows how to use different configuration presets:
- FAST: Quantized, no temporal (fastest)
- BALANCED: Full precision, no temporal (balanced)
- ACCURATE: Full precision + temporal (best accuracy)
"""

import sys


def demo_presets():
    """Demonstrate VPRTempo presets."""
    print("\n" + "="*70)
    print("VPRTempo Configuration Presets")
    print("="*70)
    
    try:
        from live_vpr.vprtempo_config import (
            VPRTempoConfig,
            VPRTEMPO_FAST,
            VPRTEMPO_BALANCED,
            VPRTEMPO_ACCURATE,
        )
        from live_vpr.vprtempo_utils import get_preset_config
        print("\n✅ VPRTempo modules loaded successfully!\n")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("This is expected if you're just testing the integration.")
        print("Presets are available once you 'pip install vprtempo'\n")
        demo_presets_info()
        return
    
    # Show all presets
    presets = {
        "FAST": VPRTEMPO_FAST,
        "BALANCED": VPRTEMPO_BALANCED,
        "ACCURATE": VPRTEMPO_ACCURATE,
    }
    
    print("-"*70)
    print("Available Presets:")
    print("-"*70)
    
    for name, config in presets.items():
        print(f"\n📋 {name}:")
        print(f"   Model:                {config.model}")
        print(f"   Temporal enabled:     {config.use_temporal}")
        print(f"   Temporal window:      {config.temporal_window_size}")
        print(f"   Aggregation method:   {config.temporal_aggregation}")
        print(f"   Batch size:           {config.batch_size}")
        print(f"   Descriptor dim:       {config.descriptor_dim}")
        
        if config.model == "VPRTempoQuant":
            print(f"   Speed:                🚀🚀🚀 (Fastest)")
            print(f"   Accuracy:             ⭐⭐⭐⭐")
        elif config.use_temporal:
            print(f"   Speed:                🚀 (Slower due to temporal)")
            print(f"   Accuracy:             ⭐⭐⭐⭐⭐ (Best)")
        else:
            print(f"   Speed:                🚀🚀 (Fast)")
            print(f"   Accuracy:             ⭐⭐⭐⭐⭐")
    
    # Usage examples
    print("\n" + "-"*70)
    print("Usage Examples:")
    print("-"*70)
    
    print("\n1️⃣  Using preset directly:")
    print("   from live_vpr.vprtempo_config import VPRTEMPO_FAST")
    print("   config = VPRTEMPO_FAST")
    
    print("\n2️⃣  Getting preset by name:")
    print("   from live_vpr.vprtempo_utils import get_preset_config")
    print("   config = get_preset_config('accurate')")
    
    print("\n3️⃣  Using with LiveLocalizer:")
    print("   localizer = LiveLocalizer(")
    print("       reference_map=ref_map,")
    print("       descriptor_name=config.model,")
    print("       use_temporal=config.use_temporal,")
    print("       temporal_window_size=config.temporal_window_size")
    print("   )")
    
    print("\n4️⃣  Custom configuration:")
    print("   config = VPRTempoConfig(")
    print("       model='VPRTempo',")
    print("       use_temporal=True,")
    print("       temporal_window_size=10,")
    print("       temporal_aggregation='weighted_mean'")
    print("   )")
    
    print("\n5️⃣  From dictionary:")
    print("   config_dict = {")
    print("       'model': 'VPRTempo',")
    print("       'use_temporal': True,")
    print("       'temporal_window_size': 7,")
    print("   }")
    print("   config = VPRTempoConfig.from_dict(config_dict)")
    
    print("\n" + "-"*70)
    print("Preset Comparison:")
    print("-"*70)
    
    comparison = """
    ┌──────────────────────────────────────────────────────────┐
    │ Preset    │ Speed    │ Accuracy │ Latency │ Memory       │
    ├───────────┼──────────┼──────────┼─────────┼──────────────┤
    │ FAST      │ 🚀🚀🚀  │ ⭐⭐⭐⭐ │ 30-50ms │ ~500MB       │
    │ BALANCED  │ 🚀🚀    │ ⭐⭐⭐⭐⭐│ 50-100ms│ ~1GB         │
    │ ACCURATE  │ 🚀      │ ⭐⭐⭐⭐⭐│ 60-150ms│ ~1GB + buffer│
    └──────────────────────────────────────────────────────────┘
    
    Notes:
    • FAST uses VPRTempoQuant (int8 quantization)
    • BALANCED and ACCURATE use full precision (fp32)
    • ACCURATE adds temporal aggregation overhead
    • Latency improves with GPU (CUDA/MPS)
    • Memory shown for GPU; CPU usage slightly different
    """
    print(comparison)


def demo_presets_info():
    """Show preset information without importing."""
    print("-"*70)
    print("VPRTempo Presets (Information):")
    print("-"*70)
    
    presets_info = {
        "FAST": {
            "model": "VPRTempoQuant",
            "temporal": False,
            "speed": "🚀🚀🚀 Fastest",
            "accuracy": "⭐⭐⭐⭐ Very Good",
            "use_case": "Real-time embedded systems, mobile",
        },
        "BALANCED": {
            "model": "VPRTempo",
            "temporal": False,
            "speed": "🚀🚀 Fast",
            "accuracy": "⭐⭐⭐⭐⭐ Excellent",
            "use_case": "Most applications, good balance",
        },
        "ACCURATE": {
            "model": "VPRTempo",
            "temporal": True,
            "speed": "🚀 Moderate",
            "accuracy": "⭐⭐⭐⭐⭐ Best",
            "use_case": "High-accuracy offline processing",
        },
    }
    
    for name, info in presets_info.items():
        print(f"\n📋 {name}:")
        print(f"   Model:       {info['model']}")
        print(f"   Temporal:    {info['temporal']}")
        print(f"   Speed:       {info['speed']}")
        print(f"   Accuracy:    {info['accuracy']}")
        print(f"   Best for:    {info['use_case']}")
    
    print("\n" + "-"*70)
    print("Installation:")
    print("-"*70)
    print("\nTo use VPRTempo with these presets:")
    print("  pip install vprtempo")
    print("\nThen:")
    print("  from live_vpr.vprtempo_utils import get_preset_config")
    print("  config = get_preset_config('balanced')  # or 'fast', 'accurate'")


def main():
    """Run demo."""
    print("\n" + "="*70)
    print("VPRTempo Configuration & Presets Demo")
    print("="*70)
    
    try:
        demo_presets()
    except Exception as e:
        print(f"⚠️  Error during full demo: {e}")
        print("\nFalling back to information-only mode...")
        demo_presets_info()
    
    print("\n" + "="*70)
    print("For more information:")
    print("="*70)
    print("📖 User guide:     docs/VPRTEMPO_INTEGRATION.md")
    print("⚡ Quick ref:      docs/VPRTEMPO_QUICKREF.md")
    print("🏗️  Architecture:  docs/VPRTEMPO_ARCHITECTURE.md")
    print("📝 Implementation: docs/VPRTEMPO_IMPLEMENTATION.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
