# VPRTempo Integration Architecture Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VPR_Tutorial                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    User Interface Layer                            │  │
│  │                     live_vpr_test.py                              │  │
│  │                                                                   │  │
│  │  Commands: build_map, live, video, check_source, list_sources   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  ↓                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                 Feature Extraction Layer                           │  │
│  │              live_vpr/extractors.py                              │  │
│  │                                                                   │  │
│  │  SUPPORTED_DESCRIPTORS:                                         │  │
│  │  ["HDC-DELF", "AlexNet", "CosPlace", "EigenPlaces",            │  │
│  │   "NetVLAD", "PatchNetVLAD", "SAD",                            │  │
│  │   "VPRTempo", "VPRTempoQuant"]  ← NEW                          │  │
│  │                                                                   │  │
│  │  create_feature_extractor(descriptor_name)                      │  │
│  │     ├─→ VPRTempoFeatureExtractor (new)                          │  │
│  │     ├─→ VPRTempoFeatureExtractor (quantized) (new)             │  │
│  │     ├─→ CosPlaceFeatureExtractor                               │  │
│  │     ├─→ NetVLADFeatureExtractor                                │  │
│  │     └─→ ... (other extractors)                                  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  ↓                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │              Feature Extraction Implementation                     │  │
│  │                                                                   │  │
│  │  feature_extraction/                                            │  │
│  │  ├─ feature_extractor_cosplace.py                              │  │
│  │  ├─ feature_extractor_vprtempo.py  ← NEW                       │  │
│  │  ├─ feature_extractor_eigenplaces.py                           │  │
│  │  └─ ...                                                         │  │
│  │                                                                   │  │
│  │  VPRTempoFeatureExtractor                                       │  │
│  │  ├─ __init__(quantized, device)                                │  │
│  │  ├─ _compute_features_small_batch()                            │  │
│  │  └─ compute_features(images)  → (N, 256) float32              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  ↓                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │              Online Localization Layer                            │  │
│  │              live_vpr/online.py                                  │  │
│  │                                                                   │  │
│  │  LocalizationResult                                             │  │
│  │  ├─ best_match_idx                                              │  │
│  │  ├─ best_score                                                  │  │
│  │  ├─ recognized                                                  │  │
│  │  ├─ top_k_indices / top_k_scores                               │  │
│  │  ├─ is_temporal  ← NEW                                         │  │
│  │  └─ buffer_size  ← NEW                                         │  │
│  │                                                                   │  │
│  │  LiveLocalizer                                                  │  │
│  │  ├─ __init__(ref_map, descriptor, use_temporal, ...)  ← MOD    │  │
│  │  ├─ temporal_localizer  ← NEW                                  │  │
│  │  ├─ localize_rgb(frame)  ← MOD                                 │  │
│  │  └─ reset_temporal()  ← NEW                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  ↓                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │            Temporal Processing Module (NEW)                       │  │
│  │            live_vpr/temporal_vprtempo.py                         │  │
│  │                                                                   │  │
│  │  TemporalVPRTempoBuffer                                         │  │
│  │  ├─ add_descriptor(descriptor)                                  │  │
│  │  ├─ get_aggregated()  → aggregated descriptor                  │  │
│  │  ├─ is_ready() / buffer_size() / reset()                       │  │
│  │  └─ aggregation_method: "mean" | "max" | "weighted_mean"      │  │
│  │                                                                   │  │
│  │  TemporalVPRTempoLocalizer                                      │  │
│  │  ├─ __init__(ref_descriptors, window_size, aggregation)        │  │
│  │  ├─ localize(query_descriptor)                                 │  │
│  │  ├─ get_last_result_info()                                     │  │
│  │  └─ reset()                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  ↓                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │           Configuration & Utilities Layer (NEW)                   │  │
│  │                                                                   │  │
│  │  live_vpr/vprtempo_config.py                                    │  │
│  │  ├─ VPRTempoConfig (dataclass)                                 │  │
│  │  ├─ VPRTEMPO_FAST preset                                        │  │
│  │  ├─ VPRTEMPO_BALANCED preset                                    │  │
│  │  └─ VPRTEMPO_ACCURATE preset                                    │  │
│  │                                                                   │  │
│  │  live_vpr/vprtempo_utils.py                                     │  │
│  │  ├─ add_vprtempo_args(parser)                                   │  │
│  │  ├─ build_vprtempo_config_from_args(args)                      │  │
│  │  ├─ get_preset_config(preset_name)                             │  │
│  │  └─ print_vprtempo_info()                                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow - Building a Map

```
Reference Images
       ↓
[VPRTempoFeatureExtractor]
  - Load images
  - Resize to 480x640
  - Normalize
  - Extract descriptors (256-dim)
       ↓
[Normalize Descriptors]
  - L2 normalize each descriptor
       ↓
[save_reference_map]
  - Save descriptors + metadata to .npz
       ↓
Saved Reference Map
```

## Data Flow - Online Localization (Standard Mode)

```
Live Frame (H, W, 3)
       ↓
[VPRTempoFeatureExtractor.compute_features]
  - Resize to 480x640
  - Normalize
  - Extract descriptor (256-dim)
       ↓
[Normalize Descriptor]
  - L2 normalize
       ↓
[Cosine Similarity]
  - descriptors @ query.T
       ↓
[Top-K Selection]
  - Find best matches
       ↓
LocalizationResult
  {best_match_idx, best_score, recognized, ...}
```

## Data Flow - Online Localization (Temporal Mode)

```
Live Frame 1 (H, W, 3)
       ↓
[Feature Extraction] → descriptor_1
       ↓
[TemporalVPRTempoBuffer]
  buffer = [descriptor_1]
  ready = False
       ↓
[Use descriptor_1 for localization]
       ↓
└───────┬───────┘

Live Frame 2 (H, W, 3)
       ↓
[Feature Extraction] → descriptor_2
       ↓
[TemporalVPRTempoBuffer]
  buffer = [descriptor_1, descriptor_2]
  ready = False
       ↓
[Use descriptor_2 for localization]
       ↓
└───────┬───────┘

Live Frame 3 (H, W, 3)
       ↓
[Feature Extraction] → descriptor_3
       ↓
[TemporalVPRTempoBuffer]
  buffer = [descriptor_1, descriptor_2, descriptor_3]
  (assuming window_size=3)
  ready = True
       ↓
[Aggregate: weighted_mean([d1, d2, d3])]
  → aggregated_descriptor
       ↓
[Cosine Similarity with aggregated]
       ↓
LocalizationResult
  {best_match_idx, best_score, is_temporal=True, buffer_size=3, ...}
```

## Class Hierarchy

```
FeatureExtractor (ABC)
├─ HDCDELFExtractor
├─ AlexNetConv3Extractor
├─ CosPlaceFeatureExtractor
├─ EigenPlacesFeatureExtractor
├─ PatchNetVLADFeatureExtractor
└─ VPRTempoFeatureExtractor  ← NEW
     ├─ Uses VPRTempoImageDataset for preprocessing
     ├─ Supports quantized=False (fp32) and quantized=True (int8)
     └─ Auto-detects device (CUDA, MPS, CPU)


LiveLocalizer
├─ extractor: FeatureExtractor
├─ reference_map: ReferenceMap
├─ temporal_localizer: Optional[TemporalVPRTempoLocalizer]  ← NEW
└─ Methods:
   ├─ localize_rgb(frame) → LocalizationResult
   ├─ reset_temporal()  ← NEW
   └─ set_threshold()


TemporalVPRTempoLocalizer  ← NEW
├─ buffer: TemporalVPRTempoBuffer
├─ reference_descriptors: (N, 256)
└─ Methods:
   ├─ localize(descriptor) → (idx, score, agg_descriptor)
   ├─ get_last_result_info() → dict
   └─ reset()


TemporalVPRTempoBuffer  ← NEW
├─ descriptor_buffer: deque
├─ window_size: int
├─ aggregation_method: str
└─ Methods:
   ├─ add_descriptor(descriptor)
   ├─ get_aggregated() → Optional[descriptor]
   ├─ is_ready() → bool
   ├─ reset()
   └─ buffer_size() → int


VPRTempoConfig  ← NEW (dataclass)
├─ model: "VPRTempo" | "VPRTempoQuant"
├─ use_temporal: bool
├─ temporal_window_size: int
├─ temporal_aggregation: str
└─ Methods:
   ├─ to_dict()
   └─ from_dict(config_dict)
```

## Configuration Presets

```
VPRTEMPO_FAST (quantized, no temporal)
├─ model: VPRTempoQuant
├─ use_temporal: False
└─ ~2x faster than full precision

VPRTEMPO_BALANCED (full precision, no temporal)
├─ model: VPRTempo
├─ use_temporal: False
└─ Default choice

VPRTEMPO_ACCURATE (full precision + temporal)
├─ model: VPRTempo
├─ use_temporal: True
├─ temporal_window_size: 5
├─ temporal_aggregation: weighted_mean
└─ ~5-15% accuracy improvement
```

## File Organization

```
VPR_Tutorial/
├── feature_extraction/
│   ├── feature_extractor.py (ABC)
│   ├── feature_extractor_cosplace.py
│   ├── feature_extractor_vprtempo.py  ← NEW
│   ├── feature_extractor_eigenplaces.py
│   └── ...
│
├── live_vpr/
│   ├── extractors.py (MODIFIED)
│   ├── online.py (MODIFIED)
│   ├── offline.py
│   ├── database.py
│   ├── sources.py
│   ├── ui.py
│   ├── temporal_vprtempo.py  ← NEW
│   ├── vprtempo_config.py  ← NEW
│   ├── vprtempo_utils.py  ← NEW
│   └── capture.py
│
├── docs/
│   ├── LIVE_VPR_PIPELINE.md
│   ├── PROJECT_ARCHITECTURE_GUIDE.md
│   ├── VPRTEMPO_INTEGRATION.md  ← NEW
│   ├── VPRTEMPO_QUICKREF.md  ← NEW
│   └── VPRTEMPO_IMPLEMENTATION.md  ← NEW
│
├── test_vprtempo.py  ← NEW
├── VPRTEMPO_CHANGES.md  ← NEW
└── ...
```

## Dependencies Graph

```
vprtempo (optional)
    ↓
torch, torchvision
    ↓
feature_extraction/feature_extractor_vprtempo.py
    ↓
live_vpr/extractors.py
    ↓
live_vpr/online.py
    ↓
live_vpr/temporal_vprtempo.py (optional temporal mode)
    ↓
live_vpr_test.py (CLI entry point)
```

## Backward Compatibility

✅ All changes are backward compatible

```
Existing Code Path (unchanged):
  create_feature_extractor("CosPlace")
    → CosPlaceFeatureExtractor() ← unchanged
    → compute_features(images)
    → LiveLocalizer() ← uses defaults (use_temporal=False)

New Code Path (opt-in):
  create_feature_extractor("VPRTempo")
    → VPRTempoFeatureExtractor() ← NEW
    → compute_features(images)
    → LiveLocalizer(..., use_temporal=True) ← NEW
    → TemporalVPRTempoLocalizer() ← NEW (internal)
```
