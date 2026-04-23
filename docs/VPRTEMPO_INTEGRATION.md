# VPRTempo Integration Guide

This guide explains how to use VPRTempo (a Spiking Neural Network for Visual Place Recognition) in the live VPR pipeline.

## Overview

VPRTempo is a temporally encoded spiking neural network that achieves fast and efficient visual place recognition. This integration adds VPRTempo as a descriptor option alongside CosPlace, NetVLAD, and others.

**Reference**: Hines et al., 2024 ICRA - "VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition"

## Installation

VPRTempo is provided as an optional dependency. Install it with:

```bash
pip install vprtempo
```

Or add to your `requirements.txt`:

```
vprtempo>=1.1.11
```

## Quick Start

### 1. Build a Map with VPRTempo

```bash
# Using the Python CLI
python live_vpr_test.py --mode build_map \
  --data_dir images/reference_folder \
  --descriptor VPRTempo \
  --map_path artifacts/maps/my_map_vprtempo.npz

# Or the bash wrapper
bash scripts/live_vpr_cli.sh build-map \
  --data_dir images/reference_folder \
  --descriptor VPRTempo \
  --map_path artifacts/maps/my_map_vprtempo.npz
```

### 2. Run Live Localization

```bash
# Real-time camera localization
python live_vpr_test.py --mode live \
  --source 0 \
  --map_path artifacts/maps/my_map_vprtempo.npz \
  --descriptor VPRTempo

# Or with temporal aggregation (more accurate, slightly slower)
python live_vpr_test.py --mode live \
  --source 0 \
  --map_path artifacts/maps/my_map_vprtempo.npz \
  --descriptor VPRTempo \
  --use_temporal \
  --temporal_window_size 5
```

### 3. Video Evaluation

```bash
python live_vpr_test.py --mode video \
  --video_path path/to/video.mp4 \
  --map_path artifacts/maps/my_map_vprtempo.npz \
  --descriptor VPRTempo
```

## Configuration Options

### Model Variants

- **VPRTempo**: Full precision (fp32) - better accuracy
- **VPRTempoQuant**: Quantized (int8) - faster inference, slightly lower accuracy

### Temporal Aggregation

VPRTempo supports temporal aggregation across consecutive frames to exploit spike timing information:

```bash
# Enable temporal mode with weighted mean aggregation
--use_temporal \
--temporal_window_size 5 \
--temporal_aggregation weighted_mean
```

Aggregation methods:
- `mean`: Simple average of descriptors
- `max`: Element-wise maximum (emphasis on strongest spikes)
- `weighted_mean`: Recent frames weighted higher (default, recommended)

### Preset Configurations

Use presets for common scenarios:

```python
from live_vpr.vprtempo_utils import get_preset_config

# Fast inference (quantized, no temporal)
config = get_preset_config("fast")

# Balanced accuracy/speed
config = get_preset_config("balanced")

# Maximum accuracy (full precision, temporal)
config = get_preset_config("accurate")
```

## Python API

### Basic Usage

```python
from live_vpr.extractors import create_feature_extractor, compute_global_descriptors
import numpy as np

# Create extractor
extractor = create_feature_extractor("VPRTempo")

# Extract descriptors from images
images = [image1, image2, ...]  # List of (H, W, 3) uint8 arrays
descriptors = compute_global_descriptors(extractor, images)
# Returns: (N, 256) float32 array
```

### Temporal Localization

```python
from live_vpr.temporal_vprtempo import TemporalVPRTempoLocalizer
from live_vpr.database import load_reference_map

# Load reference map
ref_map = load_reference_map("artifacts/maps/my_map_vprtempo.npz")

# Create temporal localizer
localizer = TemporalVPRTempoLocalizer(
    reference_descriptors=ref_map.descriptors,
    window_size=5,
    aggregation_method="weighted_mean"
)

# Process frames
for frame in video_stream:
    descriptor = compute_global_descriptors(extractor, [frame])
    best_idx, best_score, aggregated = localizer.localize(descriptor.flatten())
    print(f"Match: {best_idx}, Score: {best_score:.4f}, Temporal: {aggregated is not None}")
```

### Live Localizer with Temporal Support

```python
from live_vpr.online import LiveLocalizer
from live_vpr.database import load_reference_map

# Load map
ref_map = load_reference_map("artifacts/maps/my_map_vprtempo.npz")

# Create localizer with temporal support
localizer = LiveLocalizer(
    reference_map=ref_map,
    descriptor_name="VPRTempo",
    threshold=0.5,
    top_k=5,
    use_temporal=True,
    temporal_window_size=5
)

# Localize frames
result = localizer.localize_rgb(frame_rgb)
print(f"Match: {result.best_match_idx}")
print(f"Score: {result.best_score:.4f}")
print(f"Temporal: {result.is_temporal} (buffer: {result.buffer_size}/5)")
print(f"Time: {result.extraction_time_ms:.2f} ms")

# Reset temporal buffer on location jumps
localizer.reset_temporal()
```

## Comparing Descriptors

To benchmark VPRTempo against other descriptors:

```bash
# Build maps for different descriptors
python live_vpr_test.py --mode build_map \
  --data_dir images/ref \
  --descriptor CosPlace \
  --map_path artifacts/maps/cosplace.npz

python live_vpr_test.py --mode build_map \
  --data_dir images/ref \
  --descriptor VPRTempo \
  --map_path artifacts/maps/vprtempo.npz

# Run evaluation on test video
python live_vpr_test.py --mode video \
  --video_path test.mp4 \
  --map_path artifacts/maps/cosplace.npz \
  --descriptor CosPlace

python live_vpr_test.py --mode video \
  --video_path test.mp4 \
  --map_path artifacts/maps/vprtempo.npz \
  --descriptor VPRTempo
```

## Performance Characteristics

### VPRTempo (fp32)
- **Descriptor Dimension**: 256
- **Typical Latency**: ~50-100ms per frame (GPU), ~200-500ms (CPU)
- **Memory**: ~1GB GPU VRAM
- **Accuracy**: Excellent on diverse datasets

### VPRTempoQuant (int8)
- **Descriptor Dimension**: 256
- **Typical Latency**: ~30-50ms per frame (GPU), ~100-300ms (CPU)
- **Memory**: ~500MB GPU VRAM
- **Accuracy**: Slightly lower than fp32, still very competitive

### Temporal Mode
- **Additional Latency**: Minimal (just aggregation)
- **Accuracy Improvement**: ~5-15% depending on dataset
- **Memory**: O(window_size) additional descriptors

## Troubleshooting

### VPRTempo not installed
```
ImportError: VPRTempo not installed. Install with: pip install vprtempo
```

**Solution**: Install VPRTempo or ensure it's in your environment.

### Model download hangs
VPRTempo downloads pretrained weights on first use (~600MB). Ensure:
- Internet connectivity
- Sufficient disk space
- Proper permissions in `~/.cache/`

**Workaround**: Download weights separately and set `VPRTEMPO_MODEL_PATH` environment variable.

### Out of memory
- Use `VPRTempoQuant` instead of `VPRTempo`
- Reduce batch size (edit source code)
- Run on CPU (slower but uses less VRAM)

### Temporal mode not improving accuracy
- Increase `temporal_window_size` (5-10 frames)
- Try `weighted_mean` aggregation
- Ensure camera motion is smooth (not jumping locations)

## Advanced Customization

### Custom Aggregation
```python
from live_vpr.temporal_vprtempo import TemporalVPRTempoBuffer
import numpy as np

# Extend temporal buffer with custom aggregation
class CustomAggregation(TemporalVPRTempoBuffer):
    def get_aggregated(self):
        # Your custom aggregation logic
        descriptors = np.stack(list(self.descriptor_buffer))
        return my_custom_aggregate(descriptors)
```

### Using with Benchmark Pipeline
VPRTempo works with the standard evaluation pipeline:

```bash
# Evaluate on Nordland, Oxford RobotCar, etc.
python demo.py \
  --dataset nordland \
  --descriptor VPRTempo \
  --save_results
```

## File Structure

```
live_vpr/
  extractors.py                 # Main integration point
  online.py                     # Temporal localizer support
  vprtempo_config.py            # Configuration classes
  vprtempo_utils.py             # Helper utilities
  temporal_vprtempo.py          # Temporal aggregation module

feature_extraction/
  feature_extractor_vprtempo.py # VPRTempo wrapper class
```

## Citation

If you use VPRTempo, please cite the original paper:

```bibtex
@inproceedings{hines2024vprtempo,
  title={VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition},
  author={Adam D. Hines and Peter G. Stratton and Michael Milford and Tobias Fischer},
  year={2024},
  pages={10200-10207},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}
}
```

## References

- VPRTempo GitHub: https://github.com/QVPR/VPRTempo
- VPRTempo Paper: https://arxiv.org/abs/2402.17764
- Live VPR Pipeline: See `docs/LIVE_VPR_PIPELINE.md`
