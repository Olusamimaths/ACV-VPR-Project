# VPRTempo Quick Reference

Fast lookup for common VPRTempo operations.

## Installation

```bash
pip install vprtempo
```

## Key Commands

### Build Map
```bash
python live_vpr_test.py --mode build_map \
  --data_dir images/ref \
  --descriptor VPRTempo \
  --map_path map.npz
```

### Live Localization
```bash
# Basic
python live_vpr_test.py --mode live \
  --source 0 --map_path map.npz --descriptor VPRTempo

# With temporal aggregation
python live_vpr_test.py --mode live \
  --source 0 --map_path map.npz --descriptor VPRTempo \
  --use_temporal --temporal_window_size 5
```

### Video Evaluation
```bash
python live_vpr_test.py --mode video \
  --video_path test.mp4 --map_path map.npz --descriptor VPRTempo
```

## Code Snippets

### Extract Descriptors
```python
from live_vpr.extractors import create_feature_extractor, compute_global_descriptors

extractor = create_feature_extractor("VPRTempo")
descriptors = compute_global_descriptors(extractor, images)  # (N, 256)
```

### Temporal Localization
```python
from live_vpr.temporal_vprtempo import TemporalVPRTempoLocalizer

localizer = TemporalVPRTempoLocalizer(
    reference_descriptors=ref_map.descriptors,
    window_size=5
)

best_idx, score, agg = localizer.localize(descriptor.flatten())
```

### Use Presets
```python
from live_vpr.vprtempo_utils import get_preset_config

config = get_preset_config("fast")      # Quantized, no temporal
config = get_preset_config("balanced")  # Full precision, no temporal
config = get_preset_config("accurate")  # Full precision + temporal
```

## Model Variants

| Model | Precision | Speed | Accuracy | Memory |
|-------|-----------|-------|----------|--------|
| VPRTempo | fp32 | ~75ms | Excellent | ~1GB |
| VPRTempoQuant | int8 | ~40ms | Very Good | ~500MB |

## Temporal Aggregation

| Method | Use Case |
|--------|----------|
| `mean` | Uniform importance across frames |
| `max` | Emphasize strongest spikes |
| `weighted_mean` | Recent frames more important (recommended) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError: vprtempo | `pip install vprtempo` |
| Out of memory | Use VPRTempoQuant or CPU |
| Temporal not helping | Increase window size or improve camera smoothness |
| Model download hangs | Check internet, disk space, permissions |

## File Locations

- Feature extraction: `feature_extraction/feature_extractor_vprtempo.py`
- Temporal module: `live_vpr/temporal_vprtempo.py`
- Configuration: `live_vpr/vprtempo_config.py`
- Utilities: `live_vpr/vprtempo_utils.py`
- Documentation: `docs/VPRTEMPO_INTEGRATION.md`
- Tests: `test_vprtempo.py`

## API Reference

### VPRTempoFeatureExtractor
```python
extractor = VPRTempoFeatureExtractor(quantized=False, device="auto")
descriptors = extractor.compute_features(images)  # (N, 256)
```

### TemporalVPRTempoBuffer
```python
buffer = TemporalVPRTempoBuffer(window_size=5, aggregation_method="weighted_mean")
buffer.add_descriptor(descriptor)
agg = buffer.get_aggregated()  # None if not ready
```

### TemporalVPRTempoLocalizer
```python
localizer = TemporalVPRTempoLocalizer(reference_descriptors, window_size=5)
idx, score, agg = localizer.localize(query_descriptor)
info = localizer.get_last_result_info()
localizer.reset()
```

### LiveLocalizer (temporal mode)
```python
localizer = LiveLocalizer(
    reference_map=ref_map,
    descriptor_name="VPRTempo",
    use_temporal=True,
    temporal_window_size=5
)
result = localizer.localize_rgb(frame)
print(f"Temporal: {result.is_temporal}, Buffer: {result.buffer_size}/5")
localizer.reset_temporal()
```

## Performance Tips

1. **GPU Inference**: 2-5x faster than CPU
2. **Quantized Models**: 2x faster, minimal accuracy loss
3. **Small Batches**: Bypass DataLoader overhead for <8 frames
4. **Temporal Mode**: ~5-15% accuracy improvement, negligible latency
5. **Batch Processing**: Use larger batches for map building

## Citation

```bibtex
@inproceedings{hines2024vprtempo,
  title={VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition},
  author={Hines, Adam D and Stratton, Peter G and Milford, Michael and Fischer, Tobias},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={10200--10207},
  year={2024}
}
```

## See Also

- Full docs: `docs/VPRTEMPO_INTEGRATION.md`
- Implementation details: `docs/VPRTEMPO_IMPLEMENTATION.md`
- Live pipeline: `docs/LIVE_VPR_PIPELINE.md`
- Project architecture: `docs/PROJECT_ARCHITECTURE_GUIDE.md`
