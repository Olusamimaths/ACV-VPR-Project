# Campus Dataset - User Guide

## Overview

This guide covers how to use the Visual Place Recognition (VPR) system with your custom campus dataset.

## Dataset Structure

```
custom_dataset/
├── day_images/      # 50 reference images (database)
│   └── image001.jpg ... image050.jpg
└── night_images/    # 64 query images
    ├── image001.jpg          # Has matching day image
    ├── image003-npm.jpg      # No perfect match (-npm suffix)
    └── ...
```

### Naming Convention
- `image042.jpg` (night) matches `image042.jpg` (day)
- `image043-npm.jpg` has no corresponding day image
- `PXL_*.jpg` files have no matches

## Quick Start

### Run VPR Test
```bash
python test_campus_dataset.py --descriptor CosPlace --save_results
```

### Command Options
```bash
--descriptor    # Feature extractor: CosPlace, EigenPlaces, HDC-DELF, AlexNet, NetVLAD
--n_correct N   # Number of correct matches to display (default: 3)
--n_wrong N     # Number of wrong matches to display (default: 5)
--save_results  # Save all visualizations to output_images/
```

### Examples
```bash
# Basic test with CosPlace
python test_campus_dataset.py --descriptor CosPlace --save_results

# Show more wrong matches for failure analysis
python test_campus_dataset.py --descriptor CosPlace --n_correct 2 --n_wrong 15 --save_results

# Compare different descriptors
python test_campus_dataset.py --descriptor EigenPlaces --save_results
```

## Output Files

When using `--save_results`, files are saved to `output_images/`:

| File | Description |
|------|-------------|
| `campus_matches_examples.png` | Correct (green) and wrong (red) match visualizations |
| `campus_similarity_matrix.png` | Heatmap of all day-night similarities |
| `campus_pr_curve.png` | Precision-Recall curve |
| `campus_results.txt` | Performance metrics (AUC, R@K) |

## Understanding Results

### Visualization Colors
- **Green border** = Correct match (same location)
- **Red border** = Wrong match (different locations)

### Performance Metrics
- **AUC**: Area under PR curve (0-1, higher is better)
- **R@100P**: Max recall at 100% precision
- **R@1, R@5, R@10**: Top-K recall rates

### Expected Performance (Day-to-Night)
| Descriptor | AUC | R@1 |
|------------|-----|-----|
| CosPlace | 0.7-0.8 | 0.6-0.7 |
| EigenPlaces | 0.7-0.8 | 0.6-0.7 |
| HDC-DELF | 0.6-0.7 | 0.4-0.5 |

## Troubleshooting

### Missing dependencies
```bash
pip install scipy numpy matplotlib pillow scikit-image torch torchvision
```

### "No module named torch"
CosPlace and other deep learning descriptors require PyTorch:
```bash
pip install torch torchvision
```

### Low performance
Day-to-night matching is inherently challenging due to lighting changes. Try different descriptors to find the best one for your dataset.

## Files Reference

- `test_campus_dataset.py` - Main test script for campus dataset
- `demo.py` - Original VPR tutorial demo
- `live_vpr_test.py` - Real-time VPR with camera
- `datasets/load_dataset.py` - Dataset loaders (includes CampusDataset)
- `evaluation/show_correct_and_wrong_matches.py` - Match visualization
