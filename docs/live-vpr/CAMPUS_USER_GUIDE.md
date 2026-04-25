# Campus Dataset User Guide

This guide explains how the custom campus dataset fits into the project and how to use it for both evaluation and live testing.

Read this if you want to answer:

- what the campus dataset is
- how the campus benchmark script works
- how to build a live map from campus imagery
- which files matter when changing the campus workflow

## 1. What The Campus Dataset Is

The campus dataset is a local day-to-night place-recognition dataset stored in [custom_dataset/](../../custom_dataset).

High-level structure:

```text
custom_dataset/
├── day_images/
└── night_images/
```

The intended use is:

- `day_images`: reference database
- `night_images`: query set

Unlike the original tutorial datasets, this one was collected locally and includes explicit no-match cases.

## 2. How Ground Truth Works

Ground truth is created by [CampusDataset](../../datasets/load_dataset.py) in [datasets/load_dataset.py](../../datasets/load_dataset.py).

Key rules:

- `image042.jpg` matches `image042.jpg`
- `image043-npm.jpg` is treated as a no-match query
- `npmXX.jpg` is treated as a no-match query
- `PXL_*.jpg` is treated as a no-match query

The loader also creates:

- `GThard`: strict matches
- `GTsoft`: a small tolerance window for nearby matches

## 3. Main Campus Evaluation Script

The main script is [test_campus_dataset.py](../../test_campus_dataset.py).

It follows the same evaluation pattern as [demo.py](../../demo.py):

```text
load campus dataset
-> extract descriptors
-> compute similarity matrix
-> apply matching
-> compute metrics
-> save plots / examples
```

The main difference from `demo.py` is that it uses `CampusDataset` instead of the built-in tutorial datasets.

## 4. Typical Campus Evaluation Command

```bash
python test_campus_dataset.py --descriptor CosPlace --save_results
```

This will:

- load the campus day and night images
- compute descriptors
- build the similarity matrix
- compute metrics like AUC, `R@100P`, and `R@K`
- save the plots and summary files into `output_images/`

Useful variants:

```bash
python test_campus_dataset.py --descriptor EigenPlaces --save_results
python test_campus_dataset.py --descriptor CosPlace --n_correct 2 --n_wrong 15 --save_results
```

## 5. Output Files

When `--save_results` is used, the campus script writes files into [output_images/runs/](../../output_images/runs), and also refreshes the latest legacy copies in [output_images/legacy/campus_strict/](../../output_images/legacy/campus_strict).

Common outputs:

- `output_images/legacy/campus_strict/campus_similarity_matrix.png`
- `output_images/legacy/campus_strict/campus_matching_results.png`
- `output_images/legacy/campus_strict/campus_pr_curve.png`
- `output_images/legacy/campus_strict/campus_matches_examples.png`
- `output_images/legacy/campus_strict/campus_results.txt`

These are useful for:

- qualitative debugging
- comparing descriptors
- including figures in reports or slides

## 6. Using The Campus Data In The Live Pipeline

The live system does not use `GThard` or `GTsoft` during runtime. It uses a saved reference map instead.

That means the typical campus live workflow is:

1. build a map from `custom_dataset/day_images`
2. run live or video localization against that map

### Build A Map From The Existing Day Folder

```bash
python live_vpr_test.py \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_cosplace.npz
```

### Build A Map From A Recorded Traversal

Use this when the original day images are not enough.

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --source 0 \
  --recording_path artifacts/reference_videos/campus_day_walk.mp4 \
  --capture_dir artifacts/reference_captures/campus_day_walk \
  --sample_fps 1.0 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_live.npz
```

### Run Live Localization

```bash
python live_vpr_test.py \
  --mode live \
  --source 0 \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --process_fps 2.0
```

## 7. When To Use Which Path

Use [test_campus_dataset.py](../../test_campus_dataset.py) when you want:

- metrics
- PR curves
- recall values
- qualitative correct/wrong match plots

Use [live_vpr_test.py](../../live_vpr_test.py) when you want:

- map building
- webcam testing
- phone webcam testing
- stream testing
- demo preparation

## 8. Common Files For Campus Work

- [test_campus_dataset.py](../../test_campus_dataset.py): campus benchmark script
- [datasets/load_dataset.py](../../datasets/load_dataset.py): `CampusDataset` loader
- [live_vpr_test.py](../../live_vpr_test.py): live map building and live localization
- [live_vpr/offline.py](../../live_vpr/offline.py): reference-map construction
- [live_vpr/online.py](../../live_vpr/online.py): live localization
- [live_vpr/ui.py](../../live_vpr/ui.py): on-screen and saved live results

## 9. Troubleshooting

### Missing PyTorch

Some descriptors require PyTorch:

```bash
pip install torch torchvision
```

### Missing scientific packages

```bash
pip install scipy numpy matplotlib pillow scikit-image
```

### Weak day-to-night performance

That is expected to some degree. The campus set is harder and noisier than a clean benchmark, especially because it includes no-match cases.

### Live demo is slow

Lower:

- `--process_fps`
- source resolution
- map density if the reference map is extremely large

And confirm which runtime backend is printed during live startup:

- `cuda`
- `mps`
- or `cpu`
