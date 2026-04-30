# Nocturnal VPR Quick Start

This project evaluates visual place recognition for day-to-night campus localization and also supports a live demo pipeline.

## 1. Setup

Create and activate an environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you use Conda or Mamba, create your environment first, then run the same install command.

## 2. Run Benchmark Experiments

### GardensPoint

Run a benchmark descriptor on the GardensPoint dataset:

```bash
python demo.py --descriptor EigenPlaces --dataset GardensPoint --save_results
```

Other useful descriptors include:
- `CosPlace`
- `SALAD`
- `SelaVPR++`
- `NetVLAD`
- `PatchNetVLAD`

### Campus strict dataset

This is the original 1-to-1 campus evaluation:

```bash
python test_campus_dataset.py --descriptor SALAD --save_results
```

### Campus place-level dataset

This is the rematched place-level campus evaluation:

```bash
python test_campus_dataset_place_level.py --descriptor SALAD --save_results
```

## 3. Optional CLAHE preprocessing

To test CLAHE on query images:

```bash
python demo.py --descriptor EigenPlaces --dataset GardensPoint --preprocess clahe_query --save_results
```

The same `--preprocess clahe_query` option also works with the campus scripts.

## 4. Run the Live Demo

The live pipeline builds a day map, then localizes live camera frames against that map.

Example config-driven run:

```bash
python live_vpr_test.py --config configs/live_vpr.yaml
```

For our setup, we stream camera frames from the TurboPi robot to a laptop, and run inference on the laptop.

## 5. Where Results Go

Saved outputs are written under:

```text
output_images/runs/
```

Typical contents include:
- precision-recall plots
- similarity matrices
- qualitative match visualizations
- text summaries of metrics
