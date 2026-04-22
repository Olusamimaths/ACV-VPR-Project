# Live VPR Command Reference

This is the copy-paste command guide for the live VPR pipeline.

It covers:

- raw Python execution via `live_vpr_test.py`
- bash launcher execution via `scripts/live_vpr_cli.sh`
- YAML config support
- map building, live inference, source aliases, and search backends

For architecture and code flow, see [LIVE_VPR_PIPELINE.md](./LIVE_VPR_PIPELINE.md).

## 1. Command Styles

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh <command> [extra args...]
```

### Python CLI

```bash
python live_vpr_test.py --mode <mode> [args...]
```

## 2. Config Support

The live CLI now supports YAML configuration.

- If [configs/live_vpr.yaml](../../configs/live_vpr.yaml) exists, it is loaded automatically.
- Any explicit CLI flag overrides the YAML value.
- You can also pass a different config file with `--config`.

### Bash

```bash
bash scripts/live_vpr_cli.sh live --config configs/live_vpr.yaml
```

Runs `live` using the YAML defaults, then applies any extra CLI overrides.

```bash
bash scripts/live_vpr_cli.sh live --config configs/live_vpr.yaml --source phone --threshold 0.55
```

Uses the YAML config, but overrides `source` and `threshold` for this run.

### Python

```bash
python live_vpr_test.py --config configs/live_vpr.yaml
```

Works if the YAML file already defines `mode`.

```bash
python live_vpr_test.py --config configs/live_vpr.yaml --mode live --source phone --threshold 0.55
```

Uses the YAML config, then overrides selected values on the command line.

### Config Sections In `configs/live_vpr.yaml`

- `general`: mode, descriptor, map path, data dir
- `source`: webcam index, stream URL, frame size, mirror
- `map_build`: capture directory, recording path, sample rate, saved-image downsampling
- `inference`: threshold, top-k, processing cadence, output folders
- `search`: exact / HNSW / FAISS backend settings

## 3. Check A Source

Use this before map building or live localization.

### Bash

```bash
bash scripts/live_vpr_cli.sh check-source --source 0
```

Checks whether camera `0` opens and returns frames.

```bash
bash scripts/live_vpr_cli.sh check-source --source "http://192.168.1.20:4747/video"
```

Checks whether a stream URL is readable.

### Python

```bash
python live_vpr_test.py --mode check_source --source 0
```

```bash
python live_vpr_test.py --mode check_source --source "http://192.168.1.20:4747/video"
```

## 4. Discover Cameras Systematically

Use this when you do not know which numeric source corresponds to your laptop webcam, phone webcam, or capture card.

### Bash

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10
```

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
```

The second version also saves one preview image per detected source.

### Python

```bash
python live_vpr_test.py --mode list_sources --source_scan_max 10
```

```bash
python live_vpr_test.py --mode list_sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
```

## 5. Save, List, And Delete Source Aliases

Use aliases when you want a stable source name like `phone` or `turbopi` instead of remembering an index or long URL.

### Bash

```bash
bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source 3
```

```bash
bash scripts/live_vpr_cli.sh save-source-alias --alias turbopi --source "http://<robot-ip>:8080/?action=stream"
```

```bash
bash scripts/live_vpr_cli.sh list-source-aliases
```

```bash
bash scripts/live_vpr_cli.sh delete-source-alias --alias phone
```

### Python

```bash
python live_vpr_test.py --mode save_source_alias --alias phone --source 3
```

```bash
python live_vpr_test.py --mode save_source_alias --alias turbopi --source "http://<robot-ip>:8080/?action=stream"
```

```bash
python live_vpr_test.py --mode list_source_aliases
```

```bash
python live_vpr_test.py --mode delete_source_alias --alias phone
```

## 6. Build A Map From An Existing Image Folder

Use this when you already have a curated reference-image directory.

### Bash

```bash
bash scripts/live_vpr_cli.sh build-map --data_dir custom_dataset/day_images --descriptor CosPlace --map_path artifacts/live_maps/campus_day_cosplace.npz
```

```bash
bash scripts/live_vpr_cli.sh build-map --config configs/live_vpr.yaml --data_dir custom_dataset/day_images --map_path artifacts/live_maps/campus_day_from_yaml.npz
```

### Python

```bash
python live_vpr_test.py \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_cosplace.npz
```

```bash
python live_vpr_test.py \
  --config configs/live_vpr.yaml \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --map_path artifacts/live_maps/campus_day_from_yaml.npz
```

## 7. Record A Traversal And Build A Live Map

Use this when you want to walk the environment first, then build the map from the recorded traversal.

Recorder controls:

- `r`: start or pause recording
- `q`: stop recording and continue to map building

### Bash

```bash
bash scripts/live_vpr_cli.sh record-map --source 0 --mirror
```

```bash
bash scripts/live_vpr_cli.sh record-map --source phone --sample_fps 1.5 --capture_save_scale 0.5
```

```bash
bash scripts/live_vpr_cli.sh record-map --source turbopi --map_path artifacts/live_maps/turbopi_live_map.npz
```

### Python

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --source 0 \
  --recording_path artifacts/reference_videos/campus_day_walk.mp4 \
  --capture_dir artifacts/reference_captures/campus_day_walk \
  --sample_fps 1.0 \
  --capture_save_scale 0.5 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_live.npz
```

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --source turbopi \
  --recording_path artifacts/reference_videos/turbopi_walk.mp4 \
  --capture_dir artifacts/reference_captures/turbopi_walk \
  --sample_fps 1.0 \
  --capture_save_scale 0.5 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/turbopi_live_map.npz
```

## 8. Build A Map From An Existing Traversal Video

Use this when the traversal is already recorded and you want to rebuild the map later with different sampling settings.

### Bash

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map
```

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map --sample_fps 2.0 --capture_save_scale 0.4 --map_path artifacts/live_maps/campus_day_dense.npz
```

### Python

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --video recordings/campus_walk.mp4 \
  --use_video_for_live_build \
  --capture_dir artifacts/reference_captures/campus_day_from_video \
  --sample_fps 1.0 \
  --capture_save_scale 0.5 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_from_video.npz
```

## 9. Run Live Localization

Viewer controls:

- `i`: start or pause inference
- `q`: quit
- `t`: toggle top-k panel
- `s`: save current frame
- `+` or `-`: adjust threshold

### Bash

```bash
bash scripts/live_vpr_cli.sh live --source 0 --map_path artifacts/live_maps/campus_day_live.npz --mirror
```

```bash
bash scripts/live_vpr_cli.sh live --source phone --map_path artifacts/live_maps/campus_day_live.npz --process_fps 2.0
```

```bash
bash scripts/live_vpr_cli.sh live --source turbopi --map_path artifacts/live_maps/turbopi_live_map.npz
```

### Python

```bash
python live_vpr_test.py \
  --mode live \
  --source 0 \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --process_fps 2.0 \
  --threshold 0.50 \
  --mirror
```

```bash
python live_vpr_test.py \
  --mode live \
  --source turbopi \
  --map_path artifacts/live_maps/turbopi_live_map.npz \
  --process_fps 2.0 \
  --threshold 0.50
```

## 10. Run Localization On A Recorded Query Video

Use this when you want repeatable, non-live testing.

### Bash

```bash
VPR_VIDEO_PATH=recordings/query_walk.mp4 bash scripts/live_vpr_cli.sh video
```

```bash
VPR_VIDEO_PATH=recordings/query_walk.mp4 bash scripts/live_vpr_cli.sh video --map_path artifacts/live_maps/campus_day_live.npz --process_fps 1.0
```

### Python

```bash
python live_vpr_test.py \
  --mode video \
  --video recordings/query_walk.mp4 \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --process_fps 2.0
```

## 11. Search Backend Options

The live system now supports multiple search backends for map lookup:

- `exact`
- `hnsw`
- `faiss_ivf_flat`
- `faiss_ivf_pq`

### Exact Search

Good default for small to medium maps. No extra dependency required.

#### Bash

```bash
bash scripts/live_vpr_cli.sh live --source phone --search_backend exact
```

#### Python

```bash
python live_vpr_test.py --mode live --source phone --map_path artifacts/live_maps/campus_day_live.npz --search_backend exact
```

### HNSW Search

Good first ANN option for larger maps. Requires `hnswlib`.

#### Bash

```bash
bash scripts/live_vpr_cli.sh live --source phone --search_backend hnsw --search_candidate_k 64 --search_hnsw_ef_search 96
```

#### Python

```bash
python live_vpr_test.py \
  --mode live \
  --source phone \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --search_backend hnsw \
  --search_candidate_k 64 \
  --search_hnsw_m 16 \
  --search_hnsw_ef_construction 200 \
  --search_hnsw_ef_search 96
```

### FAISS IVF Flat

Good for large static maps. Requires `faiss-cpu`.

#### Bash

```bash
bash scripts/live_vpr_cli.sh live --source phone --search_backend faiss_ivf_flat --search_ivf_nlist 100 --search_ivf_nprobe 10
```

#### Python

```bash
python live_vpr_test.py \
  --mode live \
  --source phone \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --search_backend faiss_ivf_flat \
  --search_ivf_nlist 100 \
  --search_ivf_nprobe 10
```

### FAISS IVF PQ

Useful when memory efficiency matters. Requires `faiss-cpu`.

#### Bash

```bash
bash scripts/live_vpr_cli.sh live --source phone --search_backend faiss_ivf_pq --search_ivf_nlist 100 --search_ivf_nprobe 10 --search_pq_m 16 --search_pq_bits 8
```

#### Python

```bash
python live_vpr_test.py \
  --mode live \
  --source phone \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --search_backend faiss_ivf_pq \
  --search_ivf_nlist 100 \
  --search_ivf_nprobe 10 \
  --search_pq_m 16 \
  --search_pq_bits 8 \
  --search_train_limit 10000
```

### Search Reranking

Use reranking to retrieve approximate candidates quickly, then rescore them exactly.

#### Bash

```bash
bash scripts/live_vpr_cli.sh live --source phone --search_backend hnsw --search_candidate_k 64 --search_rerank
```

```bash
bash scripts/live_vpr_cli.sh live --source phone --search_backend hnsw --search_candidate_k 64 --no_search_rerank
```

#### Python

```bash
python live_vpr_test.py --mode live --source phone --map_path artifacts/live_maps/campus_day_live.npz --search_backend hnsw --search_candidate_k 64 --search_rerank
```

```bash
python live_vpr_test.py --mode live --source phone --map_path artifacts/live_maps/campus_day_live.npz --search_backend hnsw --search_candidate_k 64 --no_search_rerank
```

## 12. TurboPi Workflow

Recommended order:

1. start the TurboPi stream on the robot
2. check the stream from your laptop
3. save a `turbopi` alias
4. record a reference traversal
5. run live localization

### Bash

```bash
bash scripts/live_vpr_cli.sh check-source --source "http://<robot-ip>:8080/?action=stream"
bash scripts/live_vpr_cli.sh save-source-alias --alias turbopi --source "http://<robot-ip>:8080/?action=stream"
bash scripts/live_vpr_cli.sh record-map --source turbopi --map_path artifacts/live_maps/turbopi_live_map.npz
bash scripts/live_vpr_cli.sh live --source turbopi --map_path artifacts/live_maps/turbopi_live_map.npz
```

### Python

```bash
python live_vpr_test.py --mode check_source --source "http://<robot-ip>:8080/?action=stream"
python live_vpr_test.py --mode save_source_alias --alias turbopi --source "http://<robot-ip>:8080/?action=stream"
python live_vpr_test.py --mode build_live_map --source turbopi --map_path artifacts/live_maps/turbopi_live_map.npz --recording_path artifacts/reference_videos/turbopi_walk.mp4 --capture_dir artifacts/reference_captures/turbopi_walk --sample_fps 1.0
python live_vpr_test.py --mode live --source turbopi --map_path artifacts/live_maps/turbopi_live_map.npz
```

## 13. Common Useful Flags

- `--config`: load a YAML config file before applying CLI overrides
- `--descriptor`: choose descriptor backend
- `--map_path`: input or output reference map path
- `--data_dir`: reference-image directory for folder-based map building
- `--source`: camera index, alias, or stream URL
- `--video`: recorded video used in `video` mode or `build_live_map --use_video_for_live_build`
- `--process_fps`: localization cadence during `live` or `video`
- `--sample_fps`: frame sampling density when building a map from video
- `--capture_save_scale`: downsample factor for saved sampled reference images
- `--threshold`: match/unknown decision threshold
- `--top_k`: number of reference candidates shown and returned
- `--mirror`: mirror displayed live frames
- `--start_recording`: start traversal recording immediately
- `--start_inference`: start live inference immediately
- `--hide_top_k`: hide the top-k preview panel at startup
- `--output_video`: save the annotated live or recorded session
- `--inference_stats_dir`: folder for annotated per-inference report images
- `--no_save_inference_images`: disable saved inference reports
- `--snapshot_dir`: folder for frames saved with the `s` key
- `--search_backend`: choose `exact`, `hnsw`, `faiss_ivf_flat`, or `faiss_ivf_pq`
- `--search_metric`: choose `cosine` or `inner_product`
- `--search_candidate_k`: ANN candidate pool size before optional reranking
- `--search_rerank` / `--no_search_rerank`: enable or disable exact reranking
- `--search_hnsw_m`, `--search_hnsw_ef_construction`, `--search_hnsw_ef_search`: HNSW tuning knobs
- `--search_ivf_nlist`, `--search_ivf_nprobe`: FAISS IVF tuning knobs
- `--search_pq_m`, `--search_pq_bits`, `--search_train_limit`: FAISS IVFPQ tuning knobs

## 14. Notes On Dependencies

- YAML config support requires `PyYAML`
- `search_backend=exact` uses only the base repo dependencies
- `search_backend=hnsw` requires `hnswlib`
- `search_backend=faiss_ivf_flat` and `search_backend=faiss_ivf_pq` require `faiss-cpu`

If an ANN backend is selected without its dependency installed, the CLI will raise a clear import error when that backend is initialized.
