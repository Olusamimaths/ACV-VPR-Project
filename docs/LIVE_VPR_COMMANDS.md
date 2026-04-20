# Live VPR Command Reference

This is the quick command cookbook for the live VPR pipeline.

It includes:

- bash-launcher commands
- equivalent raw Python commands

Use this when you want copy-paste commands more than architectural explanation.

For how the system works, see [LIVE_VPR_PIPELINE.md](./LIVE_VPR_PIPELINE.md).

## 1. Command Styles

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh <command> [extra args...]
```

### Python CLI

```bash
python live_vpr_test.py --mode <mode> [args...]
```

## 2. Check A Source

Use this before recording or live localization.

### Bash

```bash
bash scripts/live_vpr_cli.sh check-source --source 0
```

Checks whether camera `0` opens and returns a frame.

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

## 3. Discover Cameras Systematically

Use this when you do not know which numeric source is your phone or webcam.

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

## 4. Save A Source Alias

Use this when you want a stable name instead of remembering a camera index or long stream URL.

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

## 5. Build A Map From An Existing Image Folder

Use this when you already have curated reference images.

### Bash

```bash
bash scripts/live_vpr_cli.sh build-map --data_dir custom_dataset/day_images --descriptor CosPlace --map_path artifacts/live_maps/campus_day_cosplace.npz
```

### Python

```bash
python live_vpr_test.py \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_cosplace.npz
```

## 6. Record A Traversal And Build A Map

Use this when you want to walk the environment first and build the map afterward.

Recorder controls:

- `r`: start or pause recording
- `q`: stop recording and continue to map building

### Bash

```bash
bash scripts/live_vpr_cli.sh record-map --source 0 --mirror
```

```bash
bash scripts/live_vpr_cli.sh record-map --source phone --sample_fps 1.5
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
  --descriptor CosPlace \
  --map_path artifacts/live_maps/turbopi_live_map.npz
```

## 7. Build A Map From An Existing Video

Use this when you already recorded a traversal and want to rebuild the map later.

### Bash

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map
```

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map --sample_fps 2.0 --map_path artifacts/live_maps/campus_day_dense.npz
```

### Python

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --video recordings/campus_walk.mp4 \
  --use_video_for_live_build \
  --capture_dir artifacts/reference_captures/campus_day_from_video \
  --sample_fps 1.0 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_from_video.npz
```

## 8. Run Live Localization

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
bash scripts/live_vpr_cli.sh live --source phone --map_path artifacts/live_maps/campus_day_live.npz
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

## 9. Run Localization On A Recorded Query Video

Use this when you want repeatable testing.

### Bash

```bash
VPR_VIDEO_PATH=recordings/query_walk.mp4 bash scripts/live_vpr_cli.sh video
```

### Python

```bash
python live_vpr_test.py \
  --mode video \
  --video recordings/query_walk.mp4 \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --process_fps 2.0
```

## 10. TurboPi Workflow

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

## 11. Common Useful Flags

- `--descriptor`: choose descriptor backend
- `--map_path`: input or output reference map
- `--source`: camera index, alias, or stream URL
- `--process_fps`: localization cadence during `live` or `video`
- `--sample_fps`: sampling density when building a map from video
- `--threshold`: match/unknown decision threshold
- `--mirror`: mirror displayed live frames
- `--start_recording`: start traversal recording immediately
- `--start_inference`: start live inference immediately
- `--no_save_inference_images`: disable saved inference reports
