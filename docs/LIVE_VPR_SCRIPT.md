# Live VPR Launcher Script

This document explains the bash launcher at `scripts/live_vpr_cli.sh`.

The script is a lightweight wrapper around `live_vpr_test.py`. It gives you short commands for the main workflows without needing to type the full Python command every time.

For a broader command cookbook that includes both launcher commands and raw Python commands, see `docs/LIVE_VPR_COMMANDS.md`.

## Basic Usage

Run it with:

```bash
bash scripts/live_vpr_cli.sh <command> [extra args...]
```

Available commands:

- `build-map`
- `record-map`
- `video-map`
- `live`
- `video`
- `check-source`
- `list-sources`
- `save-source-alias`
- `list-source-aliases`
- `delete-source-alias`
- `help`

## Commands

### `build-map`

Build a map from an existing image folder.

```bash
bash scripts/live_vpr_cli.sh build-map
```

Equivalent Python flow:

```bash
python live_vpr_test.py --mode build_map ...
```

### `record-map`

Record a traversal video from a live source, sample the video, and build a reference map.

```bash
bash scripts/live_vpr_cli.sh record-map --source 0 --mirror
```

The recorder opens paused by default. Press `r` to start recording and `r` again to pause. If you want it to begin immediately, pass `--start_recording`.

Equivalent Python flow:

```bash
python live_vpr_test.py --mode build_live_map ...
```

### `video-map`

Build a map from an already-recorded traversal video.

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map
```

### `live`

Run live localization from a webcam, virtual webcam, or stream.

```bash
bash scripts/live_vpr_cli.sh live --source 0 --mirror
```

The live viewer opens with inference paused by default. Press `i` to start or pause inference. If you want it to start immediately, pass `--start_inference`.

### `video`

Run localization on a recorded query video.

```bash
VPR_VIDEO_PATH=recordings/query_walk.mp4 bash scripts/live_vpr_cli.sh video
```

### `check-source`

Check whether a webcam or stream is available.

```bash
bash scripts/live_vpr_cli.sh check-source --source 0
```

## Source Discovery And Naming

### `list-sources`

Probe a range of numeric camera indexes and report which ones open successfully.

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10
```

You can also save preview frames while scanning:

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
```

This helps you systematically identify which index corresponds to your phone.

### `save-source-alias`

Save a friendly alias such as `phone` or `laptop`.

```bash
bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source 3
```

You can also save an alias for a stream URL:

```bash
bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source http://192.168.1.20:4747/video
```

### `list-source-aliases`

List all saved aliases:

```bash
bash scripts/live_vpr_cli.sh list-source-aliases
```

### `delete-source-alias`

Remove an alias you no longer want:

```bash
bash scripts/live_vpr_cli.sh delete-source-alias --alias phone
```

Once an alias is saved, you can use it anywhere a source is accepted:

```bash
bash scripts/live_vpr_cli.sh record-map --source phone
bash scripts/live_vpr_cli.sh live --source phone
```

## Default Configuration Through Environment Variables

The script uses environment variables for defaults, so you can keep a preferred setup without rewriting commands.

Supported variables:

- `PYTHON_BIN`
  Python executable to use.
- `VPR_DESCRIPTOR`
  Descriptor name, for example `CosPlace`.
- `VPR_MAP_PATH`
  Reference-map path used for both building and localization.
- `VPR_REFERENCE_DIR`
  Folder path for `build-map`.
- `VPR_SOURCE`
  Webcam index or stream URL.
- `VPR_THRESHOLD`
  Live recognition threshold.
- `VPR_PROCESS_FPS`
  Inference cadence for `live` and `video`.
- `VPR_SAMPLE_FPS`
  Sampling rate for `record-map` and `video-map`.
- `VPR_RECORDING_PATH`
  Traversal-video path for `record-map`.
- `VPR_CAPTURE_DIR`
  Directory for sampled reference frames.
- `VPR_VIDEO_PATH`
  Input video path for `video` and `video-map`.

## Example Presets

### Laptop Webcam

```bash
export VPR_DESCRIPTOR=CosPlace
export VPR_MAP_PATH=artifacts/live_maps/campus_day_live.npz
export VPR_SOURCE=0
export VPR_PROCESS_FPS=2.0
```

Then run:

```bash
bash scripts/live_vpr_cli.sh record-map
bash scripts/live_vpr_cli.sh live --mirror
```

### Phone As Virtual Webcam

```bash
export VPR_SOURCE=1
bash scripts/live_vpr_cli.sh check-source
bash scripts/live_vpr_cli.sh live
```

### Build A Map From An Existing Traversal Video

```bash
export VPR_VIDEO_PATH=recordings/campus_walk.mp4
export VPR_MAP_PATH=artifacts/live_maps/campus_from_video.npz
export VPR_SAMPLE_FPS=1.0
bash scripts/live_vpr_cli.sh video-map
```

## Passing Extra Args

The script forwards any extra arguments directly to `live_vpr_test.py`.

Examples:

```bash
bash scripts/live_vpr_cli.sh record-map --sample_fps 2.0 --descriptor EigenPlaces
bash scripts/live_vpr_cli.sh live --threshold 0.45 --top_k 8
```

This means the script gives you convenient defaults, but you can still override settings per run.

## Recommended Workflow

1. Use `record-map` to build a richer reference map from a traversal video.
2. Use `live` for real-time testing.
3. Use `video` when you want repeatable evaluation on a saved query traversal.
4. Use `video-map` if you want to rebuild the map later with a different `sample_fps`.

## Where To Change The Script

If you want to change command presets or add more launcher commands, edit:

- `scripts/live_vpr_cli.sh`

If you want to change actual pipeline behavior, edit the Python modules instead:

- `live_vpr_test.py`
- `live_vpr/`
