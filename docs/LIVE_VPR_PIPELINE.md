# Live VPR Pipeline

This document explains the modular live Visual Place Recognition pipeline in this repository. The pipeline now supports:

- building a reference map from an existing image folder
- building a reference map by recording a traversal video and sampling frames after recording stops
- live localization from a webcam, phone webcam, stream URL, or recorded video
- configurable inference cadence so live preview and model processing are decoupled
- systematic camera-source discovery and friendly source aliases like `phone`

For the bash launcher that wraps the common commands, see `docs/LIVE_VPR_SCRIPT.md`.
For a ready-to-run command cookbook with both bash and Python examples, see `docs/LIVE_VPR_COMMANDS.md`.

## Architecture

The code is organized into two phases.

```text
OFFLINE PHASE (Map Building)

Option A
reference image folder
-> load images
-> resize/preprocess
-> extract descriptors
-> normalize descriptors
-> save map (.npz)

Option B
live camera / phone webcam / stream
-> record traversal video
-> stop recording
-> sample frames from video at sample_fps
-> resize/preprocess
-> extract descriptors
-> normalize descriptors
-> save map (.npz)

ONLINE PHASE (Live Localization)

webcam / phone webcam / stream / video
-> display every frame
-> run localization at process_fps
-> reuse latest prediction between inference steps
-> show top-k results and recognition decision
```

## Module Layout

The modular implementation lives in `live_vpr/`.

- `live_vpr/database.py`
  Reference-map structure, serialization, and descriptor normalization.
- `live_vpr/extractors.py`
  Descriptor factory and conversion into global descriptor matrices.
- `live_vpr/offline.py`
  Folder-based and path-list-based map building.
- `live_vpr/capture.py`
  Traversal video recording and post-record frame sampling.
- `live_vpr/online.py`
  Query localization against a saved map.
- `live_vpr/sources.py`
  OpenCV capture-source abstraction.
- `live_vpr/ui.py`
  Live overlay and top-k visualization.
- `live_vpr_test.py`
  Python CLI entrypoint that ties everything together.
- `scripts/live_vpr_cli.sh`
  Bash launcher for common workflows.

## Offline Phase

### Option A: Build From An Existing Reference Folder

```bash
python live_vpr_test.py \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --map_path artifacts/live_maps/campus_day_cosplace.npz \
  --descriptor CosPlace
```

Use this when you already have a curated reference-image set.

### Option B: Record A Traversal Video And Build The Map After Stop

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source 0 \
  --recording_path artifacts/reference_videos/campus_day_walk.mp4 \
  --capture_dir artifacts/reference_captures/campus_day_walk \
  --sample_fps 1.0 \
  --descriptor CosPlace
```

This mode works well when the existing reference images are not enough and you want to walk through the environment once and then build the map from that traversal.

How it works:

1. The pipeline records a traversal video from the selected source.
2. The recorder opens in `PAUSED` mode by default so nothing is recorded until you start it.
3. Press `r` to start recording, and press `r` again whenever you want to pause.
4. You stop recording and continue to map building by pressing `q`.
5. The pipeline samples frames from the recorded video at `--sample_fps`.
6. Those sampled frames become the reference set used for descriptor extraction.
7. The reference map is saved as `.npz`.

Recorder controls:

- `r`: start recording if paused, or pause if already recording
- `q`: stop recording and continue to map building

If you do want the recorder to begin immediately, pass `--start_recording`.

Why this design is useful:

- the raw traversal video is preserved for reproducibility
- you can rebuild the map later with a different `sample_fps`
- you do not need to manually capture images while walking
- map creation becomes more consistent across runs

### Build A Map From An Existing Traversal Video

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --video recordings/campus_walk.mp4 \
  --use_video_for_live_build \
  --map_path artifacts/live_maps/campus_day_from_video.npz \
  --capture_dir artifacts/reference_captures/campus_day_from_video \
  --sample_fps 1.0 \
  --descriptor CosPlace
```

This skips the recording step and directly samples an already-recorded video.

## Online Phase

### Live Webcam Or Phone Webcam

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source 0 \
  --threshold 0.50 \
  --process_fps 2.0 \
  --mirror
```

The live viewer opens with inference paused by default.

- press `i` to start inference
- press `i` again to pause inference
- pass `--start_inference` if you want inference to begin immediately

If your phone appears as a virtual webcam, use another device index:

```bash
python live_vpr_test.py --mode live --map_path artifacts/live_maps/campus_day_live.npz --source 1
```

If your phone exposes a stream URL:

```bash
python live_vpr_test.py --mode live --map_path artifacts/live_maps/campus_day_live.npz --source http://192.168.1.20:4747/video
```

### Recorded Video Playback

```bash
python live_vpr_test.py \
  --mode video \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --video recordings/query_walk.mp4 \
  --process_fps 2.0
```

## Inference Cadence

The live pipeline no longer needs to run inference on every displayed frame.

- every camera frame is still shown in the preview
- the live viewer opens paused unless `--start_inference` is used
- localization only runs at `--process_fps`
- the most recent localization result is reused between inference steps
- the overlay shows the age of the current result in milliseconds

This is usually better for laptop and phone-webcam testing because:

- compute cost is controlled
- UI responsiveness improves
- latency becomes more predictable
- weak hardware can still run a usable demo

Recommended defaults:

- `--sample_fps 1.0` for traversal-to-map building
- `--process_fps 2.0` for live localization

You can set `--process_fps 0` or a negative value to process every frame.

## Important CLI Flags

### Source Discovery And Aliases

- `--mode list_sources`
  Probe numeric camera indexes systematically instead of guessing `0`, `1`, `2`.
- `--source_scan_max`
  Highest index to scan when probing available cameras.
- `--source_snapshot_dir`
  Optional directory for saving one preview image per detected source.
- `--mode save_source_alias --alias <name> --source <value>`
  Save a friendly alias such as `phone -> 3` or `phone -> http://...`.

Once an alias is saved, you can use it anywhere a source is accepted:

```bash
python live_vpr_test.py --mode live --source phone --map_path artifacts/live_maps/campus_day_live.npz
```

### Map Building

- `--descriptor`
  Descriptor family used for map creation and later localization.
- `--resize_width`, `--resize_height`
  Canonical model input size stored in map metadata.
- `--recording_path`
  Output path for the traversal video recorded during live map building.
- `--capture_dir`
  Directory for sampled reference frames extracted from the traversal video.
- `--sample_fps`
  Number of sampled reference frames per second from the recorded video.
- `--min_captures`
  Minimum number of sampled frames required before the map is accepted.
- `--max_captures`
  Optional upper bound on sampled reference frames.

### Live Localization

- `--threshold`
  Similarity threshold for accepting a match.
- `--top_k`
  Number of best reference matches shown in the overlay.
- `--process_fps`
  Localization cadence during live or video runtime.
- `--mirror`
  Mirror the live display for a more natural webcam experience.
- `--output_video`
  Save the annotated live session.

## Data Products

The pipeline may produce three related artifacts:

1. traversal video
2. sampled reference frames
3. final reference map

The saved reference map contains:

- normalized descriptors
- absolute reference image paths
- metadata including descriptor, target size, and live-build provenance such as source video path and sampling rate

## How To Modify The Pipeline

### Change The Descriptor

Edit:

- `live_vpr/extractors.py`

Update `SUPPORTED_DESCRIPTORS` and `create_feature_extractor()`. The live pipeline expects a 2D global descriptor matrix with shape `[N, D]`.

### Change Video Recording Behavior

Edit:

- `live_vpr/capture.py`

This is where you can:

- change recorder controls
- adjust codec or container
- add maximum recording duration
- add timestamp overlays
- save per-frame timestamps or motion metadata

### Change Video Sampling Policy

Edit:

- `live_vpr/capture.py`

The current approach samples by time using `sample_fps`. This is the right place to add:

- motion-based frame filtering
- blur filtering
- keyframe selection
- scene-change filtering

### Change Map Storage

Edit:

- `live_vpr/database.py`
- `live_vpr/offline.py`

This is the right place to add building labels, GPS tags, route IDs, timestamps, or user annotations.

### Change Online Recognition Logic

Edit:

- `live_vpr/online.py`
- `live_vpr_test.py`

This is where to add:

- top-1 vs top-2 margin checks
- temporal smoothing
- history-aware voting
- confidence stabilization

### Change The UI

Edit:

- `live_vpr/ui.py`

This is where to add:

- building names
- confidence bars
- route progress hints
- result-age visualization

## Suggested Workflow For Your Campus

1. Start with `CosPlace`.
2. Run `list_sources` and save an alias such as `phone` once you identify the correct camera or stream.
3. Record a traversal with `build_live_map` instead of relying only on the existing day-image folder.
4. Sample at `1.0 fps` first.
5. Run live localization at `2.0 fps`.
6. Increase `sample_fps` if you need denser reference coverage.
7. Increase `process_fps` only if the hardware can keep up.
8. Tune `--threshold` after observing false accepts and misses.

## Known Runtime Requirements

- OpenCV is required for recording, video playback, webcam access, and UI.
- The descriptor dependencies for the chosen model must be installed.
- The runtime descriptor must match the descriptor used to build the map.
- If you change preprocessing or resize settings, rebuild the map.
