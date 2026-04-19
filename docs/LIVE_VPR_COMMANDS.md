# Live VPR Command Reference

This document is a practical command reference for the live VPR pipeline. It includes both:

- commands using the bash launcher script
- equivalent commands using raw Python

Use this as a quick copy-paste guide when building maps, checking camera sources, recording reference traversals, or running live localization.

## Command Styles

There are two supported ways to run the project.

### Option A: Bash Launcher

Use:

```bash
bash scripts/live_vpr_cli.sh <command> [extra args...]
```

This is the shorter and easier interface for day-to-day use.

### Option B: Python CLI

Use:

```bash
python live_vpr_test.py --mode <mode> [args...]
```

This is the direct interface and gives full control over all options.

## 1. Check A Camera Or Stream Source

Use this before live testing or live map recording to confirm that the selected webcam, phone webcam, or stream is accessible.

If you do not know which numeric camera index is your phone, use the source-discovery commands in the next section instead of guessing.

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh check-source --source 0
```

Brief explanation:
Checks whether camera source `0` can be opened and whether a frame can be read.

```bash
bash scripts/live_vpr_cli.sh check-source --source 1
```

Brief explanation:
Checks a second camera source, which is often where a phone-as-webcam device appears.

```bash
bash scripts/live_vpr_cli.sh check-source --source http://192.168.1.20:4747/video
```

Brief explanation:
Checks a phone IP camera stream exposed as a URL.

### Python CLI

```bash
python live_vpr_test.py --mode check_source --source 0
```

Brief explanation:
Opens camera `0`, reads one frame, and prints basic source information.

```bash
python live_vpr_test.py --mode check_source --source 1
```

Brief explanation:
Checks another camera index, useful for external webcams or phone webcams.

```bash
python live_vpr_test.py --mode check_source --source http://192.168.1.20:4747/video
```

Brief explanation:
Checks whether an HTTP stream from a phone camera app is reachable and readable.

## 1b. Discover Available Camera Indexes Systematically

Use this when your phone is not obviously `0`, `1`, or `2`.

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10
```

Brief explanation:
Scans camera indexes `0..10` and prints which ones can actually be opened.

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
```

Brief explanation:
Scans the same range and saves one preview image per detected source so you can identify which camera is your phone.

### Python CLI

```bash
python live_vpr_test.py --mode list_sources --source_scan_max 10
```

Brief explanation:
Probes available numeric sources and prints frame size and backend information for each detected camera.

```bash
python live_vpr_test.py --mode list_sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
```

Brief explanation:
Scans available camera indexes and writes preview snapshots that help you tell which source is the phone webcam.

## 1c. Save A Friendly Name For A Source

Use this after identifying the right source so you can stop remembering numeric indexes or long URLs.

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source 3
```

Brief explanation:
Saves the alias `phone` so you can use `--source phone` later.

```bash
bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source http://192.168.1.20:4747/video
```

Brief explanation:
Saves the alias `phone` for a stream URL instead of a numeric camera index.

```bash
bash scripts/live_vpr_cli.sh list-source-aliases
```

Brief explanation:
Lists the aliases currently saved in the local alias file.

### Python CLI

```bash
python live_vpr_test.py --mode save_source_alias --alias phone --source 3
```

Brief explanation:
Stores `phone -> 3` in the source alias file.

```bash
python live_vpr_test.py --mode save_source_alias --alias phone --source http://192.168.1.20:4747/video
```

Brief explanation:
Stores `phone -> stream URL` in the source alias file.

```bash
python live_vpr_test.py --mode list_source_aliases
```

Brief explanation:
Shows all saved source aliases.

```bash
python live_vpr_test.py --mode delete_source_alias --alias phone
```

Brief explanation:
Deletes the alias `phone` if you want to replace or remove it.

## 2. Build A Map From An Existing Reference Image Folder

Use this when you already have a folder of day images or curated reference images.

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh build-map
```

Brief explanation:
Builds a reference map using the launcher defaults such as `VPR_REFERENCE_DIR`, `VPR_DESCRIPTOR`, and `VPR_MAP_PATH`.

```bash
bash scripts/live_vpr_cli.sh build-map --data_dir custom_dataset/day_images --descriptor CosPlace --map_path artifacts/live_maps/campus_day_cosplace.npz
```

Brief explanation:
Builds a map from the specified folder using the chosen descriptor and output path.

### Python CLI

```bash
python live_vpr_test.py \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_cosplace.npz
```

Brief explanation:
Loads all reference images from the folder, extracts descriptors, normalizes them, and saves the resulting reference map.

```bash
python live_vpr_test.py \
  --mode build_map \
  --data_dir custom_dataset/day_images \
  --descriptor EigenPlaces \
  --map_path artifacts/live_maps/campus_day_eigenplaces.npz
```

Brief explanation:
Same workflow as above, but uses a different descriptor backend.

## 3. Record A Traversal Video And Build A Map After Stopping

Use this when you want to walk through the environment with a webcam or phone webcam and build the database from that traversal.

The recorder opens paused by default.

- press `r` to start recording
- press `r` again to pause
- press `q` to stop and continue to map building

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh record-map --source 0 --mirror
```

Brief explanation:
Opens the traversal recorder using camera `0`, lets you record a route, then samples frames from the saved video and builds the map.

```bash
bash scripts/live_vpr_cli.sh record-map --source phone --sample_fps 1.5
```

Brief explanation:
Uses the saved alias `phone` and samples the traversal video more densely when building the map.

```bash
bash scripts/live_vpr_cli.sh record-map --source http://192.168.1.20:4747/video --descriptor CosPlace
```

Brief explanation:
Records a traversal from a phone stream URL and builds the map using `CosPlace`.

### Python CLI

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

Brief explanation:
Records a traversal video from camera `0`, samples one frame per second after recording stops, and builds the map from those sampled frames.

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --source phone \
  --recording_path artifacts/reference_videos/campus_day_phone.mp4 \
  --capture_dir artifacts/reference_captures/campus_day_phone \
  --sample_fps 2.0 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_phone.npz
```

Brief explanation:
Uses the saved alias `phone` and samples at `2.0 fps` for denser map coverage.

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --source 0 \
  --start_recording \
  --recording_path artifacts/reference_videos/campus_day_walk.mp4 \
  --capture_dir artifacts/reference_captures/campus_day_walk \
  --sample_fps 1.0 \
  --descriptor CosPlace \
  --map_path artifacts/live_maps/campus_day_live.npz
```

Brief explanation:
Same as above, but begins recording immediately instead of opening in paused mode.

## 4. Build A Map From An Existing Traversal Video

Use this when you already recorded a traversal and want to rebuild the map later with a different sampling rate or descriptor.

### Bash Launcher

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map
```

Brief explanation:
Builds a map from an existing video using the launcher defaults for descriptor, map path, and sampling rate.

```bash
VPR_VIDEO_PATH=recordings/campus_walk.mp4 bash scripts/live_vpr_cli.sh video-map --sample_fps 2.0 --map_path artifacts/live_maps/campus_day_dense.npz
```

Brief explanation:
Rebuilds the map from the same video, but samples more densely and writes to a different output file.

### Python CLI

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

Brief explanation:
Skips live recording and directly samples an existing video to build the reference map.

```bash
python live_vpr_test.py \
  --mode build_live_map \
  --video recordings/campus_walk.mp4 \
  --use_video_for_live_build \
  --capture_dir artifacts/reference_captures/campus_day_dense \
  --sample_fps 2.0 \
  --descriptor EigenPlaces \
  --map_path artifacts/live_maps/campus_day_eigen_dense.npz
```

Brief explanation:
Builds a map from an existing traversal video using a denser frame-sampling rate and a different descriptor.

## 5. Run Live Localization From A Webcam Or Phone Webcam

Use this after you already have a reference map.

The live viewer opens with inference paused by default.

- press `i` to start inference
- press `i` again to pause inference
- use `--start_inference` if you want the old immediate-start behavior
- the controls are shown directly on the live screen
- an annotated image is saved after each inference by default

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh live --source 0 --mirror
```

Brief explanation:
Runs live localization using camera `0`, with the display mirrored for easier interaction.

```bash
bash scripts/live_vpr_cli.sh live --source phone --process_fps 2.0 --threshold 0.50
```

Brief explanation:
Runs live localization from the saved alias `phone` at a throttled inference rate of `2 fps`.

```bash
bash scripts/live_vpr_cli.sh live --source http://192.168.1.20:4747/video --process_fps 1.5
```

Brief explanation:
Runs live localization from a phone stream URL, processing frames at `1.5 fps`.

### Python CLI

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source 0 \
  --threshold 0.50 \
  --process_fps 2.0 \
  --mirror
```

Brief explanation:
Opens the live viewer on camera `0`, starts with inference paused, and runs localization at `2 fps` once you press `i`.

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source phone \
  --threshold 0.45 \
  --process_fps 3.0
```

Brief explanation:
Opens the live viewer for the saved alias `phone`, with slightly denser inference once you start it.

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source phone \
  --threshold 0.50 \
  --process_fps 2.0 \
  --start_inference
```

Brief explanation:
Runs live localization from the saved alias `phone` and starts inference immediately instead of opening paused.

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source phone \
  --process_fps 2.0 \
  --inference_stats_dir artifacts/inference_stats/phone_live
```

Brief explanation:
Runs live localization and saves an annotated image after every inference into the chosen directory.

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source http://192.168.1.20:4747/video \
  --threshold 0.50 \
  --process_fps 1.5
```

Brief explanation:
Runs live localization from a phone stream URL using a lower inference cadence to reduce compute load.

## 6. Run Localization On A Recorded Query Video

Use this when you want repeatable testing on a saved traversal rather than a live camera feed.

### Bash Launcher

```bash
VPR_VIDEO_PATH=recordings/query_walk.mp4 bash scripts/live_vpr_cli.sh video
```

Brief explanation:
Runs localization on a recorded query video using the current default reference map and inference settings.

```bash
VPR_VIDEO_PATH=recordings/query_walk.mp4 bash scripts/live_vpr_cli.sh video --process_fps 1.0 --threshold 0.45
```

Brief explanation:
Processes the recorded video more slowly and with a lower match threshold.

### Python CLI

```bash
python live_vpr_test.py \
  --mode video \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --video recordings/query_walk.mp4 \
  --threshold 0.50 \
  --process_fps 2.0
```

Brief explanation:
Runs localization on a recorded video while throttling inference to `2 fps`.

```bash
python live_vpr_test.py \
  --mode video \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --video recordings/query_walk.mp4 \
  --threshold 0.45 \
  --process_fps 1.0 \
  --output_video artifacts/live_runs/query_walk_annotated.mp4
```

Brief explanation:
Runs localization on a recorded query video and saves the annotated output video to disk.

## 7. Save Annotated Output Or Snapshots

Use these options when you want to keep results from a run.

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh live --source 0 --output_video artifacts/live_runs/session.mp4
```

Brief explanation:
Runs live localization and saves the annotated preview stream to a video file.

### Python CLI

```bash
python live_vpr_test.py \
  --mode live \
  --map_path artifacts/live_maps/campus_day_live.npz \
  --source 0 \
  --output_video artifacts/live_runs/session.mp4 \
  --snapshot_dir artifacts/live_captures
```

Brief explanation:
Saves the annotated session video, and lets you save raw frames to `--snapshot_dir` using the `s` key during the run.

## 8. Useful Environment Variable Setup For The Bash Launcher

Use these if you want the bash launcher to remember your usual setup.

```bash
export VPR_DESCRIPTOR=CosPlace
export VPR_MAP_PATH=artifacts/live_maps/campus_day_live.npz
export VPR_SOURCE=1
export VPR_PROCESS_FPS=2.0
export VPR_SAMPLE_FPS=1.0
export VPR_RECORDING_PATH=artifacts/reference_videos/campus_day_phone.mp4
export VPR_CAPTURE_DIR=artifacts/reference_captures/campus_day_phone
```

Brief explanation:
Sets a default phone-webcam workflow so you can run the launcher with very short commands afterward.

Then run:

```bash
bash scripts/live_vpr_cli.sh record-map
bash scripts/live_vpr_cli.sh live
```

## Recommended Starting Workflow

If you are using your phone as a webcam, a good first sequence is:

### Bash Launcher

```bash
bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source 3
bash scripts/live_vpr_cli.sh record-map --source phone --mirror
bash scripts/live_vpr_cli.sh live --source phone --mirror
```

### Python CLI

```bash
python live_vpr_test.py --mode list_sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews
python live_vpr_test.py --mode save_source_alias --alias phone --source 3
python live_vpr_test.py --mode build_live_map --source phone --recording_path artifacts/reference_videos/campus_day_phone.mp4 --capture_dir artifacts/reference_captures/campus_day_phone --sample_fps 1.0 --descriptor CosPlace --map_path artifacts/live_maps/campus_day_phone.npz
python live_vpr_test.py --mode live --source phone --map_path artifacts/live_maps/campus_day_phone.npz --process_fps 2.0 --threshold 0.50 --mirror
```

Brief explanation:
This discovers available cameras, saves a friendly alias for the phone source, records a reference traversal, builds the map from that traversal, and then runs live localization using the alias.
