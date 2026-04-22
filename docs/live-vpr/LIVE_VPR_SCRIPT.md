# Live VPR Launcher Script

This document explains the bash launcher at [scripts/live_vpr_cli.sh](../../scripts/live_vpr_cli.sh).

Use this guide if you want the shortest way to run the project day to day.

The launcher is a thin wrapper around [live_vpr_test.py](../../live_vpr_test.py). It does not implement VPR logic itself. It simply:

- chooses a high-level command
- fills in defaults from environment variables
- forwards extra arguments to the Python CLI

For a command cookbook with copy-paste examples, see [LIVE_VPR_COMMANDS.md](./LIVE_VPR_COMMANDS.md).

## 1. Basic Usage

Run:

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

## 2. What Each Command Does

### `build-map`

Build a reference map from an existing image folder.

Equivalent Python mode:

```bash
python live_vpr_test.py --mode build_map ...
```

### `record-map`

Record a traversal video from a live source, sample frames from that video, and build a map.

Equivalent Python mode:

```bash
python live_vpr_test.py --mode build_live_map ...
```

Important behavior:

- recorder opens paused by default
- `r` starts or pauses recording
- `q` stops recording and continues to map building

### `video-map`

Build a map from an already-recorded video.

This uses:

```bash
python live_vpr_test.py --mode build_live_map --use_video_for_live_build ...
```

### `live`

Run live localization from a webcam, phone webcam, or stream.

Equivalent Python mode:

```bash
python live_vpr_test.py --mode live ...
```

Important behavior:

- viewer opens with inference paused by default
- `i` starts or pauses inference
- `q` quits

### `video`

Run localization on a recorded query video instead of a live source.

### `check-source`

Open a camera or stream and confirm that one frame can be read.

### `list-sources`

Probe a range of numeric camera indexes and report which ones actually open.

### `save-source-alias`

Save a friendly name like `phone` or `turbopi` for a source.

### `list-source-aliases`

Print the aliases currently saved in `artifacts/live_vpr_sources.json`.

### `delete-source-alias`

Remove an alias you no longer want.

## 3. Why The Launcher Exists

The launcher is useful because it gives you:

- shorter commands
- reusable defaults
- one consistent interface for common workflows

This is especially helpful when repeatedly testing:

- the same descriptor
- the same reference map
- the same camera or stream source

## 4. Default Configuration

The script reads defaults from environment variables.

Common ones:

- `PYTHON_BIN`
- `VPR_DESCRIPTOR`
- `VPR_MAP_PATH`
- `VPR_REFERENCE_DIR`
- `VPR_SOURCE`
- `VPR_THRESHOLD`
- `VPR_PROCESS_FPS`
- `VPR_SAMPLE_FPS`
- `VPR_RECORDING_PATH`
- `VPR_CAPTURE_DIR`
- `VPR_VIDEO_PATH`

That means you can set up a preferred workflow once and then run very short commands after that.

Example:

```bash
export VPR_DESCRIPTOR=CosPlace
export VPR_MAP_PATH=artifacts/live_maps/campus_day_live.npz
export VPR_SOURCE=phone
export VPR_PROCESS_FPS=2.0
```

Then:

```bash
bash scripts/live_vpr_cli.sh record-map
bash scripts/live_vpr_cli.sh live
```

## 5. Passing Extra Arguments

Any extra arguments after the launcher command are forwarded directly to [live_vpr_test.py](../../live_vpr_test.py).

Example:

```bash
bash scripts/live_vpr_cli.sh live --threshold 0.45 --top_k 8
```

This is useful because the launcher stays convenient without hiding the full Python CLI.

## 6. Recommended Usage Pattern

For most users, the best pattern is:

1. identify the right source with `check-source` or `list-sources`
2. save a source alias
3. build a map with `build-map`, `record-map`, or `video-map`
4. run localization with `live` or `video`

## 7. Where To Edit The Launcher

If you want to change launcher behavior, edit:

- [scripts/live_vpr_cli.sh](../../scripts/live_vpr_cli.sh)

If you want to change actual VPR behavior, edit the Python code instead:

- [live_vpr_test.py](../../live_vpr_test.py)
- [live_vpr/](../../live_vpr)
