# Live VPR Pipeline

This document explains the newer live Visual Place Recognition pipeline in this repository.

Read this after [PROJECT_ARCHITECTURE_GUIDE.md](../architecture/PROJECT_ARCHITECTURE_GUIDE.md) if you want a more focused explanation of the live system only.

Use this document to understand:

- how maps are built
- how live localization works
- which modules own which part of the flow
- where to edit the system when behavior changes

For commands, see [LIVE_VPR_COMMANDS.md](./LIVE_VPR_COMMANDS.md).  
For the bash wrapper, see [LIVE_VPR_SCRIPT.md](./LIVE_VPR_SCRIPT.md).

## 1. What The Live Pipeline Does

The live system splits the project into two phases.

### Offline Phase

Build a reusable reference map from either:

- an existing image folder
- a recorded traversal video

The output is a saved `.npz` map containing:

- normalized reference descriptors
- reference image paths
- metadata such as descriptor, target size, and map provenance

When artifact grouping is enabled, generated outputs are placed under a timestamped run folder inside `artifacts/runs/` so each map-building session stays separate. The build step also updates a stable map alias under `artifacts/live_maps/` so later live inference can still use the most recent map through the usual command flow.

### Online Phase

Use a camera, stream, or video to localize incoming frames against that saved map.

The output is:

- a best-match index
- a best similarity score
- a top-k ranked list
- a thresholded `MATCH` or `UNKNOWN` decision
- an on-screen overlay and optional saved inference reports

Those runtime outputs are also grouped into a per-run folder by default, which keeps captures, inference reports, and annotated videos from different sessions separated.

## 2. End-To-End Flow

```text
OFFLINE

reference folder or traversal recording
-> load / sample images
-> resize to map target size
-> extract descriptors
-> normalize descriptors
-> save ReferenceMap (.npz)

ONLINE

webcam / phone webcam / stream / video
-> read frame
-> resize to map target size
-> extract query descriptor
-> cosine similarity vs saved map
-> top-k ranking + thresholded decision
-> live overlay + saved inference report
```

## 3. Main Entry Point

The main entrypoint is [live_vpr_test.py](../../live_vpr_test.py).

It is mostly an orchestration layer. It wires together smaller modules from [live_vpr/](../../live_vpr).

The most important functions are:

- `build_map(...)`
- `build_live_map(...)`
- `run_online(...)`
- `check_source(...)`
- `list_sources(...)`

If you want to understand the live system, read those functions first.

## 4. Module Map

The modular implementation lives in [live_vpr/](../../live_vpr).

### [live_vpr/database.py](../../live_vpr/database.py)

Owns the saved map format.

Important pieces:

- `ReferenceMap`
- `normalize_descriptors(...)`
- `save_reference_map(...)`
- `load_reference_map(...)`

This is where the live pipeline stops thinking in terms of raw datasets and starts thinking in terms of reusable maps.

### [live_vpr/extractors.py](../../live_vpr/extractors.py)

Owns descriptor creation for the live pipeline.

Important pieces:

- `SUPPORTED_DESCRIPTORS`
- `create_feature_extractor(...)`
- `compute_global_descriptors(...)`

If you add a new live descriptor, this is one of the first files you must touch.

### [live_vpr/offline.py](../../live_vpr/offline.py)

Owns map building from reference images.

Important pieces:

- `MapBuildConfig`
- `MapBuilder`

Core flow:

```python
images = [_load_rgb_image(path, target_size) for path in image_paths]
descriptors = compute_global_descriptors(self.extractor, images)
descriptors = normalize_descriptors(descriptors)
reference_map = ReferenceMap(...)
```

### [live_vpr/capture.py](../../live_vpr/capture.py)

Owns traversal recording and post-record frame sampling.

Important pieces:

- `VideoRecordingConfig`
- `LiveReferenceRecorder`
- `FrameSamplingConfig`
- `sample_video_to_frames(...)`

This is the file to read if you want to understand the “record first, build map after stop” workflow.

### [live_vpr/online.py](../../live_vpr/online.py)

Owns runtime localization against a saved map.

Important pieces:

- `LocalizationResult`
- `LiveLocalizer`

Core logic:

```python
descriptor = compute_global_descriptors(self.extractor, [rgb_image])
descriptor = descriptor / (np.linalg.norm(descriptor, axis=1, keepdims=True) + 1e-8)
similarities = (self.reference_map.descriptors @ descriptor.T).reshape(-1)
```

This file is where the actual live VPR decision happens.

### [live_vpr/sources.py](../../live_vpr/sources.py)

Owns camera and stream source handling.

Important pieces:

- `OpenCVFrameSource`
- `probe_capture_source(...)`
- `list_available_capture_sources(...)`
- `save_source_alias(...)`
- `resolve_capture_source(...)`

This file abstracts:

- numeric camera indexes
- saved aliases like `phone` or `turbopi`
- stream URLs

### [live_vpr/ui.py](../../live_vpr/ui.py)

Owns the live overlay and the saved inference-report images.

Important pieces:

- `LiveDisplay.render(...)`
- `LiveDisplay.render_inference_report(...)`

This is where to modify:

- labels
- text padding
- top-k display
- saved inference-image layout

## 5. Offline Phase In Detail

There are three ways to build a map.

### Option A: Existing Reference Folder

Used when you already have curated reference images.

Flow:

```text
image folder
-> load paths
-> resize images
-> extract descriptors
-> normalize descriptors
-> save .npz map
```

Owned by:

- [live_vpr/offline.py](../../live_vpr/offline.py)
- `build_map(...)` in [live_vpr_test.py](../../live_vpr_test.py)

### Option B: Live Traversal Recording

Used when existing reference images are not enough.

Flow:

```text
camera / phone webcam / stream
-> record traversal video
-> stop recording
-> sample frames from video
-> build map from sampled frames
```

Owned by:

- [live_vpr/capture.py](../../live_vpr/capture.py)
- `build_live_map(...)` in [live_vpr_test.py](../../live_vpr_test.py)

Important behavior:

- the recorder opens paused by default
- `r` starts or pauses recording
- `q` stops recording and moves on to map building

### Option C: Existing Traversal Video

Used when you already recorded a route and want to rebuild the map later.

Flow:

```text
saved video
-> sample frames at sample_fps
-> build map
```

This is useful when you want to compare:

- different descriptors
- different sampling rates
- different map densities

## 6. Online Phase In Detail

The online flow is driven by `run_online(...)` in [live_vpr_test.py](../../live_vpr_test.py).

The key runtime steps are:

1. Load a saved map with `load_reference_map(...)`
2. Create a `LiveLocalizer`
3. Open a capture source with `OpenCVFrameSource`
4. Read frames continuously
5. Run localization only at `process_fps`
6. Reuse the most recent result between inference steps
7. Draw the overlay and save inference reports if enabled

This separation is important:

- frame display can stay smooth
- localization can run at a slower rate
- the system remains usable on weaker hardware

## 7. Runtime Signals vs Offline Metrics

The live session does not compute benchmark metrics like:

- AUC
- PR curves
- `R@100P`
- `R@K`

Instead, it uses runtime quantities:

- best cosine similarity score
- top-k ranked reference images
- thresholded recognition decision
- descriptor extraction latency
- age of the latest inference result

That is why the live system feels more like a demo application than a benchmark script.

## 8. Source Discovery And Aliases

The live pipeline supports more than just `--source 0`.

You can use:

- camera indexes like `0`, `1`, `2`
- stream URLs
- aliases like `phone` or `turbopi`

Aliases are saved in:

- `artifacts/live_vpr_sources.json`

This is especially useful for:

- phone webcams
- TurboPi streams
- unstable numeric camera indexes

## 9. Files To Edit For Common Tasks

### Add or change a descriptor

Edit:

- [live_vpr/extractors.py](../../live_vpr/extractors.py)
- the corresponding file in [feature_extraction/](../../feature_extraction)

### Change map metadata or map format

Edit:

- [live_vpr/database.py](../../live_vpr/database.py)

### Change traversal recording behavior

Edit:

- [live_vpr/capture.py](../../live_vpr/capture.py)

### Change localization logic

Edit:

- [live_vpr/online.py](../../live_vpr/online.py)

### Change overlays or saved inference images

Edit:

- [live_vpr/ui.py](../../live_vpr/ui.py)

### Change camera/stream handling

Edit:

- [live_vpr/sources.py](../../live_vpr/sources.py)

### Change CLI flags or add a new mode

Edit:

- [live_vpr_test.py](../../live_vpr_test.py)

## 10. Recommended Reading Order

For a new developer working on the live system, read in this order:

1. [live_vpr_test.py](../../live_vpr_test.py)
2. [live_vpr/offline.py](../../live_vpr/offline.py)
3. [live_vpr/capture.py](../../live_vpr/capture.py)
4. [live_vpr/database.py](../../live_vpr/database.py)
5. [live_vpr/online.py](../../live_vpr/online.py)
6. [live_vpr/ui.py](../../live_vpr/ui.py)
7. [live_vpr/sources.py](../../live_vpr/sources.py)

That path mirrors how the system actually operates.
