# Project Architecture Guide

This document is the shortest useful onboarding guide to this repository.

It is written for a junior developer who needs to answer questions like:

- Where does the code start?
- What is the difference between the original tutorial pipeline and the newer live pipeline?
- Which files should I open first?
- If I want to change datasets, descriptors, matching, live UI, or camera input, where do I work?

For the exact metric definitions, see [VPR_EVALUATION_FLOW.md](./VPR_EVALUATION_FLOW.md).  
For live usage commands, see [LIVE_VPR_COMMANDS.md](../live-vpr/LIVE_VPR_COMMANDS.md).  
For the live pipeline walkthrough, see [LIVE_VPR_PIPELINE.md](../live-vpr/LIVE_VPR_PIPELINE.md).

## 1. Start Here

There are three main entrypoints:

- [demo.py](../../demo.py): original benchmark-evaluation pipeline for tutorial datasets
- [test_campus_dataset.py](../../test_campus_dataset.py): benchmark-style evaluation on the custom campus dataset
- [live_vpr_test.py](../../live_vpr_test.py): newer offline/online live VPR pipeline for map building and live localization

If you are new to the repo, read them in this order:

1. [demo.py](../../demo.py)
2. [datasets/load_dataset.py](../../datasets/load_dataset.py)
3. [matching/matching.py](../../matching/matching.py)
4. [evaluation/metrics.py](../../evaluation/metrics.py)
5. [live_vpr_test.py](../../live_vpr_test.py)
6. [live_vpr/](../../live_vpr)

That gives you the old tutorial flow first, then the newer production-style live flow.

## 2. Architecture At A Glance

The repo now has two closely related architectures.

### A. Benchmark / Evaluation Architecture

Used by:

- [demo.py](../../demo.py)
- [test_campus_dataset.py](../../test_campus_dataset.py)

Flow:

```text
dataset loader
-> images + ground truth
-> descriptor extraction
-> similarity matrix S
-> matching decisions
-> metrics + plots + qualitative examples
```

This path is best for research-style evaluation.

### B. Live VPR Architecture

Used by:

- [live_vpr_test.py](../../live_vpr_test.py)
- [scripts/live_vpr_cli.sh](../../scripts/live_vpr_cli.sh)

Flow:

```text
CONFIG
YAML config + CLI overrides
-> live_vpr_test.py orchestration

OFFLINE
reference images or traversal video
-> sampled reference images
-> descriptor extraction
-> normalized reference map (.npz)

ONLINE
camera / stream / video
-> periodic descriptor extraction
-> search backend (exact / HNSW / FAISS IVF)
-> optional exact rerank
-> top-k results + thresholded decision
-> live overlay + saved inference reports
```

This path is best for demos, webcam testing, phone webcam testing, and robot integration.

## 3. Important Folders

Here is the mental model for the top-level folders:

- [datasets/](../../datasets): dataset loaders and ground-truth generation
- [feature_extraction/](../../feature_extraction): descriptor implementations
- [matching/](../../matching): converts similarity scores into match decisions
- [evaluation/](../../evaluation): PR curves, recall metrics, and qualitative match display
- [live_vpr/](../../live_vpr): modular live pipeline components
- [configs/](../../configs): YAML configuration files for the live pipeline
- [docs/](../../docs): usage and architecture docs
- [custom_dataset/](../../custom_dataset): your local campus dataset
- [artifacts/](../../artifacts): saved maps, aliases, inference images, recordings
- [TurboPi_Backup/](../../TurboPi_Backup): reference copy of the robot-side code, useful for stream integration

## 4. Core Concepts

These objects appear everywhere in the codebase.

### Images

Most of the repo works with Python lists of `numpy` images:

```python
imgs_db, imgs_q, GThard, GTsoft = dataset.load()
```

You will see this pattern in [demo.py](../../demo.py) and [test_campus_dataset.py](../../test_campus_dataset.py).

### Descriptors

A descriptor is one vector per image, usually shape `[N, D]`.

The repo standardizes that in [live_vpr/extractors.py](../../live_vpr/extractors.py):

```python
descriptors = extractor.compute_features(images)
descriptors = np.asarray(descriptors, dtype=np.float32)
```

### Similarity Matrix `S`

This is central to the evaluation path:

- rows = reference/database images
- columns = query images
- larger values = more similar

In [demo.py](../../demo.py), the common path is:

```python
db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic, axis=1, keepdims=True)
q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic, axis=1, keepdims=True)
S = np.matmul(db_D_holistic, q_D_holistic.transpose())
```

### Reference Map

The live pipeline replaces the raw dataset-plus-ground-truth setup with a reusable serialized map.

That object lives in [live_vpr/database.py](../../live_vpr/database.py):

```python
@dataclass
class ReferenceMap:
    descriptors: np.ndarray
    image_paths: list[str]
    metadata: dict[str, Any]
```

This is what the live system loads at runtime.

### Search Backend

The live runtime now separates descriptor extraction from search.

That abstraction lives in [live_vpr/search.py](../../live_vpr/search.py):

```python
search_result = self.search_backend.search(descriptor[0], top_k=self.top_k)
```

Supported backends:

- `exact`
- `hnsw`
- `faiss_ivf_flat`
- `faiss_ivf_pq`

This lets the repo scale from small maps to larger maps without rewriting the localization code.

### YAML Config

The live pipeline can load defaults from YAML before applying CLI overrides.

Start with:

- [configs/live_vpr.yaml](../../configs/live_vpr.yaml)
- [live_vpr/config.py](../../live_vpr/config.py)

Core idea:

```python
args = parse_args_with_config(parser)
```

That gives the live pipeline three layers of control:

- code defaults
- YAML defaults
- explicit CLI overrides

## 5. The Original Tutorial Flow

If you want to understand the repo from first principles, start here.

### Step 1: Load a dataset

In [demo.py](../../demo.py), the dataset is chosen by CLI and loaded from [datasets/load_dataset.py](../../datasets/load_dataset.py).

Example:

```python
if args.dataset == 'GardensPoint':
    dataset = GardensPointDataset()

imgs_db, imgs_q, GThard, GTsoft = dataset.load()
```

Dataset loaders return:

- `imgs_db`: reference images
- `imgs_q`: query images
- `GThard`: strict ground truth
- `GTsoft`: relaxed ground truth used by some metrics

### Step 2: Extract descriptors

The descriptor is chosen in [demo.py](../../demo.py) and implemented in [feature_extraction/](../../feature_extraction).

Common examples:

- [feature_extractor_cosplace.py](../../feature_extraction/feature_extractor_cosplace.py)
- [feature_extractor_eigenplaces.py](../../feature_extraction/feature_extractor_eigenplaces.py)
- [feature_extractor_holistic.py](../../feature_extraction/feature_extractor_holistic.py)
- [feature_extractor_patchnetvlad.py](../../feature_extraction/feature_extractor_patchnetvlad.py)

Each extractor exposes the same main interface:

```python
features = extractor.compute_features(images)
```

### Step 3: Compute similarity

For most global descriptors, the pipeline uses cosine similarity after L2 normalization.

### Step 4: Convert similarity into matches

[matching/matching.py](../../matching/matching.py) contains the two key strategies:

```python
M1 = matching.best_match_per_query(S)
M2 = matching.thresholding(S, 'auto')
```

Use them like this:

- `best_match_per_query`: one best database image per query
- `thresholding`: any pair above threshold becomes a match

### Step 5: Evaluate

[evaluation/metrics.py](../../evaluation/metrics.py) computes:

- precision-recall curves
- `R@100P`
- `R@K`

The central functions are:

- `createPR(...)`
- `recallAt100precision(...)`
- `recallAtK(...)`

For qualitative debugging, [evaluation/show_correct_and_wrong_matches.py](../../evaluation/show_correct_and_wrong_matches.py) displays true positives and false positives side by side.

## 6. The Campus Dataset Flow

The campus dataset is not a separate architecture. It is a custom dataset plugged into the same evaluation pattern.

The important file is [datasets/load_dataset.py](../../datasets/load_dataset.py), specifically `CampusDataset`.

What it does:

- loads `custom_dataset/day_images` as the reference database
- loads `custom_dataset/night_images` as the query set
- creates ground truth from filenames
- treats `-npm`, `npmXX`, and `PXL_*` images as no-match queries

The script [test_campus_dataset.py](../../test_campus_dataset.py) then runs the same evaluation stages as [demo.py](../../demo.py), but on your local dataset instead of GardensPoint/StLucia/SFU.

This means:

- if you understand `demo.py`, you almost understand `test_campus_dataset.py`
- the main difference is the dataset loader and the saved plots/results

## 7. The Live Pipeline

The live system is a more modular version of the same VPR idea.

The best top-level file to read is [live_vpr_test.py](../../live_vpr_test.py).

It has two main phases:

- `build_map(...)` / `build_live_map(...)`
- `run_online(...)`

### 7.1 Offline Phase: Build A Map

The map-building logic is implemented in [live_vpr/offline.py](../../live_vpr/offline.py).

The main class is `MapBuilder`:

```python
builder = MapBuilder(args.descriptor)
reference_map, stats = builder.build(config)
```

Inside `build_from_paths(...)`, the flow is:

```python
images = [_load_rgb_image(path, target_size) for path in image_paths]
descriptors = compute_global_descriptors(self.extractor, images)
descriptors = normalize_descriptors(descriptors)
reference_map = ReferenceMap(...)
save_reference_map(reference_map, output_path)
```

That means a map is just:

- a normalized descriptor matrix
- the corresponding reference image paths
- metadata such as descriptor name and target size

### 7.2 Live Map Building From A Traversal

If you do not already have reference images, the live pipeline can record a traversal video first and build the map after recording stops.

That logic lives in [live_vpr/capture.py](../../live_vpr/capture.py).

There are two important pieces:

- `LiveReferenceRecorder`
- `sample_video_to_frames(...)`

The recorder writes a traversal video:

```python
if recording:
    writer.write(frame)
    frame_count += 1
```

Then the sampler converts that video into reference images at `sample_fps`:

```python
if should_sample:
    if config.save_scale < 1.0:
        frame_to_save = cv2.resize(frame, ...)
    cv2.imwrite(str(output_path), frame_to_save)
    saved_paths.append(str(output_path))
```

This design is useful because recording and map-building are separate concerns:

- recording captures raw traversal data
- sampling controls how dense the map becomes
- saved-frame downsampling controls how much space the sampled reference image set uses

### 7.3 Online Phase: Localize Live Frames

The online localizer lives in [live_vpr/online.py](../../live_vpr/online.py).

Main class:

```python
localizer = LiveLocalizer(
    reference_map=reference_map,
    descriptor_name=descriptor,
    threshold=args.threshold,
    top_k=args.top_k,
    search_config=SearchConfig.from_namespace(args),
)
```

Core inference logic:

```python
descriptor = compute_global_descriptors(self.extractor, [rgb_image])
descriptor = descriptor / (np.linalg.norm(descriptor, axis=1, keepdims=True) + 1e-8)
search_result = self.search_backend.search(descriptor[0], top_k=self.top_k)
```

So the live phase is still doing the familiar VPR computation:

- encode one query frame
- compare it to the reference descriptors
- rank the results

The main difference is that it does not compute PR curves or recall metrics during the session. It just returns:

- best match
- best score
- top-k indices and scores
- thresholded `recognized` / `unknown` decision

### 7.4 Search Backends And Reranking

Search logic is now modular and lives in [live_vpr/search.py](../../live_vpr/search.py).

Important pieces:

- `SearchConfig`
- `SearchResult`
- `create_search_backend(...)`

Current backends:

- `exact`: full scan over the descriptor matrix with partial top-k selection
- `hnsw`: approximate nearest neighbor search using HNSW
- `faiss_ivf_flat`: FAISS IVF with exact vectors inside probed clusters
- `faiss_ivf_pq`: FAISS IVF with product quantization for larger maps

How they are currently implemented:

- `exact` uses the full descriptor matrix directly and avoids a full sort by using partial top-k selection
- `hnsw` builds one HNSW graph index and queries it for a shortlist of likely candidates
- `faiss_ivf_flat` trains a FAISS coarse quantizer, probes only relevant clusters, and keeps full vectors inside those clusters
- `faiss_ivf_pq` uses the same FAISS IVF idea but stores compressed PQ vectors to reduce memory use

The important architectural point is that approximate retrieval can still be followed by exact reranking:

```python
if self.config.rerank:
    indices, scores = _rerank_exact(self.descriptors, query, candidate_indices, top_k)
```

Why reranking matters:

- ANN backends are fast because they avoid scoring the whole map
- but the returned candidate order is only approximate
- reranking recomputes exact scores on the shortlist, which improves the final top-k order and gives more trustworthy scores for thresholding
- reranking cannot recover a true match that never made it into the shortlist, so `search_candidate_k` still matters

So the runtime flow is now:

- extract the query descriptor
- retrieve candidates with the chosen backend
- optionally rerank them exactly
- return the final top-k matches

Small mental model:

- use `exact` when the map is still small and you want the simplest baseline
- use `hnsw` first when latency becomes the main issue on CPU
- use `faiss_ivf_flat` when the map is larger and fairly static
- use `faiss_ivf_pq` when memory pressure is also becoming important

### 7.5 Live UI And Saved Inference Reports

The UI code lives in [live_vpr/ui.py](../../live_vpr/ui.py).

Main class:

```python
display = LiveDisplay(reference_map=reference_map, show_top_k=not args.hide_top_k)
```

There are two render paths:

- `render(...)`: draws the on-screen live overlay
- `render_inference_report(...)`: creates the padded saved report image for each inference

This is where you modify:

- status text
- top-k thumbnails
- labels like `Live query frame` and `Best reference match`
- margins / panel sizes

### 7.6 YAML Config And CLI Overrides

Config loading is centralized in [live_vpr/config.py](../../live_vpr/config.py), with defaults stored in [configs/live_vpr.yaml](../../configs/live_vpr.yaml).

Core idea:

```python
args = parse_args_with_config(parser)
```

This lets the repo ship default values for:

- map paths
- source defaults
- sampling rates
- inference cadence
- search backend and ANN parameters

while still letting the CLI override any of them.

### 7.7 Camera And Stream Sources

All capture-source logic is centralized in [live_vpr/sources.py](../../live_vpr/sources.py).

The key abstraction is:

```python
frame_source = OpenCVFrameSource(source, width=args.frame_width, height=args.frame_height)
```

That class supports:

- webcam indexes like `0` or `1`
- saved aliases like `phone` or `turbopi`
- stream URLs
- video file paths when wrapped by higher-level modes

This is also where source aliases are saved to:

- `artifacts/live_vpr_sources.json`

If camera selection or stream resolution is broken, start with this file.

## 8. How The CLI Ties It Together

[live_vpr_test.py](../../live_vpr_test.py) is the glue layer. It does not contain the heavy logic itself; it wires together the modular pieces.

Good examples:

- `build_map(...)` uses `MapBuilder`
- `build_live_map(...)` uses `LiveReferenceRecorder`, `sample_video_to_frames`, and `MapBuilder`
- `run_online(...)` uses `ReferenceMap`, `LiveLocalizer`, `SearchConfig`, `OpenCVFrameSource`, and `LiveDisplay`
- `main()` now uses `parse_args_with_config(...)` before dispatching modes

This is the right file to edit when:

- adding a new mode
- changing CLI arguments
- changing the overall flow between modules

## 9. Common “Where Do I Change X?” Questions

### Add a new dataset

Work in:

- [datasets/load_dataset.py](../../datasets/load_dataset.py)
- maybe [demo.py](../../demo.py) or [test_campus_dataset.py](../../test_campus_dataset.py)

Add a loader class with:

```python
def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
```

### Add a new descriptor

Work in:

- [feature_extraction/](../../feature_extraction)
- [live_vpr/extractors.py](../../live_vpr/extractors.py)
- maybe [demo.py](../../demo.py) and [test_campus_dataset.py](../../test_campus_dataset.py)

You need:

- an extractor class with `compute_features(images)`
- registration in `SUPPORTED_DESCRIPTORS`
- a branch in `create_feature_extractor(...)`

### Change live-match behavior

Work in:

- [live_vpr/online.py](../../live_vpr/online.py)
- [live_vpr/search.py](../../live_vpr/search.py)

That is where thresholding, candidate retrieval, reranking, and similarity ranking happen.

### Change search backend defaults or ANN parameters

Work in:

- [configs/live_vpr.yaml](../../configs/live_vpr.yaml)
- [live_vpr/search.py](../../live_vpr/search.py)
- [live_vpr_test.py](../../live_vpr_test.py)

Use this path when you want to switch between `exact`, `hnsw`, and `faiss` backends or tune their parameters.

### Change live overlays or saved inference images

Work in:

- [live_vpr/ui.py](../../live_vpr/ui.py)

### Change source aliases / stream handling

Work in:

- [live_vpr/sources.py](../../live_vpr/sources.py)

### Change config loading behavior

Work in:

- [live_vpr/config.py](../../live_vpr/config.py)
- [configs/live_vpr.yaml](../../configs/live_vpr.yaml)
- [live_vpr_test.py](../../live_vpr_test.py)

### Change benchmark metrics

Work in:

- [evaluation/metrics.py](../../evaluation/metrics.py)

### Change map serialization format

Work in:

- [live_vpr/database.py](../../live_vpr/database.py)

## 10. Recommended Reading Path For A New Developer

If you are onboarding this week, use this order:

1. Read [demo.py](../../demo.py) and understand the classic evaluation flow.
2. Read [datasets/load_dataset.py](../../datasets/load_dataset.py) to understand how images and ground truth enter the system.
3. Read [matching/matching.py](../../matching/matching.py) and [evaluation/metrics.py](../../evaluation/metrics.py).
4. Read [test_campus_dataset.py](../../test_campus_dataset.py) to see how the tutorial flow was adapted for the campus data.
5. Read [live_vpr_test.py](../../live_vpr_test.py) to understand the higher-level live workflow.
6. Read [live_vpr/offline.py](../../live_vpr/offline.py), [live_vpr/online.py](../../live_vpr/online.py), [live_vpr/search.py](../../live_vpr/search.py), [live_vpr/ui.py](../../live_vpr/ui.py), and [live_vpr/sources.py](../../live_vpr/sources.py).
7. Read [live_vpr/config.py](../../live_vpr/config.py) and [configs/live_vpr.yaml](../../configs/live_vpr.yaml) to understand how runtime defaults are supplied.

After that, you should be able to answer:

- how maps are built
- how live localization works
- where benchmark evaluation ends and demo logic begins
- which file to edit for most common changes

## 11. One Final Mental Model

Almost everything in this repo reduces to the same idea:

```text
image -> descriptor -> search -> decision
```

The benchmark code adds:

```text
decision -> metrics -> plots
```

The live code adds:

```text
decision -> overlay -> saved report -> demo
```

If you keep that mental model in mind, the codebase becomes much easier to navigate.
