#!/usr/bin/env python3
"""
Live VPR pipeline for offline map building and online localization.

Supported simulations:
- Laptop webcam: `--source 0`
- Phone as virtual webcam: `--source 1` or another camera index
- Phone IP stream: `--source http://<phone-ip>:<port>/video`
- Recorded video: `--video path/to/video.mp4`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from live_vpr import (
    FrameSamplingConfig,
    LiveDisplay,
    LiveLocalizer,
    LiveReferenceRecorder,
    MapBuildConfig,
    MapBuilder,
    OpenCVFrameSource,
    SUPPORTED_DESCRIPTORS,
    SUPPORTED_SEARCH_BACKENDS,
    SUPPORTED_SEARCH_METRICS,
    SearchConfig,
    VideoRecordingConfig,
    delete_source_alias,
    get_source_aliases_path,
    list_available_capture_sources,
    load_source_aliases,
    load_reference_map,
    parse_args_with_config,
    probe_capture_source,
    resolve_capture_source,
    sample_video_to_frames,
    save_source_alias,
)
from live_vpr.extractors import describe_extractor_runtime


LEGACY_MODE_ALIASES = {
    "build_db": "build_map",
    "build_live_db": "build_live_map",
    "live_test": "live",
    "video_test": "video",
    "check_camera": "check_source",
    "list_sources": "list_sources",
}


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for live/video localization. "
            "Install it with `pip install opencv-python` or reinstall from `requirements.txt`."
        ) from exc
    return cv2


def normalize_mode(mode: str) -> str:
    return LEGACY_MODE_ALIASES.get(mode, mode)


def resolve_runtime_descriptor(reference_map, descriptor_override: str | None) -> str:
    descriptor = str(reference_map.metadata.get("descriptor", "CosPlace"))
    if descriptor_override and descriptor_override != descriptor:
        raise ValueError(
            f"Descriptor mismatch: map was built with {descriptor}, "
            f"but runtime override requested {descriptor_override}."
        )
    return descriptor


def print_reference_map_summary(reference_map) -> None:
    print(f"Loaded map: {reference_map.metadata.get('map_path', 'in-memory')}")
    print(f"Descriptor: {reference_map.metadata.get('descriptor', 'unknown')}")
    print(f"Images: {reference_map.num_images}")
    print(f"Descriptor dim: {reference_map.descriptor_dim}")
    print(f"Target size: {tuple(reference_map.metadata.get('target_size', [640, 480]))}")
    print(f"Created: {reference_map.metadata.get('created_utc', 'unknown')}")


def print_search_configuration(search_config: SearchConfig) -> None:
    print(f"Search backend: {search_config.backend}")
    print(f"Search metric: {search_config.metric}")
    print(f"Search rerank: {search_config.rerank}")
    if search_config.backend == "exact":
        return
    print(f"Search candidate_k: {search_config.candidate_k}")
    if search_config.backend == "hnsw":
        print(
            "HNSW M / ef_construction / ef_search: "
            f"{search_config.hnsw_m} / {search_config.hnsw_ef_construction} / {search_config.hnsw_ef_search}"
        )
    if search_config.backend in {"faiss_ivf_flat", "faiss_ivf_pq"}:
        print(f"FAISS IVF nlist / nprobe: {search_config.ivf_nlist} / {search_config.ivf_nprobe}")
    if search_config.backend == "faiss_ivf_pq":
        print(f"FAISS PQ m / bits: {search_config.pq_m} / {search_config.pq_bits}")


def print_source_resolution(source: str | int) -> None:
    resolved = resolve_capture_source(source)
    if resolved["alias"] is not None:
        print(f"Source alias: {resolved['alias']}")
    print(f"Resolved source: {resolved['resolved_source']}")
    print(f"Aliases file: {resolved['aliases_path']}")


def build_map(args: argparse.Namespace) -> None:
    print(f"\n{'=' * 68}")
    print("OFFLINE PHASE: MAP BUILDING")
    print(f"{'=' * 68}")
    print(f"Reference directory: {args.data_dir}")
    print(f"Descriptor: {args.descriptor}")
    print(f"Output map: {args.map_path}")
    print(f"Target size: {(args.resize_width, args.resize_height)}")

    config = MapBuildConfig(
        image_dir=args.data_dir,
        output_path=args.map_path,
        descriptor=args.descriptor,
        target_size=(args.resize_width, args.resize_height),
        recursive=args.recursive,
    )
    try:
        builder = MapBuilder(args.descriptor)
        reference_map, stats = builder.build(config)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{exc.name}' while building a map with descriptor {args.descriptor}. "
            "Install the project requirements before running the live pipeline."
        ) from exc

    print("\nMap built successfully.")
    print(f"Images indexed: {reference_map.num_images}")
    print(f"Descriptor dimension: {reference_map.descriptor_dim}")
    print(f"Load time: {stats['load_time_s']:.2f}s")
    print(f"Feature extraction time: {stats['extraction_time_s']:.2f}s")
    print(f"Average extraction time: {stats['avg_extraction_ms']:.1f}ms/image")
    print(f"Saved to: {Path(args.map_path).expanduser().resolve()}")


def build_live_map(args: argparse.Namespace, use_video: bool = False) -> None:
    source = args.video if use_video else args.source
    if source is None:
        raise ValueError("A capture source is required for live map building.")

    print(f"\n{'=' * 68}")
    print("OFFLINE PHASE: LIVE MAP BUILDING")
    print(f"{'=' * 68}")
    print(f"Capture source: {source}")
    if not use_video:
        print_source_resolution(source)
    print(f"Descriptor: {args.descriptor}")
    print(f"Output map: {args.map_path}")
    print(f"Recording path: {args.recording_path}")
    print(f"Sampled frame directory: {args.capture_dir}")
    print(f"Sampling rate: {args.sample_fps:.2f} fps")
    print(f"Saved reference frame scale: {args.capture_save_scale:.2f}x")
    print(f"Target size: {(args.resize_width, args.resize_height)}")

    if use_video:
        recording_video_path = str(Path(source).expanduser().resolve())
        print(f"Using existing traversal video: {recording_video_path}")
    else:
        recording_config = VideoRecordingConfig(
            source=source,
            output_video=args.recording_path,
            window_name=args.capture_window_name,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            mirror=args.mirror,
            start_recording=args.start_recording,
        )
        recorder = LiveReferenceRecorder(recording_config)
        recording_result = recorder.run()

        if recording_result.frame_count <= 0:
            raise RuntimeError("No video frames were recorded, so a live map could not be built.")
        recording_video_path = recording_result.video_path

    sampling_result = sample_video_to_frames(
        FrameSamplingConfig(
            video_path=recording_video_path,
            output_dir=args.capture_dir,
            sample_fps=args.sample_fps,
            frame_prefix=args.capture_prefix,
            max_frames=args.max_captures,
            save_scale=args.capture_save_scale,
        )
    )

    if len(sampling_result.saved_paths) < args.min_captures:
        raise RuntimeError(
            f"Only {len(sampling_result.saved_paths)} reference frames were sampled, "
            f"but at least {args.min_captures} are required to build a map."
        )

    try:
        builder = MapBuilder(args.descriptor)
        image_paths = [Path(path) for path in sampling_result.saved_paths]
        reference_map, stats = builder.build_from_paths(
            image_paths=image_paths,
            output_path=args.map_path,
            image_dir=sampling_result.output_dir,
            target_size=(args.resize_width, args.resize_height),
            recursive=False,
            metadata_extra={
                "live_build_video_path": recording_video_path,
                "live_build_sample_fps": sampling_result.sample_fps,
                "live_build_source_fps": sampling_result.source_fps,
                "live_build_video_duration_s": sampling_result.video_duration_s,
                "live_build_saved_frame_scale": args.capture_save_scale,
            },
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{exc.name}' while building a live map with descriptor {args.descriptor}. "
            "Install the project requirements before running the live pipeline."
        ) from exc

    print("\nLive map built successfully.")
    print(f"Traversal video: {recording_video_path}")
    print(f"Sampled reference images: {len(sampling_result.saved_paths)}")
    print(f"Descriptor dimension: {reference_map.descriptor_dim}")
    print(f"Load time: {stats['load_time_s']:.2f}s")
    print(f"Feature extraction time: {stats['extraction_time_s']:.2f}s")
    print(f"Average extraction time: {stats['avg_extraction_ms']:.1f}ms/image")
    print(f"Saved to: {Path(args.map_path).expanduser().resolve()}")


def check_source(args: argparse.Namespace, use_video: bool = False) -> None:
    source = args.video if use_video else args.source
    if source is None:
        raise ValueError("A capture source is required.")

    print(f"Checking source: {source}")
    info = probe_capture_source(source, width=args.frame_width, height=args.frame_height)
    print("Source opened successfully.")
    if info.get("alias") is not None:
        print(f"Source alias: {info['alias']}")
    print(f"Resolved source: {info['source']}")
    print(f"Frame shape: {info['frame_shape']}")
    print(f"Backend: {info['backend']}")


def list_sources(args: argparse.Namespace) -> None:
    print(f"Scanning camera indexes 0..{args.source_scan_max}")
    if args.source_snapshot_dir:
        print(f"Saving preview snapshots to: {Path(args.source_snapshot_dir).expanduser().resolve()}")

    sources = list_available_capture_sources(
        max_index=args.source_scan_max,
        width=args.frame_width,
        height=args.frame_height,
        snapshot_dir=args.source_snapshot_dir,
    )
    aliases = load_source_aliases()

    if not sources:
        print("No camera indexes were successfully opened in the scanned range.")
        return

    inverse_aliases = {value: key for key, value in aliases.items()}
    print("\nDetected sources:")
    for item in sources:
        alias = inverse_aliases.get(str(item["source"]))
        alias_text = f" alias={alias}" if alias else ""
        preview_text = f" preview={item['preview_path']}" if item.get("preview_path") else ""
        print(
            f"  index={item['index']} resolved={item['source']} "
            f"shape={item['frame_shape']} backend={item['backend']}{alias_text}{preview_text}"
        )


def save_source_alias_command(args: argparse.Namespace) -> None:
    if not args.alias:
        raise ValueError("--alias is required when saving a source alias")
    if args.source is None:
        raise ValueError("--source is required when saving a source alias")

    aliases_path = save_source_alias(args.alias, args.source)
    resolved = resolve_capture_source(args.alias)
    print(f"Saved source alias '{args.alias}' -> {args.source}")
    print(f"Resolved source: {resolved['resolved_source']}")
    print(f"Aliases file: {aliases_path}")


def list_source_aliases_command() -> None:
    aliases = load_source_aliases()
    aliases_path = get_source_aliases_path()
    print(f"Aliases file: {aliases_path}")
    if not aliases:
        print("No source aliases saved yet.")
        return

    print("Saved source aliases:")
    for alias, source in sorted(aliases.items()):
        print(f"  {alias} -> {source}")


def delete_source_alias_command(args: argparse.Namespace) -> None:
    if not args.alias:
        raise ValueError("--alias is required when deleting a source alias")
    aliases_path = delete_source_alias(args.alias)
    print(f"Deleted alias '{args.alias}' if it existed.")
    print(f"Aliases file: {aliases_path}")


def create_video_writer(output_video: str | None, frame_shape: tuple[int, int, int]):
    if not output_video:
        return None

    cv2 = _import_cv2()
    output_path = Path(output_video).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    return writer


def run_online(args: argparse.Namespace, use_video: bool = False) -> None:
    cv2 = _import_cv2()

    reference_map = load_reference_map(args.map_path)
    reference_map.metadata["map_path"] = str(Path(args.map_path).expanduser().resolve())
    descriptor = resolve_runtime_descriptor(reference_map, args.descriptor)
    search_config = SearchConfig.from_namespace(args)
    print(f"\n{'=' * 68}")
    print("ONLINE PHASE: LIVE LOCALIZATION")
    print(f"{'=' * 68}")
    print_reference_map_summary(reference_map)
    print_search_configuration(search_config)

    target_size = tuple(reference_map.metadata.get("target_size", [640, 480]))
    try:
        localizer = LiveLocalizer(
            reference_map=reference_map,
            descriptor_name=descriptor,
            threshold=args.threshold,
            top_k=args.top_k,
            search_config=search_config,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{exc.name}' while loading descriptor {descriptor}. "
            "Install the project requirements before running online localization."
        ) from exc
    display = LiveDisplay(reference_map=reference_map, show_top_k=not args.hide_top_k)
    print(f"Runtime backend: {describe_extractor_runtime(localizer.extractor)}")
    print(f"Search runtime: {localizer.search_backend.describe()}")

    source = args.video if use_video else args.source
    if source is None:
        raise ValueError("A capture source is required.")

    print(f"Capture source: {source}")
    if not use_video:
        print_source_resolution(source)
    print("Controls: i=start/pause inference, q=quit, s=save frame, t=toggle top-k, +=raise threshold, -=lower threshold")

    frame_source = OpenCVFrameSource(source, width=args.frame_width, height=args.frame_height)
    writer = None
    inference_results = []
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    inference_stats_dir = Path(args.inference_stats_dir).expanduser().resolve()
    if args.save_inference_images:
        inference_stats_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_source.open()
        frame_count = 0
        fps = 0.0
        fps_start = time.time()
        last_result = None
        last_inference_time = 0.0
        process_interval_s = 0.0 if args.process_fps <= 0 else 1.0 / args.process_fps
        inference_active = args.start_inference

        while True:
            ok, frame = frame_source.read()
            if not ok or frame is None:
                if use_video:
                    break
                print("Failed to read frame from capture source.")
                break

            now = time.time()
            should_run_inference = (
                inference_active
                and (
                    last_result is None
                    or process_interval_s == 0.0
                    or (now - last_inference_time) >= process_interval_s
                )
            )
            if should_run_inference:
                frame_for_model = frame
                resized = cv2.resize(frame_for_model, target_size)
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                last_result = localizer.localize_rgb(rgb_image)
                inference_results.append(last_result)
                last_inference_time = time.time()

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start
                fps = 10.0 / elapsed if elapsed > 0 else 0.0
                fps_start = time.time()

            result_age_ms = 0.0 if last_result is None else max(0.0, (time.time() - last_inference_time) * 1000.0)
            rendered = display.render(
                frame,
                last_result,
                localizer.threshold,
                fps,
                result_age_ms=result_age_ms,
                process_fps=args.process_fps,
                inference_active=inference_active,
                inference_index=len(inference_results) if inference_results else None,
            )
            if args.mirror:
                rendered = cv2.flip(rendered, 1)

            if should_run_inference and args.save_inference_images and last_result is not None:
                save_inference_image(
                    display=display,
                    query_frame=frame,
                    inference_result=last_result,
                    inference_index=len(inference_results),
                    threshold=localizer.threshold,
                    output_dir=inference_stats_dir,
                    fps=fps,
                    result_age_ms=result_age_ms,
                    process_fps=args.process_fps,
                )

            if writer is None and args.output_video:
                writer = create_video_writer(args.output_video, rendered.shape)
            if writer is not None:
                writer.write(rendered)

            cv2.imshow(args.window_name, rendered)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("i"):
                inference_active = not inference_active
                state = "ON" if inference_active else "PAUSED"
                print(f"Inference: {state}")
            if key == ord("s"):
                filename = snapshot_dir / f"capture_{int(time.time())}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"Saved frame to {filename}")
            if key == ord("t"):
                display.show_top_k = not display.show_top_k
            if key in (ord("+"), ord("=")):
                localizer.set_threshold(localizer.threshold + 0.05)
                print(f"Threshold: {localizer.threshold:.2f}")
            if key == ord("-"):
                localizer.set_threshold(localizer.threshold - 0.05)
                print(f"Threshold: {localizer.threshold:.2f}")

        print_summary(inference_results, localizer.threshold, frame_count, args.process_fps)
        if args.save_inference_images:
            print(f"Inference images saved to: {inference_stats_dir}")
        if args.output_video:
            print(f"Annotated output saved to {Path(args.output_video).expanduser().resolve()}")
    finally:
        frame_source.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def print_summary(results, threshold: float, displayed_frames: int | None = None, process_fps: float | None = None) -> None:
    if not results:
        print(f"\n{'=' * 68}")
        print("SESSION SUMMARY")
        print(f"{'=' * 68}")
        if displayed_frames is not None:
            print(f"Frames displayed: {displayed_frames}")
        if process_fps is not None:
            cadence = "every frame" if process_fps <= 0 else f"{process_fps:.2f} fps"
            print(f"Inference cadence target: {cadence}")
        print("No inference runs were executed.")
        return

    scores = np.array([result.best_score for result in results], dtype=np.float32)
    latencies = np.array([result.extraction_time_ms for result in results], dtype=np.float32)
    recognized = sum(1 for result in results if result.recognized)

    print(f"\n{'=' * 68}")
    print("SESSION SUMMARY")
    print(f"{'=' * 68}")
    if displayed_frames is not None:
        print(f"Frames displayed: {displayed_frames}")
    print(f"Inference runs: {len(results)}")
    if process_fps is not None:
        cadence = "every frame" if process_fps <= 0 else f"{process_fps:.2f} fps"
        print(f"Inference cadence target: {cadence}")
    print(f"Recognized inference results: {recognized} ({100 * recognized / len(results):.1f}%)")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Best-score mean/median: {scores.mean():.3f} / {np.median(scores):.3f}")
    print(f"Best-score min/max: {scores.min():.3f} / {scores.max():.3f}")
    print(f"Latency mean/median: {latencies.mean():.1f}ms / {np.median(latencies):.1f}ms")


def save_inference_image(
    display,
    query_frame,
    inference_result,
    inference_index: int,
    threshold: float,
    output_dir: Path,
    fps: float,
    result_age_ms: float,
    process_fps: float | None,
) -> None:
    cv2 = _import_cv2()
    safe_score = f"{inference_result.best_score:.3f}".replace("-", "neg")
    label = "match" if inference_result.recognized else "unknown"
    filename = (
        f"inference_{inference_index:05d}_{label}_"
        f"score_{safe_score}_th_{threshold:.2f}.jpg"
    )
    output_path = output_dir / filename
    export_canvas = display.render_inference_report(
        query_frame_bgr=query_frame,
        result=inference_result,
        threshold=threshold,
        fps=fps,
        result_age_ms=result_age_ms,
        process_fps=process_fps,
        inference_index=inference_index,
    )
    cv2.imwrite(str(output_path), export_canvas)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline map building and online live VPR localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python live_vpr_test.py --mode build_map --data_dir custom_dataset/day_images --map_path artifacts/campus_day_cosplace.npz\n"
            "  python live_vpr_test.py --mode build_live_map --map_path artifacts/campus_day_live.npz --source 0 --recording_path artifacts/reference_videos/campus_day.mp4 --sample_fps 1.0\n"
            "  python live_vpr_test.py --mode live --map_path artifacts/campus_day_cosplace.npz --source 0 --process_fps 2.0\n"
            "  python live_vpr_test.py --mode live --map_path artifacts/campus_day_cosplace.npz --source 1\n"
            "  python live_vpr_test.py --mode list_sources --source_scan_max 10 --source_snapshot_dir artifacts/source_previews\n"
            "  python live_vpr_test.py --mode save_source_alias --alias phone --source 3\n"
            "  python live_vpr_test.py --mode live --map_path artifacts/campus_day_cosplace.npz --source http://192.168.1.20:4747/video\n"
            "  python live_vpr_test.py --mode video --map_path artifacts/campus_day_cosplace.npz --video demo_walk.mp4\n"
            "  python live_vpr_test.py --mode check_source --source 0\n"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        help="YAML configuration file to load before applying CLI overrides",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=[
            "build_map",
            "build_db",
            "build_live_map",
            "build_live_db",
            "live",
            "live_test",
            "video",
            "video_test",
            "check_source",
            "check_camera",
            "list_sources",
            "save_source_alias",
            "list_source_aliases",
            "delete_source_alias",
        ],
        help="Pipeline stage or runtime mode",
    )
    parser.add_argument(
        "--descriptor",
        type=str,
        default="CosPlace",
        choices=SUPPORTED_DESCRIPTORS,
        help="Descriptor used for map building and runtime localization",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="custom_dataset/day_images",
        help="Reference image directory for offline map building",
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default="artifacts/live_maps/campus_day_cosplace.npz",
        help="Path to the saved reference map",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Capture source: webcam index, saved alias, virtual webcam index, or stream URL",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Video file path for simulation playback or map building from an existing recording",
    )
    parser.add_argument(
        "--alias",
        type=str,
        help="Friendly source alias name, for example 'phone' or 'laptop'",
    )
    parser.add_argument(
        "--source_scan_max",
        type=int,
        default=10,
        help="Highest numeric camera index to probe when using list_sources",
    )
    parser.add_argument(
        "--source_snapshot_dir",
        type=str,
        help="Optional directory for saving one preview frame per detected source during list_sources",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Recognition threshold on cosine similarity",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top matches to display",
    )
    parser.add_argument(
        "--process_fps",
        type=float,
        default=2.0,
        help="How often to run localization during live/video inference. Use <=0 to process every frame",
    )
    parser.add_argument(
        "--start_inference",
        action="store_true",
        help="Start live/video inference immediately instead of opening in paused mode",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=640,
        help="Reference-map resize width",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=480,
        help="Reference-map resize height",
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=1280,
        help="Requested capture width for webcam sources",
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=720,
        help="Requested capture height for webcam sources",
    )
    parser.add_argument(
        "--capture_dir",
        type=str,
        default="artifacts/reference_captures",
        help="Directory where sampled reference frames are saved before map building",
    )
    parser.add_argument(
        "--recording_path",
        type=str,
        default="artifacts/reference_videos/live_reference_traversal.mp4",
        help="Path for the recorded traversal video used by live map building",
    )
    parser.add_argument(
        "--capture_prefix",
        type=str,
        default="ref",
        help="Filename prefix for sampled reference frames",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="How densely to sample frames from the recorded traversal video when building a live map",
    )
    parser.add_argument(
        "--capture_save_scale",
        type=float,
        default=0.5,
        help=(
            "Scale factor applied when saving sampled reference frames during live map building. "
            "Use 0.5 to halve width and height, or 1.0 to keep original size"
        ),
    )
    parser.add_argument(
        "--start_recording",
        action="store_true",
        help="Start the traversal recorder immediately instead of opening in paused mode",
    )
    parser.add_argument(
        "--max_captures",
        type=int,
        help="Optional limit on the number of sampled reference frames",
    )
    parser.add_argument(
        "--min_captures",
        type=int,
        default=5,
        help="Minimum number of sampled frames required before a live map can be built",
    )
    parser.add_argument(
        "--capture_window_name",
        type=str,
        default="Live VPR Map Recorder",
        help="OpenCV window title used during live reference recording",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively index reference images during map building",
    )
    parser.add_argument(
        "--use_video_for_live_build",
        action="store_true",
        help="Use an existing video from --video instead of recording a new traversal for build_live_map",
    )
    parser.add_argument(
        "--hide_top_k",
        action="store_true",
        help="Hide the top-k reference preview panel at startup",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the displayed live view for easier webcam use",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        help="Optional path for saving the annotated live/video session",
    )
    parser.add_argument(
        "--inference_stats_dir",
        type=str,
        default="artifacts/inference_stats",
        help="Directory where annotated images are saved after each inference",
    )
    parser.add_argument(
        "--no_save_inference_images",
        dest="save_inference_images",
        action="store_false",
        help="Disable saving an annotated image after each inference",
    )
    parser.set_defaults(save_inference_images=True)
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="artifacts/live_captures",
        help="Directory for frames saved with the 's' key",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default="Live VPR",
        help="OpenCV window title",
    )
    parser.add_argument(
        "--search_backend",
        type=str,
        default="exact",
        choices=SUPPORTED_SEARCH_BACKENDS,
        help="Search backend used for reference-map lookup",
    )
    parser.add_argument(
        "--search_metric",
        type=str,
        default="cosine",
        choices=SUPPORTED_SEARCH_METRICS,
        help="Similarity metric used by the selected search backend",
    )
    parser.add_argument(
        "--search_candidate_k",
        type=int,
        default=50,
        help="Number of ANN candidates to retrieve before optional exact reranking",
    )
    parser.add_argument(
        "--search_rerank",
        dest="search_rerank",
        action="store_true",
        help="Enable exact reranking of ANN candidates using the full descriptor matrix",
    )
    parser.add_argument(
        "--no_search_rerank",
        dest="search_rerank",
        action="store_false",
        help="Disable exact reranking for ANN search backends",
    )
    parser.add_argument(
        "--search_hnsw_m",
        type=int,
        default=16,
        help="HNSW graph connectivity parameter M",
    )
    parser.add_argument(
        "--search_hnsw_ef_construction",
        type=int,
        default=200,
        help="HNSW build-time ef parameter",
    )
    parser.add_argument(
        "--search_hnsw_ef_search",
        type=int,
        default=64,
        help="HNSW query-time ef parameter",
    )
    parser.add_argument(
        "--search_ivf_nlist",
        type=int,
        default=100,
        help="Number of FAISS IVF coarse clusters",
    )
    parser.add_argument(
        "--search_ivf_nprobe",
        type=int,
        default=10,
        help="Number of FAISS IVF clusters to probe during search",
    )
    parser.add_argument(
        "--search_pq_m",
        type=int,
        default=16,
        help="Number of product-quantization sub-vectors for FAISS IVFPQ",
    )
    parser.add_argument(
        "--search_pq_bits",
        type=int,
        default=8,
        help="Bits per codebook entry for FAISS IVFPQ",
    )
    parser.add_argument(
        "--search_train_limit",
        type=int,
        default=10000,
        help="Maximum number of descriptors used to train FAISS IVF indexes. Use <=0 to train on all descriptors",
    )
    parser.set_defaults(search_rerank=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parse_args_with_config(parser)
    if not args.mode:
        raise ValueError("--mode is required unless it is provided by the YAML config file")
    args.mode = normalize_mode(args.mode)

    if args.mode == "build_map":
        build_map(args)
        return
    if args.mode == "build_live_map":
        if args.use_video_for_live_build and not args.video:
            raise ValueError("--video is required when using --use_video_for_live_build")
        build_live_map(args, use_video=args.use_video_for_live_build)
        return
    if args.mode == "check_source":
        check_source(args, use_video=bool(args.video))
        return
    if args.mode == "list_sources":
        list_sources(args)
        return
    if args.mode == "save_source_alias":
        save_source_alias_command(args)
        return
    if args.mode == "list_source_aliases":
        list_source_aliases_command()
        return
    if args.mode == "delete_source_alias":
        delete_source_alias_command(args)
        return
    if args.mode == "live":
        run_online(args, use_video=False)
        return
    if args.mode == "video":
        if not args.video:
            raise ValueError("--video is required in video mode")
        run_online(args, use_video=True)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
