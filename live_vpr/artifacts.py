from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import time


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
LATEST_MAP_POINTER = DEFAULT_ARTIFACTS_ROOT / "live_maps" / "latest_map_path.txt"


@dataclass
class ArtifactSession:
    mode: str
    root_dir: Path
    run_name: str

    @property
    def session_dir(self) -> Path:
        return self.root_dir / self.run_name


def _slugify(value: str, *, max_len: int = 36) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    if not text:
        return "session"
    return text[:max_len].rstrip("-")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _extract_source_label(source: str | None) -> str | None:
    if not source:
        return None
    text = str(source).strip()
    if not text:
        return None
    if "://" in text:
        without_scheme = text.split("://", 1)[1]
        host = without_scheme.split("/", 1)[0]
        host = host.split(":", 1)[0]
        return _slugify(host or "stream")
    if text.isdigit():
        return f"cam{text}"
    return _slugify(text)


def _extract_map_label(map_path: str | None) -> str | None:
    if not map_path:
        return None
    return _slugify(Path(map_path).stem)


def _extract_video_label(video_path: str | None) -> str | None:
    if not video_path:
        return None
    return _slugify(Path(video_path).stem)


def _make_run_name(
    *,
    mode: str,
    descriptor: str | None = None,
    source: str | None = None,
    map_path: str | None = None,
    video_path: str | None = None,
    run_name: str | None = None,
) -> str:
    if run_name:
        return _slugify(run_name, max_len=96)

    parts = [_timestamp(), _slugify(mode, max_len=24)]
    if descriptor:
        parts.append(_slugify(descriptor, max_len=20))
    source_label = _extract_source_label(source)
    if source_label:
        parts.append(source_label)
    video_label = _extract_video_label(video_path)
    if video_label:
        parts.append(video_label)
    map_label = _extract_map_label(map_path)
    if map_label:
        parts.append(map_label)
    return "_".join(parts[:5])


def _mode_bucket(mode: str) -> str:
    buckets = {
        "build_map": "map_builds",
        "build_live_map": "live_map_builds",
        "live": "live_localization",
        "video": "video_localization",
        "list_sources": "source_scans",
    }
    return buckets.get(mode, _slugify(mode, max_len=24))


def _is_under_default_artifacts(path_value: str | None) -> bool:
    if not path_value:
        return False

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        return len(candidate.parts) > 0 and candidate.parts[0] == "artifacts"

    try:
        candidate.resolve().relative_to(DEFAULT_ARTIFACTS_ROOT.resolve())
        return True
    except Exception:
        return False


def _rebase_file(path_value: str | None, target_dir: Path, default_name: str) -> str | None:
    if not path_value:
        return None
    if not _is_under_default_artifacts(path_value):
        return str(Path(path_value).expanduser().resolve())

    original = Path(path_value)
    filename = original.name if original.suffix else default_name
    return str((target_dir / filename).resolve())


def _rebase_dir(path_value: str | None, target_dir: Path) -> str | None:
    if not path_value:
        return None
    if not _is_under_default_artifacts(path_value):
        return str(Path(path_value).expanduser().resolve())
    return str(target_dir.resolve())


def prepare_artifact_session(
    *,
    mode: str,
    descriptor: str | None,
    source: str | None,
    map_path: str | None,
    video_path: str | None,
    artifact_root: str | None,
    group_by_run: bool,
    run_name: str | None,
) -> ArtifactSession | None:
    if not group_by_run:
        return None

    root_dir = Path(artifact_root or (DEFAULT_ARTIFACTS_ROOT / "runs")).expanduser()
    if not root_dir.is_absolute():
        root_dir = (REPO_ROOT / root_dir).resolve()
    else:
        root_dir = root_dir.resolve()
    root_dir = root_dir / _mode_bucket(mode)

    return ArtifactSession(
        mode=mode,
        root_dir=root_dir,
        run_name=_make_run_name(
            mode=mode,
            descriptor=descriptor,
            source=source,
            map_path=map_path,
            video_path=video_path,
            run_name=run_name,
        ),
    )


def apply_artifact_session_to_args(args, mode: str, *, use_video: bool = False) -> ArtifactSession | None:
    source_value = str(args.video if use_video else args.source) if getattr(args, "video", None) or getattr(args, "source", None) is not None else None
    session = prepare_artifact_session(
        mode=mode,
        descriptor=getattr(args, "descriptor", None),
        source=source_value,
        map_path=getattr(args, "map_path", None) if mode in {"build_map", "build_live_map"} else None,
        video_path=getattr(args, "video", None),
        artifact_root=getattr(args, "artifact_root", None),
        group_by_run=bool(getattr(args, "group_artifacts_by_run", True)),
        run_name=getattr(args, "run_name", None),
    )
    if session is None:
        return None

    session_dir = session.session_dir

    if mode == "build_map":
        original_map_path = str(Path(args.map_path).expanduser().resolve())
        map_path = _rebase_file(args.map_path, session_dir / "maps", "reference_map.npz")
        if map_path == original_map_path:
            return None
        session_dir.mkdir(parents=True, exist_ok=True)
        setattr(args, "_artifact_requested_map_path", original_map_path)
        args.map_path = map_path
        return session

    if mode == "build_live_map":
        original_map_path = str(Path(args.map_path).expanduser().resolve())
        map_path = _rebase_file(args.map_path, session_dir / "maps", "reference_map.npz")
        recording_path = _rebase_file(args.recording_path, session_dir / "recordings", "traversal.mp4")
        capture_dir = _rebase_dir(args.capture_dir, session_dir / "reference_frames")
        changed = (
            map_path != original_map_path
            or recording_path != str(Path(args.recording_path).expanduser().resolve())
            or capture_dir != str(Path(args.capture_dir).expanduser().resolve())
        )
        if not changed:
            return None
        session_dir.mkdir(parents=True, exist_ok=True)
        setattr(args, "_artifact_requested_map_path", original_map_path)
        args.map_path = map_path
        args.recording_path = recording_path
        args.capture_dir = capture_dir
        return session

    if mode == "live":
        original_inference_stats_dir = str(Path(args.inference_stats_dir).expanduser().resolve())
        original_snapshot_dir = str(Path(args.snapshot_dir).expanduser().resolve())
        original_output_video = str(Path(args.output_video).expanduser().resolve()) if args.output_video else None
        inference_stats_dir = _rebase_dir(args.inference_stats_dir, session_dir / "inference_stats")
        snapshot_dir = _rebase_dir(args.snapshot_dir, session_dir / "captures")
        output_video = _rebase_file(args.output_video, session_dir / "videos", "annotated_session.mp4")
        changed = (
            inference_stats_dir != original_inference_stats_dir
            or snapshot_dir != original_snapshot_dir
            or output_video != original_output_video
        )
        if not changed:
            return None
        session_dir.mkdir(parents=True, exist_ok=True)
        args.inference_stats_dir = inference_stats_dir
        args.snapshot_dir = snapshot_dir
        args.output_video = output_video
        return session

    if mode == "video":
        original_inference_stats_dir = str(Path(args.inference_stats_dir).expanduser().resolve())
        original_snapshot_dir = str(Path(args.snapshot_dir).expanduser().resolve())
        original_output_video = str(Path(args.output_video).expanduser().resolve()) if args.output_video else None
        inference_stats_dir = _rebase_dir(args.inference_stats_dir, session_dir / "inference_stats")
        snapshot_dir = _rebase_dir(args.snapshot_dir, session_dir / "captures")
        output_video = _rebase_file(args.output_video, session_dir / "videos", "annotated_session.mp4")
        changed = (
            inference_stats_dir != original_inference_stats_dir
            or snapshot_dir != original_snapshot_dir
            or output_video != original_output_video
        )
        if not changed:
            return None
        session_dir.mkdir(parents=True, exist_ok=True)
        args.inference_stats_dir = inference_stats_dir
        args.snapshot_dir = snapshot_dir
        args.output_video = output_video
        return session

    if mode == "list_sources":
        source_snapshot_dir = _rebase_dir(args.source_snapshot_dir, session_dir / "source_previews")
        if source_snapshot_dir != str(Path(args.source_snapshot_dir).expanduser().resolve()):
            session_dir.mkdir(parents=True, exist_ok=True)
            args.source_snapshot_dir = source_snapshot_dir
            return session
        return None

    return session


def publish_built_reference_map(saved_map_path: str | Path, stable_map_path: str | Path | None = None) -> tuple[Path, Path | None]:
    saved = Path(saved_map_path).expanduser().resolve()
    stable: Path | None = None

    if stable_map_path is not None:
        stable = Path(stable_map_path).expanduser().resolve()
        if stable != saved:
            stable.parent.mkdir(parents=True, exist_ok=True)
            if stable.exists() or stable.is_symlink():
                stable.unlink()
            # Keep a stable concrete copy under artifacts/live_maps so later
            # run-folder reorganization does not break inference.
            shutil.copy2(saved, stable)

    LATEST_MAP_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_MAP_POINTER.write_text(str(stable or saved), encoding="utf-8")
    return saved, stable


def find_latest_built_map() -> Path | None:
    runs_root = DEFAULT_ARTIFACTS_ROOT / "runs"
    candidates = [path.resolve() for path in runs_root.glob("**/maps/*.npz") if path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def resolve_reference_map_input(map_path: str | Path) -> Path:
    requested = Path(map_path).expanduser()
    resolved = requested.resolve()
    if resolved.exists():
        return resolved

    if _is_under_default_artifacts(str(map_path)) and LATEST_MAP_POINTER.exists():
        latest = Path(LATEST_MAP_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()
        if latest.exists():
            return latest

    if _is_under_default_artifacts(str(map_path)):
        latest_built = find_latest_built_map()
        if latest_built is not None and latest_built.exists():
            return latest_built

    return resolved
