from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for webcam, phone-webcam, and video input. "
            "Install it with `pip install opencv-python` or `pip install -r requirements.txt`."
        ) from exc
    return cv2


def get_source_aliases_path(path: str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    return Path("artifacts/live_vpr_sources.json").expanduser().resolve()


def load_source_aliases(path: str | None = None) -> dict[str, str]:
    aliases_path = get_source_aliases_path(path)
    if not aliases_path.exists():
        return {}

    data = json.loads(aliases_path.read_text())
    aliases = data.get("aliases", {})
    return {str(key): str(value) for key, value in aliases.items()}


def save_source_alias(alias: str, source: str | int, path: str | None = None) -> Path:
    alias = str(alias).strip()
    if not alias:
        raise ValueError("Alias must not be empty")

    aliases_path = get_source_aliases_path(path)
    aliases_path.parent.mkdir(parents=True, exist_ok=True)
    aliases = load_source_aliases(str(aliases_path))
    aliases[alias] = str(source).strip()
    payload = {"aliases": aliases}
    aliases_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return aliases_path


def delete_source_alias(alias: str, path: str | None = None) -> Path:
    aliases_path = get_source_aliases_path(path)
    aliases = load_source_aliases(str(aliases_path))
    aliases.pop(alias, None)
    aliases_path.parent.mkdir(parents=True, exist_ok=True)
    aliases_path.write_text(json.dumps({"aliases": aliases}, indent=2, sort_keys=True) + "\n")
    return aliases_path


def parse_capture_source(source: str | int, aliases_path: str | None = None) -> str | int:
    if isinstance(source, int):
        return source
    source = str(source).strip()
    aliases = load_source_aliases(aliases_path)
    visited: set[str] = set()
    while source in aliases and source not in visited:
        visited.add(source)
        source = aliases[source].strip()
    return int(source) if source.isdigit() else source


def resolve_capture_source(source: str | int, aliases_path: str | None = None) -> dict[str, Any]:
    aliases = load_source_aliases(aliases_path)
    source_str = str(source).strip()
    alias = source_str if source_str in aliases else None
    resolved = parse_capture_source(source, aliases_path=aliases_path)
    return {
        "input_source": source,
        "alias": alias,
        "resolved_source": resolved,
        "aliases_path": str(get_source_aliases_path(aliases_path)),
    }


@dataclass
class OpenCVFrameSource:
    source: str | int
    width: int | None = None
    height: int | None = None
    aliases_path: str | None = None

    def __post_init__(self) -> None:
        self.source = parse_capture_source(self.source, aliases_path=self.aliases_path)
        self.cap: Any | None = None

    def open(self) -> None:
        cv2 = _import_cv2()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open capture source: {self.source}")

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

    def read(self):
        if self.cap is None:
            raise RuntimeError("Capture source is not open")
        ok, frame = self.cap.read()
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "OpenCVFrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def probe_capture_source(source: str | int, width: int | None = None, height: int | None = None) -> dict[str, Any]:
    cv2 = _import_cv2()
    resolved = resolve_capture_source(source)
    capture = OpenCVFrameSource(source=source, width=width, height=height)
    capture.open()
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from source: {source}")
        return {
            "input_source": source,
            "alias": resolved["alias"],
            "source": capture.source,
            "frame_shape": tuple(int(v) for v in frame.shape),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "backend": getattr(capture.cap, "getBackendName", lambda: "unknown")(),
        }
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def list_available_capture_sources(
    max_index: int = 10,
    width: int | None = None,
    height: int | None = None,
    snapshot_dir: str | None = None,
) -> list[dict[str, Any]]:
    cv2 = _import_cv2()
    snapshot_root = None
    if snapshot_dir is not None:
        snapshot_root = Path(snapshot_dir).expanduser().resolve()
        snapshot_root.mkdir(parents=True, exist_ok=True)

    discovered: list[dict[str, Any]] = []
    for index in range(max_index + 1):
        capture = OpenCVFrameSource(source=index, width=width, height=height)
        try:
            capture.open()
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            preview_path = None
            if snapshot_root is not None:
                preview_path = snapshot_root / f"source_{index}.jpg"
                cv2.imwrite(str(preview_path), frame)

            discovered.append(
                {
                    "index": index,
                    "source": capture.source,
                    "frame_shape": tuple(int(v) for v in frame.shape),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "backend": getattr(capture.cap, "getBackendName", lambda: "unknown")(),
                    "preview_path": str(preview_path) if preview_path is not None else None,
                }
            )
        except Exception:
            continue
        finally:
            capture.release()

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    return discovered
