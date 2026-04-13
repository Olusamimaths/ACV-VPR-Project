from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class ReferenceMap:
    descriptors: np.ndarray
    image_paths: list[str]
    metadata: dict[str, Any]

    @property
    def descriptor_dim(self) -> int:
        return int(self.descriptors.shape[1])

    @property
    def num_images(self) -> int:
        return int(self.descriptors.shape[0])


def normalize_descriptors(descriptors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / (norms + eps)


def list_image_paths(image_dir: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(image_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    if recursive:
        paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    return sorted(paths)


def build_map_metadata(
    descriptor_name: str,
    image_dir: str | Path,
    target_size: tuple[int, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "descriptor": descriptor_name,
        "target_size": [int(target_size[0]), int(target_size[1])],
        "source_dir": str(Path(image_dir).expanduser().resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_reference_map(reference_map: ReferenceMap, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        descriptors=reference_map.descriptors.astype(np.float32),
        image_paths=np.array(reference_map.image_paths),
        metadata=json.dumps(reference_map.metadata),
    )
    return output


def load_reference_map(map_path: str | Path) -> ReferenceMap:
    path = Path(map_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference map not found: {path}")

    data = np.load(path, allow_pickle=True)
    descriptors = np.asarray(data["descriptors"], dtype=np.float32)
    image_paths = [str(path_str) for path_str in data["image_paths"].tolist()]
    metadata_raw = data["metadata"]
    metadata = json.loads(str(metadata_raw))

    return ReferenceMap(descriptors=descriptors, image_paths=image_paths, metadata=metadata)
