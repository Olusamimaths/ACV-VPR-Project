from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .database import (
    ReferenceMap,
    build_map_metadata,
    list_image_paths,
    normalize_descriptors,
    save_reference_map,
)
from .extractors import compute_global_descriptors, create_feature_extractor


@dataclass
class MapBuildConfig:
    image_dir: str
    output_path: str
    descriptor: str = "CosPlace"
    target_size: tuple[int, int] = (640, 480)
    recursive: bool = False


def _load_rgb_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(image)


class MapBuilder:
    def __init__(self, descriptor_name: str):
        self.descriptor_name = descriptor_name
        self.extractor = create_feature_extractor(descriptor_name)

    def build(self, config: MapBuildConfig) -> tuple[ReferenceMap, dict[str, float]]:
        image_paths = list_image_paths(config.image_dir, recursive=config.recursive)
        return self.build_from_paths(
            image_paths=image_paths,
            output_path=config.output_path,
            image_dir=config.image_dir,
            target_size=config.target_size,
            recursive=config.recursive,
        )

    def build_from_paths(
        self,
        image_paths: list[Path],
        output_path: str,
        image_dir: str,
        target_size: tuple[int, int],
        recursive: bool = False,
        metadata_extra: dict | None = None,
    ) -> tuple[ReferenceMap, dict[str, float]]:
        if not image_paths:
            raise FileNotFoundError(f"No supported image files found in {image_dir}")

        load_start = time.time()
        images = [_load_rgb_image(path, target_size) for path in image_paths]
        load_time = time.time() - load_start

        extract_start = time.time()
        descriptors = compute_global_descriptors(self.extractor, images)
        descriptors = normalize_descriptors(descriptors)
        extraction_time = time.time() - extract_start

        metadata = build_map_metadata(
            descriptor_name=self.descriptor_name,
            image_dir=image_dir,
            target_size=target_size,
            extra={
                "num_images": len(image_paths),
                "descriptor_dim": int(descriptors.shape[1]),
                "recursive": bool(recursive),
                **(metadata_extra or {}),
            },
        )
        reference_map = ReferenceMap(
            descriptors=descriptors,
            image_paths=[str(path.resolve()) for path in image_paths],
            metadata=metadata,
        )
        saved_path = save_reference_map(reference_map, output_path)
        reference_map.metadata["map_path"] = str(saved_path)

        stats = {
            "load_time_s": load_time,
            "extraction_time_s": extraction_time,
            "total_time_s": load_time + extraction_time,
            "avg_extraction_ms": (extraction_time / len(images)) * 1000.0,
        }
        return reference_map, stats
