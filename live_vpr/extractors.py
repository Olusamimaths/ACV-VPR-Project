from __future__ import annotations

from typing import Any

import numpy as np

from feature_extraction.factory import SUPPORTED_DESCRIPTORS, create_feature_extractor


def describe_extractor_runtime(extractor: Any) -> str:
    device = getattr(extractor, "device", None)
    if device is not None:
        return str(device)
    return "cpu"


def compute_global_descriptors(extractor: Any, images: list[np.ndarray]) -> np.ndarray:
    descriptors = extractor.compute_features(images)

    if isinstance(descriptors, tuple):
        descriptors = descriptors[0]

    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2:
        raise ValueError(
            f"Expected 2D global descriptors for live localization, got shape {descriptors.shape}"
        )
    return descriptors
