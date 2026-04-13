from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .database import ReferenceMap
from .extractors import compute_global_descriptors, create_feature_extractor


@dataclass
class LocalizationResult:
    best_match_idx: int
    best_score: float
    recognized: bool
    top_k_indices: list[int]
    top_k_scores: list[float]
    extraction_time_ms: float
    all_similarities: np.ndarray


class LiveLocalizer:
    def __init__(
        self,
        reference_map: ReferenceMap,
        descriptor_name: str,
        threshold: float = 0.5,
        top_k: int = 5,
    ):
        self.reference_map = reference_map
        self.descriptor_name = descriptor_name
        self.threshold = threshold
        self.top_k = top_k
        self.extractor = create_feature_extractor(descriptor_name)

    def localize_rgb(self, rgb_image: np.ndarray) -> LocalizationResult:
        start = time.time()
        descriptor = compute_global_descriptors(self.extractor, [rgb_image])
        descriptor = descriptor / (np.linalg.norm(descriptor, axis=1, keepdims=True) + 1e-8)
        extraction_time_ms = (time.time() - start) * 1000.0

        similarities = (self.reference_map.descriptors @ descriptor.T).reshape(-1)
        top_k = min(self.top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        return LocalizationResult(
            best_match_idx=int(top_indices[0]),
            best_score=float(top_scores[0]),
            recognized=float(top_scores[0]) >= self.threshold,
            top_k_indices=[int(idx) for idx in top_indices.tolist()],
            top_k_scores=[float(score) for score in top_scores.tolist()],
            extraction_time_ms=float(extraction_time_ms),
            all_similarities=similarities,
        )

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(np.clip(threshold, 0.0, 1.0))
