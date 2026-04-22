from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .database import ReferenceMap
from .extractors import compute_global_descriptors, create_feature_extractor, describe_extractor_runtime
from .search import SearchConfig, create_search_backend


@dataclass
class LocalizationResult:
    best_match_idx: int
    best_score: float
    recognized: bool
    top_k_indices: list[int]
    top_k_scores: list[float]
    extraction_time_ms: float
    all_similarities: np.ndarray | None


class LiveLocalizer:
    def __init__(
        self,
        reference_map: ReferenceMap,
        descriptor_name: str,
        threshold: float = 0.5,
        top_k: int = 5,
        search_config: SearchConfig | None = None,
    ):
        self.reference_map = reference_map
        self.descriptor_name = descriptor_name
        self.threshold = threshold
        self.top_k = top_k
        self.search_config = search_config or SearchConfig()
        self.extractor = create_feature_extractor(descriptor_name)
        self.runtime_backend = describe_extractor_runtime(self.extractor)
        self.search_backend = create_search_backend(reference_map.descriptors, self.search_config)

    def localize_rgb(self, rgb_image: np.ndarray) -> LocalizationResult:
        start = time.time()
        descriptor = compute_global_descriptors(self.extractor, [rgb_image])
        descriptor = descriptor / (np.linalg.norm(descriptor, axis=1, keepdims=True) + 1e-8)
        extraction_time_ms = (time.time() - start) * 1000.0

        search_result = self.search_backend.search(descriptor[0], top_k=self.top_k)
        if not search_result.indices:
            raise RuntimeError("Search backend returned no localization candidates.")

        return LocalizationResult(
            best_match_idx=int(search_result.indices[0]),
            best_score=float(search_result.scores[0]),
            recognized=float(search_result.scores[0]) >= self.threshold,
            top_k_indices=[int(idx) for idx in search_result.indices],
            top_k_scores=[float(score) for score in search_result.scores],
            extraction_time_ms=float(extraction_time_ms),
            all_similarities=search_result.all_scores,
        )

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(np.clip(threshold, 0.0, 1.0))
