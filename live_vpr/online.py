from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np

from .database import ReferenceMap
from .extractors import compute_global_descriptors, create_feature_extractor, describe_extractor_runtime


@dataclass
class LocalizationResult:
    best_match_idx: int
    best_score: float
    recognized: bool
    top_k_indices: list[int]
    top_k_scores: list[float]
    extraction_time_ms: float
    all_similarities: np.ndarray
    is_temporal: bool = False  # True if using temporal aggregation
    buffer_size: int = 0  # For temporal methods


class LiveLocalizer:
    def __init__(
        self,
        reference_map: ReferenceMap,
        descriptor_name: str,
        threshold: float = 0.5,
        top_k: int = 5,
        use_temporal: bool = False,
        temporal_window_size: int = 5,
    ):
        self.reference_map = reference_map
        self.descriptor_name = descriptor_name
        self.threshold = threshold
        self.top_k = top_k
        self.extractor = create_feature_extractor(descriptor_name)
        self.runtime_backend = describe_extractor_runtime(self.extractor)
        
        # Initialize temporal module if VPRTempo and temporal mode enabled
        self.temporal_localizer = None
        if use_temporal and descriptor_name in {"VPRTempo", "VPRTempoQuant"}:
            from .temporal_vprtempo import TemporalVPRTempoLocalizer
            self.temporal_localizer = TemporalVPRTempoLocalizer(
                reference_descriptors=reference_map.descriptors,
                window_size=temporal_window_size,
                aggregation_method="weighted_mean",
            )

    def localize_rgb(self, rgb_image: np.ndarray) -> LocalizationResult:
        start = time.time()
        descriptor = compute_global_descriptors(self.extractor, [rgb_image])
        descriptor = descriptor / (np.linalg.norm(descriptor, axis=1, keepdims=True) + 1e-8)
        extraction_time_ms = (time.time() - start) * 1000.0

        # Use temporal localizer if available
        if self.temporal_localizer is not None:
            best_match_idx, best_score, _ = self.temporal_localizer.localize(
                descriptor.flatten()
            )
            result_info = self.temporal_localizer.get_last_result_info()
            is_temporal = result_info["is_temporal"]
            buffer_size = result_info["buffer_size"]
        else:
            # Standard frame-by-frame localization
            similarities = (self.reference_map.descriptors @ descriptor.T).reshape(-1)
            best_match_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_match_idx])
            is_temporal = False
            buffer_size = 0
            similarities = (self.reference_map.descriptors @ descriptor.T).reshape(-1)

        # Compute full similarities for top-k (use cached if temporal)
        if self.temporal_localizer is None:
            similarities = (self.reference_map.descriptors @ descriptor.T).reshape(-1)
        else:
            similarities = (self.reference_map.descriptors @ descriptor.T).reshape(-1)

        top_k = min(self.top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        return LocalizationResult(
            best_match_idx=best_match_idx,
            best_score=best_score,
            recognized=best_score >= self.threshold,
            top_k_indices=[int(idx) for idx in top_indices.tolist()],
            top_k_scores=[float(score) for score in top_scores.tolist()],
            extraction_time_ms=float(extraction_time_ms),
            all_similarities=similarities,
            is_temporal=is_temporal,
            buffer_size=buffer_size,
        )

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(np.clip(threshold, 0.0, 1.0))

    def reset_temporal(self) -> None:
        """Reset temporal buffer (useful on location jumps)."""
        if self.temporal_localizer is not None:
            self.temporal_localizer.reset()
