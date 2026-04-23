"""
Temporal VPRTempo module for sequence-based localization.

This module provides temporal windowing capabilities for VPRTempo
to exploit spike timing information across consecutive frames.

Use this for improved robustness in long sequences or videos.
"""

from typing import List, Optional, Tuple
import numpy as np
from collections import deque


class TemporalVPRTempoBuffer:
    """
    Maintains a sliding window of VPRTempo descriptors for temporal encoding.

    This allows VPRTempo to leverage temporal information by aggregating
    spike patterns across multiple consecutive frames.
    """

    def __init__(
        self,
        window_size: int = 5,
        aggregation_method: str = "mean",
    ):
        """
        Initialize temporal buffer.

        Args:
            window_size: Number of consecutive frames to maintain (default: 5)
            aggregation_method: How to combine descriptors ("mean", "max", "weighted_mean")
        """
        self.window_size = window_size
        self.aggregation_method = aggregation_method
        self.descriptor_buffer: deque = deque(maxlen=window_size)
        self.valid = False

    def add_descriptor(self, descriptor: np.ndarray) -> None:
        """
        Add a new descriptor to the buffer.

        Args:
            descriptor: (256,) float32 descriptor from VPRTempo
        """
        if descriptor.ndim != 1:
            raise ValueError(
                f"Expected 1D descriptor, got shape {descriptor.shape}"
            )
        self.descriptor_buffer.append(descriptor)
        self.valid = len(self.descriptor_buffer) == self.window_size

    def get_aggregated(self) -> Optional[np.ndarray]:
        """
        Get temporally aggregated descriptor.

        Returns:
            Aggregated (256,) descriptor if buffer is full, None otherwise
        """
        if not self.valid:
            return None

        descriptors = np.stack(list(self.descriptor_buffer), axis=0)  # (window_size, 256)

        if self.aggregation_method == "mean":
            return np.mean(descriptors, axis=0)
        elif self.aggregation_method == "max":
            return np.max(descriptors, axis=0)
        elif self.aggregation_method == "weighted_mean":
            # Give more weight to recent frames
            weights = np.linspace(0.5, 1.0, self.window_size)
            weights /= weights.sum()
            return np.average(descriptors, axis=0, weights=weights)
        else:
            raise ValueError(
                f"Unknown aggregation method: {self.aggregation_method}"
            )

    def is_ready(self) -> bool:
        """Check if buffer has enough descriptors for aggregation."""
        return self.valid

    def reset(self) -> None:
        """Clear the buffer."""
        self.descriptor_buffer.clear()
        self.valid = False

    def buffer_size(self) -> int:
        """Get current number of descriptors in buffer."""
        return len(self.descriptor_buffer)


class TemporalVPRTempoLocalizer:
    """
    Temporal localizer that uses spike sequences for improved accuracy.

    Maintains a buffer of recent frames and their VPRTempo descriptors
    to exploit temporal correlations for place recognition.
    """

    def __init__(
        self,
        reference_descriptors: np.ndarray,
        window_size: int = 5,
        aggregation_method: str = "weighted_mean",
    ):
        """
        Initialize temporal localizer.

        Args:
            reference_descriptors: (N, 256) normalized reference descriptors
            window_size: Temporal window size (default: 5 frames)
            aggregation_method: Descriptor aggregation strategy
        """
        self.reference_descriptors = reference_descriptors  # (N, 256)
        self.buffer = TemporalVPRTempoBuffer(
            window_size=window_size,
            aggregation_method=aggregation_method,
        )
        self.last_result = None

    def localize(
        self, query_descriptor: np.ndarray
    ) -> Tuple[int, float, Optional[np.ndarray]]:
        """
        Perform temporal localization with query descriptor.

        Args:
            query_descriptor: (256,) float32 descriptor from live frame

        Returns:
            Tuple of:
                - best_match_idx: Index in reference_descriptors
                - best_similarity: Cosine similarity score
                - temporal_descriptor: Aggregated descriptor if ready, else None
        """
        # Add to buffer
        self.buffer.add_descriptor(query_descriptor)

        # Use aggregated descriptor if available, else use current frame
        descriptor_to_match = (
            self.buffer.get_aggregated()
            if self.buffer.is_ready()
            else query_descriptor
        )

        # Normalize
        descriptor_norm = np.linalg.norm(
            descriptor_to_match, keepdims=True
        ) + 1e-8
        descriptor_normalized = descriptor_to_match / descriptor_norm

        # Cosine similarity
        similarities = (
            self.reference_descriptors @ descriptor_normalized
        ).flatten()
        best_match_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_match_idx])

        self.last_result = {
            "best_match_idx": best_match_idx,
            "best_similarity": best_similarity,
            "is_temporal": self.buffer.is_ready(),
            "buffer_size": self.buffer.buffer_size(),
        }

        return best_match_idx, best_similarity, descriptor_to_match

    def get_last_result_info(self) -> Optional[dict]:
        """Get metadata about the last localization result."""
        return self.last_result

    def reset(self) -> None:
        """Reset temporal buffer (e.g., on location jump)."""
        self.buffer.reset()
        self.last_result = None
