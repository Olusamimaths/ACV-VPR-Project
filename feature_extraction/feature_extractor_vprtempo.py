from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from .feature_extractor import FeatureExtractor
from .feature_extractor_torchhub import select_torch_device


def _parse_dims(dims: str | Iterable[int] | None) -> tuple[int, int]:
    if dims is None:
        return (56, 56)
    if isinstance(dims, str):
        parts = [part.strip() for part in dims.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected VPRTempo dims like '56,56', got: {dims}")
        return (int(parts[0]), int(parts[1]))
    dims_list = list(dims)
    if len(dims_list) != 2:
        raise ValueError(f"Expected two VPRTempo dims, got: {dims_list}")
    return (int(dims_list[0]), int(dims_list[1]))


class VPRTempoFeatureExtractor(FeatureExtractor):
    """Adapter for VPRTempo checkpoints.

    VPRTempo is not a universal pretrained global descriptor like SALAD. Its
    released package is built around dataset-specific trained checkpoints. To
    integrate it into this repository's retrieval flow, we expose the hidden
    feature-layer activations from each trained module and concatenate them into
    one global descriptor.
    """

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        dims: str | Iterable[int] | None = None,
        patches: int = 15,
        batch_size: int = 8,
    ):
        try:
            from vprtempo.src.dataset import ProcessImage
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VPRTempo is not installed. Install it with: "
                "`python -m pip install vprtempo==1.1.10`"
            ) from exc

        resolved_checkpoint = checkpoint_path or os.environ.get("VPRTEMPO_MODEL_PATH")
        if not resolved_checkpoint:
            raise ValueError(
                "VPRTempo requires a trained checkpoint. Pass one with "
                "`--vprtempo_model_path` or set the `VPRTEMPO_MODEL_PATH` environment variable."
            )

        self.checkpoint_path = Path(resolved_checkpoint).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"VPRTempo checkpoint not found: {self.checkpoint_path}")

        self.device = select_torch_device()
        self.dims = _parse_dims(dims)
        self.patches = int(patches)
        self.batch_size = int(batch_size)
        self.preprocess = ProcessImage(self.dims, self.patches)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError(
                f"VPRTempo checkpoint at {self.checkpoint_path} did not contain the expected module state dicts."
            )

        self.feature_layers: list[torch.nn.Linear] = []
        feature_dim = 0
        for module_name in sorted(checkpoint):
            state_dict = checkpoint[module_name]
            if not isinstance(state_dict, dict) or "feature_layer.w.weight" not in state_dict:
                raise ValueError(
                    f"Checkpoint module '{module_name}' is missing 'feature_layer.w.weight'."
                )
            weight = state_dict["feature_layer.w.weight"].detach().to(torch.float32)
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight = torch.nn.Parameter(weight)
            layer = layer.to(self.device)
            layer.eval()
            self.feature_layers.append(layer)
            feature_dim += int(weight.shape[0])

        if not self.feature_layers:
            raise ValueError(f"No feature layers could be loaded from {self.checkpoint_path}")
        self.dim = feature_dim

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image shape for VPRTempo: {image.shape}")

        processed = self.preprocess(tensor)
        if getattr(processed, "is_quantized", False):
            processed = processed.dequantize()
        return processed.to(torch.float32)

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        parts = []
        with torch.inference_mode():
            for layer in self.feature_layers:
                parts.append(layer(batch))
        descriptor = torch.cat(parts, dim=1)
        return descriptor.detach().cpu().numpy().astype(np.float32, copy=False)

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        if not imgs:
            return np.empty((0, self.dim), dtype=np.float32)

        descriptors: list[np.ndarray] = []
        for start in range(0, len(imgs), self.batch_size):
            batch_imgs = imgs[start : start + self.batch_size]
            batch = torch.stack([self._preprocess_image(img) for img in batch_imgs], dim=0)
            batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
            descriptors.append(self._forward(batch))
        return np.concatenate(descriptors, axis=0)
