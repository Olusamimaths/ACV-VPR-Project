from __future__ import annotations

import os
from typing import Any, Iterable, List

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from .feature_extractor import FeatureExtractor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def select_torch_device(
    *,
    allow_mps: bool = True,
    mps_disabled_reason: str | None = None,
) -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS")
        return torch.device("mps")
    if not allow_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        if mps_disabled_reason:
            print(f"Using CPU ({mps_disabled_reason})")
        else:
            print("Using CPU")
        return torch.device("cpu")
    print("Using CPU")
    return torch.device("cpu")


def _resolve_interpolation(interpolation: str | InterpolationMode) -> InterpolationMode:
    if isinstance(interpolation, InterpolationMode):
        return interpolation

    normalized = interpolation.lower()
    if normalized == "bicubic":
        return InterpolationMode.BICUBIC
    if normalized == "nearest":
        return InterpolationMode.NEAREST
    return InterpolationMode.BILINEAR


def build_rgb_transform(
    *,
    resize: int | tuple[int, int],
    interpolation: str | InterpolationMode = InterpolationMode.BILINEAR,
    mean: Iterable[float] = IMAGENET_MEAN,
    std: Iterable[float] = IMAGENET_STD,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(resize, interpolation=_resolve_interpolation(interpolation)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


class TorchHubImageDataset(data.Dataset):
    def __init__(self, images: list[np.ndarray], transform: transforms.Compose):
        super().__init__()
        self.images = images
        self.transform = transform

    def __getitem__(self, index: int):
        return self.transform(self.images[index]), index

    def __len__(self) -> int:
        return len(self.images)


class TorchHubGlobalFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        *,
        repo: str,
        hub_model: str,
        hub_kwargs: dict[str, Any] | None = None,
        dim: int | None = None,
        resize: int | tuple[int, int] = 480,
        interpolation: str | InterpolationMode = InterpolationMode.BILINEAR,
        small_batch_size: int = 8,
        large_batch_size: int = 8,
        missing_dependency_message: str | None = None,
        allow_mps: bool = True,
        mps_disabled_reason: str | None = None,
    ):
        self.device = select_torch_device(
            allow_mps=allow_mps,
            mps_disabled_reason=mps_disabled_reason,
        )
        self.dim = dim
        self.small_batch_size = small_batch_size
        self.large_batch_size = large_batch_size
        self.preprocess = build_rgb_transform(resize=resize, interpolation=interpolation)
        try:
            self.model = torch.hub.load(repo, hub_model, **(hub_kwargs or {}))
        except ModuleNotFoundError as exc:
            if missing_dependency_message is not None:
                raise ModuleNotFoundError(missing_dependency_message) from exc
            raise
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.dim is None:
            self.dim = self._infer_dim()

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            encoding = self.model(batch)

        if isinstance(encoding, tuple):
            encoding = encoding[0]

        encoding = encoding.detach().cpu().numpy().astype(np.float32, copy=False)
        if encoding.ndim != 2:
            raise ValueError(f"Expected 2D descriptor output, got shape {encoding.shape}")
        return encoding

    def _infer_dim(self) -> int:
        sample = np.zeros((32, 32, 3), dtype=np.uint8)
        batch = self.preprocess(sample).unsqueeze(0).to(self.device)
        return int(self._forward(batch).shape[1])

    def _compute_features_small_batch(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim or 0), dtype=np.float32)

        batch = torch.stack([self.preprocess(image) for image in images], dim=0)
        batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
        return self._forward(batch)

    def compute_features(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim or 0), dtype=np.float32)

        if len(images) <= self.small_batch_size:
            return self._compute_features_small_batch(images)

        dataset = TorchHubImageDataset(images, self.preprocess)
        num_workers = 0 if len(dataset) < 32 else min(4, os.cpu_count() or 1)
        batch_size = min(self.large_batch_size, len(dataset))
        loader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        show_progress = len(dataset) >= batch_size * 4
        iterator = tqdm(loader) if show_progress else loader

        descriptors = np.empty((len(dataset), self.dim or 0), dtype=np.float32)
        for batch, indices in iterator:
            batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
            descriptors[indices.numpy(), :] = self._forward(batch)
        return descriptors
