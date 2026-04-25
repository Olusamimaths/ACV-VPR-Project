from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - fallback path depends on env
    cv2 = None

from skimage import color, exposure


PreprocessMode = Literal["none", "clahe_query", "clahe_all"]


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def apply_clahe_rgb(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    rgb = _ensure_uint8_rgb(image)
    if cv2 is not None:
        if rgb.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            return clahe.apply(rgb)

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        l_equalized = clahe.apply(l_channel)
        merged = cv2.merge((l_equalized, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    normalized = rgb.astype(np.float32) / 255.0
    if normalized.ndim == 2:
        equalized = exposure.equalize_adapthist(
            normalized,
            clip_limit=min(1.0, clip_limit / 10.0),
            kernel_size=tile_grid_size,
        )
        return np.clip(equalized * 255.0, 0, 255).astype(np.uint8)

    lab = color.rgb2lab(normalized)
    l_channel = lab[..., 0] / 100.0
    l_equalized = exposure.equalize_adapthist(
        l_channel,
        clip_limit=min(1.0, clip_limit / 10.0),
        kernel_size=tile_grid_size,
    )
    lab[..., 0] = l_equalized * 100.0
    rgb_equalized = color.lab2rgb(lab)
    return np.clip(rgb_equalized * 255.0, 0, 255).astype(np.uint8)


def apply_preprocessing(
    imgs_db: list[np.ndarray],
    imgs_q: list[np.ndarray],
    *,
    mode: PreprocessMode = "none",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if mode == "none":
        return imgs_db, imgs_q

    kwargs = {
        "clip_limit": clahe_clip_limit,
        "tile_grid_size": clahe_tile_grid_size,
    }
    if mode == "clahe_query":
        return imgs_db, [apply_clahe_rgb(image, **kwargs) for image in imgs_q]
    if mode == "clahe_all":
        return (
            [apply_clahe_rgb(image, **kwargs) for image in imgs_db],
            [apply_clahe_rgb(image, **kwargs) for image in imgs_q],
        )
    raise ValueError(f"Unsupported preprocessing mode: {mode}")


def preprocessing_suffix(mode: PreprocessMode) -> str:
    return "" if mode == "none" else f"_{mode}"


def preprocess_summary(mode: PreprocessMode, *, clip_limit: float, tile_grid_size: int) -> str:
    if mode == "none":
        return "none"
    return f"{mode} (clip_limit={clip_limit}, tile_grid_size={tile_grid_size})"
