from __future__ import annotations

import configparser
import os
from typing import Any

import numpy as np


SUPPORTED_DESCRIPTORS = [
    "HDC-DELF",
    "AlexNet",
    "NetVLAD",
    "PatchNetVLAD",
    "CosPlace",
    "EigenPlaces",
    "SAD",
    "VPRTempo",
    "VPRTempoQuant",
]


def _load_patchnetvlad_config(descriptor_name: str) -> configparser.ConfigParser:
    from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

    if descriptor_name == "NetVLAD":
        config_path = os.path.join(PATCHNETVLAD_ROOT_DIR, "configs/netvlad_extract.ini")
    else:
        config_path = os.path.join(PATCHNETVLAD_ROOT_DIR, "configs/speed.ini")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"PatchNetVLAD config not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def create_feature_extractor(descriptor_name: str) -> Any:
    if descriptor_name not in SUPPORTED_DESCRIPTORS:
        raise ValueError(f"Unsupported descriptor: {descriptor_name}")

    if descriptor_name == "HDC-DELF":
        from feature_extraction.feature_extractor_holistic import HDCDELF

        return HDCDELF()
    if descriptor_name == "AlexNet":
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor

        return AlexNetConv3Extractor()
    if descriptor_name == "SAD":
        from feature_extraction.feature_extractor_holistic import SAD

        return SAD()
    if descriptor_name in {"NetVLAD", "PatchNetVLAD"}:
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor

        return PatchNetVLADFeatureExtractor(_load_patchnetvlad_config(descriptor_name))
    if descriptor_name == "CosPlace":
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor

        return CosPlaceFeatureExtractor()
    if descriptor_name == "EigenPlaces":
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor

        return EigenPlacesFeatureExtractor()
    if descriptor_name == "VPRTempo":
        from feature_extraction.feature_extractor_vprtempo import VPRTempoFeatureExtractor

        return VPRTempoFeatureExtractor(quantized=False)
    if descriptor_name == "VPRTempoQuant":
        from feature_extraction.feature_extractor_vprtempo import VPRTempoFeatureExtractor

        return VPRTempoFeatureExtractor(quantized=True)

    raise ValueError(f"Unsupported descriptor: {descriptor_name}")


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
