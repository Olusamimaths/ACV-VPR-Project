from __future__ import annotations

import configparser
import os
from typing import Any


PATCH_DESCRIPTOR_NAMES = {"PatchNetVLAD"}
PAIRWISE_DISTANCE_DESCRIPTOR_NAMES = {"SAD"}

SUPPORTED_DESCRIPTORS = [
    "HDC-DELF",
    "AlexNet",
    "NetVLAD",
    "PatchNetVLAD",
    "CosPlace",
    "EigenPlaces",
    "SALAD",
    "SAD",
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
    if descriptor_name == "NetVLAD":
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor

        return PatchNetVLADFeatureExtractor(_load_patchnetvlad_config(descriptor_name))
    if descriptor_name in PATCH_DESCRIPTOR_NAMES:
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor

        return PatchNetVLADFeatureExtractor(_load_patchnetvlad_config(descriptor_name))
    if descriptor_name == "CosPlace":
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor

        return CosPlaceFeatureExtractor()
    if descriptor_name == "EigenPlaces":
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor

        return EigenPlacesFeatureExtractor()
    if descriptor_name == "SALAD":
        from feature_extraction.feature_extractor_salad import SALADFeatureExtractor

        return SALADFeatureExtractor()

    raise ValueError(f"Unsupported descriptor: {descriptor_name}")
