"""Modular live VPR pipeline components."""

from .capture import (
    FrameSamplingConfig,
    FrameSamplingResult,
    LiveReferenceRecorder,
    VideoRecordingConfig,
    VideoRecordingResult,
    sample_video_to_frames,
)
from .database import ReferenceMap, load_reference_map, save_reference_map
from .extractors import SUPPORTED_DESCRIPTORS, create_feature_extractor
from .offline import MapBuilder, MapBuildConfig
from .online import LiveLocalizer, LocalizationResult
from .sources import OpenCVFrameSource, parse_capture_source, probe_capture_source
from .sources import (
    delete_source_alias,
    get_source_aliases_path,
    list_available_capture_sources,
    load_source_aliases,
    resolve_capture_source,
    save_source_alias,
)
from .ui import LiveDisplay

__all__ = [
    "FrameSamplingConfig",
    "FrameSamplingResult",
    "LiveDisplay",
    "LiveLocalizer",
    "LiveReferenceRecorder",
    "LocalizationResult",
    "MapBuildConfig",
    "MapBuilder",
    "OpenCVFrameSource",
    "ReferenceMap",
    "SUPPORTED_DESCRIPTORS",
    "VideoRecordingConfig",
    "VideoRecordingResult",
    "create_feature_extractor",
    "delete_source_alias",
    "get_source_aliases_path",
    "load_reference_map",
    "list_available_capture_sources",
    "load_source_aliases",
    "parse_capture_source",
    "probe_capture_source",
    "resolve_capture_source",
    "sample_video_to_frames",
    "save_source_alias",
    "save_reference_map",
]
