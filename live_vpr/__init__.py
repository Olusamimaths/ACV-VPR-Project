"""Modular live VPR pipeline components."""

from .config import DEFAULT_CONFIG_PATH, load_config_defaults, parse_args_with_config
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
from .search import (
    SUPPORTED_SEARCH_BACKENDS,
    SUPPORTED_SEARCH_METRICS,
    SearchConfig,
    create_search_backend,
)
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
    "DEFAULT_CONFIG_PATH",
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
    "SUPPORTED_SEARCH_BACKENDS",
    "SUPPORTED_SEARCH_METRICS",
    "SearchConfig",
    "VideoRecordingConfig",
    "VideoRecordingResult",
    "create_feature_extractor",
    "create_search_backend",
    "delete_source_alias",
    "get_source_aliases_path",
    "load_config_defaults",
    "load_reference_map",
    "list_available_capture_sources",
    "load_source_aliases",
    "parse_args_with_config",
    "parse_capture_source",
    "probe_capture_source",
    "resolve_capture_source",
    "sample_video_to_frames",
    "save_source_alias",
    "save_reference_map",
]
