#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRYPOINT="$ROOT_DIR/live_vpr_test.py"

DEFAULT_DESCRIPTOR="${VPR_DESCRIPTOR:-CosPlace}"
DEFAULT_MAP_PATH="${VPR_MAP_PATH:-artifacts/live_maps/campus_day_cosplace.npz}"
DEFAULT_REFERENCE_DIR="${VPR_REFERENCE_DIR:-custom_dataset/day_images}"
DEFAULT_SOURCE="${VPR_SOURCE:-0}"
DEFAULT_THRESHOLD="${VPR_THRESHOLD:-0.50}"
DEFAULT_PROCESS_FPS="${VPR_PROCESS_FPS:-2.0}"
DEFAULT_SAMPLE_FPS="${VPR_SAMPLE_FPS:-1.0}"
DEFAULT_RECORDING_PATH="${VPR_RECORDING_PATH:-artifacts/reference_videos/live_reference_traversal.mp4}"
DEFAULT_CAPTURE_DIR="${VPR_CAPTURE_DIR:-artifacts/reference_captures/live_reference_traversal}"

usage() {
  cat <<EOF
Usage:
  bash scripts/live_vpr_cli.sh <command> [extra args...]

Commands:
  build-map        Build a map from an existing reference-image folder
  record-map       Record a traversal video from a live source, sample it, and build a map
  video-map        Build a map from an existing traversal video
  live             Run live localization from a webcam, phone webcam, or stream
  video            Run localization on a recorded video
  check-source     Check whether a webcam, virtual webcam, or stream is accessible
  list-sources     Probe a range of numeric camera indexes to find available cameras
  save-source-alias  Save a friendly alias like 'phone' for a source
  list-source-aliases List the saved source aliases
  delete-source-alias Remove a saved source alias
  help             Show this help

Environment variable defaults:
  PYTHON_BIN           Python executable to use
  VPR_DESCRIPTOR       Descriptor name, default: ${DEFAULT_DESCRIPTOR}
  VPR_MAP_PATH         Map output/input path, default: ${DEFAULT_MAP_PATH}
  VPR_REFERENCE_DIR    Folder-based reference image directory, default: ${DEFAULT_REFERENCE_DIR}
  VPR_SOURCE           Webcam index or stream URL, default: ${DEFAULT_SOURCE}
  VPR_THRESHOLD        Live recognition threshold, default: ${DEFAULT_THRESHOLD}
  VPR_PROCESS_FPS      Live inference cadence, default: ${DEFAULT_PROCESS_FPS}
  VPR_SAMPLE_FPS       Frame sampling rate for video-first map building, default: ${DEFAULT_SAMPLE_FPS}
  VPR_RECORDING_PATH   Traversal video path, default: ${DEFAULT_RECORDING_PATH}
  VPR_CAPTURE_DIR      Sampled frame directory, default: ${DEFAULT_CAPTURE_DIR}
  VPR_VIDEO_PATH       Video path used by the 'video' and 'video-map' commands

Examples:
  bash scripts/live_vpr_cli.sh build-map
  bash scripts/live_vpr_cli.sh record-map --source 0 --mirror
  bash scripts/live_vpr_cli.sh list-sources --source_scan_max 10
  bash scripts/live_vpr_cli.sh save-source-alias --alias phone --source 3
  bash scripts/live_vpr_cli.sh video-map --video recordings/campus_walk.mp4
  bash scripts/live_vpr_cli.sh live --source phone
  bash scripts/live_vpr_cli.sh check-source --source http://192.168.1.20:4747/video
EOF
}

require_video_path() {
  if [[ -z "${VPR_VIDEO_PATH:-}" ]]; then
    echo "Tip: set VPR_VIDEO_PATH or pass --video <path> in the extra args." >&2
  fi
}

video_arg=()
if [[ -n "${VPR_VIDEO_PATH:-}" ]]; then
  video_arg=(--video "$VPR_VIDEO_PATH")
fi

command="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command" in
  build-map)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode build_map \
      --descriptor "$DEFAULT_DESCRIPTOR" \
      --data_dir "$DEFAULT_REFERENCE_DIR" \
      --map_path "$DEFAULT_MAP_PATH" \
      "$@"
    ;;
  record-map)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode build_live_map \
      --descriptor "$DEFAULT_DESCRIPTOR" \
      --map_path "$DEFAULT_MAP_PATH" \
      --source "$DEFAULT_SOURCE" \
      --recording_path "$DEFAULT_RECORDING_PATH" \
      --capture_dir "$DEFAULT_CAPTURE_DIR" \
      --sample_fps "$DEFAULT_SAMPLE_FPS" \
      "$@"
    ;;
  video-map)
    require_video_path
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode build_live_map \
      --descriptor "$DEFAULT_DESCRIPTOR" \
      --map_path "$DEFAULT_MAP_PATH" \
      --use_video_for_live_build \
      --capture_dir "$DEFAULT_CAPTURE_DIR" \
      --sample_fps "$DEFAULT_SAMPLE_FPS" \
      "${video_arg[@]}" \
      "$@"
    ;;
  live)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode live \
      --descriptor "$DEFAULT_DESCRIPTOR" \
      --map_path "$DEFAULT_MAP_PATH" \
      --source "$DEFAULT_SOURCE" \
      --threshold "$DEFAULT_THRESHOLD" \
      --process_fps "$DEFAULT_PROCESS_FPS" \
      "$@"
    ;;
  video)
    require_video_path
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode video \
      --descriptor "$DEFAULT_DESCRIPTOR" \
      --map_path "$DEFAULT_MAP_PATH" \
      --threshold "$DEFAULT_THRESHOLD" \
      --process_fps "$DEFAULT_PROCESS_FPS" \
      "${video_arg[@]}" \
      "$@"
    ;;
  check-source)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode check_source \
      --source "$DEFAULT_SOURCE" \
      "$@"
    ;;
  list-sources)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode list_sources \
      "$@"
    ;;
  save-source-alias)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode save_source_alias \
      "$@"
    ;;
  list-source-aliases)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode list_source_aliases \
      "$@"
    ;;
  delete-source-alias)
    exec "$PYTHON_BIN" "$ENTRYPOINT" \
      --mode delete_source_alias \
      "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 1
    ;;
esac
