from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

from .sources import OpenCVFrameSource, _import_cv2


@dataclass
class VideoRecordingConfig:
    source: str | int
    output_video: str
    window_name: str = "Live VPR Map Recorder"
    frame_width: int | None = None
    frame_height: int | None = None
    mirror: bool = False
    start_recording: bool = False


@dataclass
class VideoRecordingResult:
    video_path: str
    frame_count: int
    duration_s: float
    stopped_by_user: bool


@dataclass
class FrameSamplingConfig:
    video_path: str
    output_dir: str
    sample_fps: float = 1.0
    frame_prefix: str = "ref"
    max_frames: int | None = None


@dataclass
class FrameSamplingResult:
    saved_paths: list[str]
    output_dir: str
    sample_fps: float
    source_fps: float
    video_duration_s: float


class LiveReferenceRecorder:
    def __init__(self, config: VideoRecordingConfig):
        self.config = config
        self.output_video = Path(config.output_video).expanduser().resolve()
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        self.source = OpenCVFrameSource(
            config.source,
            width=config.frame_width,
            height=config.frame_height,
        )

    def run(self) -> VideoRecordingResult:
        cv2 = _import_cv2()
        writer = None
        frame_count = 0
        recording = self.config.start_recording
        started_at = time.time()
        stopped_by_user = False

        print(f"Capture source: {self.config.source}")
        print(f"Recording traversal video to: {self.output_video}")
        print("Controls: r=start/pause recording, q=stop and build map")

        try:
            self.source.open()
            source_fps = getattr(self.source.cap, "get", lambda *_: 0.0)(cv2.CAP_PROP_FPS)
            if not source_fps or math.isnan(source_fps) or source_fps <= 0:
                source_fps = 20.0

            while True:
                ok, frame = self.source.read()
                if not ok or frame is None:
                    print("Failed to read frame from capture source.")
                    break

                if writer is None:
                    writer = cv2.VideoWriter(
                        str(self.output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(source_fps),
                        (frame.shape[1], frame.shape[0]),
                    )

                if recording:
                    writer.write(frame)
                    frame_count += 1

                status_frame = self._render_overlay(frame, recording, frame_count)
                if self.config.mirror:
                    status_frame = cv2.flip(status_frame, 1)

                cv2.imshow(self.config.window_name, status_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    stopped_by_user = True
                    break
                if key == ord("r"):
                    recording = not recording
                    state = "ON" if recording else "PAUSED"
                    print(f"Recording: {state}")

            duration_s = time.time() - started_at
            return VideoRecordingResult(
                video_path=str(self.output_video),
                frame_count=frame_count,
                duration_s=duration_s,
                stopped_by_user=stopped_by_user,
            )
        finally:
            self.source.release()
            if writer is not None:
                writer.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _render_overlay(self, frame, recording: bool, frame_count: int):
        cv2 = _import_cv2()
        display = frame.copy()
        state = "RECORDING" if recording else "PAUSED"
        state_color = (0, 220, 0) if recording else (0, 170, 255)
        cv2.rectangle(display, (0, 0), (display.shape[1], 78), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"Traversal recorder  state={state}  frames={frame_count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            state_color,
            2,
        )
        cv2.putText(
            display,
            "r=start/pause recording  q=stop and build map",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
        return display


def sample_video_to_frames(config: FrameSamplingConfig) -> FrameSamplingResult:
    cv2 = _import_cv2()
    video_path = Path(config.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Recorded traversal video not found: {video_path}")

    if config.sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")

    saved_paths: list[str] = []
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps or math.isnan(source_fps) or source_fps <= 0:
            source_fps = 20.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_s = frame_count / source_fps if frame_count > 0 else 0.0

        sample_interval_s = 1.0 / config.sample_fps
        next_sample_time_s = 0.0
        current_frame_idx = 0

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            current_time_s = current_frame_idx / source_fps
            should_sample = current_time_s + (0.5 / source_fps) >= next_sample_time_s
            if should_sample:
                timestamp_ms = int(round(current_time_s * 1000.0))
                filename = f"{config.frame_prefix}_{len(saved_paths):04d}_{timestamp_ms:08d}ms.jpg"
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(str(output_path))
                next_sample_time_s += sample_interval_s

                if config.max_frames is not None and len(saved_paths) >= config.max_frames:
                    break

            current_frame_idx += 1

        return FrameSamplingResult(
            saved_paths=saved_paths,
            output_dir=str(output_dir),
            sample_fps=float(config.sample_fps),
            source_fps=float(source_fps),
            video_duration_s=float(video_duration_s),
        )
    finally:
        capture.release()
