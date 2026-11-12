"""Video details and validation (plan Part B step 2). PyAV reads the container; a bad file is refused.

Refusals use the error codes of D1 (media_unsupported, media_too_large, media_undecodable) and the
one new code, media_too_long. Every refusal is a PermanentError: retrying cannot help, so the video
is held and listed in rejected_videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import av
from av.error import FFmpegError

from moderation.config import Settings
from moderation.errors import (
    MEDIA_TOO_LARGE,
    MEDIA_TOO_LONG,
    MEDIA_UNDECODABLE,
    MEDIA_UNSUPPORTED,
    PermanentError,
)

# Container names PyAV reports for the formats we accept (MP4/MOV family, WebM/Matroska family).
ACCEPTED_FORMATS: Final = frozenset({"mov", "mp4", "matroska", "webm"})


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    avg_fps: float  # total frames / total seconds, used for variable-FPS videos too (decision 2)
    width: int
    height: int
    codec: str
    container: str


def probe(path: Path, cfg: Settings) -> VideoInfo:
    """Read and check the file. Raises PermanentError with the code the backend is told."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PermanentError(MEDIA_UNDECODABLE, f"cannot read file: {exc}") from exc
    if size > cfg.video_max_bytes:
        raise PermanentError(MEDIA_TOO_LARGE, f"file exceeds {cfg.video_max_bytes} bytes")
    try:
        container = av.open(str(path), mode="r")
    except (FFmpegError, OSError, ValueError) as exc:
        raise PermanentError(MEDIA_UNDECODABLE, f"{type(exc).__name__}: {exc}") from exc
    with container:
        fmt = container.format.name.split(",")
        if not ACCEPTED_FORMATS.intersection(fmt):
            raise PermanentError(MEDIA_UNSUPPORTED, f"container {container.format.name!r} is not MP4, MOV, WebM or MKV")
        if not container.streams.video:
            raise PermanentError(MEDIA_UNSUPPORTED, "no video stream")
        stream = container.streams.video[0]
        codec = stream.codec_context
        width, height = int(codec.width or 0), int(codec.height or 0)
        if width <= 0 or height <= 0:
            raise PermanentError(MEDIA_UNDECODABLE, "video has no frame size")
        if width * height > cfg.video_max_pixels:
            raise PermanentError(MEDIA_TOO_LARGE, f"frame size {width}x{height} exceeds {cfg.video_max_pixels} pixels")
        duration_s = _duration_seconds(container, stream)
        if duration_s <= 0:
            raise PermanentError(MEDIA_UNDECODABLE, "video has no duration")
        if duration_s > cfg.video_max_duration_s:
            raise PermanentError(MEDIA_TOO_LONG, f"{duration_s:.1f} s is longer than {cfg.video_max_duration_s} s")
        avg_fps = _average_fps(stream, container, duration_s)
        if avg_fps <= 0:
            raise PermanentError(MEDIA_UNDECODABLE, "video has no frame rate")
        return VideoInfo(
            duration_s=duration_s,
            avg_fps=avg_fps,
            width=width,
            height=height,
            codec=codec.name,
            container=container.format.name,
        )


def _duration_seconds(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return 0.0


def _average_fps(stream: av.video.stream.VideoStream, container: av.container.InputContainer, duration_s: float) -> float:
    """Total frames divided by total seconds. Falls back to the stream's rate when the frame count is unknown."""
    if stream.frames and duration_s > 0:
        return stream.frames / duration_s
    rate = stream.average_rate
    return float(rate) if rate is not None else 0.0
