"""Frame selection (plan section 4). One pass over the video: sample, scene change, duplicates, cap.

Per frame j, the decision uses only grey 64x64 thumbnails (for the comparisons) and the frame itself
(converted to colour only when it is kept):

  1. Sample: keep frames where j % N == 0, N = round(average FPS / VIDEO_SAMPLE_FPS).
  2. Scene change: the mean grey difference between frame j and the frame 2 before, or 2 after
     (plan section 4.3: "compares frames within 2 frames before and after"), is above
     VIDEO_SCENE_THRESHOLD. A scene frame is kept even if it is not a sample frame.
  3. Duplicates: an 8x8 average-brightness fingerprint (64 bits). A candidate whose fingerprint is
     within VIDEO_DEDUP_DISTANCE bits of the last kept frame is dropped.
  4. Cap: above VIDEO_MAX_FRAMES, scene frames are kept first and the rest are thinned evenly.

A frame is decided once the frame 2 after it has been decoded (or at the end of the video), so the
video is read once. Kept frames are stored as JPEG bytes at VIDEO_FRAME_SIDE on the longest side.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import av
import numpy as np
from av.error import FFmpegError
from PIL import Image

from moderation.config import Settings
from moderation.errors import MEDIA_UNDECODABLE, PermanentError
from moderation.video.probe import VideoInfo

WINDOW: Final = 2  # a frame is compared with the frames this far before and after it
THUMB_SIDE: Final = 64  # grey thumbnail used for scene and duplicate checks
HASH_SIDE: Final = 8  # 8 x 8 = 64-bit fingerprint
JPEG_QUALITY: Final = 95  # one extra JPEG pass on top of the video's own compression


@dataclass(frozen=True)
class Frame:
    frame_no: int  # index of the decoded frame in the video (0-based)
    time_ms: int  # position in the video
    source: Literal["sample", "scene"]
    jpeg: bytes  # RGB, longest side = VIDEO_FRAME_SIDE, kept in memory only


@dataclass(frozen=True)
class FrameSelection:
    frames: tuple[Frame, ...]  # in time order, at most VIDEO_MAX_FRAMES
    decoded: int  # frames read from the video
    sampled: int  # frames that were sample frames
    scene: int  # frames that were scene changes
    duplicates: int  # candidates dropped as near-copies
    candidates: int  # candidates before the cap


def select_frames(path: Path, cfg: Settings, info: VideoInfo) -> FrameSelection:
    """Decode the video once and keep the frames the checks need. Raises PermanentError on a corrupt file."""
    step = max(1, round(info.avg_fps / cfg.video_sample_fps))
    out_w, out_h = _target_size(info.width, info.height, cfg.video_frame_side)
    thumbs: list[np.ndarray] = []
    times: list[float | None] = []
    raw_recent: dict[int, av.VideoFrame] = {}
    candidates: list[Frame] = []
    counts = {"sampled": 0, "scene": 0, "duplicates": 0}
    last_kept: tuple[int, np.ndarray] | None = None  # (fingerprint, thumbnail) of the last frame kept
    next_to_decide = 0

    def decide(j: int) -> None:
        nonlocal last_kept
        before = thumbs[j - WINDOW] if j - WINDOW >= 0 else None
        after = thumbs[j + WINDOW] if j + WINDOW < len(thumbs) else None
        is_scene = any(
            other is not None and scene_difference(thumbs[j], other) > cfg.video_scene_threshold for other in (before, after)
        )
        is_sample = j % step == 0
        if not (is_scene or is_sample):
            raw_recent.pop(j, None)
            return
        if is_sample:
            counts["sampled"] += 1
        if is_scene:
            counts["scene"] += 1
        fingerprint = average_hash(thumbs[j])
        if last_kept is not None and _is_duplicate(fingerprint, thumbs[j], last_kept, cfg):
            counts["duplicates"] += 1
            raw_recent.pop(j, None)
            return
        last_kept = (fingerprint, thumbs[j])
        raw = raw_recent.pop(j)
        candidates.append(
            Frame(
                frame_no=j,
                time_ms=_time_ms(times[j], j, info.avg_fps),
                source="scene" if is_scene else "sample",
                jpeg=_to_jpeg(raw, out_w, out_h),
            )
        )

    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for raw in container.decode(stream):
                i = len(thumbs)
                thumbs.append(_thumbnail(raw))
                times.append(raw.time)
                raw_recent[i] = raw
                while next_to_decide <= i - WINDOW:
                    decide(next_to_decide)
                    next_to_decide += 1
    except FFmpegError as exc:
        raise PermanentError(MEDIA_UNDECODABLE, f"video is corrupt: {type(exc).__name__}: {exc}") from exc
    if not thumbs:
        raise PermanentError(MEDIA_UNDECODABLE, "no frames could be decoded")
    while next_to_decide < len(thumbs):
        decide(next_to_decide)
        next_to_decide += 1

    chosen = cap_frames(candidates, cfg.video_max_frames)
    return FrameSelection(
        frames=tuple(chosen),
        decoded=len(thumbs),
        sampled=counts["sampled"],
        scene=counts["scene"],
        duplicates=counts["duplicates"],
        candidates=len(candidates),
    )


def cap_frames(candidates: Sequence[Frame], max_frames: int) -> list[Frame]:
    """At most max_frames. Scene frames are kept first; the rest are thinned evenly. Time order kept."""
    if len(candidates) <= max_frames:
        return list(candidates)
    scenes = [f for f in candidates if f.source == "scene"]
    if len(scenes) >= max_frames:
        return _evenly(scenes, max_frames)
    samples = [f for f in candidates if f.source == "sample"]
    chosen = [*scenes, *_evenly(samples, max_frames - len(scenes))]
    return sorted(chosen, key=lambda f: f.frame_no)


def _evenly[T](items: Sequence[T], k: int) -> list[T]:
    """k items spread evenly over the sequence, first and last included. k < len(items)."""
    if k <= 0:
        return []
    if k == 1:
        return [items[0]]
    last = len(items) - 1
    return [items[round(i * last / (k - 1))] for i in range(k)]


def scene_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute grey difference, 0 to 255."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def average_hash(thumb: np.ndarray) -> int:
    """64-bit fingerprint: each 8x8 block is 1 if brighter than the average block, else 0."""
    block = THUMB_SIDE // HASH_SIDE
    small = thumb.reshape(HASH_SIDE, block, HASH_SIDE, block).mean(axis=(1, 3))
    bits = (small > small.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _is_duplicate(fingerprint: int, thumb: np.ndarray, last: tuple[int, np.ndarray], cfg: Settings) -> bool:
    """Near-identical to the last kept frame: close fingerprint AND close pixels. The pixel check matters for
    flat frames, which all hash to 0: a black frame and a white frame must not count as copies."""
    last_fingerprint, last_thumb = last
    return (
        hamming(fingerprint, last_fingerprint) <= cfg.video_dedup_distance
        and scene_difference(thumb, last_thumb) <= cfg.video_scene_threshold
    )


def _thumbnail(raw: av.VideoFrame) -> np.ndarray:
    """Grey thumbnail by point sampling, not averaging: averaging blurs the contrast away, so a cut between two
    different scenes measured only about 18 of 255 and was missed at the threshold of 30."""
    return raw.reformat(width=THUMB_SIDE, height=THUMB_SIDE, format="gray", interpolation="POINT").to_ndarray()


def _target_size(width: int, height: int, side: int) -> tuple[int, int]:
    """Longest side = side; the other keeps the aspect ratio. Never zero."""
    if width >= height:
        return side, max(1, round(height * side / width))
    return max(1, round(width * side / height)), side


def _to_jpeg(raw: av.VideoFrame, width: int, height: int) -> bytes:
    rgb = raw.reformat(width=width, height=height, format="rgb24").to_ndarray()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def _time_ms(seconds: float | None, frame_no: int, avg_fps: float) -> int:
    if seconds is not None:
        return max(0, round(seconds * 1000))
    return round(frame_no / avg_fps * 1000)
