"""Synthetic videos for the D2 tests, made with PyAV at test time. Nothing here is harmful footage.

Frames are seeded random patterns, not flat colours: a flat frame has no structure, so its fingerprint
is the same for every colour, and it cannot test scene or duplicate logic on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

SIZE = (128, 128)  # height, width: even, so H.264 accepts it


def pattern(seed: int, size: tuple[int, int] = SIZE) -> np.ndarray:
    """A structured RGB frame, different for each seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)


def shifted(frame: np.ndarray, pixels: int = 1) -> np.ndarray:
    """The same picture moved a little: a near-copy of `frame`."""
    return np.roll(frame, pixels, axis=1)


def write_clip(path: Path, frames: Sequence[np.ndarray], fps: int = 30) -> Path:
    """Encode frames to an MP4 (H.264) at a constant frame rate."""
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width = int(frames[0].shape[1])
        stream.height = int(frames[0].shape[0])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for i, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def scene_frames(scenes: Sequence[tuple[int, int]], size: tuple[int, int] = SIZE) -> list[np.ndarray]:
    """(seed, frame count) per scene: each scene is a hard cut to a new pattern, held still."""
    frames: list[np.ndarray] = []
    for seed, count in scenes:
        frames.extend([pattern(seed, size)] * count)
    return frames
