"""D2 frame selection (plan section 4): sampling, scene change, duplicates, cap. Synthetic clips, no models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import video_fixtures as fx

from moderation.config import Settings
from moderation.errors import MEDIA_UNDECODABLE, PermanentError
from moderation.video.frames import (
    Frame,
    average_hash,
    cap_frames,
    hamming,
    scene_difference,
    select_frames,
)
from moderation.video.probe import probe

DB = "postgresql://u:p@localhost:5432/x"


def cfg(**overrides: object) -> Settings:
    return Settings(_env_file=None, database_url=DB, **overrides)  # type: ignore[call-arg]


def run(path: Path, settings: Settings):
    return select_frames(path, settings, probe(path, settings))


# --- pure helpers -----------------------------------------------------------------


def test_identical_frames_have_zero_difference() -> None:
    a = fx.pattern(1)[:, :, 0].astype(np.uint8)
    assert scene_difference(a, a) == 0


def test_different_patterns_differ_strongly() -> None:
    a = fx.pattern(1)[:, :, 0].astype(np.uint8)
    b = fx.pattern(2)[:, :, 0].astype(np.uint8)
    assert scene_difference(a, b) > 30  # above the default scene threshold


def test_average_hash_of_a_flat_frame_is_zero() -> None:
    # Known limit: every flat frame hashes to 0, so the pixel check in the duplicate rule is what tells them apart.
    assert average_hash(np.full((64, 64), 200, dtype=np.uint8)) == 0


def test_average_hash_is_64_bits_and_stable() -> None:
    thumb = fx.pattern(3, (64, 64))[:, :, 0].astype(np.uint8)
    value = average_hash(thumb)
    assert 0 <= value < 2**64
    assert average_hash(thumb.copy()) == value


def test_hamming_counts_differing_bits() -> None:
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(0, 2**64 - 1) == 64


def _frame(n: int, source: str = "sample") -> Frame:
    return Frame(frame_no=n, time_ms=n * 1000, source=source, jpeg=b"")  # type: ignore[arg-type]


def test_cap_keeps_everything_under_the_limit() -> None:
    frames = [_frame(n) for n in range(5)]
    assert cap_frames(frames, 10) == frames


def test_cap_thins_evenly_and_keeps_first_and_last() -> None:
    frames = [_frame(n) for n in range(100)]
    kept = cap_frames(frames, 5)
    assert len(kept) == 5
    assert kept[0].frame_no == 0
    assert kept[-1].frame_no == 99
    assert [f.frame_no for f in kept] == sorted(f.frame_no for f in kept)


def test_cap_keeps_scene_frames_first() -> None:
    samples = [_frame(n) for n in range(50)]
    scenes = [_frame(n, "scene") for n in (7, 13, 29)]
    kept = cap_frames(sorted([*samples, *scenes], key=lambda f: f.frame_no), 10)
    assert len(kept) == 10
    kept_scenes = {f.frame_no for f in kept if f.source == "scene"}
    assert kept_scenes == {7, 13, 29}  # every scene frame survives while room is left
    assert [f.frame_no for f in kept] == sorted(f.frame_no for f in kept)


def test_cap_with_more_scene_frames_than_room_thins_the_scenes() -> None:
    scenes = [_frame(n, "scene") for n in range(40)]
    kept = cap_frames(scenes, 4)
    assert len(kept) == 4
    assert all(f.source == "scene" for f in kept)


def test_cap_of_one_prefers_the_scene_frame() -> None:
    assert cap_frames([_frame(0), _frame(1, "scene")], 1)[0].source == "scene"


# --- synthetic clips ----------------------------------------------------------------


def test_static_clip_keeps_one_frame(tmp_path: Path) -> None:
    clip = fx.write_clip(tmp_path / "static.mp4", [fx.pattern(1)] * 90)
    sel = run(clip, cfg())
    assert [f.frame_no for f in sel.frames] == [0]  # 0, 30 and 60 are copies of frame 0
    assert sel.decoded == 90
    assert sel.sampled == 3
    assert sel.duplicates == 2


def test_hard_cut_keeps_the_frame_at_the_cut(tmp_path: Path) -> None:
    clip = fx.write_clip(tmp_path / "cut.mp4", fx.scene_frames([(1, 60), (2, 60)]))
    sel = run(clip, cfg())
    assert [f.frame_no for f in sel.frames] == [0, 60]
    assert [f.source for f in sel.frames] == ["sample", "scene"]
    assert sel.scene >= 1
    assert sel.frames[1].time_ms == 2000  # 60 frames at 30 fps


def test_flat_colour_cut_is_not_mistaken_for_a_duplicate(tmp_path: Path) -> None:
    """Regression: flat frames all hash to 0. A black-to-white cut must still keep the white frame."""
    black = np.zeros((128, 128, 3), dtype=np.uint8)
    white = np.full((128, 128, 3), 255, dtype=np.uint8)
    clip = fx.write_clip(tmp_path / "flat.mp4", [black] * 30 + [white] * 30)
    sel = run(clip, cfg())
    assert 30 in [f.frame_no for f in sel.frames]


@pytest.mark.parametrize("sample_fps,step", [(1.0, 30), (2.0, 15), (0.5, 60)])
def test_sampling_rate_is_one_frame_per_n(tmp_path: Path, sample_fps: float, step: int) -> None:
    clip = fx.write_clip(tmp_path / "sample.mp4", [fx.pattern(4)] * 120)
    sel = run(clip, cfg(video_sample_fps=sample_fps))
    assert sel.sampled == -(-120 // step)  # ceil(120 / step) sample frames


def test_scene_threshold_at_maximum_finds_no_scene(tmp_path: Path) -> None:
    clip = fx.write_clip(tmp_path / "cut2.mp4", fx.scene_frames([(1, 30), (2, 30)]))
    sel = run(clip, cfg(video_scene_threshold=255))
    assert sel.scene == 0


def test_cap_applies_to_a_real_clip(tmp_path: Path) -> None:
    scenes = [(seed, 3) for seed in range(1, 21)]  # 20 hard cuts
    clip = fx.write_clip(tmp_path / "many.mp4", fx.scene_frames(scenes))
    sel = run(clip, cfg(video_max_frames=5))
    assert len(sel.frames) == 5
    assert sel.candidates > 5
    assert [f.frame_no for f in sel.frames] == sorted(f.frame_no for f in sel.frames)


def test_kept_frames_are_jpeg_at_the_configured_side(tmp_path: Path) -> None:
    from io import BytesIO

    from PIL import Image

    clip = fx.write_clip(tmp_path / "side.mp4", [fx.pattern(5)] * 10)
    sel = run(clip, cfg(video_frame_side=64))
    image = Image.open(BytesIO(sel.frames[0].jpeg))
    assert image.format == "JPEG"
    assert max(image.size) == 64


def test_corrupt_bytes_refused_as_undecodable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video at all" * 50)
    with pytest.raises(PermanentError) as info:
        run(bad, cfg())
    assert info.value.code == MEDIA_UNDECODABLE
