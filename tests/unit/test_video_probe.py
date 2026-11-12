"""D2 probe (plan Part B step 2): every refusal is the planned error code, and a good file passes."""

from __future__ import annotations

from pathlib import Path

import av
import pytest
import video_fixtures as fx

from moderation.config import Settings
from moderation.errors import (
    MEDIA_TOO_LARGE,
    MEDIA_TOO_LONG,
    MEDIA_UNDECODABLE,
    MEDIA_UNSUPPORTED,
    PermanentError,
)
from moderation.video.probe import probe

DB = "postgresql://u:p@localhost:5432/x"


def cfg(**overrides: object) -> Settings:
    return Settings(_env_file=None, database_url=DB, **overrides)  # type: ignore[call-arg]


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return fx.write_clip(tmp_path / "ok.mp4", [fx.pattern(1)] * 120)  # 4 s at 30 fps


def test_good_clip_is_described(clip: Path) -> None:
    info = probe(clip, cfg())
    assert info.duration_s == pytest.approx(4.0, abs=0.05)
    assert info.avg_fps == pytest.approx(30.0, abs=0.5)
    assert (info.width, info.height) == (128, 128)
    assert info.codec == "h264"


def test_longer_than_the_limit_is_too_long(clip: Path) -> None:
    with pytest.raises(PermanentError) as info:
        probe(clip, cfg(video_max_duration_s=2))
    assert info.value.code == MEDIA_TOO_LONG


def test_at_the_limit_is_accepted(clip: Path) -> None:
    assert probe(clip, cfg(video_max_duration_s=5)).duration_s > 0


def test_file_larger_than_the_limit_is_too_large(clip: Path) -> None:
    with pytest.raises(PermanentError) as info:
        probe(clip, cfg(video_max_bytes=1024))
    assert info.value.code == MEDIA_TOO_LARGE


def test_frame_larger_than_the_pixel_limit_is_too_large(clip: Path) -> None:
    with pytest.raises(PermanentError) as info:
        probe(clip, cfg(video_max_pixels=100))
    assert info.value.code == MEDIA_TOO_LARGE


def test_text_file_is_undecodable(tmp_path: Path) -> None:
    text = tmp_path / "note.mp4"
    text.write_bytes(b"hello, this is a note, not a video")
    with pytest.raises(PermanentError) as info:
        probe(text, cfg())
    assert info.value.code == MEDIA_UNDECODABLE


def test_missing_file_is_undecodable(tmp_path: Path) -> None:
    with pytest.raises(PermanentError) as info:
        probe(tmp_path / "gone.mp4", cfg())
    assert info.value.code == MEDIA_UNDECODABLE


def test_avi_container_is_unsupported(tmp_path: Path) -> None:
    avi = tmp_path / "old.avi"
    with av.open(str(avi), mode="w", format="avi") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 128, 128, "yuv420p"
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(fx.pattern(i), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with pytest.raises(PermanentError) as info:
        probe(avi, cfg())
    assert info.value.code == MEDIA_UNSUPPORTED
