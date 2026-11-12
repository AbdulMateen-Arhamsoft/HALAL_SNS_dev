"""D2 business rules through the Moderator: a video is allowed, held or blocked on what its frames show.

The frame scorer is a stand-in that reads brightness, not a model: a white frame scores as G-03. That is
enough to test the rules (one frame blocks a hard-block category, a bad file is held and listed, one bad
video does not stop the others, an animated image is checked as a video). Real models: tests/models.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import video_fixtures as fx
from PIL import Image

from moderation.config import Settings
from moderation.domain import (
    CATEGORIES,
    ImageRef,
    ImageScore,
    ItemSource,
    Label,
    ModerationRequest,
    VideoRef,
)
from moderation.errors import MEDIA_TOO_LONG, MEDIA_UNDECODABLE
from moderation.fetch import is_animated
from moderation.moderate import (
    ANIMATED_REASON,
    REJECT_VIDEO_DETAILS,
    Moderator,
    VideoPipeline,
)
from moderation.policy import D1_POLICY
from moderation.video.check import VideoChecker
from moderation.video.fetch import VideoFetcher
from tests.fakes import FakeFetcher, FakeOcr, FakeText

DB = "postgresql://u:p@localhost:5432/x"
HARD = frozenset({"G-01", "G-03", "G-05"})
CLEAN_BASE = 0.02


class BrightnessImage:
    """Stand-in frame scorer: a bright frame (mean above 200) is G-03 at 0.97, anything else is clean."""

    async def score(self, images: list[bytes]) -> list[ImageScore]:
        out = []
        for blob in images:
            grey = np.asarray(Image.open(io.BytesIO(blob)).convert("L"))
            scores = {code: CLEAN_BASE for code in CATEGORIES}
            if grey.mean() > 200:
                scores["G-03"] = 0.97
            out.append(ImageScore(scores, (1.0, 0.0)))
        return out


async def no_text(texts: list[str]) -> list[list[Any]]:
    return [[] for _ in texts]


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "image_allowed_root": str(tmp_path / "root"),
        "video_temp_dir": str(tmp_path / "tmp"),
        "video_evidence_root": str(tmp_path / "evidence"),
    }
    base.update(overrides)
    return Settings(_env_file=None, database_url=DB, **base)


def pipeline(cfg: Settings) -> VideoPipeline:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    checker = VideoChecker(
        image=BrightnessImage(),
        ocr=FakeOcr(),
        judge_texts=no_text,
        policy=D1_POLICY,
        hard_block=HARD,
        text_unreadable_share=0.5,
    )
    return VideoPipeline(cfg=cfg, fetcher=VideoFetcher(cfg, client), checker=checker)


def moderator(video: VideoPipeline | None, images: dict[str, bytes] | None = None) -> Moderator:
    return Moderator(
        text=FakeText(),
        image=BrightnessImage(),
        ocr=FakeOcr(),
        fetcher=FakeFetcher(images or {}),
        policy=D1_POLICY,
        video=video,
    )


def request(videos: list[VideoRef] | None = None, images: list[ImageRef] | None = None) -> ModerationRequest:
    return ModerationRequest(
        request_id="req-1",
        surface="post",
        content_id="post_9",
        content_version=1,
        author_id="u_1",
        text=None if videos or images else "a caption",
        videos=videos or [],
        images=images or [],
    )


def upload(tmp_path: Path, name: str, frames: list[np.ndarray] | None = None, raw: bytes | None = None) -> str:
    """Write a file under the upload root and return the url the request uses for it."""
    root = tmp_path / "root" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if raw is not None:
        target.write_bytes(raw)
    else:
        fx.write_clip(target, frames or [])
    return f"uploads/{name}"


def flat_cut() -> list[np.ndarray]:
    black = np.zeros((128, 128, 3), dtype=np.uint8)
    white = np.full((128, 128, 3), 255, dtype=np.uint8)
    return [black] * 30 + [white] * 30


async def test_a_clean_video_is_allowed_and_its_frames_are_kept_without_images(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    url = upload(tmp_path, "clean.mp4", fx.scene_frames([(1, 30), (2, 30)]))
    decision = await moderator(pipeline(cfg)).moderate(request(videos=[VideoRef(media_id="v1", url=url)]))
    assert decision.label is Label.ALLOW
    assert decision.rejected_videos == ()
    assert decision.video_frames
    assert not any(f.flagged for f in decision.video_frames)
    assert all(f.jpeg == b"" for f in decision.video_frames)  # only flagged frames keep their bytes


async def test_one_bright_frame_blocks_a_hard_block_category(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    url = upload(tmp_path, "cut.mp4", flat_cut())
    decision = await moderator(pipeline(cfg)).moderate(request(videos=[VideoRef(media_id="v1", url=url)]))
    assert decision.label is Label.BLOCK
    assert decision.category == "G-03"
    assert decision.reason == "Video v1: Sexual content and nudity"
    flagged = [f for f in decision.video_frames if f.flagged]
    assert flagged, "the bright frame is flagged"
    assert all(f.jpeg for f in flagged), "flagged frames keep their bytes as evidence"


async def test_an_unreadable_file_is_held_and_listed_in_rejected_videos(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    url = upload(tmp_path, "broken.mp4", raw=b"this is not a video" * 40)
    decision = await moderator(pipeline(cfg)).moderate(request(videos=[VideoRef(media_id="v1", url=url)]))
    assert decision.label is Label.HOLD
    assert [r.error_code for r in decision.rejected_videos] == [MEDIA_UNDECODABLE]
    assert decision.rejected_videos[0].detail == REJECT_VIDEO_DETAILS[MEDIA_UNDECODABLE]
    assert decision.reason == f"Video v1: {REJECT_VIDEO_DETAILS[MEDIA_UNDECODABLE]}"


async def test_a_video_over_the_duration_limit_is_held_as_media_too_long(tmp_path: Path) -> None:
    cfg = settings(tmp_path, video_max_duration_s=1)
    url = upload(tmp_path, "long.mp4", fx.scene_frames([(1, 120)]))  # 4 s
    decision = await moderator(pipeline(cfg)).moderate(request(videos=[VideoRef(media_id="v1", url=url)]))
    assert decision.label is Label.HOLD
    assert [r.error_code for r in decision.rejected_videos] == [MEDIA_TOO_LONG]


async def test_one_bad_video_does_not_stop_the_other_video(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    bad = upload(tmp_path, "bad.mp4", raw=b"garbage" * 100)
    good = upload(tmp_path, "good.mp4", fx.scene_frames([(3, 60)]))
    decision = await moderator(pipeline(cfg)).moderate(
        request(videos=[VideoRef(media_id="bad", url=bad), VideoRef(media_id="good", url=good)])
    )
    assert decision.label is Label.HOLD
    assert [r.media_id for r in decision.rejected_videos] == ["bad"]
    assert {f.media_id for f in decision.video_frames} == {"good"}  # the good video was still checked


async def test_a_worker_without_the_video_pipeline_refuses_video_posts(tmp_path: Path) -> None:
    url = upload(tmp_path, "clip.mp4", fx.scene_frames([(1, 30)]))
    with pytest.raises(RuntimeError, match="not configured"):
        await moderator(None).moderate(request(videos=[VideoRef(media_id="v1", url=url)]))


def animated_gif(frames: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    frames[0].save(buffer, "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buffer.getvalue()


async def test_an_animated_image_is_checked_frame_by_frame_when_video_checks_are_on(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    gif = animated_gif([Image.new("RGB", (64, 64), (255, 255, 255))] * 6)
    assert is_animated(gif)
    url = "https://cdn.example/anim.gif"
    decision = await moderator(pipeline(cfg), {url: gif}).moderate(
        request(images=[ImageRef(media_id="a1", url=url)])
    )
    assert decision.label is Label.BLOCK
    assert decision.category == "G-03"
    assert decision.rejected_images == ()
    assert any(i.source is ItemSource.VIDEO and i.media_id == "a1" for i in decision.items)


async def test_an_animated_image_is_still_held_when_video_checks_are_off(tmp_path: Path) -> None:
    """D1 behaviour, kept for a worker without the video pipeline."""
    gif = animated_gif([Image.new("RGB", (64, 64), (10, 10, 10))] * 3)
    url = "https://cdn.example/anim.gif"
    decision = await moderator(None, {url: gif}).moderate(request(images=[ImageRef(media_id="a1", url=url)]))
    assert decision.label is Label.HOLD
    assert decision.items[0].note == ANIMATED_REASON


async def test_a_still_image_is_not_sent_to_the_video_pipeline(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    still = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 200, 200)).save(still, "JPEG")
    url = "https://cdn.example/still.jpg"
    decision = await moderator(pipeline(cfg), {url: still.getvalue()}).moderate(
        request(images=[ImageRef(media_id="s1", url=url)])
    )
    assert decision.video_frames == ()
    assert not any(i.source is ItemSource.VIDEO for i in decision.items)

