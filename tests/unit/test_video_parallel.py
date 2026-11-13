"""Several videos at once: the CPU steps run together, the GPU step takes one video at a time, and the
number of prepared videos that may wait for the GPU is limited (that limit is the memory limit).

The scorer and the text reader are stand-ins that count how many calls are running at the same moment.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import video_fixtures as fx
from PIL import Image

from moderation.classifiers import OcrPool
from moderation.config import Settings
from moderation.domain import (
    CATEGORIES,
    ImageRef,
    ImageScore,
    Label,
    ModerationRequest,
    OcrResult,
    VideoRef,
)
from moderation.errors import RetryableError
from moderation.moderate import Moderator, VideoPipeline
from moderation.policy import D1_POLICY
from moderation.video.check import VideoChecker
from moderation.video.fetch import VideoFetcher
from tests.fakes import FakeFetcher, FakeText

DB = "postgresql://u:p@localhost:5432/x"
HARD = frozenset({"G-01", "G-03", "G-05"})


class CountingOcr:
    """Counts the reads running at once; each read takes `delay` seconds."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.running = 0
        self.peak = 0

    async def read(self, images: list[bytes]) -> list[OcrResult]:
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(self.delay)
            return [OcrResult("", None, 0) for _ in images]
        finally:
            self.running -= 1


class GateScorer:
    """Stand-in frame scorer. A bright frame is G-03. Scoring waits for `release` when one is given."""

    def __init__(
        self, release: asyncio.Event | None = None, on_score: Any = None
    ) -> None:
        self.release = release
        self.on_score = on_score
        self.running = 0
        self.peak = 0
        self.calls = 0

    async def score(self, images: list[bytes]) -> list[ImageScore]:
        self.calls += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.on_score is not None:
                self.on_score()
            if self.release is not None:
                await self.release.wait()
            out = []
            for blob in images:
                grey = np.asarray(Image.open(io.BytesIO(blob)).convert("L"))
                scores = {code: 0.02 for code in CATEGORIES}
                if grey.mean() > 200:
                    scores["G-03"] = 0.97
                out.append(ImageScore(scores, (1.0, 0.0)))
            return out
        finally:
            self.running -= 1


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


def build(
    cfg: Settings, scorer: Any, ocr: Any, images: dict[str, bytes] | None = None
) -> Moderator:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    )
    checker = VideoChecker(
        image=scorer,
        ocr=ocr,
        judge_texts=no_text,
        policy=D1_POLICY,
        hard_block=HARD,
        text_unreadable_share=0.5,
    )
    return Moderator(
        text=FakeText(),
        image=scorer,
        ocr=ocr,
        fetcher=FakeFetcher(images or {}),
        policy=D1_POLICY,
        video=VideoPipeline(
            cfg=cfg, fetcher=VideoFetcher(cfg, client), checker=checker
        ),
    )


def upload(tmp_path: Path, name: str, frames: list[np.ndarray]) -> str:
    root = tmp_path / "root" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    fx.write_clip(root / name, frames)
    return f"uploads/{name}"


def request(refs: list[VideoRef]) -> ModerationRequest:
    return ModerationRequest(
        request_id="req-1",
        surface="post",
        content_id="post_9",
        content_version=1,
        author_id="u_1",
        text=None,
        videos=refs,
    )


def three_clips(tmp_path: Path) -> list[VideoRef]:
    return [
        VideoRef(
            media_id=f"v{i}",
            url=upload(tmp_path, f"c{i}.mp4", fx.scene_frames([(i, 30), (i + 5, 30)])),
        )
        for i in (1, 2, 3)
    ]


# --- the CPU steps run together ---------------------------------------------------


async def test_the_videos_of_one_post_read_their_text_at_the_same_time(
    tmp_path: Path,
) -> None:
    ocr = CountingOcr()
    mod = build(settings(tmp_path, video_parallel=3), GateScorer(), ocr)
    decision = await mod.moderate(request(three_clips(tmp_path)))
    assert decision.label is Label.ALLOW
    assert ocr.peak == 3


async def test_video_parallel_one_keeps_them_one_after_the_other(
    tmp_path: Path,
) -> None:
    ocr = CountingOcr()
    mod = build(settings(tmp_path, video_parallel=1), GateScorer(), ocr)
    await mod.moderate(request(three_clips(tmp_path)))
    assert ocr.peak == 1


# --- waiting for the GPU is limited -----------------------------------------------


async def test_prepared_videos_wait_and_only_the_limit_may_wait(tmp_path: Path) -> None:
    release = asyncio.Event()
    scorer = GateScorer(release)
    mod = build(
        settings(tmp_path, video_parallel=3, video_max_waiting=2),
        scorer,
        CountingOcr(0.0),
    )
    assert mod._video is not None
    gate = mod._video.gate

    task = asyncio.create_task(mod.moderate(request(three_clips(tmp_path))))
    for _ in range(
        200
    ):  # let the CPU steps finish; two videos reach the scorer, the third must wait for a slot
        await asyncio.sleep(0.01)
        if gate.waiting_videos == 2:
            break
    await asyncio.sleep(0.2)
    assert (
        gate.waiting_videos == 2
    )  # never more than video_max_waiting hold frames at the GPU
    assert scorer.calls == 2
    assert gate.waiting_bytes > 0  # the frames they hold are counted

    release.set()
    decision = await task
    assert decision.label is Label.ALLOW
    assert scorer.calls == 3
    assert gate.peak_waiting_videos == 2
    assert (gate.waiting_videos, gate.waiting_bytes) == (
        0,
        0,
    )  # everything is let go afterwards


async def test_waiting_bytes_are_the_jpeg_frames_of_the_videos_that_wait(
    tmp_path: Path,
) -> None:
    release = asyncio.Event()
    mod = build(
        settings(tmp_path, video_parallel=3, video_max_waiting=4),
        GateScorer(release),
        CountingOcr(0.0),
    )
    assert mod._video is not None
    gate = mod._video.gate
    task = asyncio.create_task(mod.moderate(request(three_clips(tmp_path))))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if gate.waiting_videos == 3:
            break
    assert gate.waiting_videos == 3
    held = gate.waiting_bytes
    release.set()
    decision = await task
    assert held > 0
    assert gate.peak_waiting_bytes == held
    assert decision.video_frames  # the decision itself keeps no unflagged frame bytes
    assert all(f.jpeg == b"" for f in decision.video_frames)


# --- the answer does not depend on who finishes first -----------------------------


async def test_results_keep_the_order_the_videos_were_sent_in(tmp_path: Path) -> None:
    slow = upload(
        tmp_path, "slow.mp4", [np.full((128, 128, 3), 255, dtype=np.uint8)] * 30
    )
    fast = upload(tmp_path, "fast.mp4", fx.scene_frames([(1, 30), (2, 30)]))

    class SlowForBright(GateScorer):
        async def score(self, images: list[bytes]) -> list[ImageScore]:
            grey = np.asarray(Image.open(io.BytesIO(images[0])).convert("L"))
            if grey.mean() > 200:
                await asyncio.sleep(0.3)  # the first video finishes last
            return await super().score(images)

    mod = build(settings(tmp_path, video_parallel=2), SlowForBright(), CountingOcr(0.0))
    decision = await mod.moderate(
        request(
            [VideoRef(media_id="slow", url=slow), VideoRef(media_id="fast", url=fast)]
        )
    )
    ids = [f.media_id for f in decision.video_frames]
    assert ids == sorted(ids, key=["slow", "fast"].index)  # all of "slow" first
    assert decision.label is Label.BLOCK  # worst piece wins, whatever the order


# --- failures ----------------------------------------------------------------------


async def test_a_retryable_error_stops_the_other_videos_and_lets_everything_go(
    tmp_path: Path,
) -> None:
    class Failing(GateScorer):
        async def score(self, images: list[bytes]) -> list[ImageScore]:
            grey = np.asarray(Image.open(io.BytesIO(images[0])).convert("L"))
            if grey.mean() > 200:
                raise RetryableError("model_oom", "no memory")
            await asyncio.sleep(5)  # the others would take long; they must be cancelled
            return await super().score(images)

    bright = upload(
        tmp_path, "b.mp4", [np.full((128, 128, 3), 255, dtype=np.uint8)] * 30
    )
    other = upload(tmp_path, "o.mp4", fx.scene_frames([(1, 30), (2, 30)]))
    mod = build(settings(tmp_path, video_parallel=2), Failing(), CountingOcr(0.0))
    assert mod._video is not None
    gate = mod._video.gate
    started = asyncio.get_running_loop().time()
    with pytest.raises(RetryableError):
        await mod.moderate(
            request(
                [VideoRef(media_id="b", url=bright), VideoRef(media_id="o", url=other)]
            )
        )
    assert (
        asyncio.get_running_loop().time() - started < 3
    )  # did not wait for the slow one
    assert (gate.waiting_videos, gate.waiting_bytes) == (0, 0)
    # every slot is free again: the next post is not blocked
    assert gate.cpu._value == 2  # type: ignore[attr-defined]
    assert gate.waiting._value == mod._video.cfg.video_max_waiting  # type: ignore[attr-defined]


async def test_a_bad_video_is_listed_and_the_good_one_is_still_checked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root" / "uploads"
    root.mkdir(parents=True)
    (root / "broken.mp4").write_bytes(b"not a video at all" * 20)
    good = upload(tmp_path, "good.mp4", fx.scene_frames([(1, 30), (2, 30)]))
    mod = build(settings(tmp_path, video_parallel=2), GateScorer(), CountingOcr(0.0))
    decision = await mod.moderate(
        request(
            [
                VideoRef(media_id="broken", url="uploads/broken.mp4"),
                VideoRef(media_id="good", url=good),
            ]
        )
    )
    assert [r.media_id for r in decision.rejected_videos] == ["broken"]
    assert decision.label is Label.HOLD
    assert any(f.media_id == "good" for f in decision.video_frames)


# --- the temporary copy is gone before the video waits ----------------------------


async def test_the_temporary_copy_is_deleted_before_scoring_starts(
    tmp_path: Path,
) -> None:
    tmp = tmp_path / "tmp"
    seen: list[list[str]] = []
    scorer = GateScorer(
        on_score=lambda: seen.append(
            [p.name for p in tmp.glob("*")] if tmp.exists() else []
        )
    )
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (64, 64), (i * 20, 10, 10)) for i in range(6)]
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    url = "https://cdn.example/anim.gif"
    mod = build(settings(tmp_path), scorer, CountingOcr(0.0), {url: buffer.getvalue()})
    await mod.moderate(
        ModerationRequest(
            request_id="r",
            surface="post",
            content_id="p",
            content_version=1,
            author_id="u",
            text=None,
            images=[ImageRef(media_id="a1", url=url)],
        )
    )
    assert seen == [
        []
    ]  # nothing left in the temp folder while the frames were being scored
    assert list(tmp.glob("*")) == []


# --- the OCR pool ------------------------------------------------------------------


class SlowEngine:
    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.uses = 0
        self.closed = False

    async def read(self, images: list[bytes]) -> list[str]:
        self.uses += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.05)
        self.running -= 1
        return [f"t{len(blob)}" for blob in images]

    def close(self) -> None:
        self.closed = True


async def test_ocr_pool_runs_one_call_per_engine_and_never_two_on_the_same_engine() -> (
    None
):
    engines = [SlowEngine() for _ in range(3)]
    pool = OcrPool(engines)
    results = await asyncio.gather(*(pool.read([b"x" * n]) for n in range(1, 10)))
    assert results == [
        [f"t{n}"] for n in range(1, 10)
    ]  # each caller gets its own answer
    assert all(e.peak == 1 for e in engines)  # an engine is never used twice at once
    assert all(e.uses >= 1 for e in engines)  # the work is spread over all engines
    assert pool.size == 3
    pool.close()
    assert all(e.closed for e in engines)


async def test_ocr_pool_gives_the_engine_back_after_an_error() -> None:
    class Broken(SlowEngine):
        async def read(self, images: list[bytes]) -> list[str]:
            raise ValueError("boom")

    pool = OcrPool([Broken()])
    for _ in range(3):
        with pytest.raises(ValueError):
            await pool.read(
                [b"x"]
            )  # would hang on the second call if the engine were not returned


def test_ocr_pool_needs_an_engine() -> None:
    with pytest.raises(ValueError):
        OcrPool([])
