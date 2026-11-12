"""Check one video: score its frames, read their on-screen text, combine into one verdict (plan section 8, steps 5-8).

    python -m moderation.video.check clip.mp4     # one clip from the command line; prints the verdict and timings

The models are the D1 models, unchanged: SigLIP 2 scores each frame, RapidOCR reads its text, and the
text goes through the caption rules (Qwen3Guard, via Moderator.judge_texts). Nothing here is stored;
the worker saves the result (Stage 3).

Combine rule (decision 9): for a hard-block category the video's score is the highest frame score, so
one flagged frame is enough. For every other category it is the mean of the 3 highest frame scores.
The combined score per category is then judged with the video thresholds (policy.judge_video).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from moderation.config import Settings
from moderation.domain import CATEGORIES, ItemResult, ItemSource, Label, OcrResult
from moderation.policy import Policy, worst_item
from moderation.text_clean import clean_text
from moderation.video.frames import Frame

log = logging.getLogger("moderation.video")

SCORE_BATCH: Final = 32  # frames per SigLIP 2 / RapidOCR call; the models also split a batch on GPU out-of-memory
TOP_FRAMES: Final = 3  # soft categories: the mean of this many highest frame scores
TextJudge = Callable[[list[str]], Awaitable[list[list[ItemResult]]]]


@dataclass(frozen=True)
class FrameResult:
    frame_no: int
    time_ms: int
    source: str
    scores: dict[str, float]
    flagged: bool  # any category at or above its hold threshold, on this frame alone
    text: str  # "none": no text; "readable": text judged; "unreadable": text found but not readable


@dataclass(frozen=True)
class VideoVerdict:
    items: tuple[ItemResult, ...]  # the video item, its on-screen text items, any hold for unreadable text
    frames: tuple[FrameResult, ...]
    combined: dict[str, float] | None  # None when a frame could not be scored
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step; logged, never stored


def combine_scores(frame_scores: Sequence[Mapping[str, float]], hard_block: frozenset[str]) -> dict[str, float]:
    """One score per category for the whole video. Hard-block categories: the highest frame score.
    Other categories: the mean of the TOP_FRAMES highest. Callers check finiteness first."""
    if not frame_scores:
        raise ValueError("no frame scores to combine")
    combined: dict[str, float] = {}
    for code in CATEGORIES:
        values = sorted((scores[code] for scores in frame_scores), reverse=True)
        if code in hard_block:
            combined[code] = values[0]
        else:
            top = values[:TOP_FRAMES]
            combined[code] = sum(top) / len(top)
    return combined


def text_state(result: OcrResult, policy: Policy) -> str:
    """none: fewer letters than ocr_min_chars (no text). unreadable: text found, low or missing confidence."""
    if len("".join(result.text.split())) < policy.ocr_min_chars:
        return "none"
    if result.mean_confidence is None or result.mean_confidence < policy.ocr_min_confidence:
        return "unreadable"
    return "readable"


class VideoChecker:
    def __init__(
        self,
        *,
        image: Any,
        ocr: Any,
        judge_texts: TextJudge,
        policy: Policy,
        hard_block: frozenset[str],
        text_unreadable_share: float,
    ) -> None:
        self._image = image
        self._ocr = ocr
        self._judge_texts = judge_texts
        self._policy = policy
        self._hard_block = hard_block
        self._text_unreadable_share = text_unreadable_share

    async def check(self, media_id: str, frames: Sequence[Frame]) -> VideoVerdict:
        if not frames:
            raise ValueError("a video with no frames cannot be checked")
        jpegs = [frame.jpeg for frame in frames]
        timings: dict[str, float] = {}

        started = time.perf_counter()
        scores = await self._score_frames(jpegs)
        timings["score_s"] = time.perf_counter() - started

        started = time.perf_counter()
        ocr_results = await _in_batches(self._ocr.read, jpegs, SCORE_BATCH)
        timings["ocr_s"] = time.perf_counter() - started

        started = time.perf_counter()
        text_items, states = await self._judge_video_text(media_id, ocr_results)
        timings["text_s"] = time.perf_counter() - started

        if scores is None:
            # A broken score is never allowed (plan: a video that cannot be checked is held).
            video_item = ItemResult(
                ItemSource.VIDEO, media_id, Label.HOLD, None, None, f"Video {media_id}: frames could not be scored"
            )
            combined = None
            frame_scores: list[dict[str, float]] = [{} for _ in frames]
            flagged = [True] * len(frames)
        else:
            combined = combine_scores(scores, self._hard_block)
            video_item = self._policy.judge_video(media_id, combined)
            frame_scores = scores
            flagged = [self._policy.judge_video(media_id, s).label is not Label.ALLOW for s in scores]
        frame_results = tuple(
            FrameResult(
                frame_no=frame.frame_no,
                time_ms=frame.time_ms,
                source=frame.source,
                scores=frame_scores[i],
                flagged=flagged[i],
                text=states[i],
            )
            for i, frame in enumerate(frames)
        )
        return VideoVerdict(items=(video_item, *text_items), frames=frame_results, combined=combined, timings=timings)

    async def _score_frames(self, jpegs: list[bytes]) -> list[dict[str, float]] | None:
        """Category scores per frame, or None when any frame has a score that is not a finite number."""
        results = await _in_batches(self._image.score, jpegs, SCORE_BATCH)
        scores = [dict(result.categories) for result in results]
        if any(not math.isfinite(value) for frame in scores for value in frame.values()):
            return None
        return scores

    async def _judge_video_text(self, media_id: str, ocr_results: Sequence[OcrResult]) -> tuple[list[ItemResult], list[str]]:
        """Unreadable frames are skipped. The video is held only when most text-bearing frames are unreadable.
        Readable text is judged once per distinct text, since the same caption stays on screen for many frames."""
        states = [text_state(result, self._policy) for result in ocr_results]
        bearing = [s for s in states if s != "none"]
        unreadable = sum(1 for s in states if s == "unreadable")
        items: list[ItemResult] = []
        if bearing and unreadable / len(bearing) > self._text_unreadable_share:
            items.append(
                ItemResult(
                    ItemSource.VIDEO_TEXT,
                    media_id,
                    Label.HOLD,
                    None,
                    None,
                    f"Text in video {media_id}: could not be read in {unreadable} of {len(bearing)} frames",
                )
            )
        texts = _distinct(
            clean_text(result.text.replace("\n", " "))
            for result, state in zip(ocr_results, states, strict=True)
            if state == "readable"
        )
        if texts:
            judged = await self._judge_texts(texts)
            for text_items in judged:
                items.extend(_as_video_text(item, media_id) for item in text_items)
        return items, states



def _as_video_text(item: ItemResult, media_id: str) -> ItemResult:
    return replace(item, source=ItemSource.VIDEO_TEXT, media_id=media_id)


def _distinct(texts: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for text in texts:
        if text:
            seen.setdefault(text, None)
    return list(seen)


async def _in_batches[T, R](call: Callable[[list[T]], Awaitable[Sequence[R]]], items: Sequence[T], size: int) -> list[R]:
    out: list[R] = []
    for start in range(0, len(items), size):
        out.extend(await call(list(items[start : start + size])))
    return out


# --- command line -------------------------------------------------------------


async def _run_clip(path: Path) -> dict[str, Any]:
    # Imported here: the API and the unit tests never load the model runtime.
    from moderation.config import get_settings
    from moderation.logging_setup import configure_logging
    from moderation.moderate import Moderator, build_reason
    from moderation.policy import D1_POLICY
    from moderation.video.frames import select_frames
    from moderation.video.probe import probe
    from moderation.worker import load_models

    cfg: Settings = get_settings()
    configure_logging(cfg.log_level)
    started = time.perf_counter()
    info = probe(path, cfg)
    selection = select_frames(path, cfg, info)
    t_frames = time.perf_counter() - started

    text, image, ocr = await asyncio.to_thread(load_models, cfg)
    moderator = Moderator(text=text, image=image, ocr=ocr, fetcher=None, policy=D1_POLICY)
    checker = VideoChecker(
        image=image,
        ocr=ocr,
        judge_texts=moderator.judge_texts,
        policy=D1_POLICY,
        hard_block=cfg.video_hard_block_codes,
        text_unreadable_share=cfg.video_text_unreadable_share,
    )
    verdict = await checker.check(path.name, selection.frames)
    worst = worst_item(verdict.items)
    total = time.perf_counter() - started
    result = {
        "file": path.name,
        "label": worst.label.value,
        "category": worst.category,
        "reason": build_reason(worst),
        "video": {"duration_s": round(info.duration_s, 2), "avg_fps": round(info.avg_fps, 2), "size": f"{info.width}x{info.height}"},
        "frames": {
            "decoded": selection.decoded,
            "sampled": selection.sampled,
            "scene": selection.scene,
            "duplicates_removed": selection.duplicates,
            "candidates": selection.candidates,
            "checked": len(selection.frames),
            "flagged": sum(1 for f in verdict.frames if f.flagged),
        },
        "combined": verdict.combined,
        "timings_s": {**{k: round(v, 3) for k, v in verdict.timings.items()}, "frames_s": round(t_frames, 3), "total_s": round(total, 3)},
        "policy_version": D1_POLICY.version,
    }
    log.info("video clip checked", extra={"label": worst.label.value, "checked": len(selection.frames), "total_s": round(total, 3)})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check one video clip with the D2 pipeline and print the verdict.")
    parser.add_argument("path", type=Path, help="a local video file (MP4, MOV, WebM or MKV)")
    args = parser.parse_args(argv)
    # The CLI never connects to the database, but Settings requires DATABASE_URL to exist.
    os.environ.setdefault("DATABASE_URL", "postgresql://unused@localhost/unused")
    result = asyncio.run(_run_clip(args.path))
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
