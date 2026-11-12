"""D2 video check (plan section 8, steps 5-8): the combine rule, the verdicts and the text rules, with fake models."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from moderation.domain import (
    CATEGORIES,
    ImageScore,
    ItemResult,
    ItemSource,
    Label,
    OcrResult,
)
from moderation.policy import D1_POLICY
from moderation.video.check import VideoChecker, combine_scores, text_state
from moderation.video.frames import Frame
from tests.fakes import FakeOcr

HARD = frozenset({"G-01", "G-03", "G-05"})
BASE = 0.02


def scores(**over: float) -> dict[str, float]:
    base = {code: BASE for code in CATEGORIES}
    base.update(over)
    return base


class ScriptedImage:
    """Returns one score dict per frame, looked up by the frame's bytes."""

    def __init__(self, by_frame: dict[bytes, dict[str, float]]) -> None:
        self._by_frame = by_frame

    async def score(self, images: Sequence[bytes]) -> list[ImageScore]:
        return [ImageScore(self._by_frame.get(blob, scores()), (1.0, 0.0)) for blob in images]


class RecordingTextJudge:
    def __init__(self, label: Label = Label.ALLOW) -> None:
        self.label = label
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[ItemResult]]:
        self.calls.append(list(texts))
        return [[ItemResult(ItemSource.TEXT, None, self.label, None, 0.9 if self.label is not Label.ALLOW else None)] for _ in texts]


def frames(count: int) -> list[Frame]:
    return [Frame(frame_no=i * 30, time_ms=i * 1000, source="sample", jpeg=f"frame-{i}".encode()) for i in range(count)]


def checker(image: ScriptedImage, ocr: FakeOcr, judge: RecordingTextJudge, share: float = 0.5) -> VideoChecker:
    return VideoChecker(
        image=image,
        ocr=ocr,
        judge_texts=judge,
        policy=D1_POLICY,
        hard_block=HARD,
        text_unreadable_share=share,
    )


# --- combine rule -------------------------------------------------------------------


def test_hard_block_category_uses_the_highest_frame() -> None:
    frame_scores = [scores(**{"G-03": 0.10}), scores(**{"G-03": 0.97}), scores()]
    assert combine_scores(frame_scores, HARD)["G-03"] == 0.97


def test_soft_category_uses_the_mean_of_the_three_highest() -> None:
    frame_scores = [scores(**{"G-06": v}) for v in (0.9, 0.3, 0.6, 0.0, 0.0)]
    assert combine_scores(frame_scores, HARD)["G-06"] == pytest.approx((0.9 + 0.6 + 0.3) / 3)


def test_soft_category_with_fewer_than_three_frames_averages_what_there_is() -> None:
    frame_scores = [scores(**{"G-06": 0.8}), scores(**{"G-06": 0.4})]
    assert combine_scores(frame_scores, HARD)["G-06"] == pytest.approx(0.6)


def test_combine_needs_at_least_one_frame() -> None:
    with pytest.raises(ValueError):
        combine_scores([], HARD)


def test_hard_block_set_decides_which_rule_applies() -> None:
    frame_scores = [scores(**{"G-03": 0.97}), scores(), scores()]
    assert combine_scores(frame_scores, frozenset())["G-03"] == pytest.approx((0.97 + BASE + BASE) / 3)


# --- verdicts ------------------------------------------------------------------------


async def test_clean_video_is_allowed() -> None:
    fs = frames(4)
    verdict = await checker(ScriptedImage({}), FakeOcr(), RecordingTextJudge()).check("v1", fs)
    assert verdict.items[0].label is Label.ALLOW
    assert verdict.items[0].source is ItemSource.VIDEO
    assert all(not f.flagged for f in verdict.frames)


async def test_one_flagged_frame_blocks_a_hard_block_category() -> None:
    fs = frames(5)
    image = ScriptedImage({fs[2].jpeg: scores(**{"G-03": 0.97})})
    verdict = await checker(image, FakeOcr(), RecordingTextJudge()).check("v1", fs)
    assert verdict.items[0].label is Label.BLOCK
    assert verdict.items[0].category == "G-03"
    assert [f.flagged for f in verdict.frames] == [False, False, True, False, False]


async def test_one_odd_frame_does_not_flag_a_soft_category() -> None:
    fs = frames(10)
    image = ScriptedImage({fs[0].jpeg: scores(**{"G-06": 0.9})})
    verdict = await checker(image, FakeOcr(), RecordingTextJudge()).check("v1", fs)
    assert verdict.items[0].label is Label.ALLOW  # (0.9 + 0.02 + 0.02) / 3 is below the 0.5 hold
    assert verdict.frames[0].flagged  # but the frame alone is still flagged for review


async def test_three_frames_at_hold_level_hold_a_soft_category() -> None:
    fs = frames(6)
    image = ScriptedImage({f.jpeg: scores(**{"G-06": 0.7}) for f in fs[:3]})
    verdict = await checker(image, FakeOcr(), RecordingTextJudge()).check("v1", fs)
    assert verdict.items[0].label is Label.HOLD
    assert verdict.items[0].category == "G-06"


async def test_a_frame_that_cannot_be_scored_holds_the_video() -> None:
    fs = frames(3)
    image = ScriptedImage({fs[1].jpeg: scores(**{"G-01": math.nan})})
    verdict = await checker(image, FakeOcr(), RecordingTextJudge()).check("v1", fs)
    assert verdict.items[0].label is Label.HOLD
    assert "could not be scored" in (verdict.items[0].note or "")
    assert verdict.combined is None
    assert all(f.flagged for f in verdict.frames)


async def test_readable_on_screen_text_is_judged_like_a_caption() -> None:
    fs = frames(2)
    ocr = FakeOcr({fs[0].jpeg: OcrResult("kill them all", 0.95, 1)})
    judge = RecordingTextJudge(Label.BLOCK)
    verdict = await checker(ScriptedImage({}), ocr, judge).check("v1", fs)
    assert judge.calls == [["kill them all"]]
    text_items = [i for i in verdict.items if i.source is ItemSource.VIDEO_TEXT]
    assert text_items and text_items[0].label is Label.BLOCK
    assert text_items[0].media_id == "v1"


async def test_the_same_on_screen_text_is_judged_once() -> None:
    fs = frames(5)
    ocr = FakeOcr({f.jpeg: OcrResult("Ramadan Mubarak", 0.95, 1) for f in fs})
    judge = RecordingTextJudge()
    await checker(ScriptedImage({}), ocr, judge).check("v1", fs)
    assert judge.calls == [["Ramadan Mubarak"]]


async def test_frames_without_text_are_not_judged_or_counted() -> None:
    fs = frames(4)
    ocr = FakeOcr({fs[0].jpeg: OcrResult("ab", 0.99, 1)})  # two letters: no text
    judge = RecordingTextJudge()
    verdict = await checker(ScriptedImage({}), ocr, judge).check("v1", fs)
    assert judge.calls == []
    assert verdict.items[0].label is Label.ALLOW
    assert [f.text for f in verdict.frames] == ["none"] * 4


async def test_video_held_when_most_text_frames_are_unreadable() -> None:
    fs = frames(4)
    ocr = FakeOcr({f.jpeg: OcrResult("#$%^ &*()", None, 3) for f in fs[:3]})  # three unreadable
    verdict = await checker(ScriptedImage({}), ocr, RecordingTextJudge()).check("v1", fs)
    held = [i for i in verdict.items if i.source is ItemSource.VIDEO_TEXT]
    assert held and held[0].label is Label.HOLD
    assert "3 of 3" in (held[0].note or "")


async def test_a_few_unreadable_text_frames_are_skipped_not_held() -> None:
    fs = frames(4)
    ocr = FakeOcr(
        {
            fs[0].jpeg: OcrResult("#$%^ &*()", None, 3),  # unreadable
            fs[1].jpeg: OcrResult("hello friends", 0.95, 1),  # readable
            fs[2].jpeg: OcrResult("hello friends", 0.95, 1),
        }
    )
    judge = RecordingTextJudge()
    verdict = await checker(ScriptedImage({}), ocr, judge, share=0.5).check("v1", fs)
    assert not any(i.label is Label.HOLD for i in verdict.items)
    assert judge.calls == [["hello friends"]]
    assert [f.text for f in verdict.frames] == ["unreadable", "readable", "readable", "none"]


def test_text_state_rules() -> None:
    policy = D1_POLICY
    assert text_state(OcrResult("ab", 0.99, 1), policy) == "none"
    assert text_state(OcrResult("hello", None, 1), policy) == "unreadable"
    assert text_state(OcrResult("hello", 0.3, 1), policy) == "unreadable"
    assert text_state(OcrResult("hello", 0.9, 1), policy) == "readable"


async def test_timings_are_recorded_for_each_step() -> None:
    fs = frames(2)
    verdict = await checker(ScriptedImage({}), FakeOcr(), RecordingTextJudge()).check("v1", fs)
    assert set(verdict.timings) == {"score_s", "ocr_s", "text_s"}
