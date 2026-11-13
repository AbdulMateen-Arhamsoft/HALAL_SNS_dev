"""D2 settings: every limit from plan section 9 has its planned value, and bad values are refused."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from moderation.config import Settings

DB = "postgresql://u:p@localhost:5432/x"


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, database_url=DB, **overrides)  # type: ignore[call-arg]


def test_defaults_match_the_plan() -> None:
    cfg = make()
    assert cfg.video_max_duration_s == 300  # decision 6
    assert cfg.video_sample_fps == 1.0  # decision 2
    assert cfg.video_max_frames == 240  # decision 5
    assert cfg.video_frame_side == 384  # section 4.6
    assert cfg.video_scene_threshold == 30  # decision 3, assumption
    assert cfg.video_dedup_distance == 5  # decision 4, assumption
    assert cfg.video_workers == 3  # raised from 1: videos are prepared in parallel, scored one at a time on the GPU
    assert cfg.video_parallel == 3
    assert cfg.video_max_waiting == 4
    assert cfg.api_max_videos == 3  # the plan (section 7) started at 1; 3 by decision, changeable by configuration
    assert cfg.video_hard_block_codes == frozenset({"G-01", "G-03", "G-05"})  # open item 4, proposed


def test_placeholders_have_values() -> None:
    cfg = make()
    assert cfg.video_max_bytes == 100 * 1024 * 1024
    assert cfg.video_max_pixels == 1920 * 1080
    assert cfg.video_fetch_timeout_s == 60.0
    assert cfg.video_max_pending_per_user == 3
    assert cfg.video_retention_days == 30


@pytest.mark.parametrize(
    "field,value",
    [
        ("video_max_duration_s", 0),
        ("video_sample_fps", 0),
        ("video_sample_fps", 31),
        ("video_max_frames", 0),
        ("video_frame_side", 16),
        ("video_scene_threshold", 0),
        ("video_dedup_distance", 65),
        ("video_workers", 0),
        ("video_workers", 9),
        ("video_parallel", 0),
        ("video_parallel", 9),
        ("video_max_waiting", 0),
        ("video_max_waiting", 17),
        ("video_text_unreadable_share", 0),
        ("video_text_unreadable_share", 1.5),
        ("api_max_videos", 101),
    ],
)
def test_out_of_range_values_are_refused(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_hard_block_codes_are_parsed_and_normalised() -> None:
    cfg = make(video_hard_block_categories=" g-03 , G-05 ")
    assert cfg.video_hard_block_codes == frozenset({"G-03", "G-05"})


def test_empty_hard_block_list_is_allowed() -> None:
    assert make(video_hard_block_categories="").video_hard_block_codes == frozenset()


@pytest.mark.parametrize("bad", ["G-9", "G-08", "H-01", "violence", "G-01,G-99"])
def test_unknown_hard_block_code_is_an_error_not_ignored(bad: str) -> None:
    cfg = make(video_hard_block_categories=bad)
    with pytest.raises(ValueError, match="unknown codes"):
        _ = cfg.video_hard_block_codes
