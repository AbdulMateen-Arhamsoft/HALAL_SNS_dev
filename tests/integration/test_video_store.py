"""D2 against real Postgres: the video lane, the frame rows written with the decision, the per-user count, evidence retention."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace

import pytest

from moderation.domain import (
    Decision,
    Label,
    ModerationRequest,
    VideoFrameRecord,
    VideoRef,
)
from moderation.store import LANE_FAST, LANE_VIDEO, Store
from tests.integration.conftest import run_sql


def video_request(content_id: str = "reel_1", author: str = "u_1", version: int = 1) -> ModerationRequest:
    return ModerationRequest(
        request_id=str(uuid.uuid4()),
        surface="reel",
        content_id=content_id,
        content_version=version,
        author_id=author,
        videos=[VideoRef(media_id="v1", url="https://clips.local/v1.mp4")],
        callback_url="http://cb.local/hook",
    )


def fast_request(content_id: str = "post_1") -> ModerationRequest:
    return ModerationRequest(
        request_id=str(uuid.uuid4()),
        surface="post",
        content_id=content_id,
        content_version=1,
        text="hello",
        callback_url="http://cb.local/hook",
    )


def frame(frame_no: int, flagged: bool, evidence: str | None = None) -> VideoFrameRecord:
    return VideoFrameRecord(
        media_id="v1",
        frame_no=frame_no,
        time_ms=frame_no * 1000,
        source="scene" if flagged else "sample",
        scores={"G-03": 0.97 if flagged else 0.02},
        flagged=flagged,
        evidence_path=evidence,
    )


def video_decision(*frames: VideoFrameRecord) -> Decision:
    return Decision(
        label=Label.BLOCK,
        category="G-03",
        reason="Video v1: Sexual content and nudity",
        policy_version="d2.0.0",
        video_frames=tuple(frames),
    )


@pytest.fixture
async def video_store(store: Store, db_url: str) -> Store:
    """The D1 store fixture already applied both migrations; nothing else to set up."""
    return store


async def test_a_video_post_goes_to_the_video_lane(video_store: Store, db_url: str) -> None:
    await video_store.create_job(video_request())
    [(lane,)] = await run_sql(db_url, "SELECT lane FROM job")
    assert lane == LANE_VIDEO


async def test_a_text_or_image_post_stays_in_the_fast_lane(video_store: Store, db_url: str) -> None:
    await video_store.create_job(fast_request())
    [(lane,)] = await run_sql(db_url, "SELECT lane FROM job")
    assert lane == LANE_FAST


async def test_each_loop_claims_only_its_own_lane(video_store: Store) -> None:
    await video_store.create_job(video_request("reel_a"))
    await video_store.create_job(fast_request("post_b"))
    assert await video_store.claim_job(LANE_FAST) is not None  # the fast post
    assert await video_store.claim_job(LANE_FAST) is None  # never the video
    video = await video_store.claim_job(LANE_VIDEO)
    assert video is not None and video.request.videos[0].media_id == "v1"


async def test_a_video_notification_carries_its_lane(video_store: Store) -> None:
    lanes: list[str] = []

    async def listen() -> None:
        async for lane in video_store.listen_for_jobs():
            lanes.append(lane)
            return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0.3)  # let LISTEN start
    await video_store.create_job(video_request("reel_n"))
    await asyncio.wait_for(listener, timeout=5)
    assert lanes == [LANE_VIDEO]


async def test_saving_a_decision_writes_its_frames(video_store: Store, db_url: str) -> None:
    await video_store.create_job(video_request("reel_frames"))
    job = await video_store.claim_job(LANE_VIDEO)
    assert job is not None
    saved = await video_store.save_decision(
        job, video_decision(frame(0, False), frame(12, True, evidence=f"{job.job_id}/v1/00000012_12000ms.jpg"))
    )
    assert saved is not None
    rows = await run_sql(
        db_url, "SELECT frame_no, time_ms, source, flagged, evidence_path FROM video_frame ORDER BY frame_no"
    )
    assert rows == [
        (0, 0, "sample", False, None),
        (12, 12000, "scene", True, f"{job.job_id}/v1/00000012_12000ms.jpg"),
    ]


async def test_a_stale_worker_writes_no_frames(video_store: Store, db_url: str) -> None:
    """Lease fencing: when another worker re-claimed the job, nothing (not even the frames) is written."""
    await video_store.create_job(video_request("reel_stale"))
    job = await video_store.claim_job(LANE_VIDEO)
    assert job is not None
    await run_sql(db_url, "UPDATE job SET attempts = attempts + 1")  # another worker took it over
    saved = await video_store.save_decision(job, video_decision(frame(0, True)))
    assert saved is None
    assert await run_sql(db_url, "SELECT count(*) FROM video_frame") == [(0,)]
    assert await run_sql(db_url, "SELECT count(*) FROM decision") == [(0,)]


async def test_a_video_decision_reserves_a_callback_with_rejected_videos(video_store: Store) -> None:
    await video_store.create_job(video_request("reel_cb"))
    job = await video_store.claim_job(LANE_VIDEO)
    assert job is not None
    decision = replace(
        video_decision(frame(0, False)),
        label=Label.HOLD,
        category=None,
        reason="Video v1: video file is unreadable or corrupt",
        rejected_videos=(),
    )
    saved = await video_store.save_decision(job, decision)
    assert saved is not None and saved.callback is not None
    assert saved.callback.body["event_type"] == "moderation.decided"
    assert "rejected_videos" not in saved.callback.body  # none in this decision


async def test_count_pending_videos_counts_only_that_authors_waiting_video_posts(video_store: Store) -> None:
    await video_store.create_job(video_request("reel_1", author="u_1"))
    await video_store.create_job(video_request("reel_2", author="u_1"))
    await video_store.create_job(video_request("reel_3", author="u_2"))
    await video_store.create_job(fast_request("post_9"))  # no video: not counted
    assert await video_store.count_pending_videos("u_1") == 2
    assert await video_store.count_pending_videos("u_2") == 1
    assert await video_store.count_pending_videos("nobody") == 0


async def test_a_finished_video_post_is_no_longer_pending(video_store: Store) -> None:
    await video_store.create_job(video_request("reel_done", author="u_3"))
    job = await video_store.claim_job(LANE_VIDEO)
    assert job is not None
    await video_store.save_decision(job, video_decision(frame(0, False)))
    assert await video_store.count_pending_videos("u_3") == 0


async def test_expired_evidence_is_listed_and_cleared(video_store: Store, db_url: str) -> None:
    await video_store.create_job(video_request("reel_ev"))
    job = await video_store.claim_job(LANE_VIDEO)
    assert job is not None
    old = f"{job.job_id}/v1/00000005_5000ms.jpg"
    fresh = f"{job.job_id}/v1/00000009_9000ms.jpg"
    await video_store.save_decision(job, video_decision(frame(5, True, old), frame(9, True, fresh)))
    await run_sql(
        db_url,
        "UPDATE video_frame SET created_at = now() - interval '40 days' WHERE evidence_path = %(p)s",
        {"p": old},
    )
    assert await video_store.expired_evidence(30) == [old]
    await video_store.clear_evidence([old])
    assert await video_store.expired_evidence(30) == []
    rows = dict(await run_sql(db_url, "SELECT frame_no, evidence_path FROM video_frame"))
    assert rows == {5: None, 9: fresh}  # the fresh evidence is kept


async def test_unflagged_frames_are_never_listed_as_evidence(video_store: Store, db_url: str) -> None:
    await video_store.create_job(video_request("reel_nf"))
    job = await video_store.claim_job(LANE_VIDEO)
    assert job is not None
    await video_store.save_decision(job, video_decision(frame(1, False)))
    await run_sql(db_url, "UPDATE video_frame SET created_at = now() - interval '90 days'")
    assert await video_store.expired_evidence(30) == []
