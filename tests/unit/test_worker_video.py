"""D2 worker: flagged-frame evidence, its retention sweep, and the lane wake-ups. No database, no models."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from moderation.config import Settings
from moderation.domain import Decision, Label, VideoFrameRecord
from moderation.store import LANE_FAST, LANE_VIDEO
from moderation.worker import Worker

DB = "postgresql://u:p@localhost:5432/x"


def cfg(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=DB,
        video_evidence_root=str(tmp_path / "evidence"),
        video_retention_days=30,
    )


class EvidenceStore:
    """Only the calls the evidence sweep and the listener make."""

    def __init__(self, expired: list[str] | None = None, lanes: list[str] | None = None) -> None:
        self.expired = list(expired or [])
        self.cleared: list[list[str]] = []
        self.lanes = list(lanes or [])
        self.retention_asked: list[int] = []

    async def expired_evidence(self, days: int) -> list[str]:
        self.retention_asked.append(days)
        return list(self.expired)

    async def clear_evidence(self, paths: list[str]) -> None:
        self.cleared.append(list(paths))

    async def listen_for_jobs(self) -> AsyncIterator[str]:
        for lane in self.lanes:
            yield lane


def worker(tmp_path: Path, store: Any = None) -> Worker:
    return Worker(cfg(tmp_path), store or EvidenceStore(), moderator=None, callbacks=None)  # type: ignore[arg-type]


def decision_with(records: list[VideoFrameRecord]) -> Decision:
    return Decision(label=Label.BLOCK, category="G-03", reason="Video v1: Sexual content and nudity", video_frames=tuple(records))


def record(frame_no: int, time_ms: int, flagged: bool, jpeg: bytes = b"") -> VideoFrameRecord:
    return VideoFrameRecord(media_id="v1", frame_no=frame_no, time_ms=time_ms, source="sample", scores={}, flagged=flagged, jpeg=jpeg)


class Job:
    job_id = uuid.UUID("6f1c0c8e-3d52-4d0e-8c56-0a1f0c3b9e11")


# --- evidence ---------------------------------------------------------------------------


def test_only_flagged_frames_are_written_as_evidence(tmp_path: Path) -> None:
    decision = decision_with([record(12, 400, True, b"JPEG-BYTES"), record(13, 433, False)])
    saved = worker(tmp_path)._write_evidence(Job(), decision)  # type: ignore[arg-type]

    flagged, unflagged = saved.video_frames
    expected = f"{Job.job_id}/v1/00000012_400ms.jpg"
    assert flagged.evidence_path == expected
    assert flagged.jpeg == b""  # the bytes are on disk now, not kept in memory
    assert (tmp_path / "evidence" / expected).read_bytes() == b"JPEG-BYTES"
    assert unflagged.evidence_path is None
    assert not (tmp_path / "evidence" / str(Job.job_id) / "v1" / "00000013_433ms.jpg").exists()


def test_a_decision_without_frames_is_returned_unchanged(tmp_path: Path) -> None:
    decision = decision_with([])
    assert worker(tmp_path)._write_evidence(Job(), decision) is decision  # type: ignore[arg-type]


# --- retention sweep -----------------------------------------------------------------------


async def test_expired_evidence_is_deleted_and_its_path_cleared(tmp_path: Path) -> None:
    rel = f"{Job.job_id}/v1/00000012_400ms.jpg"
    target = tmp_path / "evidence" / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    store = EvidenceStore(expired=[rel])
    await worker(tmp_path, store)._sweep_evidence()
    assert not target.exists()
    assert store.cleared == [[rel]]
    assert store.retention_asked == [30]


async def test_a_missing_file_is_still_cleared_from_the_database(tmp_path: Path) -> None:
    rel = "gone/v1/0_0ms.jpg"
    store = EvidenceStore(expired=[rel])
    await worker(tmp_path, store)._sweep_evidence()
    assert store.cleared == [[rel]]


async def test_a_path_outside_the_evidence_folder_is_never_deleted(tmp_path: Path) -> None:
    outside = tmp_path / "keep-me.txt"
    outside.write_text("do not delete")
    (tmp_path / "evidence").mkdir()
    store = EvidenceStore(expired=["../keep-me.txt"])
    await worker(tmp_path, store)._sweep_evidence()
    assert outside.exists()
    assert store.cleared == [[]]  # nothing it may delete, so nothing cleared


async def test_an_old_file_with_no_database_row_is_removed_by_age(tmp_path: Path) -> None:
    """A worker that lost its lease wrote a file but no row points to it: the age sweep removes it."""
    import os

    orphan = tmp_path / "evidence" / "lost-job" / "v1" / "00000001_1000ms.jpg"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    long_ago = 0.0  # 1970: far beyond the 30-day retention
    os.utime(orphan, (long_ago, long_ago))
    recent = tmp_path / "evidence" / "lost-job" / "v1" / "00000002_2000ms.jpg"
    recent.write_bytes(b"recent")
    await worker(tmp_path, EvidenceStore())._sweep_evidence()
    assert not orphan.exists()
    assert recent.exists()  # inside the retention period: kept


async def test_nothing_expired_does_nothing(tmp_path: Path) -> None:
    store = EvidenceStore(expired=[])
    await worker(tmp_path, store)._sweep_evidence()
    assert store.cleared == [[]]


# --- lanes ----------------------------------------------------------------------------------


async def test_a_video_notification_wakes_only_the_video_loop(tmp_path: Path) -> None:
    w = worker(tmp_path, EvidenceStore(lanes=[LANE_VIDEO]))
    await w.listen_for_jobs()
    assert w._wakes[LANE_VIDEO].is_set()
    assert not w._wakes[LANE_FAST].is_set()


async def test_a_fast_notification_wakes_only_the_fast_loop(tmp_path: Path) -> None:
    w = worker(tmp_path, EvidenceStore(lanes=[LANE_FAST]))
    await w.listen_for_jobs()
    assert w._wakes[LANE_FAST].is_set()
    assert not w._wakes[LANE_VIDEO].is_set()


async def test_an_unknown_lane_wakes_nothing(tmp_path: Path) -> None:
    w = worker(tmp_path, EvidenceStore(lanes=["mystery"]))
    await w.listen_for_jobs()
    assert not any(wake.is_set() for wake in w._wakes.values())


def test_stop_wakes_every_loop_so_they_can_exit(tmp_path: Path) -> None:
    w = worker(tmp_path)
    w.stop()
    assert w._stop.is_set()
    assert all(wake.is_set() for wake in w._wakes.values())
