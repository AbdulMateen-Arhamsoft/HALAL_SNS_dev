"""D2 API: videos in a post, the video limits, the per-user limit, the video upload. Uses the D1 fake store."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from moderation.store import JobSnapshot
from tests.unit.test_api import FakeStore, client_for, post_body


class VideoStore(FakeStore):
    """The D1 fake plus the one call D2 adds: how many video posts an author has waiting."""

    def __init__(self, pending: int = 0) -> None:
        super().__init__()
        self.pending = pending
        self.counted: list[str] = []

    async def count_pending_videos(self, author_id: str) -> int:
        self.counted.append(author_id)
        return self.pending


def video_body(**overrides: Any) -> dict[str, Any]:
    body = post_body(images=[], text="look at this", videos=[{"media_id": "v1", "url": "https://cdn.example/v1.mp4"}])
    body["author_id"] = "u_1"
    body.update(overrides)
    return body


async def test_post_with_a_video_is_answered_pending_at_once() -> None:
    store = VideoStore()
    async with client_for(store, api_wait_s=5.0) as client:  # would wait 5 s for a fast-lane post
        resp = await client.post("/v1/moderations", json=video_body())
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"
    assert store.counted == ["u_1"]


async def test_finished_video_post_answers_200() -> None:
    store = VideoStore()
    job_id = uuid.uuid4()
    store.rows[job_id] = JobSnapshot(job_id, "done", None, "allow", None, "Allowed", 900)
    store.by_key[("post", "post_123", 1)] = job_id
    async with client_for(store) as client:
        resp = await client.post("/v1/moderations", json=video_body())
    assert resp.status_code == 200
    assert resp.json()["label"] == "allow"


async def test_video_post_without_author_is_422() -> None:
    body = video_body()
    del body["author_id"]
    async with client_for(VideoStore()) as client:
        resp = await client.post("/v1/moderations", json=body)
    assert resp.status_code == 422
    assert "author_id is required" in resp.text


@pytest.mark.parametrize("surface", ["story", "reel"])
async def test_story_and_reel_are_accepted(surface: str) -> None:
    async with client_for(VideoStore()) as client:
        resp = await client.post("/v1/moderations", json=video_body(surface=surface))
    assert resp.status_code == 202


async def test_more_videos_than_api_max_videos_is_422() -> None:
    body = video_body(videos=[{"media_id": "v1", "url": "https://cdn.example/1.mp4"}, {"media_id": "v2", "url": "https://cdn.example/2.mp4"}])
    async with client_for(VideoStore(), api_max_videos=1) as client:
        resp = await client.post("/v1/moderations", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"] == "limit_exceeded"


async def test_no_videos_allowed_when_api_max_videos_is_zero() -> None:
    async with client_for(VideoStore(), api_max_videos=0) as client:
        resp = await client.post("/v1/moderations", json=video_body())
    assert resp.status_code == 422


async def test_author_at_the_pending_limit_gets_429() -> None:
    async with client_for(VideoStore(pending=3), video_max_pending_per_user=3) as client:
        resp = await client.post("/v1/moderations", json=video_body())
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited"


async def test_author_below_the_pending_limit_is_accepted() -> None:
    async with client_for(VideoStore(pending=2), video_max_pending_per_user=3) as client:
        resp = await client.post("/v1/moderations", json=video_body())
    assert resp.status_code == 202


async def test_fast_posts_are_not_counted_against_the_video_limit() -> None:
    store = VideoStore(pending=99)
    async with client_for(store) as client:
        resp = await client.post("/v1/moderations", json=post_body())
    assert resp.status_code == 202
    assert store.counted == []


async def test_image_and_video_share_one_media_id_namespace() -> None:
    body = video_body(images=[{"media_id": "m1", "url": "https://cdn.example/m1.jpg"}], videos=[{"media_id": "m1", "url": "https://cdn.example/v.mp4"}])
    async with client_for(VideoStore()) as client:
        resp = await client.post("/v1/moderations", json=body)
    assert resp.status_code == 422


# --- video upload ----------------------------------------------------------------------

MP4_HEADER = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
EBML_HEADER = b"\x1a\x45\xdf\xa3" + b"\x00" * 64


async def test_video_upload_disabled_without_image_allowed_root() -> None:
    async with client_for(VideoStore(), image_allowed_root="") as client:
        resp = await client.post("/v1/moderations/uploads/video", files={"file": ("a.mp4", MP4_HEADER, "video/mp4")})
    assert resp.status_code == 503


async def test_video_upload_stores_an_mp4_and_returns_its_url(tmp_path: Path) -> None:
    async with client_for(VideoStore(), image_allowed_root=str(tmp_path)) as client:
        resp = await client.post("/v1/moderations/uploads/video", files={"file": ("a.mp4", MP4_HEADER, "video/mp4")})
    assert resp.status_code == 201
    body = resp.json()
    assert body["url"] == f"uploads/{body['media_id']}.mp4"
    assert (tmp_path / body["url"]).read_bytes() == MP4_HEADER


async def test_video_upload_accepts_webm_and_mkv_as_mkv(tmp_path: Path) -> None:
    async with client_for(VideoStore(), image_allowed_root=str(tmp_path)) as client:
        resp = await client.post("/v1/moderations/uploads/video", files={"file": ("a.webm", EBML_HEADER, "video/webm")})
    assert resp.status_code == 201
    assert resp.json()["url"].endswith(".mkv")


async def test_non_video_upload_is_422(tmp_path: Path) -> None:
    async with client_for(VideoStore(), image_allowed_root=str(tmp_path)) as client:
        resp = await client.post("/v1/moderations/uploads/video", files={"file": ("a.txt", b"hello world", "text/plain")})
    assert resp.status_code == 422
    assert list(tmp_path.iterdir()) == []


async def test_video_upload_larger_than_video_max_bytes_is_refused(tmp_path: Path) -> None:
    """Over the limit plus the small margin: 413 from the body cap (as D1's images). Just over the limit: 422."""
    async with client_for(VideoStore(), image_allowed_root=str(tmp_path), video_max_bytes=1024 * 1024) as client:
        too_big = await client.post(
            "/v1/moderations/uploads/video", files={"file": ("a.mp4", MP4_HEADER + b"\x00" * (2 * 1024 * 1024), "video/mp4")}
        )
        just_over = await client.post(
            "/v1/moderations/uploads/video", files={"file": ("a.mp4", MP4_HEADER + b"\x00" * (1024 * 1024), "video/mp4")}
        )
    assert too_big.status_code == 413
    assert just_over.status_code == 422


async def test_video_upload_is_not_capped_at_the_json_body_limit(tmp_path: Path) -> None:
    """Regression: a 300 KB video is far above the 128 KB JSON cap, but a legal upload must still get through."""
    payload = MP4_HEADER + b"\x00" * (300 * 1024)
    async with client_for(VideoStore(), image_allowed_root=str(tmp_path)) as client:
        resp = await client.post("/v1/moderations/uploads/video", files={"file": ("a.mp4", payload, "video/mp4")})
    assert resp.status_code == 201


async def test_json_post_still_capped_at_the_json_body_limit() -> None:
    big = video_body(text="x" * 200_000)
    async with client_for(VideoStore()) as client:
        resp = await client.post("/v1/moderations", json=big)
    assert resp.status_code == 413


async def test_video_upload_needs_the_api_key_when_keys_are_set(tmp_path: Path) -> None:
    async with client_for(VideoStore(), image_allowed_root=str(tmp_path), api_keys="alice:secret") as client:
        resp = await client.post("/v1/moderations/uploads/video", files={"file": ("a.mp4", MP4_HEADER, "video/mp4")})
    assert resp.status_code == 401

