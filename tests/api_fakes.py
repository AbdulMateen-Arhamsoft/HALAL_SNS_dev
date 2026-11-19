"""Shared doubles for tests that build the HTTP app: settings without a database and an in-memory store."""

from __future__ import annotations

import uuid
from typing import Any

import psycopg

from moderation.config import Settings
from moderation.domain import ModerationRequest
from moderation.store import JobSnapshot


def make_settings(**overrides: Any) -> Settings:
    overrides.setdefault("api_wait_s", 0.0)
    overrides.setdefault("api_keys", "")  # tests must not depend on a real .env's API_KEYS
    return Settings(database_url="postgresql://test:test@localhost:5432/test", **overrides)


class FakeStore:
    """In-memory stand-in for the store calls the moderation API makes."""

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, JobSnapshot] = {}
        self.by_key: dict[tuple[str, str, int], uuid.UUID] = {}
        self.down = False

    async def create_job(self, request: ModerationRequest) -> tuple[uuid.UUID, str]:
        self._check()
        key = (request.surface, request.content_id, request.content_version)
        if key not in self.by_key:
            job_id = uuid.uuid4()
            self.by_key[key] = job_id
            self.rows[job_id] = JobSnapshot(job_id, "queued", None, None, None, None)
        job_id = self.by_key[key]
        return job_id, self.rows[job_id].status

    async def get_job_status(self, job_id: uuid.UUID) -> JobSnapshot | None:
        self._check()
        return self.rows.get(job_id)

    async def ping(self) -> None:
        self._check()

    def _check(self) -> None:
        if self.down:
            raise psycopg.OperationalError("connection refused")


def post_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "request_id": "req-1",
        "surface": "post",
        "content_id": "post_123",
        "content_version": 1,
        "text": "a caption",
        "images": [{"media_id": "m1", "url": "https://cdn.example/m1.jpg"}],
        "callback_url": "http://backend.local/moderation-result",
    }
    body.update(overrides)
    return body
