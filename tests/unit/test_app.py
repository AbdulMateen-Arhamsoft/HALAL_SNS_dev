"""The root app: moderation and recommendation behind one door."""

import uuid

import httpx
import psycopg
from api_fakes import FakeStore, make_settings, post_body

from app import create_app
from recommendation.config import RecommendationSettings


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def make_app():
    rec = RecommendationSettings(_env_file=None)
    return create_app(settings=make_settings(), store=FakeStore(), recommendation_settings=rec)  # type: ignore[arg-type]


REC_BODY = {
    "user_id": "u1",
    "candidates": [{"post_id": "p1", "creator_id": "c1", "age_hours": 1, "source": "follow"}],
}


async def test_serves_moderation_and_recommendation() -> None:
    async with client_for(make_app()) as client:
        mod = await client.post("/v1/moderations", json=post_body())
        rec = await client.post("/v1/recommendations", json=REC_BODY)
    assert mod.status_code == 202
    assert rec.status_code == 200
    assert rec.json()["ranked_posts"][0]["post_id"] == "p1"


async def test_healthz() -> None:
    async with client_for(make_app()) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_openapi_lists_both_paths() -> None:
    async with client_for(make_app()) as client:
        resp = await client.get("/openapi.json")
    spec = resp.json()
    assert "/v1/moderations" in spec["paths"]
    assert "/v1/recommendations" in spec["paths"]
    assert spec["info"]["title"] == "Halal SNS AI"
    assert "Recommendation" in [t["name"] for t in spec["tags"]]


class DownStore:
    """A Store whose database never answers."""

    def __init__(self, cfg) -> None:
        pass

    async def open(self) -> None:
        raise psycopg.OperationalError("connection refused")

    async def close(self) -> None:
        pass

    async def create_job(self, request):
        raise psycopg.OperationalError("connection refused")

    async def get_job_status(self, job_id):
        raise psycopg.OperationalError("connection refused")

    async def ping(self) -> None:
        raise psycopg.OperationalError("connection refused")


async def test_starts_with_database_down_recommendation_works_moderation_503(monkeypatch) -> None:
    monkeypatch.setattr("moderation.api.app.Store", DownStore)
    app = create_app(settings=make_settings(), recommendation_settings=RecommendationSettings(_env_file=None))
    async with app.router.lifespan_context(app), client_for(app) as client:  # startup must not raise
        rec = await client.post("/v1/recommendations", json=REC_BODY)
        mod = await client.get(f"/v1/moderations/{uuid.uuid4()}")
        ready = await client.get("/readyz")
        live = await client.get("/healthz")
    assert rec.status_code == 200
    assert mod.status_code == 503
    assert mod.json()["error"] == "unavailable"
    assert ready.status_code == 503
    assert live.status_code == 200
