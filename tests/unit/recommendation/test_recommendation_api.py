import json
from pathlib import Path
from typing import Any

import httpx
from api_fakes import FakeStore, make_settings

from app import create_app
from recommendation.config import RecommendationSettings

DATA = Path(__file__).resolve().parents[2] / "data" / "recommendation"
URL = "/v1/recommendations"


def client_for(*, api_keys: str = "", max_body_bytes: int = 1024 * 1024) -> httpx.AsyncClient:
    rec = RecommendationSettings(_env_file=None).with_overrides(max_body_bytes=max_body_bytes)
    app = create_app(settings=make_settings(api_keys=api_keys), store=FakeStore(), recommendation_settings=rec)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def candidate(pid: str, **kw: Any) -> dict[str, Any]:
    base = {"post_id": pid, "creator_id": f"c_{pid}", "topics": ["t"], "age_hours": 1, "source": "follow"}
    base.update(kw)
    return base


def body(n: int = 3, **kw: Any) -> dict[str, Any]:
    base = {"user_id": "u1", "candidates": [candidate(str(i)) for i in range(n)]}
    base.update(kw)
    return base


async def test_happy_path_with_200_candidates_file() -> None:
    payload = load("recommendation_candidates_test.json")
    assert (DATA / "recommendation_candidates_test.json").stat().st_size > 100_000  # ~108 KB, must not trip the cap
    assert len(payload["candidates"]) == 200
    async with client_for() as client:
        resp = await client.post(URL, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == payload["request_id"]
    assert data["user_id"] == payload["user_id"]
    assert data["model"] == "rules_v1"
    assert data["total_candidates"] == 200
    sent = sorted(c["post_id"] for c in payload["candidates"])
    assert sorted(p["post_id"] for p in data["ranked_posts"]) == sent
    for ranked in data["ranked_posts"]:
        assert 1 <= len(ranked["reasons"]) <= 3
        assert ranked["score"] >= 0


async def test_new_user_without_follows_or_interests() -> None:
    payload = load("new_user_no_follows_no_interests.json")
    async with client_for() as client:
        resp = await client.post(URL, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] is None
    assert data["total_candidates"] == len(payload["candidates"])
    assert len(data["ranked_posts"]) == len(payload["candidates"])


async def test_request_id_is_echoed() -> None:
    async with client_for() as client:
        resp = await client.post(URL, json=body(request_id="abc-123"))
    assert resp.json()["request_id"] == "abc-123"


async def test_response_is_best_first() -> None:
    async with client_for() as client:
        resp = await client.post(URL, json=body(5))
    scores = [p["score"] for p in resp.json()["ranked_posts"]]
    assert scores == sorted(scores, reverse=True)


async def test_same_request_twice_gives_identical_output() -> None:
    payload = load("recommendation_candidates_test.json")
    async with client_for() as client:
        first = await client.post(URL, json=payload)
        second = await client.post(URL, json=payload)
    assert first.json() == second.json()


async def test_duplicate_post_ids_are_422() -> None:
    payload = body(0, candidates=[candidate("x"), candidate("x", creator_id="other")])
    async with client_for() as client:
        resp = await client.post(URL, json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"] == "duplicate_post_ids"
    assert "detail" in resp.json()


async def test_zero_candidates_is_422() -> None:
    async with client_for() as client:
        resp = await client.post(URL, json=body(0))
    assert resp.status_code == 422


async def test_201_candidates_is_422() -> None:
    async with client_for() as client:
        resp = await client.post(URL, json=body(201))
    assert resp.status_code == 422
    assert resp.json()["error"] == "limit_exceeded"


async def test_200_candidates_is_accepted() -> None:
    async with client_for() as client:
        resp = await client.post(URL, json=body(200))
    assert resp.status_code == 200
    assert resp.json()["total_candidates"] == 200


async def test_negative_age_hours_is_422() -> None:
    payload = body(0, candidates=[candidate("a", age_hours=-1)])
    async with client_for() as client:
        resp = await client.post(URL, json=payload)
    assert resp.status_code == 422


async def test_missing_user_id_is_422() -> None:
    payload = body()
    del payload["user_id"]
    async with client_for() as client:
        resp = await client.post(URL, json=payload)
    assert resp.status_code == 422


async def test_api_key_required_when_configured() -> None:
    async with client_for(api_keys="alice:secret123") as client:
        missing = await client.post(URL, json=body())
        wrong = await client.post(URL, json=body(), headers={"X-API-Key": "nope"})
        ok = await client.post(URL, json=body(), headers={"X-API-Key": "secret123"})
    for resp in (missing, wrong):
        assert resp.status_code == 401
        assert resp.json() == {"error": "unauthorized", "detail": "missing or invalid X-API-Key"}
    assert ok.status_code == 200


async def test_open_when_no_api_keys() -> None:
    async with client_for(api_keys="") as client:
        resp = await client.post(URL, json=body())
    assert resp.status_code == 200


async def test_body_over_cap_is_413() -> None:
    async with client_for(max_body_bytes=1024) as client:
        resp = await client.post(URL, json=body(50))
    assert resp.status_code == 413
    assert resp.json()["error"] == "body_too_large"


async def test_small_body_under_small_cap_is_accepted() -> None:
    async with client_for(max_body_bytes=1024) as client:
        resp = await client.post(URL, json=body(1))
    assert resp.status_code == 200
