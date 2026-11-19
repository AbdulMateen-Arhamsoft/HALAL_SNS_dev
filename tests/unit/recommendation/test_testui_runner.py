"""The test page's runner must agree with the real ranking code."""

import importlib.util
import json
from pathlib import Path

import pytest

from recommendation.config import get_settings
from recommendation.ranking import rank_posts

ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "testui" / "recommendation" / "engine_runner.py"
SCENARIOS = sorted((ROOT / "tests" / "data" / "recommendation").glob("*.json"))


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("testui_recommendation_engine_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["candidates"], data["user_features"]


def test_scenarios_exist():
    assert len(SCENARIOS) >= 2


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.name)
def test_ranked_order_matches_rank_posts(runner, path):
    candidates, user_features = _load(path)
    result = runner.run_pipeline(candidates, user_features, dict(runner.DEFAULT_OVERRIDES))
    expected = rank_posts(candidates, user_features, get_settings())
    assert [p["post_id"] for p in result["ranked"]] == [p["post_id"] for p in expected]
    assert [p["score"] for p in result["ranked"]] == [p["score"] for p in expected]


def test_overrides_change_scores(runner):
    candidates, user_features = _load(SCENARIOS[-1])
    base = dict(runner.DEFAULT_OVERRIDES)
    changed = {**base, "TOPIC_WEIGHT": 0.0, "FRESHNESS_WEIGHT": 1.0}
    before = {p["post_id"]: p["score"] for p in runner.run_pipeline(candidates, user_features, base)["scored"]}
    after = {p["post_id"]: p["score"] for p in runner.run_pipeline(candidates, user_features, changed)["scored"]}
    assert before != after


def test_overrides_do_not_leak(runner):
    candidates, user_features = _load(SCENARIOS[0])
    runner.run_pipeline(candidates, user_features, {**runner.DEFAULT_OVERRIDES, "TOPIC_WEIGHT": 0.9})
    assert get_settings().topic_weight == 0.30


def test_explain_post_matches_score(runner):
    candidates, user_features = _load(SCENARIOS[-1])
    overrides = {**runner.DEFAULT_OVERRIDES, "SAVE_WEIGHT": 4}
    result = runner.run_pipeline(candidates, user_features, overrides)
    scores = {p["post_id"]: p["score"] for p in result["scored"]}
    post = candidates[0]
    explained = runner.explain_post(post, user_features, overrides)
    assert round(explained["final"]["score"], 6) == scores[post["post_id"]]


def test_runner_does_not_import_streamlit():
    assert "streamlit" not in RUNNER_PATH.read_text(encoding="utf-8")
