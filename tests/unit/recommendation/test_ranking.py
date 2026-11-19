import pytest

from recommendation.config import RecommendationSettings
from recommendation.ranking import generate_reasons, rank_posts

CFG = RecommendationSettings(_env_file=None)


def cand(pid: str, creator: str = "c", **kw) -> dict:
    base = {"post_id": pid, "creator_id": creator, "topics": [], "age_hours": 1000, "source": "other"}
    base.update(kw)
    return base


def breakdown(**kw) -> dict:
    base = {"topic": 0.0, "relationship": 0.0, "freshness": 0.0, "engagement": 0.0, "exploration": 0.0}
    base.update(kw)
    return base


def test_sorted_best_first() -> None:
    cands = [
        cand("low", "a"),
        cand("high", "b", topics=["t"], age_hours=0, is_followed=True),
        cand("mid", "c", age_hours=0),
    ]
    out = rank_posts(cands, {"topic_weights": {"t": 1.0}}, CFG)
    assert [p["post_id"] for p in out] == ["high", "mid", "low"]
    scores = [p["score"] for p in out]
    assert scores == sorted(scores, reverse=True)


def test_every_candidate_returned_once() -> None:
    cands = [cand(str(i), "same") for i in range(7)]
    out = rank_posts(cands, {}, CFG)
    assert sorted(p["post_id"] for p in out) == sorted(str(i) for i in range(7))


def test_empty_candidates() -> None:
    assert rank_posts([], {}, CFG) == []


def test_too_many_candidates_raises() -> None:
    cfg = CFG.with_overrides(max_candidates=2)
    with pytest.raises(ValueError, match="Maximum 2"):
        rank_posts([cand("1"), cand("2"), cand("3")], {}, cfg)


def test_exactly_max_candidates_is_fine() -> None:
    cfg = CFG.with_overrides(max_candidates=2)
    assert len(rank_posts([cand("1", "a"), cand("2", "b")], {}, cfg)) == 2


def test_reasons_followed_topic_recent() -> None:
    out = rank_posts(
        [cand("1", age_hours=0, topics=["t"], is_followed=True, source="exploration", likes=50, impressions=50)],
        {"topic_weights": {"t": 0.5}},
        CFG,
    )
    # five reasons apply; only the first three are kept
    assert out[0]["reasons"] == ["followed creator", "topic match", "recent post"]


def test_reasons_individual() -> None:
    assert generate_reasons(breakdown(topic=0.1), False) == ["topic match"]
    assert generate_reasons(breakdown(freshness=0.5), False) == ["recent post"]
    assert generate_reasons(breakdown(freshness=0.49), False) == ["general relevance"]
    assert generate_reasons(breakdown(engagement=0.10), False) == ["good engagement"]
    assert generate_reasons(breakdown(exploration=1.0), False) == ["exploration"]
    assert generate_reasons(breakdown(), True) == ["followed creator"]


def test_general_relevance_when_nothing_applies() -> None:
    out = rank_posts([cand("1", impressions=10000)], {}, CFG)
    assert out[0]["reasons"] == ["general relevance"]


def test_reasons_never_exceed_three() -> None:
    reasons = generate_reasons(breakdown(topic=1, freshness=1, engagement=1, exploration=1), True)
    assert len(reasons) == 3


def test_diversity_is_applied() -> None:
    cands = [
        cand("1", "A", age_hours=0, is_followed=True),
        cand("2", "A", age_hours=1, is_followed=True),
        cand("3", "A", age_hours=2, is_followed=True),
        cand("4", "B", age_hours=500),
    ]
    out = rank_posts(cands, {}, CFG)
    assert [p["post_id"] for p in out] == ["1", "2", "4", "3"]
