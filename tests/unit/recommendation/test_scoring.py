import math

import pytest

from recommendation.config import RecommendationSettings
from recommendation.scoring import (
    calculate_engagement_score,
    calculate_exploration_score,
    calculate_freshness_score,
    calculate_post_score,
    calculate_relationship_score,
    calculate_topic_score,
    clamp,
)

CFG = RecommendationSettings(_env_file=None)


def test_clamp() -> None:
    assert clamp(-1) == 0.0
    assert clamp(2) == 1.0
    assert clamp(0.4) == 0.4
    assert clamp(5, 0, 10) == 5


def test_topic_score_sums_matching_weights() -> None:
    assert calculate_topic_score(["a", "b"], {"a": 0.4, "b": 0.5, "c": 0.9}) == pytest.approx(0.9)


def test_topic_score_is_capped_at_one() -> None:
    assert calculate_topic_score(["a", "b"], {"a": 0.8, "b": 0.7}) == 1.0


def test_topic_score_no_topics_or_unknown_topics_is_zero() -> None:
    assert calculate_topic_score([], {"a": 1.0}) == 0.0
    assert calculate_topic_score(["z"], {"a": 1.0}) == 0.0


def test_topic_score_ignores_negative_weights() -> None:
    assert calculate_topic_score(["a", "b"], {"a": -0.5, "b": 0.3}) == pytest.approx(0.3)


def test_relationship_followed_is_one() -> None:
    assert calculate_relationship_score(True, "c1", {}) == 1.0


def test_relationship_is_half_of_recent_engagement() -> None:
    assert calculate_relationship_score(False, "c1", {"c1": 0.6}) == pytest.approx(0.3)


def test_relationship_clamped_and_unknown_creator() -> None:
    assert calculate_relationship_score(False, "c1", {"c1": 3.0}) == 1.0
    assert calculate_relationship_score(False, "c1", {"c1": -3.0}) == 0.0
    assert calculate_relationship_score(False, "c2", {"c1": 0.6}) == 0.0


def test_freshness_default_half_life_is_24h() -> None:
    assert calculate_freshness_score(0, False, CFG) == 1.0
    assert calculate_freshness_score(24, False, CFG) == pytest.approx(0.5)
    assert calculate_freshness_score(48, False, CFG) == pytest.approx(0.25)


def test_freshness_followed_half_life_is_72h() -> None:
    assert calculate_freshness_score(72, True, CFG) == pytest.approx(0.5)
    assert calculate_freshness_score(24, True, CFG) == pytest.approx(0.5 ** (1 / 3))
    assert calculate_freshness_score(24, True, CFG) > calculate_freshness_score(24, False, CFG)


def test_freshness_negative_age_is_treated_as_zero() -> None:
    assert calculate_freshness_score(-10, False, CFG) == 1.0


def test_engagement_no_data_is_the_prior() -> None:
    # (0 + 20 * 0.05) / (0 + 20)
    assert calculate_engagement_score(0, 0, 0, CFG) == pytest.approx(0.05)


def test_engagement_golden_value() -> None:
    # (10 + 2*5 + 1) / (100 + 20) = 21 / 120
    assert calculate_engagement_score(10, 5, 100, CFG) == pytest.approx(0.175)


def test_engagement_saves_count_double() -> None:
    assert calculate_engagement_score(0, 10, 100, CFG) > calculate_engagement_score(10, 0, 100, CFG)


def test_engagement_clamped_to_one_and_negatives_floored() -> None:
    assert calculate_engagement_score(1000, 0, 1, CFG) == 1.0
    assert calculate_engagement_score(-5, -5, -5, CFG) == pytest.approx(0.05)


def test_engagement_zero_denominator_is_zero() -> None:
    cfg = CFG.with_overrides(engagement_smoothing_k=0)
    assert calculate_engagement_score(0, 0, 0, cfg) == 0.0


def test_exploration_score() -> None:
    assert calculate_exploration_score("exploration", CFG) == 1.0
    assert calculate_exploration_score("followed", CFG) == 0.0
    assert calculate_exploration_score("", CFG) == 0.0


POST = {
    "post_id": "p1",
    "creator_id": "c1",
    "topics": ["tech"],
    "age_hours": 24,
    "likes": 10,
    "saves": 5,
    "impressions": 100,
    "is_followed": False,
    "source": "exploration",
}
FEATURES = {"topic_weights": {"tech": 0.5}, "recent_creator_engagement": {"c1": 0.4}}


def test_final_score_hand_computed() -> None:
    # 0.30*0.5 + 0.25*0.2 + 0.20*0.5 + 0.15*0.175 + 0.10*1 = 0.42625
    result = calculate_post_score(POST, FEATURES, CFG)
    assert result["score"] == pytest.approx(0.42625)
    assert result["score_breakdown"] == {
        "topic": 0.5,
        "relationship": 0.2,
        "freshness": 0.5,
        "engagement": 0.175,
        "exploration": 1.0,
        "penalty": 0.0,
    }
    assert result["post_id"] == "p1"
    assert result["creator_id"] == "c1"
    assert result["topics"] == ["tech"]


def test_penalty_is_subtracted() -> None:
    result = calculate_post_score({**POST, "penalty": 0.1}, FEATURES, CFG)
    assert result["score"] == pytest.approx(0.32625)
    assert result["score_breakdown"]["penalty"] == 0.1


def test_penalty_never_pushes_score_below_zero() -> None:
    weak = {"post_id": "p", "creator_id": "c", "topics": [], "age_hours": 10000, "source": "x"}
    result = calculate_post_score({**weak, "penalty": 1.0}, {}, CFG)
    assert result["score"] == 0.0


def test_penalty_is_clamped_to_one() -> None:
    result = calculate_post_score({**POST, "penalty": 5}, FEATURES, CFG)
    assert result["score_breakdown"]["penalty"] == 1.0
    assert result["score"] == 0.0


def test_default_penalty_comes_from_settings() -> None:
    cfg = CFG.with_overrides(default_penalty=0.1)
    assert calculate_post_score(POST, FEATURES, cfg)["score"] == pytest.approx(0.32625)


def test_missing_user_features_do_not_crash() -> None:
    result = calculate_post_score({"post_id": "p", "creator_id": "c"}, {}, CFG)
    assert math.isfinite(result["score"])
    assert result["score_breakdown"]["topic"] == 0.0


def test_followed_post_uses_longer_half_life_in_final_score() -> None:
    followed = calculate_post_score({**POST, "is_followed": True}, FEATURES, CFG)
    assert followed["score_breakdown"]["relationship"] == 1.0
    assert followed["score_breakdown"]["freshness"] == pytest.approx(0.5 ** (1 / 3), abs=1e-6)
