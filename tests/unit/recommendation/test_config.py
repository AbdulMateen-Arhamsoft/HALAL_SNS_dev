import pytest
from pydantic import ValidationError

from recommendation.config import RecommendationSettings, get_settings


def defaults() -> RecommendationSettings:
    return RecommendationSettings(_env_file=None)


def test_defaults_equal_plan_values() -> None:
    cfg = defaults()
    assert cfg.topic_weight == 0.30
    assert cfg.relationship_weight == 0.25
    assert cfg.freshness_weight == 0.20
    assert cfg.engagement_weight == 0.15
    assert cfg.exploration_weight == 0.10
    assert cfg.default_freshness_half_life_hours == 24
    assert cfg.followed_freshness_half_life_hours == 72
    assert cfg.engagement_smoothing_k == 20
    assert cfg.engagement_prior == 0.05
    assert cfg.save_weight == 2
    assert cfg.max_consecutive_same_creator == 2
    assert cfg.max_consecutive_same_topic == 2
    assert cfg.max_candidates == 200
    assert cfg.model_name == "rules_v1"


def test_weights_sum_to_one() -> None:
    cfg = defaults()
    total = (
        cfg.topic_weight + cfg.relationship_weight + cfg.freshness_weight + cfg.engagement_weight + cfg.exploration_weight
    )
    assert total == pytest.approx(1.0)


def test_settings_are_frozen() -> None:
    with pytest.raises(ValidationError):
        defaults().topic_weight = 0.5  # type: ignore[misc]


def test_with_overrides_returns_a_changed_copy() -> None:
    cfg = defaults()
    changed = cfg.with_overrides(topic_weight=0.5, max_candidates=10)
    assert changed.topic_weight == 0.5
    assert changed.max_candidates == 10
    assert changed.relationship_weight == 0.25
    assert cfg.topic_weight == 0.30


def test_with_overrides_validates() -> None:
    cfg = defaults()
    with pytest.raises(ValidationError):
        cfg.with_overrides(topic_weight=-1)
    with pytest.raises(ValidationError):
        cfg.with_overrides(default_freshness_half_life_hours=0)
    with pytest.raises(ValidationError):
        cfg.with_overrides(max_consecutive_same_creator=0)
    with pytest.raises(ValidationError):
        cfg.with_overrides(max_body_bytes=10)


def test_rec_env_variable_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REC_TOPIC_WEIGHT", "0.5")
    monkeypatch.setenv("REC_MAX_CANDIDATES", "50")
    get_settings.cache_clear()
    try:
        cfg = get_settings()
        assert cfg.topic_weight == 0.5
        assert cfg.max_candidates == 50
        assert get_settings() is cfg  # cached
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
