"""The five signals and the final score of one post. Pure functions: same input, same score."""

from __future__ import annotations

from typing import Any

from recommendation.config import RecommendationSettings, get_settings


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Keep a numeric value within a range."""
    return max(minimum, min(value, maximum))


def calculate_topic_score(post_topics: list[str], user_topic_weights: dict[str, float]) -> float:
    """How relevant the post's topics are to the user. No topics: 0. Capped at 1."""
    if not post_topics:
        return 0.0
    total = sum(max(0.0, float(user_topic_weights.get(topic, 0.0))) for topic in post_topics)
    return clamp(total)


def calculate_relationship_score(
    is_followed: bool,
    creator_id: str,
    recent_creator_engagement: dict[str, float],
) -> float:
    """Followed creators get 1.0. Others get half of the user's recent engagement with that creator."""
    if is_followed:
        return 1.0
    engagement = float(recent_creator_engagement.get(str(creator_id), 0.0))
    return clamp(0.5 * engagement)


def calculate_freshness_score(
    age_hours: float,
    is_followed: bool,
    cfg: RecommendationSettings | None = None,
) -> float:
    """Exponential decay: 72 h half-life for followed creators, 24 h for the rest. A negative age is an error."""
    cfg = cfg or get_settings()
    if age_hours < 0:
        raise ValueError("age_hours must not be negative")
    half_life = cfg.followed_freshness_half_life_hours if is_followed else cfg.default_freshness_half_life_hours
    return 0.5 ** (age_hours / half_life)


def calculate_engagement_score(
    likes: int,
    saves: int,
    impressions: int,
    cfg: RecommendationSettings | None = None,
) -> float:
    """Smoothed engagement quality: (likes + save_weight*saves + k*prior) / (impressions + k), clamped to [0, 1].
    A negative count is an error."""
    cfg = cfg or get_settings()
    if min(likes, saves, impressions) < 0:
        raise ValueError("likes, saves and impressions must not be negative")
    numerator = likes + cfg.save_weight * saves + cfg.engagement_smoothing_k * cfg.engagement_prior
    denominator = impressions + cfg.engagement_smoothing_k
    if denominator == 0:
        return 0.0
    return clamp(numerator / denominator)


def calculate_exploration_score(source: str, cfg: RecommendationSettings | None = None) -> float:
    """1 for posts that came from the exploration source, else 0."""
    cfg = cfg or get_settings()
    return 1.0 if source == cfg.exploration_source else 0.0


def score_signals(
    post: dict[str, Any],
    user_features: dict[str, Any],
    cfg: RecommendationSettings,
) -> dict[str, float]:
    """The five signals (each 0 to 1) and the penalty of one post, before any weighting or rounding.

    post: creator_id, topics, age_hours, likes, saves, impressions, is_followed, source, penalty (optional)
    user_features: topic_weights, recent_creator_engagement
    """
    is_followed = bool(post.get("is_followed", False))
    return {
        "topic": calculate_topic_score(post.get("topics", []), user_features.get("topic_weights", {})),
        "relationship": calculate_relationship_score(
            is_followed, str(post.get("creator_id", "")), user_features.get("recent_creator_engagement", {})
        ),
        "freshness": calculate_freshness_score(post.get("age_hours", 0), is_followed, cfg),
        "engagement": calculate_engagement_score(
            post.get("likes", 0), post.get("saves", 0), post.get("impressions", 0), cfg
        ),
        "exploration": calculate_exploration_score(post.get("source", ""), cfg),
        "penalty": clamp(float(post.get("penalty", cfg.default_penalty))),
    }


def weighted_parts(signals: dict[str, float], cfg: RecommendationSettings) -> dict[str, float]:
    """Each signal times its weight. Their sum, minus the penalty, is the score before the floor at 0."""
    return {
        "topic": cfg.topic_weight * signals["topic"],
        "relationship": cfg.relationship_weight * signals["relationship"],
        "freshness": cfg.freshness_weight * signals["freshness"],
        "engagement": cfg.engagement_weight * signals["engagement"],
        "exploration": cfg.exploration_weight * signals["exploration"],
    }


def final_score(signals: dict[str, float], cfg: RecommendationSettings) -> float:
    """The weighted signals added up, minus the penalty, never below 0."""
    return max(0.0, sum(weighted_parts(signals, cfg).values()) - signals["penalty"])


def calculate_post_score(
    post: dict[str, Any],
    user_features: dict[str, Any],
    cfg: RecommendationSettings | None = None,
) -> dict[str, Any]:
    """The final score of one post, with the rounded breakdown."""
    cfg = cfg or get_settings()
    signals = score_signals(post, user_features, cfg)
    return {
        "post_id": post["post_id"],
        "creator_id": post.get("creator_id"),
        "topics": post.get("topics", []),
        "score": round(final_score(signals, cfg), 6),
        "score_breakdown": {name: round(value, 6) for name, value in signals.items()},
    }
