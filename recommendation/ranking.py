"""Score, sort, diversify and explain the candidates. No database: the backend supplies the candidates."""

from __future__ import annotations

from typing import Any

from recommendation.config import RecommendationSettings, get_settings
from recommendation.diversity import apply_diversity
from recommendation.scoring import calculate_post_score


def generate_reasons(score_breakdown: dict[str, float], is_followed: bool) -> list[str]:
    """Up to three short, readable reasons for a ranking."""
    reasons = []
    if is_followed:
        reasons.append("followed creator")
    if score_breakdown["topic"] > 0:
        reasons.append("topic match")
    if score_breakdown["freshness"] >= 0.5:
        reasons.append("recent post")
    if score_breakdown["engagement"] >= 0.10:
        reasons.append("good engagement")
    if score_breakdown["exploration"] > 0:
        reasons.append("exploration")
    if not reasons:
        reasons.append("general relevance")
    return reasons[:3]


def rank_posts(
    candidates: list[dict[str, Any]],
    user_features: dict[str, Any],
    cfg: RecommendationSettings | None = None,
) -> list[dict[str, Any]]:
    """Score every candidate, sort by score, apply diversity and add reasons.
    The candidate limit is checked by the route (recommendation/api/routes.py), not here."""
    cfg = cfg or get_settings()
    scored = [calculate_post_score(candidate, user_features, cfg) for candidate in candidates]
    scored.sort(key=lambda post: post["score"], reverse=True)
    diversified = apply_diversity(scored, cfg)

    followed = {str(c["post_id"]): bool(c.get("is_followed", False)) for c in candidates}
    return [
        {
            "post_id": post["post_id"],
            "score": post["score"],
            "reasons": generate_reasons(post["score_breakdown"], followed[str(post["post_id"])]),
        }
        for post in diversified
    ]
