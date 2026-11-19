"""Helpers for the recommendation engine test UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommendation import diversity, ranking, scoring
from recommendation.config import RecommendationSettings, get_settings

DATA_DIR = ROOT / "tests" / "data" / "recommendation"

_DEFAULTS = get_settings()

# Keys are the names the sidebar uses; `build_config` lower-cases them into setting names.
DEFAULT_OVERRIDES = {
    "TOPIC_WEIGHT": _DEFAULTS.topic_weight,
    "RELATIONSHIP_WEIGHT": _DEFAULTS.relationship_weight,
    "FRESHNESS_WEIGHT": _DEFAULTS.freshness_weight,
    "ENGAGEMENT_WEIGHT": _DEFAULTS.engagement_weight,
    "EXPLORATION_WEIGHT": _DEFAULTS.exploration_weight,
    "DEFAULT_FRESHNESS_HALF_LIFE_HOURS": _DEFAULTS.default_freshness_half_life_hours,
    "FOLLOWED_FRESHNESS_HALF_LIFE_HOURS": _DEFAULTS.followed_freshness_half_life_hours,
    "ENGAGEMENT_SMOOTHING_K": _DEFAULTS.engagement_smoothing_k,
    "ENGAGEMENT_PRIOR": _DEFAULTS.engagement_prior,
    "SAVE_WEIGHT": _DEFAULTS.save_weight,
    "MAX_CONSECUTIVE_SAME_CREATOR": _DEFAULTS.max_consecutive_same_creator,
    "MAX_CONSECUTIVE_SAME_TOPIC": _DEFAULTS.max_consecutive_same_topic,
    "EXPLORATION_SOURCE": _DEFAULTS.exploration_source,
}


def list_scenario_files() -> list[Path]:
    """Return all JSON scenario files in the data folder."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.json"))


def load_scenario(path: Path) -> dict[str, Any]:
    """Load one candidate scenario JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_config(overrides: dict[str, Any]) -> RecommendationSettings:
    """The settings with the sidebar values applied (UPPER_CASE key -> snake_case setting). Nothing global changes."""
    return get_settings().with_overrides(**{key.lower(): value for key, value in overrides.items()})


def score_all(
    candidates: list[dict[str, Any]],
    user_features: dict[str, Any],
    cfg: RecommendationSettings,
) -> list[dict[str, Any]]:
    return [
        scoring.calculate_post_score(post=candidate, user_features=user_features, cfg=cfg)
        for candidate in candidates
    ]


def run_pipeline(
    candidates: list[dict[str, Any]],
    user_features: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the full ranking pipeline and return intermediate stages
    so the UI can show how each step works.
    """
    cfg = build_config(overrides)
    scored = score_all(candidates, user_features, cfg)
    sorted_posts = sorted(scored, key=lambda post: post["score"], reverse=True)
    diversified = diversity.apply_diversity(sorted_posts, cfg)

    candidate_lookup = {
        str(candidate["post_id"]): candidate for candidate in candidates
    }

    ranked = []
    for post in diversified:
        original = candidate_lookup[str(post["post_id"])]
        reasons = ranking.generate_reasons(
            score_breakdown=post["score_breakdown"],
            is_followed=bool(original.get("is_followed", False)),
        )
        ranked.append(
            {
                "post_id": post["post_id"],
                "creator_id": post.get("creator_id"),
                "topics": post.get("topics", []),
                "score": post["score"],
                "score_breakdown": post["score_breakdown"],
                "reasons": reasons,
                "source": original.get("source", ""),
                "is_followed": bool(original.get("is_followed", False)),
                "age_hours": original.get("age_hours", 0),
                "likes": original.get("likes", 0),
                "saves": original.get("saves", 0),
                "impressions": original.get("impressions", 0),
            }
        )

    diversity_moves = _find_diversity_moves(sorted_posts, diversified)

    return {
        "scored": scored,
        "sorted": sorted_posts,
        "diversified": diversified,
        "ranked": ranked,
        "diversity_moves": diversity_moves,
        "overrides": dict(overrides),
    }


def explain_post(
    post: dict[str, Any],
    user_features: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Build a step-by-step formula explanation for one post."""
    cfg = build_config(overrides)
    topic_weights = user_features.get("topic_weights", {})
    creator_engagement = user_features.get("recent_creator_engagement", {})

    topics = post.get("topics", [])
    topic_parts = [
        {
            "topic": topic,
            "weight": float(topic_weights.get(topic, 0.0)),
        }
        for topic in topics
    ]
    topic_raw = sum(max(0.0, part["weight"]) for part in topic_parts)
    topic_score = scoring.calculate_topic_score(topics, topic_weights)

    is_followed = bool(post.get("is_followed", False))
    creator_id = str(post.get("creator_id", ""))
    engagement_with_creator = float(
        creator_engagement.get(creator_id, 0.0)
    )
    relationship_score = scoring.calculate_relationship_score(
        is_followed=is_followed,
        creator_id=creator_id,
        recent_creator_engagement=creator_engagement,
    )

    age_hours = float(post.get("age_hours", 0))
    half_life = (
        cfg.followed_freshness_half_life_hours
        if is_followed
        else cfg.default_freshness_half_life_hours
    )
    freshness_score = scoring.calculate_freshness_score(age_hours, is_followed, cfg)

    likes = int(post.get("likes", 0))
    saves = int(post.get("saves", 0))
    impressions = int(post.get("impressions", 0))
    k = cfg.engagement_smoothing_k
    prior = cfg.engagement_prior
    save_weight = cfg.save_weight
    engagement_score = scoring.calculate_engagement_score(
        likes=likes,
        saves=saves,
        impressions=impressions,
        cfg=cfg,
    )

    source = post.get("source", "")
    exploration_score = scoring.calculate_exploration_score(source, cfg)
    penalty = scoring.clamp(float(post.get("penalty", cfg.default_penalty)))

    weighted = scoring.weighted_parts(
        {
            "topic": topic_score,
            "relationship": relationship_score,
            "freshness": freshness_score,
            "engagement": engagement_score,
            "exploration": exploration_score,
        },
        cfg,
    )
    final_before_floor = sum(weighted.values()) - penalty
    final_score = max(0.0, final_before_floor)

    return {
        "post_id": post.get("post_id"),
        "topic": {
            "parts": topic_parts,
            "raw_sum": topic_raw,
            "score": topic_score,
            "formula": "topic = min(1, sum(user_weight[t] for t in post.topics))",
        },
        "relationship": {
            "is_followed": is_followed,
            "creator_id": creator_id,
            "creator_engagement": engagement_with_creator,
            "score": relationship_score,
            "formula": (
                "relationship = 1.0 if followed else 0.5 * recent_creator_engagement"
            ),
        },
        "freshness": {
            "age_hours": age_hours,
            "half_life": half_life,
            "score": freshness_score,
            "formula": "freshness = 0.5 ** (age_hours / half_life)",
        },
        "engagement": {
            "likes": likes,
            "saves": saves,
            "impressions": impressions,
            "k": k,
            "prior": prior,
            "save_weight": save_weight,
            "numerator": likes + save_weight * saves + k * prior,
            "denominator": impressions + k,
            "score": engagement_score,
            "formula": (
                "engagement = (likes + save_weight*saves + k*prior) / (impressions + k)"
            ),
        },
        "exploration": {
            "source": source,
            "exploration_source": cfg.exploration_source,
            "score": exploration_score,
            "formula": "exploration = 1 if source == exploration else 0",
        },
        "final": {
            "weights": {
                "topic": cfg.topic_weight,
                "relationship": cfg.relationship_weight,
                "freshness": cfg.freshness_weight,
                "engagement": cfg.engagement_weight,
                "exploration": cfg.exploration_weight,
            },
            "weighted_parts": weighted,
            "penalty": penalty,
            "before_floor": final_before_floor,
            "score": final_score,
            "formula": (
                "score = w_t*topic + w_r*rel + w_f*fresh + w_e*eng + w_x*expl - penalty"
            ),
        },
    }


def _find_diversity_moves(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """List posts whose position changed because of diversity reordering."""
    before_pos = {
        str(post["post_id"]): index for index, post in enumerate(before)
    }
    moves = []
    for index, post in enumerate(after):
        post_id = str(post["post_id"])
        old_index = before_pos[post_id]
        if old_index != index:
            moves.append(
                {
                    "post_id": post_id,
                    "from": old_index + 1,
                    "to": index + 1,
                    "creator_id": post.get("creator_id"),
                    "topics": post.get("topics", []),
                }
            )
    return moves
