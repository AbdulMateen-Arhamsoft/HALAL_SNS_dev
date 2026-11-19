"""Spread creators and topics: delay a post that would make too many in a row. Never drops a post."""

from __future__ import annotations

from typing import Any

from recommendation.config import RecommendationSettings, get_settings


def has_same_topic(first_topics: list[str], second_topics: list[str]) -> bool:
    """True when two posts share at least one topic."""
    return bool(set(first_topics) & set(second_topics))


def violates_diversity(
    candidate: dict[str, Any],
    selected_posts: list[dict[str, Any]],
    cfg: RecommendationSettings | None = None,
) -> bool:
    """Would adding the candidate make more than the allowed number of consecutive posts of one creator or topic?"""
    cfg = cfg or get_settings()
    candidate_creator = candidate.get("creator_id")
    candidate_topics = candidate.get("topics", [])
    max_creator = cfg.max_consecutive_same_creator
    max_topic = cfg.max_consecutive_same_topic

    if len(selected_posts) >= max_creator and all(
        post.get("creator_id") == candidate_creator for post in selected_posts[-max_creator:]
    ):
        return True

    return bool(
        candidate_topics
        and len(selected_posts) >= max_topic
        and all(has_same_topic(post.get("topics", []), candidate_topics) for post in selected_posts[-max_topic:])
    )


def apply_diversity(
    ranked_posts: list[dict[str, Any]],
    cfg: RecommendationSettings | None = None,
) -> list[dict[str, Any]]:
    """Reorder scored posts to reduce repeated creators and topics, keeping score order as far as possible.

    A post that violates the rule is delayed, not removed. If every remaining post violates it, the
    highest-ranked remaining post goes next, so no candidate is lost.
    """
    cfg = cfg or get_settings()
    remaining = ranked_posts.copy()
    selected: list[dict[str, Any]] = []

    while remaining:
        selected_index = next(
            (i for i, candidate in enumerate(remaining) if not violates_diversity(candidate, selected, cfg)),
            0,
        )
        selected.append(remaining.pop(selected_index))

    return selected
