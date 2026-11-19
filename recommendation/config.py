"""Every recommendation setting. Defaults are the values in the implementation plan; REC_* variables override them."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RecommendationSettings(BaseSettings):
    """Frozen: nothing changes a setting at runtime. The test page passes a modified copy (`with_overrides`)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="REC_", extra="ignore", frozen=True)

    # --- Score weights (the plan keeps them summing to 1.0) ---
    topic_weight: float = Field(default=0.30, ge=0, description="Topic relevance")
    relationship_weight: float = Field(default=0.25, ge=0, description="Followed creator, or recent engagement with the creator")
    freshness_weight: float = Field(default=0.20, ge=0)
    engagement_weight: float = Field(default=0.15, ge=0)
    exploration_weight: float = Field(default=0.10, ge=0)

    # --- Freshness (hours) ---
    default_freshness_half_life_hours: float = Field(default=24, gt=0)
    followed_freshness_half_life_hours: float = Field(default=72, gt=0)

    # --- Engagement smoothing: (likes + save_weight*saves + k*prior) / (impressions + k) ---
    engagement_smoothing_k: float = Field(default=20, ge=0)
    engagement_prior: float = Field(default=0.05, ge=0)
    save_weight: float = Field(default=2, ge=0)

    # --- Diversity ---
    max_consecutive_same_creator: int = Field(default=2, ge=1)
    max_consecutive_same_topic: int = Field(default=2, ge=1)

    # --- Source name that earns the exploration score ---
    exploration_source: str = "exploration"

    # --- API ---
    model_name: str = "rules_v1"
    max_candidates: int = Field(default=200, ge=1)
    max_body_bytes: int = Field(
        default=1024 * 1024, ge=1024, description="Largest /v1/recommendations body (200 candidates are about 110 KB)"
    )

    # The plan has not fixed the penalty size; a caller may send one per candidate.
    default_penalty: float = Field(default=0.0, ge=0)

    def with_overrides(self, **values) -> RecommendationSettings:
        """A copy with some values changed (the test page, tests). Validated like the original."""
        return type(self).model_validate({**self.model_dump(), **values})


@lru_cache
def get_settings() -> RecommendationSettings:
    return RecommendationSettings()
