"""Request and response bodies of POST /v1/recommendations."""

from typing import Any

from pydantic import BaseModel, Field


class CandidatePost(BaseModel):
    post_id: str = Field(description="The backend's post id. Unique within the request")
    creator_id: str
    topics: list[str] = Field(default_factory=list)
    age_hours: float = Field(ge=0, description="Hours since the post was published")
    likes: int = Field(default=0, ge=0)
    saves: int = Field(default=0, ge=0)
    impressions: int = Field(default=0, ge=0)
    is_followed: bool = Field(default=False, description="The user follows this creator")
    source: str = Field(description="Which candidate source found it. 'exploration' earns the exploration score")
    penalty: float = Field(default=0.0, ge=0, description="Optional, 0 to 1. Subtracted from the score")


class UserFeatures(BaseModel):
    topic_weights: dict[str, float] = Field(default_factory=dict, description="Topic -> how much the user likes it, 0 to 1")
    recent_creator_engagement: dict[str, float] = Field(
        default_factory=dict, description="Creator id -> recent engagement, 0 to 1"
    )
    impressions_count: int = Field(default=0, ge=0)
    positive_actions_count: int = Field(default=0, ge=0)


class RecommendationRequest(BaseModel):
    request_id: str | None = Field(default=None, description="Optional. Echoed in the response and the log")
    user_id: str
    candidates: list[CandidatePost] = Field(min_length=1)
    user_features: UserFeatures = Field(default_factory=UserFeatures)
    context: dict[str, Any] | None = Field(default=None, description="Optional. Accepted, not used for ranking yet")


class RankedPost(BaseModel):
    post_id: str
    score: float
    reasons: list[str]


class RecommendationResponse(BaseModel):
    request_id: str | None = None
    model: str = Field(description="Ranking model name, e.g. rules_v1")
    user_id: str
    ranked_posts: list[RankedPost] = Field(description="Best first. Every candidate is returned")
    total_candidates: int
