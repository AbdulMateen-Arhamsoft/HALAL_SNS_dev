"""POST /v1/recommendations. Auth and the body cap are applied by the root app, like for moderation."""

import logging

from fastapi import APIRouter, Request

from moderation.api.schemas import ErrorResponse, Rejected
from recommendation.api.schemas import RecommendationRequest, RecommendationResponse
from recommendation.ranking import rank_posts

log = logging.getLogger("recommendation")

recommendations = APIRouter(prefix="/v1/recommendations", tags=["Recommendation"])


@recommendations.post(
    "",
    response_model=RecommendationResponse,
    summary="Rank the candidate posts for one user",
    responses={
        401: {"model": ErrorResponse, "description": "unauthorized: missing or invalid X-API-Key"},
        413: {"model": ErrorResponse, "description": "body_too_large"},
        422: {"model": ErrorResponse, "description": "limit_exceeded, duplicate post ids, or a malformed body"},
    },
)
def recommend(body: RecommendationRequest, request: Request) -> RecommendationResponse:
    """Score, sort and diversify the candidates and return all of them, best first, each with up to three reasons.

    The backend builds the candidates and must send **only posts moderation has allowed**. This service keeps no
    state and never reads the database, so the same request always gives the same order.
    """
    cfg = request.app.state.recommendation_cfg  # set by the root app.py
    if len(body.candidates) > cfg.max_candidates:
        raise Rejected(422, "limit_exceeded", f"too many candidates: {len(body.candidates)} > {cfg.max_candidates}")
    ids = [c.post_id for c in body.candidates]
    if len(ids) != len(set(ids)):
        raise Rejected(422, "duplicate_post_ids", "post_id must be unique within the request")

    ranked = rank_posts(
        [c.model_dump() for c in body.candidates],
        body.user_features.model_dump(),
        cfg,
    )
    log.info(
        "ranked",
        extra={"request_id": body.request_id, "user_id": body.user_id, "candidates": len(ranked)},
    )
    return RecommendationResponse(
        request_id=body.request_id,
        model=cfg.model_name,
        user_id=body.user_id,
        ranked_posts=ranked,
        total_candidates=len(ranked),
    )
