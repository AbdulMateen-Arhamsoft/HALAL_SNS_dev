"""The single entry point of the Halal SNS AI service: moderation and recommendation behind one HTTP API.

Run with: uvicorn app:create_app --factory

Moderation keeps its own worker process (python -m moderation.worker); this file is only the HTTP door.
"""

from fastapi import Depends

from moderation.api import create_app as create_moderation_app
from moderation.api.routes import require_api_key
from recommendation.api import recommendations
from recommendation.config import get_settings as get_recommendation_settings

RECOMMENDATIONS_PATH = "/v1/recommendations"


def create_app(*, settings=None, store=None, recommendation_settings=None):
    rec_cfg = recommendation_settings or get_recommendation_settings()
    app = create_moderation_app(
        settings=settings,
        store=store,
        # Recommendation needs no database: a down Postgres must not take the feed ranking with it.
        tolerate_db_down=True,
        extra_body_caps={RECOMMENDATIONS_PATH: rec_cfg.max_body_bytes},
    )
    app.title = "Halal SNS AI"
    app.state.recommendation_cfg = rec_cfg
    app.include_router(recommendations, dependencies=[Depends(require_api_key)])
    app.openapi_tags = [
        *(app.openapi_tags or []),
        {"name": "Recommendation", "description": "Rank the candidate posts the backend sends for one user."},
    ]
    return app
