"""The Check door (HTTP API). Run with: uvicorn moderation.api:create_app --factory (moderation only).

The combined entry point, moderation and recommendation, is app:create_app (uvicorn app:create_app --factory).
"""

from moderation.api.app import create_app

__all__ = ["create_app"]
