"""The recommendation route. Mounted by the root app.py; there is no separate server."""

from recommendation.api.routes import recommendations

__all__ = ["recommendations"]
