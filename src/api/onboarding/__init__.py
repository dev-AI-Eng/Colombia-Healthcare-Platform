"""Spreadsheet onboarding API, mounted at /onboarding.

Behind the same gate as the review API: synthetic-data mode only, and the
server refuses to bind anywhere but loopback while it is enabled. This endpoint
accepts files and shows their contents back, so exposing it without staff
authentication would publish whatever was uploaded.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.onboarding import review, routes
from src.api.review.access import require_synthetic_review

router = APIRouter(
    prefix="/onboarding",
    tags=["onboarding (synthetic data only)"],
    dependencies=[Depends(require_synthetic_review)],
)
router.include_router(routes.router)
router.include_router(review.router)

__all__ = ["router"]
