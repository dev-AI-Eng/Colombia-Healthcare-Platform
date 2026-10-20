"""Read-only review API over synthetic data, mounted at /review.

Exists so the seeded data and the M0 safety properties can be inspected before
staff-facing features exist. It is replaced by the authenticated staff API in
M11. See access.py for when it is available.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.review import (
    audit,
    conversation,
    identity,
    live,
    patients,
    reference,
    scheduling,
    session,
)
from src.api.review.access import require_synthetic_review

router = APIRouter(
    prefix="/review",
    tags=["review (synthetic data only)"],
    dependencies=[Depends(require_synthetic_review)],
)
for module in (
    reference,
    patients,
    identity,
    audit,
    scheduling,
    session,
    conversation,
    live,
):
    router.include_router(module.router)

__all__ = ["router"]
