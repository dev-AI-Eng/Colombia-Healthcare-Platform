"""Reusable FastAPI dependency declarations."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import Settings, get_settings
from src.core.db import get_session
from src.core.pagination import PageRequest
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.registry.models import Clinic

MAX_PAGE_SIZE = 200


def page_request(
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Page size")] = 50,
    offset: Annotated[int, Query(ge=0, description="Rows to skip")] = 0,
) -> PageRequest:
    return PageRequest(limit=limit, offset=offset)


# scope="function" (FastAPI >= 0.121) closes the session when the endpoint returns,
# before the response is sent. With the default scope it would close after the
# response, so a failed commit could lose audit entries for data already delivered.
SessionDep = Annotated[AsyncSession, Depends(get_session, scope="function")]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PageDep = Annotated[PageRequest, Depends(page_request)]


async def clinic_scope(
    session: SessionDep,
    clinic_id: Annotated[uuid.UUID, Query(description="Clinic whose data this request may access")],
) -> ClinicScope:
    """Require a clinic and bind it to this request's transaction.

    Every route serving clinic data depends on this, so the database's row-level
    security has a clinic to compare against. Without it the policies match no
    row, which is the safe default but would look like an empty database.

    A clinic that does not exist is a 404 here. Left to the row-level policies
    it would match nothing, which reads as an empty list on a read, and on a
    write reaches the foreign key and becomes a 500.
    """
    scope = ClinicScope(clinic_id=clinic_id)
    await apply_clinic_scope(session, scope)
    if await session.scalar(select(Clinic.id).where(Clinic.id == clinic_id)) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clinic not found")
    return scope


ClinicScopeDep = Annotated[ClinicScope, Depends(clinic_scope)]


def require_found[T](value: T | None, what: str) -> T:
    """Return the value, or raise 404. Also covers rows hidden by clinic scope."""
    if value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found")
    return value
