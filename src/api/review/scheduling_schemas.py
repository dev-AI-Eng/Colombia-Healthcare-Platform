"""Request bodies for the scheduling write routes.

Separate from `schemas.py` because that file holds response models for the
read-only review API, and mixing request bodies into it blurs which direction
a model travels.

Times arrive as ISO 8601 with an offset and are required to carry one: a naive
datetime means the sender's zone is a guess, and a guessed appointment time is
an hour wrong twice a year in most of the world. Colombia has no daylight
saving, which makes a missing offset look harmless locally and wrong the moment
a caller runs anywhere else.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, Field, field_validator

#: An idempotency key is a caller's own identifier for one intent. A retried
#: request carrying the same key returns the original appointment instead of
#: writing a second one, which is what makes a dropped connection safe.
_IDEMPOTENCY = Field(
    default=None,
    max_length=120,
    description=(
        "A caller-chosen key for this request. Sending it again returns the "
        "appointment the first call created rather than booking twice."
    ),
)


def _require_offset(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Send the time with a UTC offset, e.g. 2026-11-03T09:00:00-05:00. "
            "A time without one leaves the zone to be guessed."
        )
    return value


class _Timed(BaseModel):
    """A request that names a time span, by end or by duration."""

    start: dt.datetime = Field(description="When it begins. ISO 8601 with an offset.")
    end: dt.datetime | None = Field(
        default=None,
        description="When it ends. Omit it and send `duration_minutes` instead.",
    )
    duration_minutes: int | None = Field(
        default=None,
        ge=1,
        le=8 * 60,
        description="How long it lasts, if `end` is not given.",
    )

    @field_validator("start", "end")
    @classmethod
    def _offset_required(cls, value: dt.datetime | None) -> dt.datetime | None:
        return None if value is None else _require_offset(value)


class BookIn(_Timed):
    """Book one appointment at a known time."""

    patient_id: uuid.UUID
    doctor_id: uuid.UUID
    location_id: uuid.UUID
    appointment_type_id: uuid.UUID
    idempotency_key: str | None = _IDEMPOTENCY


class HoldIn(_Timed):
    """Hold a slot while the patient decides."""

    patient_id: uuid.UUID
    doctor_id: uuid.UUID
    location_id: uuid.UUID
    appointment_type_id: uuid.UUID
    minutes: int = Field(
        default=15,
        ge=1,
        le=120,
        description=(
            "How long the hold lasts. The default matches the engine's: long "
            "enough for a WhatsApp reply, short enough not to block the diary."
        ),
    )
    idempotency_key: str | None = _IDEMPOTENCY


class RescheduleIn(_Timed):
    """Move an existing appointment to a new time."""

    idempotency_key: str | None = _IDEMPOTENCY


class CancelIn(BaseModel):
    """Cancel an appointment, optionally saying why.

    The reason is free text and optional here because a receptionist cancelling
    by phone has one in their own words. M4's reporting counts cancellations by
    `cancellation_reason`, so a caller that has a code should send it.
    """

    reason: str | None = Field(
        default=None,
        max_length=200,
        description="Why it was cancelled. Shown to staff and counted in reporting.",
    )
