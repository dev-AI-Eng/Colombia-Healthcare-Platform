"""Queries for consents and phone bindings.

Every function takes the clinic explicitly; row-level security enforces the same
boundary in the database. Consents and bindings are patient data, so every
function records one audit entry per record returned, and none of them returns
records belonging to a soft-deleted patient.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_accesses
from src.core.pagination import PageRequest, PageResult, fetch_page
from src.identity.models import Consent, ConsentPurpose, PhoneBinding
from src.registry.models import Patient


@dataclass(frozen=True, slots=True)
class HandsetMember:
    binding: PhoneBinding
    given_names: str
    family_names: str


@dataclass(frozen=True, slots=True)
class SharedHandset:
    """One phone number bound to more than one patient (ADR-18).

    `reference` is the first 6 bytes of the number's blind index in hex. It
    groups members without revealing the number, and is stable only while the
    blind-index key is unchanged.
    """

    reference: str
    members: list[HandsetMember]


async def _record_consents(session: AsyncSession, consents: list[Consent]) -> None:
    await record_accesses(
        session,
        AccessAction.READ,
        [
            Access(resource="consents", resource_id=str(c.id), patient_id=c.patient_id)
            for c in consents
        ],
    )


async def list_consents(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    page: PageRequest,
    patient_id: uuid.UUID | None = None,
    purpose: ConsentPurpose | None = None,
    active_only: bool = False,
) -> PageResult[Consent]:
    statement = (
        select(Consent)
        .join(Patient, Patient.id == Consent.patient_id)
        .where(Consent.clinic_id == clinic_id, Patient.deleted_at.is_(None))
        .order_by(Consent.granted_at, Consent.id)
    )
    if patient_id is not None:
        statement = statement.where(Consent.patient_id == patient_id)
    if purpose is not None:
        statement = statement.where(Consent.purpose == purpose)
    if active_only:
        statement = statement.where(Consent.revoked_at.is_(None))
    rows, total = await fetch_page(session, statement, page)
    consents = [row[0] for row in rows]
    await _record_consents(session, consents)
    return PageResult(items=consents, total=total, limit=page.limit, offset=page.offset)


async def list_patient_consents(
    session: AsyncSession, *, clinic_id: uuid.UUID, patient_id: uuid.UUID
) -> list[Consent]:
    result = await session.scalars(
        select(Consent)
        .join(Patient, Patient.id == Consent.patient_id)
        .where(
            Consent.clinic_id == clinic_id,
            Consent.patient_id == patient_id,
            Patient.deleted_at.is_(None),
        )
        .order_by(Consent.granted_at)
    )
    consents = list(result.all())
    await _record_consents(session, consents)
    return consents


async def withdraw_consents(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    patient_id: uuid.UUID,
    purpose: ConsentPurpose,
    now: dt.datetime,
) -> list[Consent]:
    """Revoke every live consent of one purpose, and return those revoked.

    The row is kept and dated rather than deleted: it is the record that the
    patient once agreed and then withdrew, and the consent gate reads
    `revoked_at` to tell a patient who opted out from one who never opted in.
    Already-revoked rows are left alone, so a second request changes nothing.
    """
    consents = list(
        (
            await session.scalars(
                select(Consent)
                .where(
                    Consent.clinic_id == clinic_id,
                    Consent.patient_id == patient_id,
                    Consent.purpose == purpose,
                    Consent.revoked_at.is_(None),
                )
                .with_for_update()
            )
        ).all()
    )
    for consent in consents:
        # The CHECK constraint requires revoked_at >= granted_at, and a consent
        # recorded with a future grant date must still be revocable today.
        consent.revoked_at = max(now, consent.granted_at)
    await session.flush()
    await record_accesses(
        session,
        AccessAction.UPDATE,
        [
            Access(resource="consents", resource_id=str(c.id), patient_id=c.patient_id)
            for c in consents
        ],
    )
    return consents


async def list_patient_bindings(
    session: AsyncSession, *, clinic_id: uuid.UUID, patient_id: uuid.UUID
) -> list[PhoneBinding]:
    result = await session.scalars(
        select(PhoneBinding)
        .join(Patient, Patient.id == PhoneBinding.patient_id)
        .where(
            PhoneBinding.clinic_id == clinic_id,
            PhoneBinding.patient_id == patient_id,
            Patient.deleted_at.is_(None),
        )
        .order_by(PhoneBinding.created_at)
    )
    bindings = list(result.all())
    await record_accesses(
        session,
        AccessAction.READ,
        [
            Access(resource="phone_bindings", resource_id=str(b.id), patient_id=b.patient_id)
            for b in bindings
        ],
    )
    return bindings


async def list_shared_handsets(
    session: AsyncSession, *, clinic_id: uuid.UUID
) -> list[SharedHandset]:
    """Numbers with more than one unrevoked binding to an active patient of the clinic."""
    shared_numbers = (
        select(PhoneBinding.phone_e164_bidx)
        .join(Patient, Patient.id == PhoneBinding.patient_id)
        .where(
            PhoneBinding.clinic_id == clinic_id,
            PhoneBinding.revoked_at.is_(None),
            Patient.deleted_at.is_(None),
        )
        .group_by(PhoneBinding.phone_e164_bidx)
        .having(func.count() > 1)
    )
    rows = await session.execute(
        select(PhoneBinding, Patient.given_names, Patient.family_names)
        .join(Patient, Patient.id == PhoneBinding.patient_id)
        .where(
            PhoneBinding.clinic_id == clinic_id,
            PhoneBinding.revoked_at.is_(None),
            Patient.deleted_at.is_(None),
            PhoneBinding.phone_e164_bidx.in_(shared_numbers),
        )
        .order_by(PhoneBinding.phone_e164_bidx, PhoneBinding.created_at)
    )

    groups: dict[bytes, list[HandsetMember]] = {}
    for binding, given, family in rows.tuples():
        groups.setdefault(binding.phone_e164_bidx, []).append(
            HandsetMember(binding=binding, given_names=given, family_names=family)
        )

    await record_accesses(
        session,
        AccessAction.READ,
        [
            Access(
                resource="phone_bindings",
                resource_id=str(member.binding.id),
                patient_id=member.binding.patient_id,
            )
            for members in groups.values()
            for member in members
        ],
    )
    return [
        SharedHandset(reference=bidx[:6].hex(), members=members) for bidx, members in groups.items()
    ]
