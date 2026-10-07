"""Clinics, their locations, their doctors and their patients.

Every table except `clinics` carries `clinic_id`. Isolation between clinics is
not enforced yet: repository queries accept an optional `clinic_id` filter, some
lookups by id do not filter by clinic, and row-level security is not enabled.
Enforcement arrives with staff authentication, which establishes a clinic per
request.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.core.crypto import EncryptedString
from src.core.db import Base, string_enum


class DocumentType(StrEnum):
    """The RIPS identification document types, in full.

    These are the values of `tipoDocumentoIdentificacion` listed in MinSalud's
    Documento Técnico 1 (versión de lanzamiento, 29 April 2026), which
    Resolución 948 de 2026 made the authoritative table; that resolution
    repealed Resolución 2275 de 2023. Codes are two uppercase letters.

    Because the table now lives in a technical document rather than in the
    resolution itself, it can change without a new norm. Re-check it before M1
    imports real clinic files.

    RIPS restricts some of these by context: CN, RC and MS identify newborns,
    and AS and MS are not accepted on an electronic invoice. Those rules belong
    to the import and billing paths, not to this enumeration.
    """

    RC = "RC"  # Registro civil
    TI = "TI"  # Tarjeta de identidad
    CC = "CC"  # Cédula de ciudadanía
    CE = "CE"  # Cédula de extranjería
    CD = "CD"  # Carné diplomático
    PA = "PA"  # Pasaporte
    SC = "SC"  # Salvoconducto de permanencia
    PE = "PE"  # Permiso especial de permanencia
    DE = "DE"  # Documento extranjero
    PT = "PT"  # Permiso por protección temporal
    CN = "CN"  # Certificado de nacido vivo
    AS = "AS"  # Adulto sin identificar
    MS = "MS"  # Menor sin identificar


class Clinic(Base):
    __tablename__ = "clinics"
    __table_args__ = ({"schema": "app"},)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    timezone: Mapped[str] = mapped_column(String(64), server_default="America/Bogota")
    locale: Mapped[str] = mapped_column(String(16), server_default="es-CO")
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Location(Base):
    __tablename__ = "locations"
    __table_args__ = (Index("ix_locations_clinic_id", "clinic_id"), {"schema": "app"})

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    name: Mapped[str] = mapped_column(String(200))
    address: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Doctor(Base):
    __tablename__ = "doctors"
    __table_args__ = (
        Index(
            "uq_doctors_clinic_id_external_ref",
            "clinic_id",
            "external_ref",
            unique=True,
            postgresql_where=text("external_ref IS NOT NULL"),
        ),
        Index("ix_doctors_clinic_id", "clinic_id"),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    full_name: Mapped[str] = mapped_column(String(200))
    # Free-text specialty as the clinic writes it. `specialty_id` points at the
    # clinic's own catalogue when their file supplies one (migration 0005);
    # both are kept because not every clinic has a catalogue.
    specialty: Mapped[str] = mapped_column(String(100))
    specialty_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app.specialties.id"))
    office_number: Mapped[str | None] = mapped_column(String(32))
    external_ref: Mapped[str | None] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Specialty(Base):
    """A clinic's own catalogue of specialties, as its export lists them.

    `external_ref` is the clinic's code ("E01"). It resolves references inside
    the uploaded file and matches the row again on a later import.
    """

    __tablename__ = "specialties"
    __table_args__ = (
        Index(
            "uq_specialties_clinic_id_external_ref",
            "clinic_id",
            "external_ref",
            unique=True,
            postgresql_where=text("external_ref IS NOT NULL"),
        ),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    external_ref: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Patient(Base):
    """A patient of one clinic.

    Direct identifiers are encrypted at rest. `document_number_bidx` and
    `phone_e164_bidx` are HMAC blind indexes that allow exact-match lookup
    without decrypting (see src/core/crypto.py).
    """

    __tablename__ = "patients"
    __table_args__ = (
        Index(
            "uq_patients_clinic_id_document",
            "clinic_id",
            "document_type",
            "document_number_bidx",
            unique=True,
        ),
        Index(
            "uq_patients_clinic_id_external_ref",
            "clinic_id",
            "external_ref",
            unique=True,
            postgresql_where=text("external_ref IS NOT NULL"),
        ),
        Index(
            "ix_patients_clinic_id_phone_e164_bidx",
            "clinic_id",
            "phone_e164_bidx",
            postgresql_where=text("deleted_at IS NULL"),
        ),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    document_type: Mapped[DocumentType] = mapped_column(
        string_enum(DocumentType, "document_type_valid")
    )
    document_number: Mapped[str] = mapped_column(EncryptedString)
    document_number_bidx: Mapped[bytes] = mapped_column(LargeBinary)
    given_names: Mapped[str] = mapped_column(EncryptedString)
    # One column holds both surnames, the usual Colombian form.
    family_names: Mapped[str] = mapped_column(EncryptedString)
    birth_date: Mapped[date | None] = mapped_column(Date)
    phone_e164: Mapped[str | None] = mapped_column(EncryptedString)
    phone_e164_bidx: Mapped[bytes | None] = mapped_column(LargeBinary)
    email: Mapped[str | None] = mapped_column(EncryptedString)
    # Fields the clinic's own export carries (migration 0005). `eps` is the
    # insurer's name as written by the clinic: matched loosely, stored verbatim,
    # because the official EPS list changes and historical rows name defunct ones.
    eps: Mapped[str | None] = mapped_column(String(120))
    secondary_contact_name: Mapped[str | None] = mapped_column(String(200))
    # Encrypted like every other direct identifier.
    secondary_contact_phone: Mapped[str | None] = mapped_column(EncryptedString)
    telegram_chat_id: Mapped[str | None] = mapped_column(String(64))
    # The clinic's own key for this row, so a re-import updates instead of duplicating.
    external_ref: Mapped[str | None] = mapped_column(String(64))
    # Soft-deletion marker. Every repository query that returns patient details
    # (registry, scheduling, identity) excludes patients where it is set.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
