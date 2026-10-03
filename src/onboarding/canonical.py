"""The fields a clinic file can be mapped onto, and the words clinics use for them.

This is the target of the whole pipeline: the matcher proposes a column for each
of these, a human confirms it, and the normalizers convert values into them.

The alias lists are the reason most files need no model call at all. They come
from the vocabulary Colombian clinic exports actually use, including the
unaccented spellings that are the norm rather than the exception: `telefono`,
`identificacion`, `medico`, `numero`. Matching strips accents before comparing,
so only one spelling of each word needs to appear here.

Adding an alias is the cheapest possible improvement to mapping accuracy, and it
is deterministic: unlike a model, a dictionary entry behaves the same way on
every file forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class Entity(StrEnum):
    """Which canonical table a sheet becomes."""

    PATIENT = "patient"
    DOCTOR = "doctor"
    SPECIALTY = "specialty"
    AVAILABILITY = "availability"
    APPOINTMENT = "appointment"


class Requirement(StrEnum):
    REQUIRED = "required"  # the entity cannot be built without it
    OPTIONAL = "optional"
    IGNORED = "ignored"  # recognised, deliberately not imported


@dataclass(frozen=True, slots=True)
class Field:
    """One canonical field, and how a clinic might have labelled it."""

    name: str
    entity: Entity
    requirement: Requirement
    #: Shown on the confirmation screen and sent to the model as the field's
    #: meaning. Written for a person, not for a schema.
    description: str
    aliases: tuple[str, ...] = ()
    #: Which normalizer converts values for this field. The matcher never picks
    #: a transform; this binding is fixed in code (ADR-08a).
    normalizer: str | None = None
    examples: tuple[str, ...] = field(default=())


_PATIENT: Final = (
    Field(
        "document_type",
        Entity.PATIENT,
        Requirement.REQUIRED,
        "The kind of identity document: cédula, tarjeta de identidad, pasaporte.",
        aliases=(
            "tipo de documento",
            "tipo documento",
            "tipo doc",
            "tipodoc",
            "td",
            "t.d.",
            "tipo de identificacion",
            "tipo identificacion",
            "clase de documento",
            "tipo de id",
            "id type",
            "document type",
        ),
        normalizer="document_type",
        examples=("CC", "TI", "C.C.", "Cédula de ciudadanía"),
    ),
    Field(
        "document_number",
        Entity.PATIENT,
        Requirement.REQUIRED,
        "The patient's identity document number. This is how a patient is matched.",
        aliases=(
            "numero de documento",
            "numero documento",
            "no documento",
            "no. documento",
            "nro documento",
            "nro doc",
            "num doc",
            # RIPS names this field `numDocumentoIdentificacion` (archivo US,
            # Resolución 3374 de 2000, and the JSON schema since). Without the
            # alias it only reaches `document_number` by fuzzy score, which is
            # too weak to pre-tick -- and an unticked document number means
            # every row is skipped at apply time for having no identifier.
            "num documento identificacion",
            "numero documento identificacion",
            "numero de identificacion",
            "identificacion",
            "documento",
            "cedula",
            "cc",
            "nuip",
            "id",
            "documento de identidad",
            "document number",
            "identification",
        ),
        normalizer="document_number",
        # Deliberately not a plausible cédula: these are shown to a reviewer
        # and sent to a model, so they must be recognisable as ours.
        examples=("0000000001", "0000000002"),
    ),
    Field(
        "full_name",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The patient's whole name in one column. Split into given names and surnames.",
        aliases=(
            "nombre completo",
            "nombres y apellidos",
            "nombre del paciente",
            "paciente",
            "nombre y apellidos",
            "nombre",
            "full name",
            "patient name",
            "nombre paciente",
        ),
        normalizer="full_name",
        examples=("Nombre Segundo Apellido Apellido",),
    ),
    Field(
        "given_names",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The patient's given names, when the file separates them from the surnames.",
        aliases=(
            "nombres",
            "primer nombre",
            "segundo nombre",
            "nombre 1",
            "nombres del paciente",
            "given names",
            "first name",
        ),
        examples=("Carlos Andrés",),
    ),
    Field(
        "family_names",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The patient's surnames. Colombians normally have two.",
        aliases=(
            "apellidos",
            "primer apellido",
            "segundo apellido",
            "apellido",
            "apellidos completos",
            "surname",
            "last name",
            "family names",
        ),
        examples=("Pérez Gómez",),
    ),
    Field(
        "birth_date",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "Date of birth. Used to tell apart two patients with the same name.",
        aliases=(
            "fecha de nacimiento",
            "fecha nacimiento",
            "f nacimiento",
            "f. nacimiento",
            "fec nac",
            "fecha nac",
            "nacimiento",
            "birth date",
            "date of birth",
            "fdn",
        ),
        normalizer="date",
        examples=("1990-01-01", "01/01/1990"),
    ),
    Field(
        "phone_e164",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The mobile number reminders are sent to. Colombian mobiles have 10 digits from 3.",
        aliases=(
            "celular",
            "cel",
            "movil",
            "telefono celular",
            "numero celular",
            "whatsapp",
            "telefono movil",
            "mobile",
            "cell",
            "cellphone",
        ),
        normalizer="phone",
        examples=("3000000000", "+573000000000"),
    ),
    Field(
        "phone_fixed",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "A landline. Colombian landlines have 10 digits starting 60.",
        aliases=("telefono", "tel", "telefono fijo", "fijo", "phone", "landline", "telefono casa"),
        normalizer="phone",
        examples=("6010000000",),
    ),
    Field(
        "email",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "Email address.",
        aliases=("correo", "correo electronico", "email", "e-mail", "mail", "correo del paciente"),
        examples=("correo@ejemplo.invalid",),
    ),
    Field(
        "eps",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The patient's health insurer (EPS). Stored as written.",
        aliases=(
            "eps",
            "entidad",
            "aseguradora",
            "asegurador",
            "empresa",
            "regimen",
            "entidad promotora de salud",
            # What the Chilean-origin suites call the same column.
            "prevision",
            "plan de salud",
            "insurer",
        ),
        examples=("Sura", "Nueva EPS", "Coosalud"),
    ),
    # ---- consent (ADR-19). Optional because not every clinic can export it;
    # what makes it safe is that an unrecognised value is refused rather than
    # widened, and a consent date that cannot be read refuses the row.
    Field(
        "consent_purpose",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "What the patient agreed to be contacted about.",
        aliases=(
            "consentimiento",
            "proposito consentimiento",
            "finalidad",
            "autorizacion para",
            "tipo de consentimiento",
            "consent purpose",
            "consent",
        ),
        normalizer="consent_purpose",
        examples=("citas", "telemedicina"),
    ),
    Field(
        "consent_granted_at",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The date the patient gave that consent.",
        aliases=(
            "fecha consentimiento",
            "fecha autorizacion",
            "fecha de autorizacion",
            "consentimiento fecha",
            "consent date",
            "consent granted at",
        ),
        normalizer="date",
        examples=("2026-03-15",),
    ),
    Field(
        "consent_evidence",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "How the consent was obtained: written, verbal and recorded, a digital "
        "form, or declared by the clinic on import.",
        aliases=(
            "evidencia consentimiento",
            "medio consentimiento",
            "como se obtuvo",
            "soporte",
            "consent evidence",
        ),
        normalizer="consent_evidence",
        examples=("escrito", "digital"),
    ),
    Field(
        "secondary_contact_name",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "Emergency contact's name.",
        aliases=(
            "contacto de emergencia",
            "nombre contacto",
            "acudiente",
            "responsable",
            "contacto secundario",
            "emergency contact",
        ),
    ),
    Field(
        "secondary_contact_phone",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "Emergency contact's phone number.",
        # Every alias here names *whose* number it is. "telefono contacto" and
        # "contacto telefono" were listed and had to go: a column headed
        # "Teléfono de contacto" is the patient's own number in most exports,
        # and it matched this field at strong confidence and arrived pre-ticked
        # -- a reviewer accepting it would file the patient's phone as their
        # emergency contact's, and the reminder would reach the wrong person.
        aliases=(
            "telefono de emergencia",
            "telefono emergencia",
            "celular acudiente",
            "telefono acudiente",
            "celular de emergencia",
            "contacto de emergencia telefono",
        ),
        normalizer="phone",
    ),
    Field(
        "telegram_chat_id",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "Telegram chat identifier, when the clinic uses Telegram.",
        aliases=("telegram", "telegram id", "chat id", "telegram chat id"),
    ),
    Field(
        "external_ref",
        Entity.PATIENT,
        Requirement.OPTIONAL,
        "The clinic's own code for this patient, so a later import updates instead of duplicating.",
        aliases=("codigo paciente", "id interno", "codigo interno", "historia clinica", "hc"),
    ),
)

_DOCTOR: Final = (
    Field(
        "external_ref",
        Entity.DOCTOR,
        Requirement.REQUIRED,
        "The clinic's own code for the doctor, used to link appointments and availability.",
        aliases=(
            "codigo",
            "codigo medico",
            "doctor id",
            "id medico",
            "codigo profesional",
            "cod medico",
            "codigo del medico",
            "id dentista",
            "cod profesional",
            "id profesional",
            "id prestador",
        ),
        examples=("M01",),
    ),
    Field(
        "full_name",
        Entity.DOCTOR,
        Requirement.REQUIRED,
        "The doctor's name.",
        aliases=(
            "nombre del medico",
            "medico",
            "profesional",
            "doctor",
            "nombre profesional",
            "especialista",
            "atendido por",
            "medico tratante",
            "nombre medico",
            "doctor name",
            "nombre",
            "nombre dentista",
            "dentista",
            "odontologo",
            "nombre del profesional",
        ),
        examples=("Dra. Patricia Gómez",),
    ),
    Field(
        "specialty",
        Entity.DOCTOR,
        Requirement.OPTIONAL,
        "The doctor's specialty, as a name or as a code into the specialties sheet.",
        aliases=(
            "especialidad",
            "especialidad medica",
            "servicio",
            "area",
            "specialty",
            "specialty id",
            "codigo especialidad",
        ),
        examples=("Ortopedia", "E02"),
    ),
    Field(
        "office_number",
        Entity.DOCTOR,
        Requirement.OPTIONAL,
        "Consulting room number.",
        aliases=(
            "consultorio",
            "oficina",
            "office",
            "numero consultorio",
            "sala",
            "box",
            "unidad",
        ),
        examples=("12",),
    ),
)

_SPECIALTY: Final = (
    Field(
        "external_ref",
        Entity.SPECIALTY,
        Requirement.REQUIRED,
        "The clinic's code for the specialty.",
        aliases=("codigo", "codigo especialidad", "specialty id", "id especialidad", "cod"),
        examples=("E01",),
    ),
    Field(
        "name",
        Entity.SPECIALTY,
        Requirement.REQUIRED,
        "The specialty's name.",
        aliases=("especialidad", "nombre", "nombre especialidad", "specialty name", "descripcion"),
        examples=("Ortopedia",),
    ),
)

_AVAILABILITY: Final = (
    Field(
        "doctor_ref",
        Entity.AVAILABILITY,
        Requirement.REQUIRED,
        "Which doctor this schedule belongs to.",
        aliases=("codigo medico", "doctor id", "medico", "id medico", "profesional"),
        examples=("M01",),
    ),
    Field(
        "weekday",
        Entity.AVAILABILITY,
        Requirement.REQUIRED,
        "Day of the week the doctor works.",
        aliases=("dia", "dia de la semana", "day", "day of week", "weekday", "dia semana"),
        examples=("Lunes", "Monday"),
        normalizer="weekday",
    ),
    Field(
        "start_time",
        Entity.AVAILABILITY,
        Requirement.REQUIRED,
        "Time the doctor starts.",
        aliases=("hora inicio", "hora de inicio", "desde", "inicio", "start time", "hora desde"),
        normalizer="time",
        examples=("07:00",),
    ),
    Field(
        "end_time",
        Entity.AVAILABILITY,
        Requirement.REQUIRED,
        "Time the doctor finishes.",
        aliases=("hora fin", "hora de fin", "hasta", "fin", "end time", "hora hasta"),
        normalizer="time",
        examples=("19:00",),
    ),
    Field(
        "valid_from",
        Entity.AVAILABILITY,
        Requirement.OPTIONAL,
        "First date this schedule applies.",
        aliases=("vigente desde", "valido desde", "desde fecha", "valid from", "fecha inicio"),
        normalizer="date",
    ),
    Field(
        "valid_to",
        Entity.AVAILABILITY,
        Requirement.OPTIONAL,
        "Last date this schedule applies.",
        aliases=("vigente hasta", "valido hasta", "hasta fecha", "valid to", "fecha fin"),
        normalizer="date",
    ),
)

_APPOINTMENT: Final = (
    Field(
        "external_ref",
        Entity.APPOINTMENT,
        Requirement.OPTIONAL,
        "The clinic's own code for the appointment, so a re-import updates it.",
        aliases=("id cita", "codigo cita", "appointment id", "numero cita", "consecutivo"),
        examples=("C0001",),
    ),
    Field(
        "patient_document",
        Entity.APPOINTMENT,
        Requirement.REQUIRED,
        "The document number of the patient the appointment is for.",
        aliases=(
            "documento paciente",
            "documento del paciente",
            "identificacion paciente",
            "cedula paciente",
            "documento",
            "identificacion",
            "patient document",
            # On an appointments sheet a bare document column is the patient's:
            # the appointment has no document of its own.
            "numero de documento",
            "numero documento",
            "document number",
            "cedula",
            "numero de identificacion",
            "no documento",
            "nro documento",
        ),
        normalizer="document_number",
    ),
    Field(
        "doctor_ref",
        Entity.APPOINTMENT,
        Requirement.REQUIRED,
        "Which doctor the appointment is with.",
        aliases=(
            "codigo medico",
            "doctor id",
            "medico",
            "profesional",
            "id medico",
            # A dental suite names the practitioner after the specialty.
            "id dentista",
            "nombre dentista",
            "cod profesional",
            "codigo profesional",
        ),
    ),
    Field(
        "appointment_date",
        Entity.APPOINTMENT,
        Requirement.REQUIRED,
        "The day of the appointment. Not the patient's date of birth.",
        aliases=(
            "fecha de la cita",
            "fecha cita",
            "fecha",
            "fecha programada",
            "fecha atencion",
            "fecha de atencion",
            "appointment date",
            "dia de la cita",
        ),
        normalizer="date",
        examples=("2026-10-15", "15/10/2026"),
    ),
    Field(
        "appointment_time",
        Entity.APPOINTMENT,
        Requirement.REQUIRED,
        "The time of the appointment.",
        aliases=(
            "hora",
            "hora de la cita",
            "hora cita",
            "appointment time",
            "hora programada",
            # An agenda that stores a start and an end: the start is the one
            # the appointment is booked for.
            "hora inicio",
            "hora de inicio",
        ),
        normalizer="time",
        examples=("07:00", "8:30 a. m."),
    ),
    Field(
        "status",
        Entity.APPOINTMENT,
        Requirement.OPTIONAL,
        "Whether the appointment is scheduled, confirmed, attended, cancelled or missed.",
        aliases=("estado", "estado de la cita", "situacion", "status", "estado cita"),
        normalizer="status",
        examples=("Confirmada", "Cancelada"),
    ),
    Field(
        "consultation_type",
        Entity.APPOINTMENT,
        Requirement.OPTIONAL,
        "First visit or follow-up.",
        aliases=(
            "tipo de consulta",
            "tipo consulta",
            "tipo de cita",
            "tipo cita",
            "consultation type",
            "modalidad",
        ),
        examples=("Primera vez", "Control"),
    ),
    Field(
        "source",
        Entity.APPOINTMENT,
        Requirement.OPTIONAL,
        "How the appointment arose: EPS referral, discharge, walk-in.",
        aliases=("origen", "fuente", "procedencia", "source", "remitido por"),
    ),
    Field(
        "cancellation_reason",
        Entity.APPOINTMENT,
        Requirement.OPTIONAL,
        "Why the appointment was cancelled.",
        aliases=(
            "motivo de cancelacion",
            "motivo cancelacion",
            "razon de cancelacion",
            "cancellation reason",
            "observacion",
        ),
    ),
    Field(
        "slot_ref",
        Entity.APPOINTMENT,
        Requirement.OPTIONAL,
        "A slot code, when the file stores the time in a separate slots sheet.",
        aliases=("slot id", "codigo slot", "id slot", "turno", "cupo"),
        examples=("S0001",),
    ),
)

ALL_FIELDS: Final[tuple[Field, ...]] = (
    *_PATIENT,
    *_DOCTOR,
    *_SPECIALTY,
    *_AVAILABILITY,
    *_APPOINTMENT,
)

FIELDS_BY_ENTITY: Final[dict[Entity, tuple[Field, ...]]] = {
    Entity.PATIENT: _PATIENT,
    Entity.DOCTOR: _DOCTOR,
    Entity.SPECIALTY: _SPECIALTY,
    Entity.AVAILABILITY: _AVAILABILITY,
    Entity.APPOINTMENT: _APPOINTMENT,
}


def field_for(entity: Entity, name: str) -> Field | None:
    return next((f for f in FIELDS_BY_ENTITY[entity] if f.name == name), None)


def required_fields(entity: Entity) -> tuple[Field, ...]:
    return tuple(f for f in FIELDS_BY_ENTITY[entity] if f.requirement is Requirement.REQUIRED)
