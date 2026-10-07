"""Explain a refusal to the receptionist who has to act on it.

A normalizer's message says what is wrong with a cell, in the shortest form
that is accurate: *"'3067891234' is not an assigned Colombian number, so it
cannot be reached."* That is correct and it is evidence, but it does not say
what the person should now do, and it is in English while the people reading it
work in Spanish.

This module adds that second half. The refusal itself is unchanged -- it is
what the transform log holds, and it stays the authority on what happened --
and the guidance sits beside it.

**The explanation is keyed on the rule name, never on the cell.** Every
normalizer stamps its outcome with a rule from a fixed vocabulary
(`phone.not_assigned`, `name.three_tokens`, `document_number.scientific_notation`),
and that name carries no patient data. The cell value does: fourteen of the
refusal messages embed it, because naming the offending value is what makes the
message actionable. Keying on the rule is what lets this module be useful
without any of that leaving the machine.

**Most of it is a table, not a model.** The vocabulary is small and fixed, so a
written Spanish explanation is better than a generated one on every axis that
matters: it is identical on every import, it costs nothing, it needs no network,
and it cannot leak. ADR-08a's reasoning applies here too -- a model is the
fallback for what the deterministic stage could not resolve, not the first
resort. `explain` therefore answers from the table, and `missing_rules` reports
what the table does not cover so a new rule is noticed rather than silently
unexplained.

The model's place is `suggest_wording`, for a rule the table has no entry for.
It receives the rule name and the field's own description -- both ours, neither
the clinic's -- and it never decides anything: a cell's status, the value it
converts to, and whether a row imports are all settled before this module is
called, and nothing here can change them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from src.core.config import get_settings
from src.core.logging import get_logger
from src.onboarding.canonical import ALL_FIELDS

log = get_logger(__name__)

#: What each rule means and what the reviewer should do about it, in the
#: Spanish a Colombian receptionist reads. Keyed by the rule a normalizer
#: stamps on its outcome.
#:
#: Only rules a person can see are listed. A rule that marks a *successful*
#: conversion (`phone.e164`, `date.iso`, `name.two_tokens`) never reaches a
#: review queue, so explaining it would be noise.
#:
#: `action` is written as an instruction to the person, and it never says
#: "correct the value": a repaired cédula or phone number belongs to somebody
#: else. It says to check the source, or to supply what the file could not.
GUIDANCE: Final[dict[str, tuple[str, str]]] = {
    # -- phone ---------------------------------------------------------------
    "phone.not_assigned": (
        "El número existe como texto pero no corresponde a ninguna línea asignada "
        "en Colombia, así que no se le puede escribir ni llamar.",
        "Verifique el número en la historia del paciente. No lo aproxime: el número "
        "más parecido es de otra persona y recibiría los recordatorios.",
    ),
    "phone.unparseable": (
        "El valor no tiene forma de número telefónico.",
        "Revise esa celda en el archivo original; suele ser una nota escrita en la "
        "columna equivocada.",
    ),
    "phone.empty": (
        "La celda está vacía y este campo es obligatorio para este tipo de registro.",
        "Complete el número en el archivo, o deje la fila por fuera si el paciente "
        "no tiene teléfono registrado.",
    ),
    # -- names ---------------------------------------------------------------
    "name.three_tokens": (
        "Tres palabras no dicen dónde termina el nombre y empiezan los apellidos: "
        "«Carlos Pérez Gómez» y «Juan Carlos Pérez» tienen la misma forma. La Ley "
        "2129 de 2021 además permite elegir el orden de los apellidos, así que "
        "ninguna regla de posición funciona.",
        "Confirme la división en pantalla, o exporte el nombre en columnas separadas "
        "y vuelva a subir el archivo: así no se vuelve a preguntar.",
    ),
    # No entry for `name.four_tokens`: four words are two given names and two
    # surnames, which is unambiguous, so it converts as valid and never reaches
    # a reviewer. The coverage test above refuses an entry for it.
    "name.unsupported_shape": (
        "El nombre tiene una forma que no se puede dividir con seguridad.",
        "Revise la celda: suele ser un nombre con un cargo, una nota o dos pacientes "
        "en la misma línea.",
    ),
    "name.looks_like_a_formula": (
        "La celda empieza con un carácter que Excel interpreta como fórmula, así que "
        "no se guarda como nombre.",
        "Revise esa celda en el archivo original. El texto se conserva tal cual y "
        "nunca se ejecuta.",
    ),
    "name.trailing_particle": (
        "El nombre termina en una partícula («de», «del», «la») que pertenece al "
        "apellido siguiente, y ese apellido no está.",
        "Revise si el apellido quedó cortado al exportar.",
    ),
    "name.comma_incomplete": (
        "Hay una coma que separa apellidos de nombres, pero falta uno de los dos lados.",
        "Complete el nombre en el archivo original.",
    ),
    "name.multiple_commas": (
        "Más de una coma hace imposible saber qué parte son los apellidos.",
        "Revise la celda: suele ser más de un paciente escrito en la misma línea.",
    ),
    "name.empty": (
        "La celda del nombre está vacía.",
        "Complete el nombre, o deje la fila por fuera si no es un paciente.",
    ),
    # -- document --------------------------------------------------------------
    "document_number.scientific_notation": (
        "Excel guardó la cédula como número y la convirtió a notación científica, "
        "con lo cual los dígitos del final se perdieron. El valor que quedó ya no "
        "es el documento de nadie.",
        "Vuelva a exportar esa columna con formato de texto. No la reconstruya a "
        "mano: los dígitos perdidos no se pueden adivinar.",
    ),
    "document_number.length": (
        "El documento tiene más o menos dígitos de los que admite el registro RIPS.",
        "Verifique el documento en la historia del paciente.",
    ),
    "document_number.unexpected": (
        "El valor no tiene forma de número de documento.",
        "Revise esa celda en el archivo original.",
    ),
    "document_number.empty": (
        "La fila no trae número de documento, que es con lo que se identifica al paciente.",
        "Complete el documento, o deje la fila por fuera si no es un paciente.",
    ),
    "document_type.unknown": (
        "El tipo de documento no corresponde a ninguno de la tabla oficial (CC, TI, "
        "CE, PA, RC, MS, AS, PE, PT).",
        "Indique cuál es. No se asume CC: eso uniría un número real con una "
        "identidad legal equivocada.",
    ),
    "document_type.empty": (
        "La fila no dice qué tipo de documento es.",
        "Indique el tipo. Dejarlo en blanco no lo convierte en cédula.",
    ),
    # -- dates and times -------------------------------------------------------
    "date.ambiguous_column": (
        "Toda la columna puede leerse como día/mes o como mes/día, y las dos "
        "lecturas dan fechas reales pero distintas.",
        "Confirme el formato una vez para toda la columna. Se aplica igual a todas "
        "las filas, nunca fila por fila.",
    ),
    "date.unrecognised": (
        "La fecha no está en un formato que se pueda leer sin adivinar.",
        "Revise esa celda. Un formato por columna es lo que evita que un archivo "
        "termine con las dos lecturas mezcladas.",
    ),
    "date.impossible": (
        "La fecha no existe en el calendario: un 31 de abril, o un 29 de febrero "
        "de un año que no es bisiesto.",
        "Verifique la fecha en el archivo original.",
    ),
    "date.implausible": (
        "La fecha se entiende, pero cae fuera del rango razonable para este campo: "
        "una fecha de nacimiento en el futuro, o un año como 1890 que suele ser un "
        "error de digitación por 1980.",
        "Confirme el año. Una persona puede ver cuál era la intención; el programa no.",
    ),
    "date.serial_out_of_range": (
        "Excel guardó la fecha como número y ese número no corresponde a una fecha válida.",
        "Vuelva a exportar la columna con formato de fecha.",
    ),
    "date.compact_invalid": (
        "El valor parece una fecha compacta (AAAAMMDD) pero las cifras no forman una fecha real.",
        "Verifique la fecha en el archivo original.",
    ),
    "date.empty": (
        "La celda de fecha está vacía y este campo es obligatorio.",
        "Complete la fecha, o deje la fila por fuera.",
    ),
    "time.unrecognised": (
        "La hora no está en un formato que se pueda leer.",
        "Use 24 horas (14:30) o 12 horas con a. m./p. m.",
    ),
    "time.out_of_range": (
        "La hora está fuera del día: más de 23:59, o minutos por encima de 59.",
        "Verifique la hora en el archivo original.",
    ),
    "time.impossible": (
        "El valor tiene forma de hora pero no corresponde a una hora real.",
        "Verifique la hora en el archivo original.",
    ),
    "time.empty": (
        "La celda de hora está vacía y este campo es obligatorio.",
        "Complete la hora, o deje la fila por fuera.",
    ),
    # -- status, weekday, booleans, consent ------------------------------------
    "status.unknown": (
        "El estado de la cita no corresponde a ninguno conocido, así que no se sabe "
        "si la cita se atendió, se canceló o sigue pendiente.",
        "Indique a cuál corresponde. Suponerlo cambiaría la historia de la cita.",
    ),
    "status.ambiguous": (
        "El estado puede significar más de una cosa.",
        "Indique cuál corresponde en esta clínica.",
    ),
    "status.empty": (
        "La cita no trae estado.",
        "Indique el estado, o deje la fila por fuera.",
    ),
    "weekday.unknown": (
        "El día de la semana no se reconoce.",
        "Escriba el día en palabras («lunes», «martes»).",
    ),
    "weekday.numeric": (
        "El día viene como número, y un número es ambiguo: 1 es lunes en unos "
        "sistemas y domingo en otros.",
        "Escriba el día en palabras.",
    ),
    "weekday.empty": (
        "La regla de disponibilidad no dice a qué día aplica.",
        "Indique el día de la semana.",
    ),
    "boolean.unknown": (
        "El valor no es un sí ni un no claro.",
        "Use SÍ/NO, o déjelo vacío si no aplica.",
    ),
    "boolean.empty": (
        "La celda está vacía y no está claro si eso significa «no».",
        "Indique SÍ o NO explícitamente.",
    ),
    "consent.purpose_unknown": (
        "La finalidad del consentimiento no corresponde a ninguna registrada.",
        "Indique cuál es. El consentimiento es la base legal del envío, así que no se asume.",
    ),
    "consent.purpose_empty": (
        "El consentimiento no dice para qué se otorgó.",
        "Indique la finalidad.",
    ),
    "consent.evidence_unknown": (
        "El tipo de soporte del consentimiento no se reconoce.",
        "Indique cómo se obtuvo el consentimiento.",
    ),
    "consent.evidence_empty": (
        "El consentimiento no dice con qué soporte se obtuvo.",
        "Indique cómo se obtuvo.",
    ),
}


@dataclass(frozen=True, slots=True)
class Explanation:
    """Why a cell was held back, and what to do about it."""

    rule: str
    #: What the rule means, in Spanish. Never contains the cell's value.
    means: str
    #: What the reviewer should do. Never "fix the value" for an identifier.
    action: str
    #: Where the wording came from: `table` or a provider's name. Shown so a
    #: reviewer can tell written guidance from generated.
    source: str = "table"


@lru_cache(maxsize=256)
def _generated(rule: str, field_means: str) -> Explanation | None:
    """One model-written explanation, cached for the life of the process.

    Cached by `(rule, field_means)` because the wording depends on nothing else
    -- not the cell, not the row, not the clinic. A transform log can hold two
    thousand cells failing for three reasons; without the cache that would be
    two thousand calls for three answers.

    None on any failure. The caller falls back to the table, so a provider that
    is down or slow costs nothing but the default wording.
    """
    from src.onboarding import llm

    try:
        return llm.explain_rule(rule, field_means)
    except llm.LLMUnavailable:
        return None
    except Exception:  # a provider fault must never break the review screen
        log.warning("guidance.wording_failed", rule=rule, exc_info=True)
        return None


def explain(rule: str, field_name: str = "") -> Explanation | None:
    """What this refusal means and what to do, or None if nothing covers it.

    A model writes it when one is configured, falling back to the written table
    when the call fails or no key is set. The table is the floor rather than the
    first resort: the client's instruction is that generated wording is worth
    paying for, and it reads the clinic's own field description, so it can be
    more specific than one sentence covering every field that shares a rule.

    What is sent is the rule name and our own field description. The cell never
    is -- see this module's docstring -- so the choice between table and model
    changes the wording and the cost, never what leaves the machine.

    `field_name` is optional so a caller with only a rule still gets the table.
    """
    written = GUIDANCE.get(rule)
    field = next((f for f in ALL_FIELDS if f.name == field_name), None)
    # The field's own description when we know the field; the rule's written
    # explanation otherwise, which still tells the model what it is wording.
    means = field.description if field else (written[0] if written else rule)
    if get_settings().mapping_llm_available and (generated := _generated(rule, means)):
        return generated
    return Explanation(rule, written[0], written[1]) if written else None


def missing_rules(rules: set[str]) -> set[str]:
    """Which of `rules` the table does not explain.

    Used by a test over the rules the normalizers can actually emit, so a new
    refusal is noticed when it is added rather than when a receptionist meets it
    with no guidance. The table stays the fallback even when a model writes the
    wording a reviewer actually sees, so a gap here is still a gap.
    """
    return {rule for rule in rules if rule not in GUIDANCE}


def suggest_wording(rule: str, field_name: str) -> Explanation:
    """Ask a model to word a rule the table does not cover.

    The fallback, reached only when `explain` returns None. What is sent is the
    rule's name and the field's own description from `canonical.py` -- both ours
    -- and never the cell, the heading, or anything the clinic wrote.

    It decides nothing. The cell's status, the value it converted to and whether
    the row imports are all settled before this is called; this only supplies
    words for a screen. A provider failure raises `LLMUnavailable`, and the
    caller shows the normalizer's own message, which is where it started.
    """
    from src.onboarding import llm

    field = next((f for f in ALL_FIELDS if f.name == field_name), None)
    means = field.description if field else field_name
    return llm.explain_rule(rule, means)
