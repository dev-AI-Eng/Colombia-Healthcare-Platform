"""Sign-in and roles for the review surface.

    Rev2, ADR-13: "Patient identifiers (document number, phone, email) are
    masked by default in every screen, API, report and download. Unmasked data
    is available only through a dedicated export: clinic-admin role of that
    clinic, step-up reauthentication, mandatory justification ... and its own
    audit action type."

This is the role and the audit action, built early. What it is not is M5: there
is no per-user account store, no step-up re-authentication, no justification
capture and no expiring export link. Those are M5's, and the docstrings below
say which piece is missing where, so nobody reads this as the finished control.

WHAT IT DOES GIVE
-----------------
Two roles. `staff` is the default and sees what the API has always returned:
identifiers masked. `admin` sees them in full, and every such read writes a
`disclose` audit entry naming the fields disclosed -- a different action type
from an ordinary read, so the log can answer "who has seen a patient's cédula",
which is the question an audit exists to answer.

HOW A SESSION WORKS
-------------------
A password is exchanged for a signed, expiring token held in an HttpOnly
cookie. The token carries the role and an expiry and is signed with the audit
chain key, so a cookie cannot be edited into a higher role: changing the role
changes the signature, and the signature is checked before the role is read.

The password lives in settings, which read it from the environment or a secrets
file. It is never written to source, a log or an audit entry.

WHY NOT A ROLE HEADER
---------------------
A header would have been less code, and it is what an internal tool often does.
It is also a role anybody can claim: `curl -H "X-Role: admin"` would unmask
every patient in the clinic. Signing the role is what makes it a credential
rather than a request for one.

FAILING CLOSED
--------------
An absent cookie, a malformed one, a bad signature, an expired token, or any
unexpected error resolves to `staff`. There is no path by which a failure
produces `admin`, which is what makes the default safe: the worst outcome of a
bug here is that an administrator sees masked data and has to sign in again.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from enum import StrEnum
from typing import Annotated, Final

from fastapi import Cookie, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.dependencies import SettingsDep
from src.core.config import Settings

#: The session cookie. HttpOnly, so page scripts cannot read it.
SESSION_COOKIE: Final = "clinic_session"

#: How long a signed session lasts. Short, because it grants unmasked patient
#: identifiers and this is a desk in a clinic, not a personal device. M5's
#: step-up re-authentication replaces this with a per-export challenge.
SESSION_SECONDS: Final = 8 * 60 * 60

#: The fields an unmasked read discloses. Recorded on the audit entry so the
#: log says what was seen, not merely that something was.
UNMASKED_FIELDS: Final = ("document_number", "phone_e164", "email")


class Role(StrEnum):
    """Who is asking.

    `STAFF` is the floor: it is what an unauthenticated request gets, and it
    sees exactly what the API returned before roles existed.
    """

    STAFF = "staff"
    ADMIN = "admin"

    @property
    def sees_unmasked(self) -> bool:
        return self is Role.ADMIN


class SignIn(BaseModel):
    """The sign-in body. One field, because there is one account.

    M5 replaces this with real accounts; until then the password identifies the
    role rather than a person, and the audit entry says `clinic-admin` rather
    than a name. That is a limitation worth stating rather than hiding behind a
    username field that means nothing.
    """

    password: str = Field(min_length=1, max_length=200)


class Session(BaseModel):
    """What the client is told about its own session. Never the token."""

    role: Role
    expires_in_seconds: int


def _sign(payload: str, settings: Settings) -> str:
    """HMAC the payload with the audit chain key.

    Reuses that key because it is already required to be a real secret outside
    development, and it is held separately from the database password -- so an
    attacker with database access cannot mint a session with it.
    """
    key = settings.audit_chain_key.get_secret_value().encode()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def issue_token(role: Role, settings: Settings, *, now: float | None = None) -> str:
    """A signed `role.expiry.signature` token."""
    expiry = int(now if now is not None else time.time()) + SESSION_SECONDS
    payload = f"{role.value}.{expiry}"
    return f"{payload}.{_sign(payload, settings)}"


def read_token(token: str | None, settings: Settings, *, now: float | None = None) -> Role:
    """The role a token proves, or `STAFF` if it proves nothing.

    Every failure path returns `STAFF` rather than raising, because a bad cookie
    is not an error a user can act on -- it is an unauthenticated request, and
    the API already has an answer for those.
    """
    if not token:
        return Role.STAFF
    parts = token.split(".")
    if len(parts) != 3:
        return Role.STAFF
    role_value, expiry_value, signature = parts

    # Constant time, so a wrong signature cannot be found a byte at a time.
    if not hmac.compare_digest(signature, _sign(f"{role_value}.{expiry_value}", settings)):
        return Role.STAFF

    try:
        expiry = int(expiry_value)
    except ValueError:
        return Role.STAFF
    if expiry <= int(now if now is not None else time.time()):
        return Role.STAFF

    try:
        return Role(role_value)
    except ValueError:
        # A signed token naming a role this version does not have. Trusting it
        # would mean trusting a name we cannot reason about.
        return Role.STAFF


def verify_password(candidate: str, settings: Settings) -> bool:
    """Whether this is the admin password.

    Compared in constant time. An unset password means no admin sign-in is
    possible at all, rather than an empty string being accepted -- a deployment
    that forgot to configure one must not get an open door.
    """
    expected = settings.review_admin_password.get_secret_value()
    if not expected:
        return False
    # Compared as UTF-8 bytes, not as strings: `compare_digest` raises
    # TypeError on a non-ASCII string, and a Colombian clinic's password may
    # well contain an ñ or an accent. Encoding first keeps the comparison
    # constant time and makes it work on any password a person would choose.
    return secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


async def current_role(
    settings: SettingsDep,
    clinic_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> Role:
    """The role for this request. `STAFF` unless a valid session says otherwise."""
    return read_token(clinic_session, settings)


RoleDep = Annotated[Role, Depends(current_role)]


def require_admin(role: RoleDep) -> Role:
    """Refuse anything but an administrator.

    403 rather than 404: the route exists and the caller is simply not allowed,
    and saying so is what lets a staff screen explain itself instead of looking
    broken.
    """
    if not role.sees_unmasked:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Esta acción requiere una sesión de administrador.",
        )
    return role


def session_cookie_arguments(token: str, request: Request) -> dict[str, object]:
    """How the session cookie is set.

    `secure` follows the scheme rather than being hard-coded: the dashboard is
    served over http on loopback in development, and a `secure` cookie would
    simply never be stored there, which looks like a broken login. In
    production the scheme is https and the flag goes on.
    """
    return {
        "key": SESSION_COOKIE,
        "value": token,
        "max_age": SESSION_SECONDS,
        "httponly": True,
        "samesite": "lax",
        "secure": request.url.scheme == "https",
        "path": "/",
    }
