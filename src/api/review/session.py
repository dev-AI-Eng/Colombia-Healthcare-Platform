"""Sign in, sign out, and ask who you are.

Three routes. They are the smallest thing that makes the admin role real: a
password is exchanged for a signed cookie, the cookie is cleared on sign-out,
and a screen can ask what role it is rendering for without guessing.

What this is not is M5's authentication. There are no per-user accounts, so the
audit log attributes an admin action to `clinic-admin` rather than to a person,
and there is no step-up challenge before an unmasked read. Both are M5's, and
both are named in `auth.py` so this does not read as finished.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from src.api.dependencies import SettingsDep
from src.api.review.auth import (
    SESSION_COOKIE,
    SESSION_SECONDS,
    Role,
    RoleDep,
    Session,
    SignIn,
    issue_token,
    session_cookie_arguments,
    verify_password,
)
from src.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/session", response_model=Session, summary="The role this browser has")
async def read_session(role: RoleDep) -> Session:
    """What the caller is. `staff` when there is no valid session.

    A screen calls this to decide whether to offer unmasking, rather than
    inferring it from whether a cookie exists -- an expired cookie exists and
    proves nothing.
    """
    return Session(
        role=role,
        expires_in_seconds=SESSION_SECONDS if role.sees_unmasked else 0,
    )


@router.post("/session", response_model=Session, summary="Sign in as administrator")
async def sign_in(
    body: SignIn, request: Request, response: Response, settings: SettingsDep
) -> Session:
    """Exchange the password for a signed session cookie.

    A wrong password answers 401 with the same Spanish sentence whatever was
    wrong with it, and the attempt is logged without the password or any part
    of it -- logging a prefix to "help debugging" is how a secret ends up in a
    log file.
    """
    if not verify_password(body.password, settings):
        log.warning(
            "review.sign_in_rejected",
            ip=request.client.host if request.client else None,
            # Whether one is configured at all, which is the difference between
            # "wrong password" and "nobody can ever sign in", and is the thing
            # worth knowing from a log.
            password_configured=bool(settings.review_admin_password.get_secret_value()),
        )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Contraseña incorrecta.",
        )

    token = issue_token(Role.ADMIN, settings)
    response.set_cookie(**session_cookie_arguments(token, request))  # type: ignore[arg-type]
    log.info(
        "review.sign_in",
        role=Role.ADMIN.value,
        ip=request.client.host if request.client else None,
    )
    return Session(role=Role.ADMIN, expires_in_seconds=SESSION_SECONDS)


@router.delete("/session", response_model=Session, summary="Sign out")
async def sign_out(request: Request, response: Response) -> Session:
    """Clear the cookie. Always succeeds, even with no session to clear.

    Signing out of nothing is not an error, and answering 404 would make a
    staff screen show a failure for doing the safe thing.
    """
    response.delete_cookie(SESSION_COOKIE, path="/", samesite="lax", httponly=True)
    return Session(role=Role.STAFF, expires_in_seconds=0)
