"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api import health
from src.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    UnhandledErrorMiddleware,
)
from src.api.onboarding import router as onboarding_router
from src.api.review import router as review_router
from src.audit.service import anchor_chain
from src.core.config import get_settings
from src.core.db import dispose_engine, get_sessionmaker
from src.core.logging import configure_logging, get_logger
from src.core.ratelimit import InProcessRateLimiter

#: Sized for the JSON bodies the API accepts (a mapping, a correction). The
#: upload route is exempt and enforces its own, larger limit while streaming.
MAX_REQUEST_BODY_BYTES = 1_000_000

#: The exact paths that receive a file rather than a JSON body. Matched exactly,
#: so the JSON sub-routes under `/onboarding/uploads/...` stay bounded.
UPLOAD_PATHS = ("/onboarding/uploads", "/onboarding/upload")

#: Host headers the review surface answers to. A browser reaching loopback from
#: another origin sends that origin's name, so the name is what rejects it.
#: Ports are stripped by TrustedHostMiddleware before matching.
#:
#: `testserver` is Starlette's name for an in-process client. It is included
#: because it resolves to nothing in DNS, so no browser can send it and no
#: rebinding attack can use it -- a rebinding host has to resolve to 127.0.0.1
#: to arrive at all. Leaving it out would make every test talk to a different
#: app than the one that ships.
LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "::1", "[::1]", "testserver"]

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info(
        "startup",
        app_env=settings.app_env,
        allow_real_patient_data=settings.allow_real_patient_data,
    )
    await _anchor_audit_chain()
    yield
    await _anchor_audit_chain()
    await dispose_engine()


async def _anchor_audit_chain() -> None:
    """Witness the audit chain's tip, so entries written since the last anchor
    cannot be removed unnoticed (see src/audit/service.py).

    Anchoring once per process lifetime bounds the exposure to one process's
    worth of entries. The scheduled anchoring that shortens that window arrives
    with background jobs in M2. A failure here must not stop the API from
    serving, so it is logged rather than raised.
    """
    try:
        async with get_sessionmaker()() as session, session.begin():
            await anchor_chain(session)
    except SQLAlchemyError:
        log.exception("audit.anchor_failed")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    # The OpenAPI document lists the review routes, so it is served only where
    # those routes are enabled.
    interactive_docs = settings.synthetic_data_mode

    app = FastAPI(
        title="Clinic Scheduler API",
        version="0.1.0",
        lifespan=_lifespan,
        docs_url="/docs" if interactive_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if interactive_docs else None,
    )

    # add_middleware wraps previously added middleware, so the last added runs first.
    app.add_middleware(
        RateLimitMiddleware, limiter=InProcessRateLimiter(settings.rate_limit_per_minute)
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=MAX_REQUEST_BODY_BYTES,
        # The upload endpoint bounds the file itself as it streams, so the
        # JSON-sized limit must not also cap a spreadsheet.
        exempt_paths=UPLOAD_PATHS,
    )
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    if settings.synthetic_data_mode:
        # The review and import screens are loopback-only, which `server.py`
        # enforces for the socket. The socket is not the whole story: a page on
        # another origin whose DNS resolves to 127.0.0.1 reaches us from the
        # developer's own browser, carrying its own Host header. Requiring a
        # loopback Host closes that, and these routes only exist here anyway.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOOPBACK_HOSTS)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(review_router)
    app.include_router(onboarding_router)
    return app
