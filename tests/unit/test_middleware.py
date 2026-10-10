from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from src.audit.context import get_context
from src.core.config import Settings
from src.core.ratelimit import InProcessRateLimiter


def _app(*, rate_limit: int = 1000, max_bytes: int = 1000) -> FastAPI:
    app = FastAPI()

    @app.get("/thing")
    async def thing() -> dict[str, str | None]:
        context = get_context()
        return {"request_id": context.request_id, "ip": context.ip}

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(RateLimitMiddleware, limiter=InProcessRateLimiter(rate_limit))
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    return app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(_app()) as test_client:
        yield test_client


# ------------------------------------------------------------------ request context


def test_valid_request_id_is_reused_and_bound_to_audit_context(client: TestClient) -> None:
    response = client.get("/thing", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert response.json()["request_id"] == "abc-123"


@pytest.mark.parametrize("supplied", ["has space", "x" * 129, "semi;colon"])
def test_invalid_request_id_is_replaced(client: TestClient, supplied: str) -> None:
    returned = client.get("/thing", headers={"X-Request-ID": supplied}).headers["X-Request-ID"]
    assert returned != supplied
    assert len(returned) == 36


def test_security_headers_are_set(client: TestClient) -> None:
    headers = client.get("/thing").headers
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Frame-Options"] == "DENY"


# ---------------------------------------------------------------------- body size


def test_declared_oversized_body_is_rejected(client: TestClient) -> None:
    response = client.post("/echo", content=b"x" * 2000)
    assert response.status_code == 413


def test_chunked_oversized_body_is_rejected(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(5):
            yield b"x" * 300

    # A generator body is sent without Content-Length.
    response = client.post("/echo", content=chunks())
    assert response.status_code == 413


def test_body_within_limit_is_accepted(client: TestClient) -> None:
    assert client.post("/echo", content=b"x" * 900).json() == {"bytes": 900}


# --------------------------------------------------------------------- rate limit


def test_requests_over_the_limit_get_429_with_retry_after() -> None:
    with TestClient(_app(rate_limit=3)) as limited:
        codes = [limited.get("/thing").status_code for _ in range(5)]
        blocked = limited.get("/thing")
    assert codes == [200, 200, 200, 429, 429]
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60


def test_health_probe_is_never_rate_limited() -> None:
    with TestClient(_app(rate_limit=1)) as limited:
        limited.get("/thing")
        limited.get("/thing")
        assert [limited.get("/healthz").status_code for _ in range(5)] == [200] * 5


async def test_idle_keys_are_swept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the sweep the window would keep an entry per client seen, forever."""
    clock = [1000.0]
    monkeypatch.setattr("src.core.ratelimit.time.monotonic", lambda: clock[0])
    limiter = InProcessRateLimiter(limit=10)
    await limiter.check("198.51.100.7")
    assert "198.51.100.7" in limiter._hits

    clock[0] += 61
    await limiter.check("203.0.113.1")

    assert "198.51.100.7" not in limiter._hits


# ------------------------------------------------- the limit the mode implies


def test_production_gets_the_strict_limit_with_nothing_configured() -> None:
    """What faces the network is sized for patients messaging a clinic."""
    settings = Settings(
        app_env="production",
        phi_encryption_key="p" * 40,
        phi_blind_index_key="b" * 40,
        audit_chain_key="a" * 40,
    )

    assert settings.synthetic_data_mode is False
    assert settings.effective_rate_limit_per_minute == Settings.PRODUCTION_RATE_LIMIT


def test_synthetic_mode_is_not_throttled_to_the_production_number() -> None:
    """The callers on a developer's machine are tools, not patients.

    The dashboard renders server-side and makes several API calls per page, and
    the browser test suite makes hundreds. At 60 a minute the suite answers
    "Too many requests" and looks broken, which is what happened before this
    existed -- the workaround was to tell whoever ran it to set an environment
    variable, and a suite that needs production settings changed to pass is a
    trap for the next person.
    """
    settings = Settings(app_env="local")

    assert settings.synthetic_data_mode is True
    assert settings.effective_rate_limit_per_minute == Settings.SYNTHETIC_RATE_LIMIT
    assert settings.effective_rate_limit_per_minute > Settings.PRODUCTION_RATE_LIMIT


def test_a_limit_that_is_set_wins_in_either_mode() -> None:
    """A deployment behind a proxy pins its own number, and that is final."""
    local = Settings(app_env="local", rate_limit_per_minute=7)
    production = Settings(
        app_env="production",
        rate_limit_per_minute=7,
        phi_encryption_key="p" * 40,
        phi_blind_index_key="b" * 40,
        audit_chain_key="a" * 40,
    )

    assert local.effective_rate_limit_per_minute == 7
    assert production.effective_rate_limit_per_minute == 7


def test_enabling_real_patient_data_restores_the_strict_limit() -> None:
    """The gate is the data, not the environment name.

    `app_env=local` with real patient data enabled is no longer a developer's
    sandbox, so it must not keep the sandbox's limit.
    """
    settings = Settings(app_env="local", allow_real_patient_data=True)

    assert settings.synthetic_data_mode is False
    assert settings.effective_rate_limit_per_minute == Settings.PRODUCTION_RATE_LIMIT


def test_synthetic_mode_still_has_a_limit() -> None:
    """Raised, not removed: a runaway loop must still be stopped."""
    settings = Settings(app_env="local")

    assert settings.effective_rate_limit_per_minute < 1_000_000
    with TestClient(_app(rate_limit=3)) as limited:
        statuses = [limited.get("/thing").status_code for _ in range(5)]
    assert 429 in statuses
