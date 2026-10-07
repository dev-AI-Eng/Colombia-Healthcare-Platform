"""ASGI middleware applied to every HTTP request.

Implemented as plain ASGI callables rather than Starlette's BaseHTTPMiddleware.
The body-size limit has to replace the `receive` callable the application reads
the body from, which BaseHTTPMiddleware's `dispatch(request, call_next)`
interface does not allow.

Order in `create_app`, outermost first: request context, security headers,
unhandled-error boundary, body size limit, rate limit.
"""

from __future__ import annotations

import re
import uuid
from typing import ClassVar

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.audit.context import ActorKind, AuditContext, reset_context, set_context
from src.core.logging import get_logger
from src.core.ratelimit import RateLimiter

_log = get_logger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestContextMiddleware:
    """Assign a request id, bind a default audit and logging context, echo the id back.

    An incoming `X-Request-ID` is reused if it is 1-128 characters of letters,
    digits, `.`, `_` or `-`; otherwise a UUID is generated. The default audit
    context attributes the request to the system; routes that serve staff
    replace it with a more specific one.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get("x-request-id")
        request_id = (
            supplied if supplied and _REQUEST_ID_PATTERN.match(supplied) else str(uuid.uuid4())
        )
        scope.setdefault("state", {})["request_id"] = request_id
        client = scope.get("client")

        token = set_context(
            AuditContext(
                actor_kind=ActorKind.SYSTEM,
                purpose="http_request",
                request_id=request_id,
                ip=client[0] if client else None,
            )
        )
        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
            reset_context(token)


class SecurityHeadersMiddleware:
    """Headers suited to an API whose responses can contain patient data.

    `Cache-Control: no-store` stops browsers and intermediaries from keeping
    copies of responses.
    """

    _HEADERS: ClassVar[dict[str, str]] = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "X-Frame-Options": "DENY",
    }

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in self._HEADERS.items():
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class UnhandledErrorMiddleware:
    """Turn an unhandled exception into a generic 500 response.

    Starlette's own handler for unhandled exceptions runs outside every user
    middleware, so its responses would lack `X-Request-ID` and the security
    headers. This middleware sits inside those two, so its 500 keeps both.

    The response body contains only a generic message and the request id. The
    exception is logged with the request id. If the response had already started
    when the exception was raised, it is logged and re-raised, because a second
    response cannot be sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            request_id = scope.get("state", {}).get("request_id")
            _log.error(
                "unhandled_exception",
                request_id=request_id,
                path=scope.get("path"),
                error_type=type(exc).__name__,
                response_started=response_started,
                exc_info=exc,
            )
            if response_started:
                raise
            response = JSONResponse(
                {"detail": "Internal server error", "request_id": request_id},
                status_code=500,
            )
            await response(scope, receive, send)


class _PayloadTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes` with 413.

    A declared Content-Length over the limit is rejected before the application
    runs. Otherwise, including for chunked bodies with no Content-Length, bytes
    are counted as the application reads them. If an endpoint reads the body
    itself (`await request.body()`), crossing the limit produces 413. FastAPI
    endpoints that declare a body parameter parse the body inside their own
    error handling and respond 400 instead. A body that is never read is never
    counted.

    `exempt_paths` are the exact paths whose job is to receive a file and which
    bound the upload themselves while streaming it. Without the exemption a
    global limit sized for a JSON body also caps every spreadsheet, so a real
    clinic export is refused with "Payload too large" before the reader sees it
    and the route's own, larger limit becomes dead code. Exact, not prefix: a
    prefix would also exempt the JSON sub-routes under the same path.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, exempt_paths: tuple[str, ...] = ()) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Matched exactly. A prefix exempted every sub-route under it, so the
        # JSON bodies of `.../mapping`, `.../validate` and `.../rows/correct`
        # were unbounded too -- and `/onboarding/uploadsomething` would have
        # matched a route that does not exist.
        if scope.get("path", "") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _PayloadTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _PayloadTooLarge:
            if response_started:
                raise
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"detail": "Payload too large"}, status_code=413)(scope, receive, send)


class RateLimitMiddleware:
    """Limit each client IP through a `RateLimiter` backend.

    Rejected requests receive 429 with `Retry-After` set to the whole seconds the
    backend says to wait. Paths in `exempt_paths` are never limited, so health
    probes keep working.

    The client address is the ASGI scope's `client`. Behind a reverse proxy,
    start the server with proxy headers enabled and the proxy's address trusted,
    otherwise every request appears to come from the proxy.
    """

    def __init__(
        self,
        app: ASGIApp,
        limiter: RateLimiter,
        exempt_paths: frozenset[str] = frozenset({"/healthz", "/readyz"}),
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        retry_after = await self.limiter.check(client[0] if client else "unknown")
        if retry_after is not None:
            response = JSONResponse(
                {"detail": "Too many requests"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
