"""Server-sent events: telling an open dashboard that something changed.

The client asked how the clinic's dashboard learns that a patient has
responded. This is the mechanism, and one decision shapes the whole file:

**The stream carries a notice, never patient data.** An event here says
"appointments changed in clinic X" and nothing more. The browser then re-fetches
through the ordinary audited route, so every disclosure still passes the same
gate and lands in the audit log. A socket streaming patient names would be a
second disclosure channel to secure, and rule 8 would need enforcing twice.

That also makes the stream cheap to reason about: it cannot leak, because there
is nothing in it to leak.

**Why SSE rather than WebSockets.** A dashboard needs the server to talk and
the browser to listen; nothing here is two-way. Browsers implement SSE natively
including reconnection, whereas a socket adds sticky sessions and proxies that
fight the upgrade, in exchange for a direction we do not use.

**Why not polling alone.** The API allows 60 requests a minute per client in
production. A 30-second refresh across a clinic's open screens spends that
budget on requests where nothing changed. Auto-refresh stays in the frontend as
the fallback for a browser or proxy that drops the stream — a screen that goes
silently stale is worse than one that polls.

**State is per process, deliberately for now.** `notify` reaches the listeners
attached to *this* worker. One worker is the current deployment (CLAUDE.md
notes Redis is needed before a second), so this is correct today and becomes a
Redis pub/sub channel when that arrives. Said plainly rather than discovered:
with two workers and no Redis, a dashboard on worker A would not hear a change
made on worker B.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from collections.abc import AsyncIterator
from typing import Final

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import ClinicScopeDep

router = APIRouter()

#: How often a comment is sent when nothing has happened. Proxies and load
#: balancers close an idle connection, and the browser's reconnect is cheap but
#: not free -- a heartbeat keeps the stream open instead.
HEARTBEAT_SECONDS: Final = 25.0

#: Dropped rather than queued once a listener is this far behind. A dashboard
#: that cannot keep up with three notices does not need the fourth: it needs to
#: re-fetch, which is what every notice tells it to do anyway.
QUEUE_LIMIT: Final = 4

#: What changed. A closed set so a screen can decide whether it cares: the
#: follow-up list does not need redrawing because the audit log grew.
TOPIC_APPOINTMENTS: Final = "appointments"
TOPIC_ESCALATIONS: Final = "escalations"

#: clinic id -> the queues of every dashboard currently watching it.
_listeners: dict[uuid.UUID, set[asyncio.Queue[str]]] = {}


def notify(clinic_id: uuid.UUID, topic: str) -> None:
    """Tell every open dashboard for this clinic that `topic` changed.

    Synchronous and non-blocking on purpose. This is called from inside a
    request that is changing an appointment, and a notification must never be
    able to slow down or fail a booking: a full queue is dropped, not awaited.
    """
    payload = json.dumps(
        {"topic": topic, "at": dt.datetime.now(tz=dt.UTC).isoformat()},
        separators=(",", ":"),
    )
    for queue in _listeners.get(clinic_id, set()):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # The listener is behind. Every notice means the same thing --
            # "re-fetch" -- so a dropped one costs nothing the next does not
            # already say.
            continue


def listener_count(clinic_id: uuid.UUID) -> int:
    """How many dashboards are watching. For the test, and for diagnosis."""
    return len(_listeners.get(clinic_id, set()))


async def _stream(request: Request, clinic_id: uuid.UUID) -> AsyncIterator[str]:
    """One dashboard's connection, until it goes away."""
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_LIMIT)
    _listeners.setdefault(clinic_id, set()).add(queue)
    try:
        # Sent immediately so the browser knows the stream is live rather than
        # waiting on a first change that may be hours away.
        yield "event: ready\ndata: {}\n\n"
        while True:
            if await request.is_disconnected():
                return
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                # A comment line, which the EventSource API ignores. Its only
                # job is to stop a proxy closing an idle connection.
                yield ": keep-alive\n\n"
                continue
            yield f"event: changed\ndata: {payload}\n\n"
    finally:
        # Always, including on cancellation: a leaked queue is a slow memory
        # leak and a notification sent to nobody.
        watching = _listeners.get(clinic_id)
        if watching is not None:
            watching.discard(queue)
            if not watching:
                del _listeners[clinic_id]


@router.get(
    "/live",
    summary="Stream a notice when this clinic's data changes",
    response_class=StreamingResponse,
)
async def live(request: Request, scope: ClinicScopeDep) -> StreamingResponse:
    """Open a server-sent event stream for one clinic.

    Scoped by the same dependency every other route uses, so a dashboard
    cannot listen to another clinic's changes: the clinic comes from the
    request's scope and the listener is filed under it.

    Not audit-logged, and that is deliberate rather than an omission. Rule 8
    audits disclosures of patient data, and this stream discloses none -- it
    says a topic changed. The re-fetch it triggers is audited, which is where
    the patient data actually travels.
    """
    return StreamingResponse(
        _stream(request, scope.clinic_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Nginx buffers a streamed response by default, which holds every
            # notice until the buffer fills and makes the stream useless.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
