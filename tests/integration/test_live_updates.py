"""The live-update stream carries notices, never patient data.

The client asked how a clinic's dashboard learns that a patient responded.
These tests pin the answer, and the first one is the important one: a stream
that cannot carry patient data cannot leak it, and every disclosure still goes
through the ordinary audited route.

The rest cover what a real deployment does to a long-lived connection —
heartbeats, a listener that falls behind, a clinic that should not hear another
clinic's changes — because those are the paths a demo never reaches.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.review import live
from src.audit.context import AuditContext
from tests.integration.factories import appointment, at, create_graph, scope_client


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    return staff_context


@pytest.fixture(autouse=True)
def _no_leaked_listeners() -> None:
    """Each test starts with no listeners.

    Module-level state, so a test that leaves a queue behind would make the
    next one's counts wrong -- and a leaked queue in production is a slow
    memory leak, which is why `_stream` discards in a `finally`.
    """
    live._listeners.clear()


# ------------------------------------------------- the defining property


def test_a_notice_carries_no_patient_data() -> None:
    """The reason this is SSE and not a patient-data socket.

    An event says a topic changed, with a timestamp. No name, no appointment,
    no id -- so the stream cannot disclose anything, and the re-fetch it
    triggers goes through the route that audits (rule 8).
    """
    clinic_id = uuid.uuid4()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=live.QUEUE_LIMIT)
    live._listeners[clinic_id] = {queue}

    live.notify(clinic_id, live.TOPIC_APPOINTMENTS)

    payload = json.loads(queue.get_nowait())
    assert set(payload) == {"topic", "at"}, f"the notice carries more than it should: {payload}"
    assert payload["topic"] == live.TOPIC_APPOINTMENTS


def test_one_clinic_is_not_told_about_another() -> None:
    """A dashboard listens to the clinic its request was scoped to, and the
    listener is filed under that clinic. There is no path by which one hears
    the other."""
    mine, theirs = uuid.uuid4(), uuid.uuid4()
    my_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=live.QUEUE_LIMIT)
    their_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=live.QUEUE_LIMIT)
    live._listeners[mine] = {my_queue}
    live._listeners[theirs] = {their_queue}

    live.notify(mine, live.TOPIC_APPOINTMENTS)

    assert my_queue.qsize() == 1
    assert their_queue.qsize() == 0, "another clinic's dashboard was notified"


# --------------------------------------------------- behaviour under load


def test_a_listener_that_falls_behind_is_dropped_not_queued() -> None:
    """Every notice says the same thing -- "re-fetch" -- so a dropped one
    costs nothing the next does not already say. Queuing without bound would
    turn a slow dashboard into unbounded memory."""
    clinic_id = uuid.uuid4()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=live.QUEUE_LIMIT)
    live._listeners[clinic_id] = {queue}

    for _ in range(live.QUEUE_LIMIT + 10):
        live.notify(clinic_id, live.TOPIC_APPOINTMENTS)

    assert queue.qsize() == live.QUEUE_LIMIT, "the queue grew past its bound"


def test_notifying_with_no_listeners_is_harmless() -> None:
    """`notify` runs inside a request that is changing an appointment. It must
    never be able to fail that request, so a clinic nobody is watching is an
    ordinary case rather than an error."""
    live.notify(uuid.uuid4(), live.TOPIC_APPOINTMENTS)


def test_notify_never_raises_on_a_full_queue() -> None:
    """Same reasoning, stated as the failure it prevents: a booking must not
    fail because a dashboard stopped reading."""
    clinic_id = uuid.uuid4()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    live._listeners[clinic_id] = {queue}
    live.notify(clinic_id, live.TOPIC_APPOINTMENTS)

    # Second one has nowhere to go. No exception reaches the caller.
    live.notify(clinic_id, live.TOPIC_APPOINTMENTS)
    assert queue.qsize() == 1


# ---------------------------------------------------------- the endpoint


async def test_the_stream_opens_and_says_so(session: AsyncSession) -> None:
    """A `ready` event is sent immediately, before any change.

    Without it a dashboard cannot distinguish "connected, nothing has
    happened" from "never connected", and the first real change may be hours
    away.

    Driven against the generator rather than through `TestClient`. The client
    runs the app in a portal on another thread and reading an endless stream
    from it deadlocks: the heartbeat keeps the response open by design, so
    there is no end for the reader to reach. That cost one hung test run to
    find, and it is why the header assertions live in their own test below.
    """
    graph = await create_graph(session)

    class _Connected:
        async def is_disconnected(self) -> bool:
            return False

    stream = live._stream(_Connected(), graph.clinic_id)  # type: ignore[arg-type]
    try:
        first = await anext(stream)
        assert first.startswith("event: ready"), first
        # And the listener was registered, which is what makes a later notify
        # reach this connection at all.
        assert live.listener_count(graph.clinic_id) == 1
    finally:
        await stream.aclose()

    # Closed cleanly: the queue is discarded in a `finally`, because a leaked
    # one is a slow memory leak and a notice sent to nobody.
    assert live.listener_count(graph.clinic_id) == 0


def test_the_stream_sets_the_headers_a_proxy_needs() -> None:
    """The headers a proxy needs, asserted without opening a connection.

    `TestClient.stream` blocks on a response that never ends, and this one is
    designed never to end -- the heartbeat keeps it open. So the headers are
    checked where they are declared. The deployment-critical one is
    `X-Accel-Buffering: no`: nginx buffers a streamed response by default,
    holding every notice until the buffer fills and making the stream useless.
    """
    import inspect

    source = inspect.getsource(live.live)
    assert '"X-Accel-Buffering": "no"' in source
    assert "no-cache" in source
    assert 'media_type="text/event-stream"' in source


async def test_a_write_reaches_an_open_stream(session: AsyncSession, client: TestClient) -> None:
    """End to end: a cancellation through the API notifies a listener.

    Asserted on the listener rather than by reading the stream, because
    `TestClient` serialises requests -- holding a stream open while making a
    second request deadlocks. What matters is that the write path calls
    `notify`, and the queue is where that is observable.
    """
    graph = await create_graph(session)
    row = appointment(graph, at(3, 9, 0))
    session.add(row)
    await session.commit()
    scoped = scope_client(client, graph)

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=live.QUEUE_LIMIT)
    live._listeners[graph.clinic_id] = {queue}

    assert scoped.post(f"/review/appointments/{row.id}/cancel", json={}).status_code == 200

    assert queue.qsize() >= 1, "cancelling an appointment notified nobody"
    assert json.loads(queue.get_nowait())["topic"] == live.TOPIC_APPOINTMENTS


def test_the_heartbeat_is_shorter_than_a_common_proxy_timeout() -> None:
    """A proxy closes an idle connection, usually at 30 or 60 seconds. The
    heartbeat has to beat the shortest of those or the stream dies quietly and
    the dashboard goes stale without saying so."""
    assert 0 < live.HEARTBEAT_SECONDS < 30


async def test_a_clinic_may_not_open_unlimited_streams(
    session: AsyncSession, client: TestClient
) -> None:
    """The per-listener queue bounds how much one slow dashboard holds; it does
    not bound how many dashboards connect.

    A QA audit measured 500 listeners registered against one clinic with
    nothing refusing the 501st, each costing a queue, a set entry and a held
    connection on a single-worker deployment. A clinic has a handful of screens,
    so the cap is small and refusing is a 503 somebody can read.
    """
    graph = await create_graph(session)
    await session.commit()
    scoped = scope_client(client, graph)

    # Fill the clinic's slots directly: opening real streams through
    # TestClient would block, since each one is a long-lived response.
    live._listeners[graph.clinic_id] = {
        asyncio.Queue(maxsize=live.QUEUE_LIMIT) for _ in range(live.LISTENERS_PER_CLINIC)
    }
    try:
        response = scoped.get("/review/live")
        assert response.status_code == 503, response.text
        assert "Demasiadas pantallas" in response.json()["detail"]
    finally:
        live._listeners.pop(graph.clinic_id, None)


async def test_a_stream_for_a_clinic_that_does_not_exist_is_refused(
    client: TestClient,
) -> None:
    """`clinic_scope` deliberately does not reject an unknown clinic -- the
    row-level policies simply match nothing -- which is right for a route that
    reads rows and wrong for this one. The stream reads none, so there is
    nothing for a policy to filter: an arbitrary UUID became a dictionary key
    that would never be notified and never collected while the connection was
    held. A typo'd dashboard URL is now a 404 rather than a dead stream.
    """
    nobody = uuid.uuid4()
    response = client.get("/review/live", params={"clinic_id": str(nobody)})
    assert response.status_code == 404, response.text
    assert live.listener_count(nobody) == 0, "a listener was registered for an unknown clinic"
