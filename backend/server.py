"""
server.py - GuardianEye Phase 7: the HTTP serving layer.

FastAPI wraps two things that already exist and belong to other layers:
cv-agent's camera/detection loop (cv/risk_engine.py) and backend-agent's own
Phase 6 SQLite persistence (db.py / persistence.py). This module owns none of
the domain logic in either - it is a thin, deliberately dumb pipe: read a
published snapshot, read the DB, or hand a command to the loop thread and
wait for it to come back. See CLAUDE.md decision 1 for why the video pipeline
and the API/UI are decoupled at all.

THE CONCURRENCY MODEL (settled, measured - do not change it; read this before
touching anything below)
--------------------------------------------------------------------------
cv/risk_engine.py's main() keeps the MAIN thread, because its local debug
window (cv2.imshow, protected by CLAUDE.md decision 1) throws
`cv2.error: Unknown C++ exception from OpenCV code` when called from a
non-main thread on this machine - Cocoa requires window/UI calls on the main
thread, and this was hit and measured, not assumed. uvicorn therefore runs on
a background DAEMON thread instead of the usual arrangement. That is the
reverse of what every uvicorn tutorial shows, and it is intentional.

Consequence: HTTP handlers must NEVER touch HazardMap / ReviewQueue /
PersonTracker / AlertManager directly - those objects are Layer A/B state and
belong to the loop thread alone. Handlers instead:
  1. read a published, immutable SNAPSHOT (SharedState) that the loop thread
     hands over once per frame, and
  2. request state changes (confirm/dismiss/skip) through a COMMAND QUEUE
     that only the loop thread drains, once per frame, so there is still
     exactly one writer to Layer A state, full stop.

Why the command queue exists at all rather than just calling
hazard_map.confirm()/.dismiss() straight from the HTTP thread: dismiss()
specifically (cv/risk_engine.py:846) needs the CURRENT camera frame to
compute the dismissal fingerprint that CLAUDE.md decision 4's "spot changed
since dismissal, re-raise" rule depends on. Only the loop thread has a
current frame. Doing the write from the HTTP thread would either need to
smuggle a frame across a thread boundary (defeating the point - by the time
the HTTP thread runs, "current" is a lie) or silently skip the fingerprint
and quietly break the re-raise rule. Routing every state-changing action
through one queue that only the loop thread drains keeps "exactly one writer"
true regardless of which action it is, rather than relying on each action to
independently remember why that matters.

/video_feed MUST use a SYNC generator, not async def
--------------------------------------------------------------------------
Starlette's StreamingResponse iterates a sync generator function in a
threadpool (not on the asyncio event loop), so blocking inside it on a
threading.Condition (SharedState.wait_for_frame) is safe - it blocks one
threadpool worker, not the event loop. If this were `async def` instead, that
same blocking wait would stall the event loop thread and freeze every other
endpoint in the process (including /risk_status, /clips, /review, etc. for
every other client) for as long as the camera stalls. This is a specific,
version-pinned Starlette behaviour (see backend/requirements.txt's comment on
why fastapi/starlette are pinned exactly, not >=) - do not "simplify" this
generator to async def later without re-reading this paragraph.

Cost this implies, named rather than left to be discovered under load: each
concurrent /video_feed viewer occupies one threadpool worker for as long as
they're connected. Starlette's default threadpool is sized 40 - comfortably
enough for "a tablet plus a laptop" on a home network, not sized for a public
stream with many simultaneous viewers. That is an acceptable, explicit limit
for this project's scope (a household), not an oversight.

Default bind host is 127.0.0.1, not 0.0.0.0
--------------------------------------------------------------------------
This is Shaked's explicit decision, on privacy grounds: /video_feed shows a
live room with a child in it, and /clips/{id}/video serves saved footage of
the same, and none of it has authentication in front of it (Phase 8's call,
not this phase's - see CLAUDE.md's "no cloud, ever" / local-network framing
in decision 8, which is about storage but reflects the same instinct here).
Binding to loopback only means nothing on the LAN - let alone the internet -
can reach any of this by default. A --host flag exists so a developer can
deliberately opt into a LAN bind while testing from a second device, but that
is an opt-in, not the default, and Phase 8 must decide on authentication
before that opt-in becomes this project's normal operating mode.
"""

import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from db import DEFAULT_DB_PATH, init_db
from persistence import (
    discard_clip,
    get_clip,
    get_clips_by_status,
    get_event_for_clip,
    get_recent_events,
    keep_clip,
    sweep_expired_clips,
)

# --- Tunables, named rather than inlined (house style) ---------------------

# How long /video_feed's generator blocks on a single wait_for_frame() call
# before looping back around to check for client disconnect / stopped state.
# 1s matches the brief exactly: long enough that a healthy ~15fps camera
# publishes many frames per wait, short enough that a stalled camera doesn't
# make the HTTP connection look dead to an intervening proxy/browser.
VIDEO_FEED_WAIT_TIMEOUT_SECONDS = 1.0

# Same idea for /review's long-poll-free command routes: how long an HTTP
# thread will wait for the loop thread to drain a submitted command before
# giving up and reporting the loop as unresponsive. 1s default per the brief;
# a real "the frame loop is wedged" condition should look exactly like a
# timeout, not hang the HTTP request forever.
COMMAND_DEFAULT_TIMEOUT_SECONDS = 1.0

# Background sweep_expired_clips() cadence. Matches risk_engine.py's own
# --sweep-interval default (60s, see backend/API.md) - the two callers are
# deliberately redundant per Phase 6's own note that this is safe, not a bug
# to dedupe.
CLIP_SWEEP_INTERVAL_SECONDS = 60.0

# The four legal clips.status values, named once here rather than repeated
# as string literals at every validation site (db.py's schema comment is the
# source of truth; this list must stay in sync with it by hand since SQLite
# has no CHECK constraint enforcing it - see db.py's SCHEMA_SQL).
VALID_CLIP_STATUSES = ("pending", "kept", "discarded", "expired")

MULTIPART_BOUNDARY = b"frame"


class SharedState:
    """The one piece of shared, mutable state HTTP handlers are allowed to
    touch directly. Guarded by a single threading.Condition rather than a
    plain Lock, because /video_feed's readers need to BLOCK until a new
    frame arrives (not just briefly hold a lock) and a Condition is exactly
    "a lock plus a place to wait/notify" - reimplementing that with a Lock
    and a busy-poll loop would burn CPU for no benefit.

    Ownership: `publish()` is called by the loop thread only, once per frame.
    Every other method may be called by any HTTP handler thread.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._status: dict | None = None
        self._frame_seq: int = 0
        self._last_publish_monotonic: float | None = None
        self._stopped: bool = False

    def publish(self, jpeg: bytes, status: dict) -> None:
        """Loop thread, once per frame. `jpeg` is the already-encoded clean
        annotated frame (no diagnostics burned in - CLAUDE.md decision 1);
        `status` is a freshly-built dict the caller promises not to mutate
        again after this call (we store the reference, not a copy, to keep
        this O(1) - see the module docstring's cost note on why copying here
        would be the wrong tradeoff at camera framerate).
        """
        with self._cond:
            self._jpeg = jpeg
            self._status = status
            self._frame_seq += 1
            self._last_publish_monotonic = time.monotonic()
            self._cond.notify_all()

    def snapshot(self) -> tuple:
        """(jpeg, frame_seq, status, age_seconds). Grabs references under the
        lock and releases IMMEDIATELY - serialization (JSON encoding,
        multipart framing) happens after release, never while holding the
        lock, so a slow HTTP response can never block the loop thread's next
        publish(). `age_seconds` is computed from a monotonic clock read
        while still holding the lock (paired with the monotonic write in
        publish()) so it reflects real elapsed time regardless of wall-clock
        adjustments.
        """
        with self._cond:
            jpeg = self._jpeg
            status = self._status
            seq = self._frame_seq
            last = self._last_publish_monotonic
        age = None if last is None else time.monotonic() - last
        return jpeg, seq, status, age

    def wait_for_frame(self, last_seq: int, timeout: float) -> tuple:
        """Block until frame_seq has advanced past `last_seq`, `timeout`
        elapses, or mark_stopped() fires. Returns (jpeg, seq) on a new frame,
        (None, last_seq) on a plain timeout (caller should loop and try
        again - used by /video_feed to keep the connection alive through a
        momentarily stalled camera), or (None, -1) once stopped (caller
        should terminate its generator).
        """
        with self._cond:
            if self._stopped:
                return None, -1
            if self._frame_seq == last_seq:
                self._cond.wait(timeout=timeout)
            if self._stopped:
                return None, -1
            if self._frame_seq == last_seq:
                return None, last_seq
            return self._jpeg, self._frame_seq

    def mark_stopped(self) -> None:
        """Camera loop is exiting (process shutdown, fatal camera error,
        etc). Wakes every thread blocked in wait_for_frame() so their
        /video_feed generators notice and terminate cleanly instead of
        hanging until their next 1s timeout tick (or, worse, looking alive
        forever on a connection nothing will ever publish to again).
        """
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def is_stopped(self) -> bool:
        with self._cond:
            return self._stopped


class CommandTimeout(Exception):
    """Raised by CommandQueue.submit() when the loop thread hasn't drained
    the command within `timeout` seconds - a real "the frame loop is wedged
    or has exited" signal, not a soft failure to paper over."""


class CommandQueue:
    """The one path by which an HTTP handler may request a Layer A state
    change (confirm / dismiss / skip a review-queue entry).

    WHY A QUEUE INSTEAD OF CALLING hazard_map.confirm()/.dismiss() DIRECTLY
    FROM THE HTTP THREAD (restated here, not just in the module docstring,
    because this is the load-bearing reason this class exists at all):
    hazard_map.dismiss(entry_id, frame) needs the CURRENT camera frame to
    compute the dismissal fingerprint CLAUDE.md decision 4's "spot changed
    since dismissal -> re-raise" rule is built on (cv/risk_engine.py:846).
    Only the loop thread has a current frame at the moment it's needed - an
    HTTP handler calling in from outside has no frame of its own to offer,
    and grabbing a stale one would silently defeat the fingerprint's purpose.
    Routing confirm/dismiss/skip alike through this queue, drained once per
    frame by the loop thread, keeps exactly one writer to Layer A state
    (HazardMap/ReviewQueue) full stop, and means the loop thread always has
    a real, current frame in hand when a handler needs one.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._queue = deque()  # each item: (action, entry_id, Event, result_box)

    def submit(self, action: str, entry_id, timeout: float = COMMAND_DEFAULT_TIMEOUT_SECONDS) -> dict:
        """HTTP thread. Enqueues the command and blocks on its own
        threading.Event until the loop thread's drain() sets it (or
        `timeout` elapses, raising CommandTimeout). `result_box` is a
        single-element list rather than a plain variable so drain() can
        write into it from the other thread without needing its own lock -
        the Event's set() already provides the happens-before edge the
        submitting thread needs to safely read result_box[0] afterwards.
        """
        event = threading.Event()
        result_box = [None]
        with self._lock:
            self._queue.append((action, entry_id, event, result_box))
        if not event.wait(timeout=timeout):
            raise CommandTimeout(f"loop thread did not drain command {action!r} within {timeout}s")
        return result_box[0]

    def drain(self, handlers: dict) -> None:
        """Loop thread, once per frame. Pops everything currently queued (a
        snapshot swap under the lock, so new submissions arriving mid-drain
        are left for the NEXT frame rather than processed against a
        half-updated hazard map) and runs each command's handler.

        A handler raising must NOT kill the frame loop - the frame loop is
        the whole product; one bad command is not worth taking the camera
        down. Any exception is caught, wrapped into an error result, and the
        waiting HTTP thread is still woken (with an error result rather than
        a hang) so submit() never blocks forever because of a handler bug.
        """
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
        for action, entry_id, event, result_box in pending:
            handler = handlers.get(action)
            try:
                if handler is None:
                    result_box[0] = {"error": f"unknown action {action!r}"}
                else:
                    result_box[0] = handler(entry_id)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
                result_box[0] = {"error": f"{type(exc).__name__}: {exc}"}
            event.set()


def _mjpeg_part(jpeg: bytes) -> bytes:
    """One multipart/x-mixed-replace part, standard MJPEG framing. Built as
    a single bytes join per frame rather than several small socket writes -
    cheap at camera framerate and keeps the boundary/headers/body atomic
    from Starlette's point of view.
    """
    header = (
        b"--" + MULTIPART_BOUNDARY + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
    )
    return header + jpeg + b"\r\n"


def _video_feed_generator(shared: SharedState):
    """SYNC generator - see the module docstring's "MUST use a sync
    generator" section for why this is load-bearing, not a style choice.
    """
    last_seq = 0
    while True:
        jpeg, seq = shared.wait_for_frame(last_seq, timeout=VIDEO_FEED_WAIT_TIMEOUT_SECONDS)
        if seq == -1:
            # mark_stopped() fired - the camera loop is gone. End the stream
            # cleanly rather than continuing to "wait" on a Condition nothing
            # will ever notify again.
            return
        if jpeg is None:
            # Plain timeout, not stopped: no new frame yet, but the camera
            # loop may just be momentarily slow. Loop back and wait again
            # rather than ending the response - keeps the connection (and
            # the browser's <img>) alive through a stall instead of forcing
            # a reconnect.
            continue
        last_seq = seq
        yield _mjpeg_part(jpeg)


def _clip_path_is_safe(path: str, clips_parent: str) -> bool:
    """SECURITY: the file path served by /clips/{id}/video comes ONLY from
    the DB row - the client never supplies a path, only an integer clip id -
    but we still verify the resolved path sits under the clips directory's
    PARENT before opening it. Why the parent and not clips_dir itself:
    persistence.keep_clip() moves a clip's file from <clips_dir>/pending/ to
    the SIBLING <clips_dir>/../kept/ (see persistence.py's _kept_dir_for),
    so a kept clip's real path is outside whatever single directory Phase 7
    was launched with as --clips-dir. Resolving against the shared parent of
    both pending/ and kept/ is the check that's actually true for both clip
    states, not a narrower one that would reject every kept clip as a
    "traversal attempt" it manifestly isn't.
    """
    real_path = os.path.realpath(path)
    real_parent = os.path.realpath(clips_parent)
    return real_path == real_parent or real_path.startswith(real_parent + os.sep)


def create_app(shared: SharedState, commands: CommandQueue, db_path: str = DEFAULT_DB_PATH,
                clips_dir: str = "clips/pending") -> FastAPI:
    """Builds the FastAPI app. Takes `shared`/`commands` as constructor
    arguments (rather than module-level globals) so tests can build a fresh,
    isolated app per test with its own SharedState/CommandQueue/db_path/
    clips_dir - see test_server.py.

    `clips_dir` is the directory risk_engine.py was launched with as its
    pending-clips directory (e.g. cv/clips/pending); /clips/{id}/video
    resolves the traversal guard against its PARENT (see
    _clip_path_is_safe's docstring) since kept clips live in the sibling
    directory.
    """
    init_db(db_path)
    clips_parent = os.path.dirname(os.path.abspath(clips_dir.rstrip(os.sep))) or os.path.abspath(clips_dir)

    app = FastAPI(title="GuardianEye backend", version="0.7.0")

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            _video_feed_generator(shared),
            media_type=f"multipart/x-mixed-replace; boundary={MULTIPART_BOUNDARY.decode()}",
        )

    @app.get("/risk_status")
    def risk_status():
        """Returns cv-agent's published status dict verbatim (see the
        reference shape in this phase's brief / cv/risk_engine.py) plus
        `frame_seq` from SharedState, which cv-agent's dict does not itself
        carry (it's a serving-layer concept, not a Layer A/B one). We do NOT
        build or reshape this dict - risk_status's fields (risk/persons/
        hazards/review_queue/alert/diagnostics) are cv-agent's contract, not
        ours; changing their shape here would silently fork the contract
        Phase 8's UI will be written against.
        """
        _jpeg, seq, status, _age = shared.snapshot()
        if status is None:
            return JSONResponse({"detail": "no frame published yet"}, status_code=503)
        body = dict(status)
        body["frame_seq"] = seq
        return body

    @app.get("/health")
    def health():
        """Distinguishes "camera loop is alive but the room is dark/blank"
        (a legitimate-looking black frame) from "the loop thread died and
        nothing will ever update again" (a frozen last frame that LOOKS the
        same from the video stream alone). `camera_running` is False both
        before any publish() has ever happened and after mark_stopped().
        """
        _jpeg, seq, status, age = shared.snapshot()
        running = (status is not None) and (not shared.is_stopped())
        return {"camera_running": running, "last_frame_age_seconds": age, "frame_seq": seq}

    @app.get("/events")
    def events(limit: int = 50):
        """Thin wrapper over persistence.get_recent_events() - the dicts it
        returns are already JSON-ready (see persistence.py). Two data-quality
        realities this endpoint deliberately does NOT paper over:

        1. There is no `ended_at` column, by Phase 6 design (db.py's module
           docstring explains why: AlertManager's "closed" signal is never
           voiced, so there is structurally nothing to persist a close time
           for). We do not synthesize one here from
           last_seen_at + ALERT_HOLD_SECONDS - that would be the serving
           layer inventing data that was deliberately never captured, and
           would quietly turn a documented decision into a fake column the
           next reader might trust as real.
        2. PersonTracker ID churn (confirmed against real data in Phase 6)
           means one continuous physical approach by the same child can
           legitimately appear here as several rows with different
           person_id values. This is not a Phase 7 bug and is not fixed
           here - flagged so whoever meets it in Phase 8 (building an
           events timeline UI) recognises it as a known, pre-existing
           characteristic of the data rather than filing it fresh.
        """
        clamped = max(1, min(500, limit))
        return {"events": get_recent_events(db_path=db_path, limit=clamped)}

    @app.get("/clips")
    def clips(status: str = "pending"):
        if status not in VALID_CLIP_STATUSES:
            return JSONResponse(
                {"detail": f"invalid status {status!r}; must be one of {VALID_CLIP_STATUSES}"},
                status_code=400,
            )
        return {"clips": get_clips_by_status(status, db_path=db_path)}

    @app.get("/clips/{clip_id}")
    def clip_detail(clip_id: int):
        clip = get_clip(clip_id, db_path=db_path)
        if clip is None:
            return JSONResponse({"detail": "clip not found"}, status_code=404)
        event = get_event_for_clip(clip_id, db_path=db_path)
        return {"clip": clip, "event": event}

    @app.post("/clips/{clip_id}/keep")
    def clip_keep(clip_id: int):
        if not keep_clip(clip_id, db_path=db_path):
            return JSONResponse({"detail": "clip not found or already gone"}, status_code=404)
        return get_clip(clip_id, db_path=db_path)

    @app.post("/clips/{clip_id}/discard")
    def clip_discard(clip_id: int):
        if not discard_clip(clip_id, db_path=db_path):
            return JSONResponse({"detail": "clip not found or already gone"}, status_code=404)
        return get_clip(clip_id, db_path=db_path)

    @app.get("/clips/{clip_id}/video")
    def clip_video(clip_id: int):
        clip = get_clip(clip_id, db_path=db_path)
        if clip is None:
            return JSONResponse({"detail": "clip not found"}, status_code=404)
        path = clip["path"]
        if not _clip_path_is_safe(path, clips_parent):
            # The path came from OUR OWN db row, never from the client - if
            # this ever fires it means a row was written pointing outside
            # the clips tree, not that a client tried to traverse. Refuse
            # anyway: this endpoint's job is to serve clip files, not
            # arbitrary paths a corrupted/hand-edited row happens to name.
            return JSONResponse({"detail": "clip path is outside the clips directory"}, status_code=403)
        real_path = os.path.realpath(path)
        if not os.path.isfile(real_path):
            # A discarded or expired clip's DB row survives on purpose
            # (persistence.py: audit trail) but its file is gone by design -
            # that is normal lifecycle, not a server error to 500 on.
            return JSONResponse({"detail": "clip file no longer exists"}, status_code=404)
        return FileResponse(real_path, media_type="video/mp4")

    @app.get("/review")
    def review():
        """Built entirely from the published status dict's `review_queue`
        and `hazards` fields - NOT from a live ReviewQueue/HazardMap object,
        which HTTP handlers may never touch (see module docstring). If a
        hazard entry the queue currently points at hasn't made it into the
        latest `hazards` list for some reason, `current` degrades to None
        rather than raising - a stale/partial status dict should produce a
        thin response, not a 500.
        """
        _jpeg, _seq, status, _age = shared.snapshot()
        if status is None:
            return JSONResponse({"detail": "no frame published yet"}, status_code=503)
        rq = status.get("review_queue") or {}
        hazards_by_id = {h["id"]: h for h in status.get("hazards", [])}
        current_id = rq.get("current_id")
        current = hazards_by_id.get(current_id) if current_id is not None else None
        # The real FIFO membership, published by cv-agent's
        # ReviewQueue.ids(). This was briefly derived here as "every hazard
        # whose state is pending" instead, which is a DIFFERENT set and was
        # wrong in a way worth recording so it isn't reintroduced: after a
        # skip ('s' / POST /review/skip) the queue is empty while every one
        # of those entries is still PENDING, so the derived version returned
        # `length: 0` alongside a non-empty `queue` in the same payload - and
        # a Phase 8 UI rendering it would offer a parent items no
        # confirm/dismiss call can act on (the loop-side handlers fail closed
        # on anything that is not the current candidate). Order was wrong for
        # the same reason: the queue is FIFO, `hazards` is id-ordered.
        queue_ids = list(rq.get("queue", []))
        return {
            "length": rq.get("length", 0),
            "current_id": current_id,
            "current": current,
            "queue": queue_ids,
        }

    def _submit_command(action: str, entry_id):
        try:
            return commands.submit(action, entry_id)
        except CommandTimeout as exc:
            return JSONResponse(
                {"detail": f"frame loop did not respond to {action!r} in time: {exc}"},
                status_code=503,
            )

    @app.post("/review/{entry_id}/confirm")
    def review_confirm(entry_id: int):
        return _submit_command("confirm", entry_id)

    @app.post("/review/{entry_id}/dismiss")
    def review_dismiss(entry_id: int):
        return _submit_command("dismiss", entry_id)

    @app.post("/review/skip")
    def review_skip():
        return _submit_command("skip", None)

    return app


def _sweep_loop(db_path: str, stop_event: threading.Event) -> None:
    """Runs sweep_expired_clips() on its own coarse schedule, redundantly
    with risk_engine.py's own --sweep-interval loop - persistence.py's own
    docstring says this is safe/expected (clips already swept are no longer
    'pending', so a second sweep is a no-op for them), not a bug to dedupe
    across processes.
    """
    while not stop_event.wait(timeout=CLIP_SWEEP_INTERVAL_SECONDS):
        try:
            sweep_expired_clips(db_path=db_path)
        except Exception as exc:  # noqa: BLE001 - a sweep failure must not kill the thread
            print(f"Warning: sweep_expired_clips failed: {exc}")


def start_server(shared: SharedState, commands: CommandQueue, *, host: str = "127.0.0.1",
                  port: int = 8000, db_path: str = DEFAULT_DB_PATH,
                  clips_dir: str = "clips/pending") -> threading.Thread:
    """Builds the app and runs uvicorn on a background DAEMON thread, leaving
    the MAIN thread free for cv/risk_engine.py's main() and its cv2.imshow
    debug window - see the module docstring's concurrency section for why
    this arrangement (not the usual one) is required on this machine.

    host defaults to 127.0.0.1, NOT 0.0.0.0 - Shaked's explicit privacy
    decision (module docstring has the full reasoning). Pass host="0.0.0.0"
    (or a specific LAN address) explicitly to opt into being reachable from
    other devices; that is a deliberate opt-in for development/testing, not
    this function's default behaviour, and Phase 8 must settle authentication
    before that opt-in becomes normal operating mode.

    Returns the started thread so the caller (risk_engine.py's main(), or a
    test) can join it, check is_alive(), etc.
    """
    app = create_app(shared, commands, db_path=db_path, clips_dir=clips_dir)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    # uvicorn's default signal handling calls signal.signal(), which Python
    # only permits from the main thread. This thread is deliberately NOT the
    # main thread (see module docstring), so uvicorn's normal SIGINT/SIGTERM
    # handlers would raise ValueError the moment they tried to install
    # themselves. Neutralising install_signal_handlers is what lets uvicorn
    # run on a background thread at all - shutdown is instead the caller's
    # responsibility (e.g. mark_stopped() plus letting the daemon thread die
    # with the process), not something this thread needs to catch itself.
    server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=server.run, name="guardianeye-uvicorn", daemon=True)
    thread.start()

    sweep_stop = threading.Event()
    sweep_thread = threading.Thread(
        target=_sweep_loop, args=(db_path, sweep_stop), name="guardianeye-clip-sweep", daemon=True
    )
    sweep_thread.start()

    return thread
