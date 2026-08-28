"""
test_server.py - headless self-test for backend/server.py.

Plain-assert script, NOT pytest (pytest is not installed in this project -
see backend/test_persistence.py / cv/test_risk_engine.py for the identical
convention). Run directly:

    cd backend && ../.venv/bin/python3 test_server.py

starlette.testclient.TestClient is used directly from plain functions - it
works fine outside pytest despite most tutorials wrapping it in a fixture.
Every test uses a fresh temp-file SQLite DB (tempfile) and, where clip
file/keep/discard/traversal behaviour is under test, a temp directory with
real files - matching test_persistence.py's "real file I/O, no mocking"
convention.

Covers: SharedState (publish/snapshot/wait_for_frame/mark_stopped, including
the timeout and stopped paths), CommandQueue (submit+drain round trip, the
timeout path, and a handler that raises not killing the queue), and every
HTTP route create_app() wires up (/risk_status's 503-before-first-publish,
/health in all three states, /events, /clips filtering + 400 + keep/discard
+ 404, the /clips/{id}/video traversal guard, /review, and the three review
command routes including the 503 command-timeout path).
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from starlette.testclient import TestClient

import db
import persistence as p
import server as srv


def _tmp_db_path(tmp: str) -> str:
    return os.path.join(tmp, "test.db")


def _fake_jpeg() -> bytes:
    # Not a real JPEG - server.py never decodes frame bytes, only frames/
    # forwards them, so any bytes object exercises the code path faithfully.
    return b"\xff\xd8\xff\xe0fakejpegbytes\xff\xd9"


def _status_dict(review_current_id=7, hazards=None, review_length=3, review_queue_ids=None):
    """Matches the reference risk_status shape from the Phase 7 brief
    (cv-agent's contract) closely enough to exercise /risk_status, /review,
    and the review command routes."""
    if hazards is None:
        hazards = [
            {"id": 7, "label": "object", "bbox": [120.0, 300.0, 260.0, 420.0],
             "state": "pending", "origin": "scan", "is_first_scan": False,
             "alerts_on_approach": True},
            {"id": 8, "label": "object", "bbox": [10.0, 10.0, 40.0, 40.0],
             "state": "confirmed", "origin": "scan", "is_first_scan": False,
             "alerts_on_approach": True},
        ]
    return {
        "timestamp": "2026-08-28T13:06:19.752431+00:00",
        "camera": {"index": 0, "name": "TestCam", "width": 1920, "height": 1080},
        "risk": {"zone": "orange", "value": 0.24, "person_id": 3, "hazard_id": 7,
                 "hazard_label": "object", "hazard_bbox": [120.0, 300.0, 260.0, 420.0]},
        "persons": [{"id": 3, "bbox": [800.0, 200.0, 1000.0, 700.0], "nearest_hazard_id": 7,
                     "distance": 0.24, "zone": "orange"}],
        "hazards": hazards,
        "hazard_counts": {"total": 9, "confirmed": 2, "pending_first_scan": 4,
                           "pending_new": 1, "dismissed": 2},
        "review_queue": {"length": review_length, "current_id": review_current_id,
                          "queue": [] if review_queue_ids is None else list(review_queue_ids)},
        "alert": {"text": "RED - person #3 near object", "active": True},
        "diagnostics": {"fps": 14.8, "model": "yolo26l.pt", "imgsz": 640, "conf": 0.35,
                         "device": "cpu", "scan_enabled": True, "socket_detect_enabled": True,
                         "persistence_enabled": True},
    }


# --- SharedState -------------------------------------------------------------


def test_shared_state_snapshot_before_any_publish():
    shared = srv.SharedState()
    jpeg, seq, status, age = shared.snapshot()
    assert jpeg is None
    assert status is None
    assert seq == 0
    assert age is None


def test_shared_state_publish_then_snapshot():
    shared = srv.SharedState()
    jpeg = _fake_jpeg()
    status = _status_dict()
    shared.publish(jpeg, status)
    got_jpeg, seq, got_status, age = shared.snapshot()
    assert got_jpeg == jpeg
    assert got_status is status  # same reference, publish() must not copy
    assert seq == 1
    assert age is not None and age >= 0.0


def test_shared_state_wait_for_frame_returns_on_publish():
    shared = srv.SharedState()

    def publisher():
        time.sleep(0.05)
        shared.publish(_fake_jpeg(), _status_dict())

    t = threading.Thread(target=publisher)
    t.start()
    jpeg, seq = shared.wait_for_frame(last_seq=0, timeout=2.0)
    t.join()
    assert jpeg is not None
    assert seq == 1


def test_shared_state_wait_for_frame_timeout_path():
    shared = srv.SharedState()
    start = time.monotonic()
    jpeg, seq = shared.wait_for_frame(last_seq=0, timeout=0.1)
    elapsed = time.monotonic() - start
    assert jpeg is None
    assert seq == 0  # last_seq echoed back unchanged
    assert elapsed >= 0.09  # actually waited, not a spurious instant return


def test_shared_state_wait_for_frame_stopped_path():
    shared = srv.SharedState()
    shared.mark_stopped()
    jpeg, seq = shared.wait_for_frame(last_seq=0, timeout=2.0)
    assert jpeg is None
    assert seq == -1


def test_shared_state_mark_stopped_wakes_blocked_waiter():
    shared = srv.SharedState()
    result = {}

    def waiter():
        result["value"] = shared.wait_for_frame(last_seq=0, timeout=5.0)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)  # let the waiter actually block first
    start = time.monotonic()
    shared.mark_stopped()
    t.join(timeout=2.0)
    elapsed = time.monotonic() - start
    assert result["value"] == (None, -1)
    assert elapsed < 1.0  # woken immediately, not after the 5s timeout


# --- CommandQueue --------------------------------------------------------


def test_command_queue_submit_drain_round_trip():
    commands = srv.CommandQueue()
    results = {}

    def submitter():
        results["result"] = commands.submit("confirm", 7, timeout=2.0)

    t = threading.Thread(target=submitter)
    t.start()
    time.sleep(0.05)  # let submit() enqueue before we drain
    commands.drain({"confirm": lambda entry_id: {"ok": True, "id": entry_id}})
    t.join(timeout=2.0)
    assert results["result"] == {"ok": True, "id": 7}


def test_command_queue_submit_times_out_if_never_drained():
    commands = srv.CommandQueue()
    raised = {}

    def submitter():
        try:
            commands.submit("confirm", 1, timeout=0.1)
        except srv.CommandTimeout as exc:
            raised["exc"] = exc

    t = threading.Thread(target=submitter)
    t.start()
    t.join(timeout=2.0)
    assert "exc" in raised


def test_command_queue_handler_raising_does_not_kill_queue():
    commands = srv.CommandQueue()
    results = {}

    def bad_handler(entry_id):
        raise ValueError("boom")

    def submitter(action, entry_id, key):
        results[key] = commands.submit(action, entry_id, timeout=2.0)

    t1 = threading.Thread(target=submitter, args=("dismiss", 1, "first"))
    t1.start()
    time.sleep(0.05)
    commands.drain({"dismiss": bad_handler})
    t1.join(timeout=2.0)
    assert "error" in results["first"]
    assert "boom" in results["first"]["error"]

    # Queue must still work for a subsequent, well-behaved command.
    t2 = threading.Thread(target=submitter, args=("skip", None, "second"))
    t2.start()
    time.sleep(0.05)
    commands.drain({"skip": lambda entry_id: {"ok": True}})
    t2.join(timeout=2.0)
    assert results["second"] == {"ok": True}


# --- App setup helper -----------------------------------------------------


def _make_app(tmp):
    db_path = _tmp_db_path(tmp)
    db.init_db(db_path)
    clips_dir = os.path.join(tmp, "clips", "pending")
    os.makedirs(clips_dir, exist_ok=True)
    shared = srv.SharedState()
    commands = srv.CommandQueue()
    app = srv.create_app(shared, commands, db_path=db_path, clips_dir=clips_dir)
    client = TestClient(app)
    return client, shared, commands, db_path, clips_dir


def _seed_event_and_clip(db_path, clips_dir, filename="clip.mp4", status=None):
    from dataclasses import dataclass

    @dataclass
    class FakeEvent:
        id: int
        kind: str
        person_id: object
        hazard_id: int
        hazard_label: str
        hazard_bbox: tuple
        zone: str
        peak_zone: str
        started_at: float = 0.0
        last_seen_at: float = 0.0
        clip_triggered: bool = False
        reason: str = ""

    @dataclass
    class FakeSignal:
        event: FakeEvent
        kind: str

    # A fresh run_started_at per call - this helper may be invoked several
    # times against the same DB within one test, and events.UNIQUE
    # (run_started_at, session_local_id) would otherwise collide on the
    # second call (each call constructs its own EventWriter, so the
    # in-process session_local_id map doesn't help here).
    writer = p.EventWriter(db_path=db_path, run_started_at=f"run-{time.monotonic_ns()}")
    event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                       hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red", clip_triggered=True)
    db_event_id = writer.record(FakeSignal(event=event, kind="opened"))
    clip_path = os.path.join(clips_dir, filename)
    with open(clip_path, "wb") as f:
        f.write(b"fake mp4 bytes")
    clip_id = p.insert_pending_clip(db_event_id, clip_path, "object", db_path=db_path)
    if status == "kept":
        p.keep_clip(clip_id, db_path=db_path)
    elif status == "discarded":
        p.discard_clip(clip_id, db_path=db_path)
    return db_event_id, clip_id, clip_path


# --- /risk_status, /health --------------------------------------------------


def test_risk_status_503_before_first_publish():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        resp = client.get("/risk_status")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "no frame published yet"


def test_risk_status_200_after_publish_includes_frame_seq():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        shared.publish(_fake_jpeg(), _status_dict())
        resp = client.get("/risk_status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["frame_seq"] == 1
        assert body["risk"]["zone"] == "orange"
        assert body["diagnostics"]["model"] == "yolo26l.pt"


def test_health_three_states():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)

        # 1. Never published.
        resp = client.get("/health")
        body = resp.json()
        assert body["camera_running"] is False
        assert body["last_frame_age_seconds"] is None
        assert body["frame_seq"] == 0

        # 2. Published, running.
        shared.publish(_fake_jpeg(), _status_dict())
        resp = client.get("/health")
        body = resp.json()
        assert body["camera_running"] is True
        assert body["last_frame_age_seconds"] is not None
        assert body["frame_seq"] == 1

        # 3. Stopped.
        shared.mark_stopped()
        resp = client.get("/health")
        body = resp.json()
        assert body["camera_running"] is False


# --- /video_feed (smoke - full MJPEG streaming loop is exercised via
# SharedState's own tests above; here we just confirm the route wires up and
# framing looks right for a bounded number of frames) -------------------


def test_video_feed_streams_mjpeg_frames():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        jpeg = _fake_jpeg()

        def publisher():
            for _ in range(3):
                time.sleep(0.02)
                shared.publish(jpeg, _status_dict())
            shared.mark_stopped()

        t = threading.Thread(target=publisher)
        t.start()
        with client.stream("GET", "/video_feed") as resp:
            assert resp.status_code == 200
            assert "multipart/x-mixed-replace" in resp.headers["content-type"]
            body = b"".join(resp.iter_bytes())
        t.join(timeout=2.0)
        assert b"--frame\r\n" in body
        assert b"Content-Type: image/jpeg\r\n" in body
        assert jpeg in body


# --- /events ----------------------------------------------------------------


def test_events_returns_recent_and_clamps_limit():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        _seed_event_and_clip(db_path, clips_dir)
        resp = client.get("/events")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) == 1
        assert body["events"][0]["hazard_label"] == "object"

        # limit clamp: absurd values must not error or explode.
        resp2 = client.get("/events?limit=99999")
        assert resp2.status_code == 200
        resp3 = client.get("/events?limit=0")
        assert resp3.status_code == 200


# --- /clips -------------------------------------------------------------


def test_clips_filtering_and_bad_status():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        _, pending_id, _ = _seed_event_and_clip(db_path, clips_dir, filename="p.mp4")
        _, discarded_id, _ = _seed_event_and_clip(db_path, clips_dir, filename="d.mp4", status="discarded")

        resp = client.get("/clips?status=pending")
        assert [c["id"] for c in resp.json()["clips"]] == [pending_id]

        resp = client.get("/clips?status=discarded")
        assert [c["id"] for c in resp.json()["clips"]] == [discarded_id]

        resp = client.get("/clips?status=not_a_real_status")
        assert resp.status_code == 400


def test_clip_detail_and_404():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        db_event_id, clip_id, _ = _seed_event_and_clip(db_path, clips_dir)

        resp = client.get(f"/clips/{clip_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["clip"]["id"] == clip_id
        assert body["event"]["id"] == db_event_id

        resp404 = client.get("/clips/999999")
        assert resp404.status_code == 404


def test_clip_keep_and_discard_and_404():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        _, keep_id, _ = _seed_event_and_clip(db_path, clips_dir, filename="k.mp4")
        _, discard_id, _ = _seed_event_and_clip(db_path, clips_dir, filename="dd.mp4")

        resp = client.post(f"/clips/{keep_id}/keep")
        assert resp.status_code == 200
        assert resp.json()["status"] == "kept"

        resp = client.post(f"/clips/{discard_id}/discard")
        assert resp.status_code == 200
        assert resp.json()["status"] == "discarded"

        resp = client.post("/clips/999999/keep")
        assert resp.status_code == 404
        resp = client.post("/clips/999999/discard")
        assert resp.status_code == 404


def test_clip_video_serves_file_and_404_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        _, clip_id, clip_path = _seed_event_and_clip(db_path, clips_dir, filename="v.mp4")

        resp = client.get(f"/clips/{clip_id}/video")
        assert resp.status_code == 200
        assert resp.content == b"fake mp4 bytes"

        os.remove(clip_path)
        resp = client.get(f"/clips/{clip_id}/video")
        assert resp.status_code == 404


def test_clip_video_traversal_guard_refuses_path_outside_clips_dir():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        db_event_id, _, _ = _seed_event_and_clip(db_path, clips_dir, filename="legit.mp4")

        # Simulate a corrupted/hand-edited row pointing outside the clips
        # tree entirely - the client never supplies this path, only the
        # server's own DB row does, but the guard must still refuse it.
        outside = os.path.join(tmp, "outside_secret.mp4")
        with open(outside, "wb") as f:
            f.write(b"should never be served")
        evil_clip_id = p.insert_pending_clip(db_event_id, outside, "object", db_path=db_path)

        resp = client.get(f"/clips/{evil_clip_id}/video")
        assert resp.status_code == 403


def test_clip_video_traversal_guard_allows_kept_sibling_dir():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        _, clip_id, _ = _seed_event_and_clip(db_path, clips_dir, filename="tokeep.mp4", status="kept")

        # keep_clip() moved the file to clips_dir/../kept/ - a sibling of
        # the pending dir, outside clips_dir itself but inside its parent.
        resp = client.get(f"/clips/{clip_id}/video")
        assert resp.status_code == 200
        assert resp.content == b"fake mp4 bytes"


# --- /review + command routes ----------------------------------------------


def test_review_before_publish_returns_503():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        resp = client.get("/review")
        assert resp.status_code == 503


def test_review_reflects_published_status():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        shared.publish(_fake_jpeg(), _status_dict(
            review_current_id=7, review_length=3, review_queue_ids=[7, 11, 12]))
        resp = client.get("/review")
        assert resp.status_code == 200
        body = resp.json()
        assert body["length"] == 3
        assert body["current_id"] == 7
        assert body["current"]["id"] == 7
        assert body["current"]["state"] == "pending"
        # `queue` is cv-agent's published ReviewQueue.ids() passed straight
        # through - NOT re-derived here from `hazards`. See the next test for
        # why that distinction is load-bearing rather than pedantic.
        assert body["queue"] == [7, 11, 12]


def test_review_queue_is_passed_through_not_derived_from_hazards():
    """Regression test for a real defect found in Phase 7 review.

    /review briefly built `queue` as "every hazard whose state is pending".
    That is a different set from the actual ReviewQueue, and the two diverge
    in exactly the case a parent hits most: after a skip, the queue is empty
    while all those entries are still PENDING. The derived version answered
    `length: 0` and a non-empty `queue` in one payload, and a Phase 8 UI
    rendering it would offer items that confirm/dismiss refuse (the loop-side
    handlers fail closed on anything that is not the current candidate).

    The queue is also strictly FIFO while `hazards` is id-ordered, so a
    derived version names the wrong "next up" too. Both are pinned here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)

        # The post-skip state: entries still pending, queue genuinely empty.
        shared.publish(_fake_jpeg(), _status_dict(
            review_current_id=None, review_length=0, review_queue_ids=[]))
        body = client.get("/review").json()
        assert body["length"] == 0
        assert body["current_id"] is None
        assert body["queue"] == [], "a skipped queue must be empty even though hazard 7 is still pending"

        # FIFO order must survive verbatim, not be re-sorted by hazard id.
        shared.publish(_fake_jpeg(), _status_dict(
            review_current_id=8, review_length=2, review_queue_ids=[8, 7]))
        body = client.get("/review").json()
        assert body["queue"] == [8, 7]
        assert body["current_id"] == 8


def test_review_command_routes_round_trip_via_drain():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        shared.publish(_fake_jpeg(), _status_dict())

        results = {}

        def confirm_call():
            results["resp"] = client.post("/review/7/confirm")

        t = threading.Thread(target=confirm_call)
        t.start()
        # Give the HTTP thread time to reach commands.submit() and block,
        # matching the real loop-thread drain cadence (once per frame).
        time.sleep(0.1)
        commands.drain({"confirm": lambda entry_id: {"ok": True, "confirmed": entry_id}})
        t.join(timeout=2.0)

        assert results["resp"].status_code == 200
        assert results["resp"].json() == {"ok": True, "confirmed": 7}


def test_review_skip_route():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        results = {}

        def skip_call():
            results["resp"] = client.post("/review/skip")

        t = threading.Thread(target=skip_call)
        t.start()
        time.sleep(0.1)
        commands.drain({"skip": lambda entry_id: {"ok": True, "skipped": True}})
        t.join(timeout=2.0)

        assert results["resp"].status_code == 200
        assert results["resp"].json() == {"ok": True, "skipped": True}


def test_review_dismiss_command_timeout_returns_503():
    with tempfile.TemporaryDirectory() as tmp:
        client, shared, commands, db_path, clips_dir = _make_app(tmp)
        # Monkeypatch this test's CommandQueue instance to a very short
        # timeout so the test doesn't have to wait a full second for the
        # default - submit() takes an explicit timeout, but the HTTP route
        # calls commands.submit(action, entry_id) with the queue's own
        # module-level default. We instead simply never drain, and shrink
        # COMMAND_DEFAULT_TIMEOUT_SECONDS for this one call via a direct
        # CommandQueue.submit override is unnecessary complexity - easier to
        # just accept the ~1s wait here, it's a single test.
        resp = client.post("/review/7/dismiss")
        assert resp.status_code == 503
        assert "did not respond" in resp.json()["detail"]


def run_all():
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"ALL TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    run_all()
