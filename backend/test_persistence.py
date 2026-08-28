"""
test_persistence.py - headless self-test for backend/persistence.py and
backend/db.py.

Plain-assert script, not pytest (pytest is not installed in this project -
see cv/test_risk_engine.py for the identical convention this file follows).
Run directly:

    python backend/test_persistence.py

Every test uses a fresh temp-file SQLite DB (tempfile) and, where clip file
move/delete behaviour is under test, a temp directory - matching
cv/test_risk_engine.py's write_clip tests (real files, no mocking).

Covers: schema creation, EventWriter's insert-then-update-on-escalate
behaviour and its session_local_id->db id map, clip insert/write-completion,
recent-events/event-for-clip/clips-by-status queries, keep/discard actions
(real file moves/deletes), and the 24h auto-delete sweep (with synthetic
`now`, not real elapsed time).
"""

import os
import sys
import tempfile
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import persistence as p


# --- Synthetic AlertEvent/AlertSignal stand-ins ---------------------------
# risk_engine.py's real AlertEvent/AlertSignal are dataclasses with this
# exact shape (see cv/risk_engine.py). Reimplemented minimally here rather
# than imported, so this test suite has no dependency on cv/ (camera/torch/
# ultralytics imports risk_engine.py would drag in) - persistence.py itself
# also has no import of risk_engine.py, by design (see its module
# docstring: the dependency direction is one-way).


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


def _tmp_db_path(tmp: str) -> str:
    return os.path.join(tmp, "test.db")


# --- Schema ----------------------------------------------------------------


def test_init_db_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        db.init_db(path)  # must not raise on a second call
        conn = db.get_connection(path)
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        assert "events" in tables
        assert "clips" in tables


# --- EventWriter -------------------------------------------------------------


def test_event_writer_inserts_new_row_on_opened():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="2026-01-01T00:00:00")
        event = FakeEvent(
            id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
            hazard_bbox=(1.0, 2.0, 3.0, 4.0), zone="yellow", peak_zone="yellow",
        )
        db_id = writer.record(FakeSignal(event=event, kind="opened"))
        assert db_id is not None
        assert writer.db_id_for(1) == db_id

        recent = p.get_recent_events(db_path=path)
        assert len(recent) == 1
        row = recent[0]
        assert row["kind"] == "proximity"
        assert row["hazard_label"] == "object"
        assert row["hazard_bbox"] == [1.0, 2.0, 3.0, 4.0]
        assert row["zone"] == "yellow"
        assert row["peak_zone"] == "yellow"
        assert row["clip_triggered"] is False
        assert row["run_started_at"] == "2026-01-01T00:00:00"
        assert row["session_local_id"] == 1


def test_event_writer_updates_existing_row_on_escalate():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="2026-01-01T00:00:00")
        event = FakeEvent(
            id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
            hazard_bbox=(1.0, 2.0, 3.0, 4.0), zone="yellow", peak_zone="yellow",
        )
        first_id = writer.record(FakeSignal(event=event, kind="opened"))

        event.zone = "red"
        event.peak_zone = "red"
        event.clip_triggered = True
        second_id = writer.record(FakeSignal(event=event, kind="escalated"))

        assert first_id == second_id  # same row updated, not a new one
        recent = p.get_recent_events(db_path=path)
        assert len(recent) == 1  # still exactly one row for this event
        assert recent[0]["zone"] == "red"
        assert recent[0]["peak_zone"] == "red"
        assert recent[0]["clip_triggered"] is True


def test_event_writer_two_different_events_get_two_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="2026-01-01T00:00:00")
        e1 = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="a",
                        hazard_bbox=(0, 0, 1, 1), zone="yellow", peak_zone="yellow")
        e2 = FakeEvent(id=2, kind="new_object", person_id=None, hazard_id=6, hazard_label="b",
                        hazard_bbox=(0, 0, 1, 1), zone="none", peak_zone="none", reason="room scan")
        writer.record(FakeSignal(event=e1, kind="opened"))
        writer.record(FakeSignal(event=e2, kind="opened"))
        recent = p.get_recent_events(db_path=path)
        assert len(recent) == 2


def test_event_writer_new_object_reason_persisted_proximity_reason_null():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="2026-01-01T00:00:00")
        e = FakeEvent(id=1, kind="new_object", person_id=None, hazard_id=6, hazard_label="b",
                      hazard_bbox=(0, 0, 1, 1), zone="none", peak_zone="none", reason="room scan")
        writer.record(FakeSignal(event=e, kind="opened"))
        row = p.get_recent_events(db_path=path)[0]
        assert row["reason"] == "room scan"

        e2 = FakeEvent(id=2, kind="proximity", person_id=1, hazard_id=7, hazard_label="c",
                       hazard_bbox=(0, 0, 1, 1), zone="yellow", peak_zone="yellow", reason="")
        writer.record(FakeSignal(event=e2, kind="opened"))
        rows = {r["session_local_id"]: r for r in p.get_recent_events(db_path=path)}
        assert rows[2]["reason"] is None


def test_two_separate_runs_do_not_collide_on_session_local_id():
    # Different run_started_at, same session_local_id=1 - two different
    # physical processes both producing "event #1". Must not violate the
    # UNIQUE(run_started_at, session_local_id) constraint or overwrite each
    # other.
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer_a = p.EventWriter(db_path=path, run_started_at="run-A")
        writer_b = p.EventWriter(db_path=path, run_started_at="run-B")
        e_a = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="a",
                        hazard_bbox=(0, 0, 1, 1), zone="yellow", peak_zone="yellow")
        e_b = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=9, hazard_label="b",
                        hazard_bbox=(0, 0, 1, 1), zone="yellow", peak_zone="yellow")
        writer_a.record(FakeSignal(event=e_a, kind="opened"))
        writer_b.record(FakeSignal(event=e_b, kind="opened"))
        recent = p.get_recent_events(db_path=path)
        assert len(recent) == 2
        labels = {r["run_started_at"]: r["hazard_label"] for r in recent}
        assert labels == {"run-A": "a", "run-B": "b"}


# --- Clips: insert / write-completion / queries -----------------------------


def test_insert_pending_clip_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red", clip_triggered=True)
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))

        clip_path = os.path.join(tmp, "clips", "pending", "clip1.mp4")
        clip_id = p.insert_pending_clip(db_event_id, clip_path, "object", db_path=path)
        clip = p.get_clip(clip_id, db_path=path)
        assert clip["status"] == "pending"
        assert clip["write_status"] == "writing"
        assert clip["frame_count"] is None
        assert clip["event_id"] == db_event_id


def test_update_clip_write_result_success_and_failure():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))

        clip_path = os.path.join(tmp, "clip.mp4")
        clip_id = p.insert_pending_clip(db_event_id, clip_path, "object", db_path=path)
        p.update_clip_write_result(clip_path, True, 105, db_path=path)
        clip = p.get_clip(clip_id, db_path=path)
        assert clip["write_status"] == "written"
        assert clip["frame_count"] == 105

        clip_path2 = os.path.join(tmp, "clip2.mp4")
        clip_id2 = p.insert_pending_clip(db_event_id, clip_path2, "object", db_path=path)
        p.update_clip_write_result(clip_path2, False, 0, db_path=path)
        clip2 = p.get_clip(clip_id2, db_path=path)
        assert clip2["write_status"] == "failed"


def test_get_event_for_clip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))
        clip_id = p.insert_pending_clip(db_event_id, os.path.join(tmp, "c.mp4"), "object", db_path=path)

        linked = p.get_event_for_clip(clip_id, db_path=path)
        assert linked["id"] == db_event_id
        assert linked["hazard_label"] == "object"


def test_get_event_for_clip_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        assert p.get_event_for_clip(999, db_path=path) is None


def test_get_clips_by_status():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))
        c1 = p.insert_pending_clip(db_event_id, os.path.join(tmp, "c1.mp4"), "object", db_path=path)
        c2 = p.insert_pending_clip(db_event_id, os.path.join(tmp, "c2.mp4"), "object", db_path=path)
        p.discard_clip(c2, db_path=path)

        pending = p.get_clips_by_status("pending", db_path=path)
        discarded = p.get_clips_by_status("discarded", db_path=path)
        assert [c["id"] for c in pending] == [c1]
        assert [c["id"] for c in discarded] == [c2]


# --- Keep / discard ----------------------------------------------------------


def test_keep_clip_moves_file_and_updates_status():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))

        pending_dir = os.path.join(tmp, "clips", "pending")
        os.makedirs(pending_dir, exist_ok=True)
        src = os.path.join(pending_dir, "clip.mp4")
        with open(src, "wb") as f:
            f.write(b"fake mp4 bytes")
        clip_id = p.insert_pending_clip(db_event_id, src, "object", db_path=path)

        assert p.keep_clip(clip_id, db_path=path) is True

        kept_path = os.path.join(tmp, "clips", "kept", "clip.mp4")
        assert os.path.isfile(kept_path)
        assert not os.path.isfile(src)

        clip = p.get_clip(clip_id, db_path=path)
        assert clip["status"] == "kept"
        assert clip["path"] == kept_path
        assert clip["decided_at"] is not None


def test_keep_clip_missing_id_returns_false():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        assert p.keep_clip(999, db_path=path) is False


def test_discard_clip_deletes_file_and_keeps_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))

        pending_dir = os.path.join(tmp, "clips", "pending")
        os.makedirs(pending_dir, exist_ok=True)
        src = os.path.join(pending_dir, "clip.mp4")
        with open(src, "wb") as f:
            f.write(b"fake mp4 bytes")
        clip_id = p.insert_pending_clip(db_event_id, src, "object", db_path=path)

        assert p.discard_clip(clip_id, db_path=path) is True
        assert not os.path.isfile(src)
        clip = p.get_clip(clip_id, db_path=path)
        assert clip["status"] == "discarded"
        assert clip["decided_at"] is not None


# --- Auto-delete sweep -------------------------------------------------------


def test_sweep_expired_clips_deletes_only_old_pending():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))

        pending_dir = os.path.join(tmp, "clips", "pending")
        os.makedirs(pending_dir, exist_ok=True)

        now = time.time()
        old_path = os.path.join(pending_dir, "old.mp4")
        with open(old_path, "wb") as f:
            f.write(b"old")
        old_id = p.insert_pending_clip(db_event_id, old_path, "object", db_path=path)
        # Force old_id's created_at to 25 hours before `now` directly, since
        # insert_pending_clip always stamps "now" by default.
        conn = db.get_connection(path)
        try:
            from datetime import datetime, timezone
            old_ts = datetime.fromtimestamp(now - 25 * 3600, tz=timezone.utc).isoformat()
            conn.execute("UPDATE clips SET created_at = ? WHERE id = ?", (old_ts, old_id))
            conn.commit()
        finally:
            conn.close()

        fresh_path = os.path.join(pending_dir, "fresh.mp4")
        with open(fresh_path, "wb") as f:
            f.write(b"fresh")
        fresh_id = p.insert_pending_clip(db_event_id, fresh_path, "object", db_path=path)

        swept = p.sweep_expired_clips(db_path=path, now=now)
        assert swept == 1

        old_clip = p.get_clip(old_id, db_path=path)
        fresh_clip = p.get_clip(fresh_id, db_path=path)
        assert old_clip["status"] == "expired"
        assert old_clip["deleted_at"] is not None
        assert not os.path.isfile(old_path)
        assert fresh_clip["status"] == "pending"
        assert os.path.isfile(fresh_path)


def test_sweep_expired_clips_never_touches_kept_or_discarded():
    with tempfile.TemporaryDirectory() as tmp:
        path = _tmp_db_path(tmp)
        db.init_db(path)
        writer = p.EventWriter(db_path=path, run_started_at="run-A")
        event = FakeEvent(id=1, kind="proximity", person_id=1, hazard_id=5, hazard_label="object",
                          hazard_bbox=(0, 0, 1, 1), zone="red", peak_zone="red")
        db_event_id = writer.record(FakeSignal(event=event, kind="opened"))

        pending_dir = os.path.join(tmp, "clips", "pending")
        os.makedirs(pending_dir, exist_ok=True)
        kept_src = os.path.join(pending_dir, "kept.mp4")
        with open(kept_src, "wb") as f:
            f.write(b"x")
        kept_id = p.insert_pending_clip(db_event_id, kept_src, "object", db_path=path)
        p.keep_clip(kept_id, db_path=path)

        now = time.time()
        conn = db.get_connection(path)
        try:
            from datetime import datetime, timezone
            old_ts = datetime.fromtimestamp(now - 48 * 3600, tz=timezone.utc).isoformat()
            conn.execute("UPDATE clips SET created_at = ? WHERE id = ?", (old_ts, kept_id))
            conn.commit()
        finally:
            conn.close()

        swept = p.sweep_expired_clips(db_path=path, now=now)
        assert swept == 0
        clip = p.get_clip(kept_id, db_path=path)
        assert clip["status"] == "kept"


def run_all():
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"ALL TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    run_all()
