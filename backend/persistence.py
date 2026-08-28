"""
persistence.py - GuardianEye Phase 6: event/clip persistence and queries.

Data-layer only: no FastAPI, no HTTP, no UI (Phase 7/8's job - see
PHASE_PLAN.md). Consumes plain data cv-agent's risk_engine.py already
produces (AlertSignal/AlertEvent) - it does not import risk_engine.py, to
keep the dependency direction one-way (cv-agent's pipeline code doesn't need
to know this module exists; risk_engine.py imports THIS, not the reverse).

Scope, restated from docs/decision-log.md's 2026-08-28 "Phase 6 kickoff"
entry (Shaked's decision 3): the `events` table persists only alert signals
that were actually voiced/bannered to the parent - never every internal
AlertManager state transition. Concretely, `EventWriter.record()` must only
be called with a signal AlertArbiter.offer() actually returned (i.e.
`voiced is not None` in risk_engine.py's handle_alert_signal()), never
speculatively.

Threading note (real correctness issue, not hypothetical): `write_clip()`'s
`on_complete` callback fires on ClipRecorder's background write thread, not
the main frame loop thread. A `sqlite3.Connection` is not safe to share
across threads by default. Every function below therefore opens its own
short-lived connection (via backend.db.get_connection) rather than being
handed one from outside - this makes every function in this module safe to
call from any thread, including from write_clip's on_complete callback.

Timestamps: every persisted timestamp is a FRESH wall-clock read
(`now_iso()`, i.e. `datetime.now(timezone.utc).isoformat()`) taken at the
moment the persistence call actually runs - never a converted
`time.monotonic()` value from AlertManager/AlertEvent, which has no fixed
relationship to a wall-clock date and isn't comparable across process
restarts. See db.py's module docstring and the decision-log entry above for
the full reasoning.
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

from db import DEFAULT_DB_PATH, get_connection

# CLAUDE.md decision 6: undecided ("pending") clips auto-delete after a
# timeout, so a parent who never opens the app doesn't accumulate clips
# forever. 24 hours (Shaked, 2026-08-28 Phase 6 kickoff decision 1) - a named
# constant, not a bare literal, matching this codebase's existing style (see
# cv/risk_engine.py's ALERT_HOLD_SECONDS / DISMISS_REAPPEAR_CHANGE_FRAC).
CLIP_PENDING_TIMEOUT_HOURS = 24

# A clip row whose write never completes (process crash mid-write, thread
# never calls on_complete) shouldn't be retried forever by a naive caller -
# not currently enforced anywhere in this module, called out here so a
# future reader isn't surprised a 'writing' row can in principle persist
# indefinitely. Not in scope for Phase 6 (no scheduler exists yet to revisit
# it); flagged for Phase 7.

# Small, bounded retry for the rare (and, per the reasoning in
# docs/decision-log.md's Phase 6 entry, expected-never-in-practice) race
# where write_clip's background thread finishes and calls on_complete before
# the synchronous INSERT that creates the clip row (done immediately after
# ClipRecorder.poll() returns, in risk_engine.py's main loop) has committed.
# The insert is a single fast statement; the write itself takes ~130-200ms
# (measured, docs/decision-log.md 2026-08-22), so in practice the insert
# always wins - this retry is a safety net, not the primary mechanism.
_CLIP_ROW_RETRY_ATTEMPTS = 5
_CLIP_ROW_RETRY_DELAY_SECONDS = 0.05


def now_iso() -> str:
    """A fresh wall-clock timestamp, ISO8601 UTC. See module docstring."""
    return datetime.now(timezone.utc).isoformat()


# --- Events -------------------------------------------------------------


class EventWriter:
    """Persists voiced AlertSignals to the `events` table for the lifetime
    of one risk_engine.py process.

    Owns the `session_local_id -> events.id` map for that process
    (docs/decision-log.md, 2026-08-28 Phase 6 kickoff: `AlertEvent.id` is
    session-local and resets to 1 every process start, so it cannot be a
    database key on its own; an in-process dict scoped to this writer's
    lifetime is sufficient to route a later "escalated" signal to the
    already-inserted row for the same event). One instance per
    risk_engine.py run - construct it once in main(), call `.record()` from
    inside `handle_alert_signal()` only when `voiced is not None`.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, run_started_at: str = None):
        self.db_path = db_path
        self.run_started_at = run_started_at or now_iso()
        self._session_to_db_id: dict[int, int] = {}

    def db_id_for(self, session_local_id: int):
        """The events.id row for a given AlertEvent.id in THIS run, or None
        if that event was never voiced (and therefore never persisted -
        decision 3's scope). Used by the clip-insert call site to link a
        clip to its triggering event.
        """
        return self._session_to_db_id.get(session_local_id)

    def record(self, signal) -> int:
        """signal is an AlertSignal already confirmed voiced by the caller
        (signal.kind is "opened" or "escalated" - never call this with a
        "closed" signal; there is nothing to persist for one, per decision
        3). Inserts a new row on "opened", updates the existing row on
        "escalated". Returns the events.id row.
        """
        event = signal.event
        ts = now_iso()
        conn = get_connection(self.db_path)
        try:
            db_id = self._session_to_db_id.get(event.id)
            if db_id is None:
                cur = conn.execute(
                    """
                    INSERT INTO events (
                        run_started_at, session_local_id, kind, person_id,
                        hazard_id, hazard_label, hazard_bbox, zone, peak_zone,
                        reason, started_at, last_seen_at, clip_triggered
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.run_started_at,
                        event.id,
                        event.kind,
                        event.person_id,
                        event.hazard_id,
                        event.hazard_label,
                        json.dumps(list(event.hazard_bbox)),
                        event.zone,
                        event.peak_zone,
                        event.reason if event.reason else None,
                        ts,
                        ts,
                        1 if event.clip_triggered else 0,
                    ),
                )
                db_id = cur.lastrowid
                self._session_to_db_id[event.id] = db_id
            else:
                conn.execute(
                    """
                    UPDATE events
                    SET zone = ?, peak_zone = ?, last_seen_at = ?, clip_triggered = ?
                    WHERE id = ?
                    """,
                    (event.zone, event.peak_zone, ts, 1 if event.clip_triggered else 0, db_id),
                )
            conn.commit()
        finally:
            conn.close()
        return db_id


def get_recent_events(db_path: str = DEFAULT_DB_PATH, limit: int = 50) -> list[dict]:
    """Most recent events first. Plain dicts (JSON-ready), not sqlite3.Row -
    Phase 7's HTTP layer can serialize these directly.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_event_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_event_for_clip(clip_id: int, db_path: str = DEFAULT_DB_PATH):
    """The event a given clip belongs to (join via clips.event_id), or None
    if the clip id doesn't exist.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT events.* FROM events
            JOIN clips ON clips.event_id = events.id
            WHERE clips.id = ?
            """,
            (clip_id,),
        ).fetchone()
        return _event_row_to_dict(row) if row is not None else None
    finally:
        conn.close()


def _event_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["hazard_bbox"] = json.loads(d["hazard_bbox"])
    d["clip_triggered"] = bool(d["clip_triggered"])
    return d


# --- Clips ----------------------------------------------------------------


def insert_pending_clip(
    event_db_id: int, path: str, hazard_label: str, db_path: str = DEFAULT_DB_PATH, created_at: str = None
) -> int:
    """A clip row created the moment ClipRecorder.poll() reports a write
    just started - status='pending', write_status='writing' (both column
    defaults), frame_count still unknown. Returns the new clips.id.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO clips (event_id, path, hazard_label, created_at) VALUES (?, ?, ?, ?)",
            (event_db_id, path, hazard_label, created_at or now_iso()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_clip_write_result(path: str, success: bool, frame_count, db_path: str = DEFAULT_DB_PATH) -> None:
    """Called from write_clip's on_complete callback, on the background
    write thread - see module docstring's threading note. Matches by `path`
    (UNIQUE) rather than a clip id handed back across the thread boundary,
    so no id has to survive the round trip through ClipRecorder's closure.

    Retries briefly if the row doesn't exist yet (see
    _CLIP_ROW_RETRY_ATTEMPTS' comment above) - the insert (main thread,
    synchronous, sub-millisecond) is expected to always win the race against
    the write itself (background thread, ~130-200ms), but this is a cheap
    safety net rather than an assumption relied on silently.
    """
    write_status = "written" if success else "failed"
    for attempt in range(_CLIP_ROW_RETRY_ATTEMPTS):
        conn = get_connection(db_path)
        try:
            cur = conn.execute(
                "UPDATE clips SET write_status = ?, frame_count = ? WHERE path = ?",
                (write_status, frame_count, path),
            )
            conn.commit()
            if cur.rowcount > 0:
                return
        finally:
            conn.close()
        if attempt < _CLIP_ROW_RETRY_ATTEMPTS - 1:
            time.sleep(_CLIP_ROW_RETRY_DELAY_SECONDS)
    print(f"Warning: update_clip_write_result found no clip row for path={path!r} after {_CLIP_ROW_RETRY_ATTEMPTS} attempts.")


def get_clips_by_status(status: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Clips filtered by status ('pending' / 'kept' / 'discarded' /
    'expired'), most recently created first.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM clips WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_clip(clip_id: int, db_path: str = DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _kept_dir_for(pending_path: str) -> str:
    """cv/clips/kept/, a sibling of the pending clip's own parent directory
    (derived from the actual clip path, not a hardcoded constant, so this
    works regardless of CWD/--clips-dir).
    """
    clips_dir = os.path.dirname(os.path.dirname(os.path.abspath(pending_path)))
    return os.path.join(clips_dir, "kept")


def keep_clip(clip_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Parent pressed "keep": move the file from cv/clips/pending/ to the
    sibling cv/clips/kept/ (created if missing), set status='kept' +
    decided_at. The DB row is never deleted - it's the audit trail. Returns
    False (no-op) if the clip id doesn't exist or the file is already gone.
    """
    clip = get_clip(clip_id, db_path)
    if clip is None:
        return False
    src = clip["path"]
    kept_dir = _kept_dir_for(src)
    os.makedirs(kept_dir, exist_ok=True)
    dest = os.path.join(kept_dir, os.path.basename(src))
    if os.path.isfile(src):
        os.replace(src, dest)
    elif not os.path.isfile(dest):
        print(f"Warning: keep_clip could not find source file {src!r} to move.")
        return False
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE clips SET status = 'kept', path = ?, decided_at = ? WHERE id = ?",
            (dest, now_iso(), clip_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def discard_clip(clip_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Parent pressed "discard": delete the file, set status='discarded' +
    decided_at. The DB row survives as an audit trail.
    """
    clip = get_clip(clip_id, db_path)
    if clip is None:
        return False
    if os.path.isfile(clip["path"]):
        os.remove(clip["path"])
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE clips SET status = 'discarded', decided_at = ? WHERE id = ?",
            (now_iso(), clip_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def sweep_expired_clips(db_path: str = DEFAULT_DB_PATH, now: float = None, timeout_hours: float = CLIP_PENDING_TIMEOUT_HOURS) -> int:
    """CLAUDE.md decision 6's auto-delete: any 'pending' clip whose
    created_at is older than `timeout_hours` gets its file deleted and its
    status set to 'expired' (+ deleted_at) - distinct from 'discarded' (an
    explicit parent decision) per the same-turn Phase 6 kickoff decision.
    The DB row survives, same as discard_clip.

    A plain function, not a background scheduler - Phase 7's FastAPI process
    doesn't exist yet, so nothing owns a recurring schedule. risk_engine.py
    calls this on a coarse interval (see cv/risk_engine.py's main loop) so
    Phase 6 is demoable without Phase 7; Phase 7 will later call it on its
    own schedule too. Safe to call repeatedly/redundantly - clips already
    swept are no longer 'pending' so a second sweep is a no-op for them.

    `now` is an explicit optional parameter (a UNIX timestamp, i.e.
    time.time()) for the same reason AlertManager/HazardMap take an
    explicit `now` - exercisable with synthetic time in a test, defaults to
    the real wall clock.
    """
    if now is None:
        now = time.time()
    cutoff = datetime.fromtimestamp(now - timeout_hours * 3600, tz=timezone.utc)
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM clips WHERE status = 'pending'").fetchall()
    finally:
        conn.close()

    swept = 0
    for row in rows:
        clip = dict(row)
        created_at = datetime.fromisoformat(clip["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at >= cutoff:
            continue
        if os.path.isfile(clip["path"]):
            os.remove(clip["path"])
        conn = get_connection(db_path)
        try:
            conn.execute(
                "UPDATE clips SET status = 'expired', deleted_at = ? WHERE id = ?",
                (now_iso(), clip["id"]),
            )
            conn.commit()
        finally:
            conn.close()
        swept += 1
    return swept
