"""
db.py - SQLite schema + connection handling for GuardianEye's persistence
layer (Phase 6). Data-layer only - no FastAPI, no HTTP, no UI (that's
Phase 7/8, see PHASE_PLAN.md).

Schema shape and the reasoning behind it are pinned down in
docs/decision-log.md, 2026-08-28 "Phase 6 kickoff: schema shape and four
decisions (Shaked)" - read that entry before changing anything here. Two
points worth restating because they're easy to get wrong later:

  - `events.session_local_id` is `AlertEvent.id` from a single
    risk_engine.py run - it resets to 1 every process start (CLAUDE.md
    decision 3: Layer A/B carry no state across sessions), so it is
    informational only and NEVER a key on its own. The real primary key is
    `events.id` (AUTOINCREMENT); `(run_started_at, session_local_id)` is
    unique so a given process's session-local id can be looked back up
    within that same run.
  - There is deliberately no `ended_at`/"closed" column on `events`. The
    events table persists only alert signals that were actually voiced/
    bannered to the parent (CLAUDE.md decision 3's scope, applied narrowly
    on Shaked's instruction - see the decision log entry above). AlertManager's
    "closed" signal (a proximity pair's hysteresis window elapsing unseen) is
    never voiced by design (`audio_for_signal()` returns None for it) and
    `handle_alert_signal()` already returns before it would ever reach
    persistence - so there is structurally nothing to write a close time
    for. `last_seen_at` on the most recent voiced row is the closest
    available "how recent is this" signal, not a precise close time. Do not
    go looking for a real close timestamp; it was deliberately never
    captured.

Threading: every function in this module (and in persistence.py) opens a
short-lived `sqlite3.Connection` per call rather than sharing one across
threads. This is a deliberate, simple default for a project at this scale
and write volume (alert/clip events, not per-frame writes) - see
persistence.py's module docstring for the specific reason this matters
(write_clip's on_complete callback fires on ClipRecorder's background
thread, and sqlite3.Connection objects are not safe to share across
threads by default).
"""

import os
import sqlite3

# Default DB location. Relative to CWD unless absolute, matching this
# repo's existing convention for --clips-dir/--audio-dir in risk_engine.py.
# Gitignored (see .gitignore) - this is runtime data from a real home, not
# source, same reasoning as cv/captures/ and cv/clips/.
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guardianeye.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_started_at    TEXT NOT NULL,
    session_local_id  INTEGER NOT NULL,
    kind              TEXT NOT NULL,
    person_id         INTEGER,
    hazard_id         INTEGER NOT NULL,
    hazard_label      TEXT NOT NULL,
    hazard_bbox       TEXT NOT NULL,
    zone              TEXT NOT NULL,
    peak_zone         TEXT NOT NULL,
    reason            TEXT,
    started_at        TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    clip_triggered    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (run_started_at, session_local_id)
);
CREATE INDEX IF NOT EXISTS idx_events_started_at ON events(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_hazard_id  ON events(hazard_id);

CREATE TABLE IF NOT EXISTS clips (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     INTEGER NOT NULL REFERENCES events(id),
    path         TEXT NOT NULL UNIQUE,
    hazard_label TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    write_status TEXT NOT NULL DEFAULT 'writing',
    frame_count  INTEGER,
    created_at   TEXT NOT NULL,
    decided_at   TEXT,
    deleted_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """A fresh, short-lived connection - never shared across threads or
    reused across calls. See the module docstring for why this is the
    deliberate default here rather than a long-lived shared connection.
    """
    directory = os.path.dirname(os.path.abspath(db_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Idempotent - safe to call at the start of every process (CREATE
    TABLE/INDEX IF NOT EXISTS throughout).
    """
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
