# backend/ — persistence layer (Phase 6)

Data layer only: no FastAPI, no HTTP, no UI (Phase 7/8's job — see
`PHASE_PLAN.md`). This is the function surface Phase 7 wraps in HTTP.

Full schema, decisions, and reasoning: `docs/decision-log.md`, 2026-08-28
"Phase 6 kickoff" entry, and the module docstrings in `backend/db.py` /
`backend/persistence.py`.

## Files

- `backend/db.py` — schema (`SCHEMA_SQL`), `init_db(db_path)`,
  `get_connection(db_path)` (a fresh, short-lived connection per call —
  never share one across threads).
- `backend/persistence.py` — `EventWriter`, clip insert/keep/discard/sweep,
  and the query functions below.
- `backend/guardianeye.db` — the actual SQLite file (gitignored — runtime
  data from a real home, not source).
- `backend/test_persistence.py` — plain-assert test suite (no pytest,
  matches `cv/test_risk_engine.py`'s style). Run: `python
  backend/test_persistence.py`.

## Events

`EventWriter(db_path, run_started_at=None)` — one instance per
`risk_engine.py` process. `.record(signal)` persists a voiced `AlertSignal`
(insert on `"opened"`, update on `"escalated"`) and returns the `events.id`
row. **Only ever call this with a signal `AlertArbiter.offer()` actually
returned** — see decision 3 in the kickoff entry.

```python
get_recent_events(db_path=DEFAULT_DB_PATH, limit=50) -> list[dict]
get_event_for_clip(clip_id, db_path=DEFAULT_DB_PATH) -> dict | None
```

Returned dicts are JSON-ready (`hazard_bbox` already decoded from its
stored JSON string to a plain list, `clip_triggered` a real bool).

Note: there is no "ended_at"/closed timestamp anywhere in this table, by
design — see `db.py`'s module docstring for why.

## Clips

```python
insert_pending_clip(event_db_id, path, hazard_label, db_path=..., created_at=None) -> int  # clips.id
update_clip_write_result(path, success: bool, frame_count, db_path=...) -> None  # matched by path (UNIQUE)
get_clip(clip_id, db_path=...) -> dict | None
get_clips_by_status(status, db_path=...) -> list[dict]  # 'pending' | 'kept' | 'discarded' | 'expired'
keep_clip(clip_id, db_path=...) -> bool   # moves file to cv/clips/kept/, status='kept'
discard_clip(clip_id, db_path=...) -> bool  # deletes file, status='discarded'
sweep_expired_clips(db_path=..., now=None, timeout_hours=CLIP_PENDING_TIMEOUT_HOURS) -> int  # count expired
```

`keep_clip`/`discard_clip`/sweep never delete the DB row — only the file
and the status/`decided_at`/`deleted_at` columns change. `expired`
(auto-timeout) and `discarded` (explicit parent "no") are kept distinct on
purpose.

`sweep_expired_clips` is a plain function, not a scheduler — Phase 7's
FastAPI process doesn't exist yet. `risk_engine.py`'s main loop currently
calls it on a coarse interval (`--sweep-interval`, default 60s) so Phase 6
is demoable without Phase 7. Phase 7 should call it on its own schedule too
(safe to call redundantly).

## Wiring into `cv/risk_engine.py` (already done, Phase 6)

- `EventWriter` constructed once in `main()`, `.record()` called from
  `handle_alert_signal()` only when `voiced is not None`.
- `ClipRecorder(..., on_write_complete=...)` — the callback fires on the
  background write thread; it calls `update_clip_write_result` (matches by
  path, so no id has to cross the thread boundary).
- `ClipRecorder.poll(now)` returns `(path, event)` pairs; the caller inserts
  the pending clip row right after, resolving `event.id` (session-local) to
  its `events.id` via `EventWriter.db_id_for()`.
- `--db-path`, `--disable-persistence`, `--sweep-interval` are new CLI flags
  on `risk_engine.py`.

## Known, deliberate scope limits (Phase 6)

- No `sessions` table — a run's identity lives in `events.run_started_at`
  (fresh wall-clock string per process) plus the in-process
  `EventWriter._session_to_db_id` map, which is exactly as long-lived as it
  needs to be.
- Every function opens its own short-lived `sqlite3.Connection` — no shared
  connection object, anywhere, on purpose (see `persistence.py`'s module
  docstring's threading note).
