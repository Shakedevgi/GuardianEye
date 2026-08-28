# backend/ — persistence layer (Phase 6) + HTTP serving layer (Phase 7)

Phase 6 (`db.py`/`persistence.py`) is the data layer: no FastAPI, no HTTP, no
UI. Phase 7 (`server.py`) wraps that data layer, plus cv-agent's live camera
loop, in HTTP — this is the function/route surface ui-agent (Phase 8) is
meant to consume. Read this file rather than guessing at shapes; if a shape
here and the actual running server ever disagree, that's a bug to report,
not a cue to reverse-engineer the live response instead.

Full schema, decisions, and reasoning: `docs/decision-log.md`, 2026-08-28
"Phase 6 kickoff" entry, and the module docstrings in `backend/db.py` /
`backend/persistence.py` / `backend/server.py` (the last of these has the
full concurrency-model writeup — read it before touching `server.py`).

## Files

- `backend/db.py` — schema (`SCHEMA_SQL`), `init_db(db_path)`,
  `get_connection(db_path)` (a fresh, short-lived connection per call —
  never share one across threads).
- `backend/persistence.py` — `EventWriter`, clip insert/keep/discard/sweep,
  and the query functions below.
- `backend/server.py` — FastAPI app: `SharedState` (published frame +
  status snapshot), `CommandQueue` (confirm/dismiss/skip routed to the loop
  thread), `create_app()`, `start_server()`. See its module docstring for
  the concurrency model — uvicorn runs on a background daemon thread,
  cv-agent's `main()` keeps the main thread, on purpose.
- `backend/guardianeye.db` — the actual SQLite file (gitignored — runtime
  data from a real home, not source).
- `backend/test_persistence.py` / `backend/test_server.py` — plain-assert
  test suites (no pytest, matches `cv/test_risk_engine.py`'s style). Run:
  `cd backend && ../.venv/bin/python3 test_persistence.py` and
  `... test_server.py`.

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

## HTTP routes (Phase 7, `server.py`)

Default bind is `127.0.0.1:8000` — deliberately not LAN-reachable by
default; see `server.py`'s module docstring for the privacy reasoning.
`create_app(shared, commands, db_path, clips_dir)` builds the app;
`start_server(...)` also runs it on a background daemon thread and starts
the redundant `sweep_expired_clips()` scheduler.

### `GET /video_feed`
MJPEG stream, `multipart/x-mixed-replace; boundary=frame`. Open directly in
a browser `<img>` tag or `<video>`-adjacent element. No JSON, no query
params. Blocks the connection open indefinitely (through camera stalls too)
until the client disconnects or the camera loop calls `mark_stopped()`.

### `GET /risk_status`
Returns cv-agent's published per-frame status dict **verbatim**, plus one
field this layer adds: `frame_seq` (int). Fields below are cv-agent's
contract, not this layer's — treat their shape as given and do not expect
this doc to be the source of truth for them if `cv/risk_engine.py` changes
first (flag that as a contract break, don't silently adapt to it):

```json
{"timestamp": "...", "frame_seq": 4821,
 "camera": {"index": 0, "name": "...", "width": 1920, "height": 1080},
 "risk": {"zone": "orange", "value": 0.24, "person_id": 3, "hazard_id": 7,
          "hazard_label": "object", "hazard_bbox": [120.0, 300.0, 260.0, 420.0]},
 "persons": [{"id": 3, "bbox": [...], "nearest_hazard_id": 7, "distance": 0.24, "zone": "orange"}],
 "hazards": [{"id": 7, "label": "object", "bbox": [...], "state": "pending",
              "origin": "scan", "is_first_scan": false, "alerts_on_approach": true}],
 "hazard_counts": {"total": 9, "confirmed": 2, "pending_first_scan": 4, "pending_new": 1, "dismissed": 2},
 "review_queue": {"length": 3, "current_id": 7},
 "alert": {"text": "RED - person #3 near object", "active": true},
 "diagnostics": {"fps": 14.8, "model": "yolo26l.pt", "imgsz": 640, "conf": 0.35,
                  "device": "mps", "scan_enabled": true, "socket_detect_enabled": true,
                  "persistence_enabled": true}}
```

`503 {"detail": "no frame published yet"}` before the camera loop has
published its first frame (e.g. server started before the camera loop, or
in front of `start_server()` in a test). ui-agent should treat 503 here as
"still booting," not an error state to alarm on.

### `GET /health`
`{"camera_running": bool, "last_frame_age_seconds": float|null, "frame_seq": int}`.
`camera_running` is false both before the first publish and after the
camera loop calls `mark_stopped()`. Exists specifically so a UI can tell "a
dark room" (camera_running true, black frame) apart from "the loop died"
(camera_running false) — poll this, don't infer liveness from `/video_feed`
alone.

### `GET /events?limit=50`
`{"events": [...]}`, thin wrapper over `persistence.get_recent_events()`.
`limit` clamped to `[1, 500]`. Two things this endpoint deliberately does
**not** paper over, worth knowing before building a timeline UI on top of
it:
- No `ended_at` — never synthesized here. `last_seen_at` on the row is the
  closest available "how recent" signal, not a precise close time (see
  `db.py`'s module docstring).
- `PersonTracker` ID churn (real, confirmed in Phase 6) means one
  continuous approach by the same child can appear as several rows with
  different `person_id`s. Not a Phase 7 bug; a UI grouping "events" visually
  should expect this.

### `GET /clips?status=pending`
`{"clips": [...]}`. `status` must be one of `pending` / `kept` / `discarded`
/ `expired`; anything else → `400`.

### `GET /clips/{clip_id}`
`{"clip": {...}, "event": {...}|null}` — `event` via `get_event_for_clip`.
`404` if `clip_id` doesn't exist.

### `POST /clips/{clip_id}/keep` / `POST /clips/{clip_id}/discard`
No body. Returns the updated clip dict (`200`), or `404` if the clip id
doesn't exist or is already gone.

### `GET /clips/{clip_id}/video`
`video/mp4` file response. `404` if the clip id is unknown OR if the row
exists but its file is gone (normal for a discarded/expired clip — the row
is an audit trail, not a promise the file still exists). `403` if the
resolved path somehow falls outside the clips directory tree (the client
never supplies a path, only an id — this guard is defense against a
corrupted row, not a client-controlled input).

### `GET /review`
`{"length": int, "current_id": int|null, "current": {...}|null, "queue": [ids]}`,
built from the latest published `/risk_status` snapshot's `review_queue` +
`hazards` fields — **not** live state. `queue` is `ReviewQueue.ids()` passed
through verbatim: real FIFO membership, in decision order.

Do **not** re-derive `queue` client-side as "every hazard whose `state` is
`pending`" — that is a different set, and it was briefly wrong here for
exactly that reason. After a skip (`POST /review/skip` or the `s` key) the
queue is empty while all those entries are still `PENDING`, so the derived
version reported `length: 0` beside a non-empty `queue` in one payload and
would offer a parent items that confirm/dismiss refuse. Ordering differs too
(the queue is FIFO; `hazards` is id-ordered), so a derived "next up" names
the wrong entry.

Only `current_id` is decidable: `confirm`/`dismiss` fail closed with
`{"ok": false, "reason": "not the current review candidate"}` for any other
id, because the queue is strictly FIFO by design (CLAUDE.md decisions 3/4).

`503` before the first frame is published, same as `/risk_status`.

### `POST /review/{entry_id}/confirm`, `POST /review/{entry_id}/dismiss`, `POST /review/skip`
No body (`skip` also ignores any path param — there is none). Submits a
command to the camera loop thread via `CommandQueue` and blocks (~1s
default) for it to be drained once-per-frame. Returns the handler's result
dict on success. `503` with a `"frame loop did not respond..."` detail
message if the loop doesn't drain it in time — this is a genuine "loop
thread is wedged or has exited" signal, not a soft/retryable error to
silently ignore in the UI.
