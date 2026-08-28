# Phase 7 — Serving layer

**Status: CLOSED (2026-08-28).** Both halves of the phase's own done-when
bar have real evidence behind them — one live-hardware, one direct-to-socket
— and the one Phase 6 debt this phase carried forward (`main()`'s alert
wiring having no unit coverage) is now closed with a test that actually
exercises the invariant, not a restatement of it. Two gaps are worth reading
precisely rather than rounding up: a real defect in `/review`'s first draft
that was found and fixed correctly but under-graded by the agent that found
it, and one pre-existing flaky test, disclosed rather than hidden. A third
thing this task asked me to treat as still-open — the "unreviewed hazard
escalating to RED" safety path — is actually **not** still open; the record
already closed it at Phase 4, and I found and fixed a stale cross-reference
that made it look otherwise. Details below.

## How this write-up was produced

I have Read/Grep/Glob/Write/Edit only, no Bash. Everything with a specific
number attached to it in this document — 103/26/16 test counts, commit
hashes, `run_started_at` timestamps, DB row dumps, the flaky-test disclosure,
the browser/keyboard live sessions — was run by the orchestrator this
session, not by me, and I'm treating it the way Phase 6's write-up treated
the orchestrator's live-camera runs: real evidence, not taken on faith
either. What I *did* do myself, directly against the files on disk rather
than against anyone's summary of them:

- Read `backend/server.py`, `backend/API.md`, `backend/requirements.txt`,
  `backend/test_server.py`, `cv/risk_engine.py`'s Phase 7 sections
  (`AlertDispatcher`, `build_risk_status`, `hazard_counts`,
  `ReviewQueue.ids()`, `main()`'s `--serve` wiring), and
  `cv/test_risk_engine.py`'s Phase 7 test block, line by line.
- Traced the exact frame-processing order in `main()`'s loop (lines
  ~2495–2641) to confirm the "diagnostics can't reach `/video_feed`" claim
  structurally, not just by reading the comment that asserts it.
- Went one level deeper than the code comments for the concurrency claim: I
  read the actual pinned `starlette==1.6.0` source in this repo's `.venv`
  (`starlette/responses.py`'s `StreamingResponse.__init__` and
  `starlette/concurrency.py`'s `iterate_in_threadpool`) to confirm, from the
  library itself rather than from a comment asserting it, that a sync
  generator passed to `StreamingResponse` really is iterated via
  `anyio.to_thread.run_sync` — i.e. one thread-pool call per frame, not on
  the event loop. This is the single most load-bearing correctness claim in
  the whole phase and it checks out against the actual installed dependency,
  not just against `server.py`'s docstring describing it.
- Read `CLAUDE.md`'s decision 1 to confirm the illustrative route list was
  actually amended to include `/review`/`/health`, not just logged as a
  proposal that never landed.
- Cross-checked `docs/phase-writeups/phase-4.md`'s own Addendum and the
  2026-08-22 decision-log entry against `PHASE_PLAN.md`'s Phase 4 status
  block, which is what surfaced the stale-cross-reference finding below.

Where I'm relying on something I did not independently re-run — the live
browser/keyboard session, the two `AlertDispatcher` live-re-verification
sessions, the real-TCP-socket preflight harness, the exact test pass counts
— I say so explicitly rather than presenting it with the same weight as code
I actually read.

## What this phase was building

`PHASE_PLAN.md`'s Phase 7 section, as amended 2026-08-28 at kickoff: FastAPI
exposing `/video_feed` plus `/events`, `/risk_status`, `/clips` (original
goal line), expanded at kickoff (Shaked approved) to also include
`/review`+confirm/dismiss/skip, `/clips/{id}/video` serving actual bytes, and
`/health`. Owner: backend-agent, working in parallel with cv-agent for the
first time in this project (see the workflow-notes entry below). Done when:
"you can open the stream URL directly in a browser and see live annotated
video, and hit the API endpoints and get correct JSON."

The phase's own framing, stated in the kickoff decision-log entry and worth
repeating because it's the right one: **the real work here is the
concurrency model, not the routes.** `cv/risk_engine.py`'s `main()` was
already a single-threaded blocking loop owning the camera, both YOLO models,
Layer A/B state, and a `cv2.imshow` debug window before Phase 7 touched it.
Making that serve HTTP concurrently, without breaking any of it, is the
actual design problem — the routes themselves are, by comparison, thin
wrappers once that's solved.

## Claim 1 — the phase's own done-when bar

**"Open the stream URL directly in a browser and see live annotated video."**
Live session 1 (Shaked, against `b768510`): opened `http://127.0.0.1:8000/video_feed`
in a browser, confirmed verbatim "the stream look good / the debug window
works / and the h/n/s behaves well." I read this at face value the way Phase
4 and 5's write-ups read a first-person confirmation of a sensory claim
("Shaked confirmed hearing it") — it's not something I can independently
re-derive from the repo, but it's a specific, first-person report against a
specific commit, not a vague "it worked." I did independently confirm the
thing that report is *about* is real: `/video_feed`'s route exists, wires to
`_video_feed_generator`, and (per the Starlette-source check above) really
does run outside the event loop.

**"Hit the API endpoints and get correct JSON."** This is where the evidence
is stronger than a single live report, because it comes from two
independent directions that don't share a failure mode:

1. **26 `test_server.py` tests, run by the orchestrator this session,
   26/26 passing**, covering every route `create_app()` wires up — I read
   each test and confirmed it exercises real behavior (real `TestClient`
   against a real FastAPI app, real temp SQLite DB, real files for the clip
   traversal-guard tests) rather than asserting against a mock.
2. **A real-TCP-socket preflight the orchestrator ran separately and
   disclosed as necessary precisely because `TestClient` doesn't prove
   this**: `TestClient` runs the ASGI app through an in-process transport,
   never a real socket and never real `uvicorn`, so it cannot demonstrate
   the actual claim this phase's concurrency model rests on — that
   `/risk_status` still answers in under a second while `/video_feed` is
   actively streaming to another client. The orchestrator's throwaway
   harness (`scratchpad/preflight.py`, not committed, real `uvicorn` on a
   background thread, a synthetic camera loop on the main thread) reported
   17/17 checks green, including that specific concurrent-request check.
   I did not run this myself and can't verify it beyond noting: (a) it's
   the correct additional check given what I confirmed above about
   `TestClient`'s limits, and (b) its non-committed nature means a future
   reader of only the repo can't re-run it, which is the same category of
   evidentiary gap Phase 3's write-up flagged for scratchpad-only
   measurements — worth a committed, even minimal, version of this check
   existing somewhere before the next time someone touches the concurrency
   model and needs to re-prove it didn't regress.

**Verdict on claim 1: met, on real evidence for both halves**, with the one
named caveat that the concurrency proof specifically (not the routes'
correctness) lives in an uncommitted script rather than the repository.

## Claim 2 — the concurrency model

I read `SharedState`, `CommandQueue`, and `start_server` in full against
what the kickoff decision-log entry and `server.py`'s own module docstring
claim, and every specific mechanism claimed is actually there:

- **One writer to Layer A/B state, enforced structurally, not by
  convention.** HTTP handlers never call `hazard_map`/`review_queue`
  methods. Reads go through `SharedState.snapshot()`/`wait_for_frame()`
  (references grabbed under a lock, released immediately, serialized
  outside it — I confirmed `snapshot()` releases the lock before computing
  `age` from a second monotonic read, and before any JSON encoding
  happens). Writes go through `CommandQueue`, drained once per frame by
  `main()`'s own loop (`commands.drain({...})`, line 2665), dispatching to
  the *exact same* `handle_confirm`/`handle_dismiss`/`handle_skip` closures
  the `h`/`n`/`s` keys already call — I checked this is literally the same
  function reference, not a parallel reimplementation that could drift.
- **The reason the queue exists at all, not just that it does, checks out.**
  `handle_dismiss`'s own docstring (which I read directly, not just the
  module docstring's restatement of it) states `hazard_map.dismiss()` needs
  `last_valid_frame` — the *current* camera frame — to compute the
  dismissal fingerprint CLAUDE.md decision 4's "spot changed since
  dismissal, re-raise" rule depends on. Only the loop thread has a current
  frame. This is the same fingerprint bug Phase 5 spent two failed
  debugging rounds on (per that phase's own write-up), so routing every
  write through one queue that only the loop thread drains isn't a generic
  "thread safety is good" instinct — it's specifically protecting a rule
  this project has already paid once to get right.
- **`SharedState` never holds its lock across serialization.** Confirmed by
  reading `snapshot()` and `wait_for_frame()` directly: both acquire
  `self._cond`, copy out references, and release before the caller does
  anything with them (JSON encoding in `/risk_status`, multipart framing in
  `_mjpeg_part`). A slow HTTP response genuinely cannot block the loop
  thread's next `publish()` call.
- **`start_server` neutralises uvicorn's signal handlers, and the code
  comment explaining why is correct.** `server.install_signal_handlers =
  lambda: None` sits directly above the daemon-thread `start()` call, with
  a comment stating `signal.signal()` can only be called from the main
  thread in Python and would raise `ValueError` otherwise. That's correct
  Python behavior, not an assumption — `signal.signal` raising outside the
  main thread is a documented CPython constraint, not something specific to
  this project that needs separate verification.
- **The main-thread-vs-background-thread choice was measured, and the
  measurement is reproducible from the docstring's own claim.** `cv2.imshow`
  throwing `cv2.error: Unknown C++ exception from OpenCV code` off the main
  thread on macOS (Cocoa's UI-thread requirement) is stated as measured
  fact, not assumption, in both the module docstring and the kickoff
  decision-log entry. I can't independently reproduce a GUI crash from a
  read-only audit, but the claim is specific (an exact exception string, a
  named cause) rather than a vague "it probably wouldn't work" — the kind of
  claim that's cheap to falsify if wrong, which is a meaningfully different
  evidentiary posture than an unfalsifiable assertion.

**Verdict on claim 2: the concurrency model is implemented exactly as
documented**, and unusually for this project, one of its core claims (sync
generator → threadpool iteration) was independently checkable against the
actual pinned library source rather than only against this project's own
code — and it held up.

## Claim 3 — CLAUDE.md decision 1, "diagnostics are JSON, boxes are pixels,"
now structurally enforced

I traced the exact ordering in `main()`'s frame loop rather than trusting
the comment that describes it:

1. Boxes are drawn onto `annotated` (hazard boxes, person boxes) —
   `cv/risk_engine.py:2511-2514`.
2. `encoded = rolling_buffer.append(annotated, now)` — line 2523. This is
   where the JPEG bytes actually get produced (`cv2.imencode`, inside
   `rolling_buffer.append`), on the pixel state as it exists at this exact
   line.
3. `if args.serve and encoded is not None: ... shared.publish(encoded.tobytes(), status)`
   — lines 2549–2576. **This is the same `encoded` object from step 2, not
   a re-encode.** No JPEG is produced a second time for `/video_feed`.
4. *Only after* `shared.publish()` has already been called does the file
   draw the connector line, FPS, model/imgsz/conf/device, the risk readout,
   and the alert banner onto `annotated` (lines 2605–2639), immediately
   before `cv2.imshow`.

The load-bearing fact is step 2 vs step 4's ordering: `encoded` is a
`numpy`/`bytes` value already produced from `cv2.imencode`, not a reference
into `annotated`'s live pixel buffer. Continuing to draw onto `annotated`
after `encoded` exists cannot retroactively change what's already in
`encoded`. This means the claim in the kickoff decision-log entry — "a
future diagnostic overlay added below this line physically cannot leak into
`/video_feed` without also corrupting saved clips" — is literally true by
construction, not aspirational: `rolling_buffer.append()` is called on the
same `annotated` object at the same point `encoded` is captured, so anyone
who moved a diagnostic draw call above line 2523 to "fix" something would
simultaneously corrupt every saved clip, which is a loud, immediate failure
mode rather than a silent one. This is a genuinely good piece of engineering
— it converts a discipline requirement (don't draw diagnostics before the
capture point) into a structural one (you can't, without breaking something
else that would be instantly obvious) — and I want to be precise that I
verified it by reading the actual line ordering and reasoning about what
`cv2.imencode`/`numpy` semantics guarantee, not by taking the comment's word
for it.

One adjacent detail worth naming, not a gap: `smoothed_fps` in the published
`/risk_status` is explicitly one frame stale (the FPS-smoothing update
happens after the publish point, in the debug-only section) — the code
comment says this plainly and calls it harmless given it's already an EMA
over many frames. I agree it's harmless; flagging only because it's a small,
honest instance of "the JSON isn't perfectly synchronous with the frame it's
attached to," worth knowing if a future debugging session ever sees
`/risk_status`'s FPS lag the on-screen FPS by exactly one frame and wonders
if something's wrong.

## Claim 4 — the `/review` bug: a real defect, correctly found, mis-graded

This is worth walking through in detail because it's the most instructive
single finding in the phase, on both the code and the process axis.

**The code finding.** An early version of `/review` derived its `queue`
field as "every hazard whose `state == 'pending'`" rather than publishing
`ReviewQueue`'s actual FIFO membership. I read `ReviewQueue` directly
(`cv/risk_engine.py:987-1035`) to confirm the two sets genuinely diverge,
not just in theory: `skip_remaining()` clears `self._ids` (the actual queue)
but does **not** touch any `HazardEntry.state` — a skipped entry stays
`PENDING` forever, just no longer queued for review. So immediately after a
skip (the `s` key, or `POST /review/skip`), the derived "pending state"
version would report a non-empty `queue` — every hazard that's still
`PENDING` — beside `length: 0`, in the same payload, which is internally
contradictory on its face. Worse, per `server.py`'s own comment on
`handle_confirm`/`handle_dismiss`: those handlers fail closed with `{"ok":
false, "reason": "not the current review candidate"}` for anything that
isn't `review_queue.current_id()`. So a Phase 8 UI rendering the derived
`queue` would show a parent items that every confirm/dismiss action would
silently refuse — not a cosmetic display bug, a UI that lies about what's
actionable. Ordering was also wrong for the same underlying reason: the real
queue is strictly FIFO, `hazards` is id-ordered, so a derived "next up" could
name the wrong entry.

**The fix.** `ReviewQueue.ids()` (new method, returns the deque as a plain
list) is now published end-to-end: `build_risk_status()`'s
`review_queue.ids()` → `SharedState`'s status dict → `/review`'s `queue`
field, passed through verbatim, not re-derived. I confirmed both regression
tests exist and actually pin the failure mode, not just the happy path:
`backend/test_server.py`'s `test_review_queue_is_passed_through_not_derived_from_hazards`
constructs exactly the post-skip state (`review_length=0`,
`review_queue_ids=[]`, but the underlying hazard still `pending`) and
asserts the empty queue survives; `cv/test_risk_engine.py`'s
`ReviewQueue.ids()` docstring states the same reasoning cv-side. This is a
real fix, not a patch that happens to make one test pass.

**The process finding, which matters as much as the code one.** Per the
task briefing, backend-agent found this itself and reported it as a
*documentation caveat* — something to note in `API.md`, not something to
fix as a defect. It was actually the latter: a real behavioral divergence
between what the endpoint would return and what the underlying state
machine actually allows, with a concrete, non-hypothetical failure mode
(Phase 8 offering unactionable items to a parent). I read this the same way
`docs/agent-workflow-notes.md` reads its other findings — not as evidence
backend-agent did sloppy work (it found the actual divergence, unprompted,
which is the hard part), but as a distinct failure mode from anything
logged so far in this project's own workflow-notes record: **correctly
identifying a real defect and then mis-classifying its severity** is
different from either "missed the bug entirely" or "found and fixed it
correctly." It's a calibration problem, not a diligence problem, and it's
worth the team watching for specifically because it's the kind of gap that
survives a superficial "did they find the issues" review — the issue
*was* found, just filed under the wrong heading, where it could plausibly
have shipped unfixed if nobody re-read the reasoning behind the caveat
rather than just noting a caveat existed.

## Claim 5 — `AlertDispatcher`: refactor with no behaviour change, closing
Phase 6's gap #4

This closes the most consequential named gap carried forward from Phase 6's
write-up (gap #4: "`main()`'s alert wiring has no test coverage, and the
specific bug this phase found and fixed lived exactly there"). I read the
extraction against both stated claims:

**Is it really behaviour-preserving?** I compared `AlertDispatcher.speak()`/
`.offer()`/`.poll()`/`.banner()` against what the kickoff/follow-up decision
log entries describe as the pre-extraction closures, and the two subtleties
the task specifically asked me to check both hold:

1. **Banner expiry uses a fresh `time.monotonic()` read, not the `now`
   passed into `offer()`/`poll()`.** Confirmed directly: `speak()` sets
   `self.banner_until = self._clock() + self._banner_seconds`, where
   `self._clock` defaults to `time.monotonic` — a separate read from
   whatever `now` the caller passed to `offer(signal, now)`. This matches
   what the entry claims the old `raise_alert()` did, and matters because
   getting it wrong (reusing the arbiter's `now`) would be a silent, subtle
   change to how long the banner stays up relative to when the underlying
   signal was actually scored.
2. **`if args.persistence_enabled` became `if self._event_recorder is not
   None`.** I confirmed these are equivalent *given* how `main()`
   constructs `event_writer` — `event_writer = EventWriter(args.db_path) if
   args.persistence_enabled else None` (line 2282) — so `event_writer` was
   already `EventWriter`-or-`None` from the same flag before it's ever
   passed to `AlertDispatcher`. The dispatcher checking the recorder
   directly rather than a second flag is a real simplification (one source
   of truth instead of two conditions that could theoretically drift), not
   a hidden behavior change, because there was never a way for the two to
   have disagreed even before this change — `args.persistence_enabled` and
   "`event_writer` is not `None`" were already the same boolean by
   construction.

**Does the new test actually test the invariant, or duplicate the
wiring?** This was my own explicit objection at Phase 6 close, and the
task specifically asked me to check whether it was actually answered rather
than assumed answered. I read
`test_alert_dispatcher_every_voiced_signal_is_also_recorded_including_held_release`
in full. It constructs a *real* `AlertArbiter` (not a stub), drives it
through the exact failure shape — voice one signal, offer a second inside
the 2.5s pacing window so the arbiter genuinely holds it, confirm nothing
was voiced or recorded yet, then call `poll()` after the window reopens and
assert the held signal is released, voiced, and recorded, in that order —
and it observes the real code path by wrapping the *instance's* `speak`
method (`dispatcher.speak = spy_speak`) rather than subclassing or
reimplementing anything. Crucially, it never imports or re-types any of
`main()`'s call sequence — it calls `dispatcher.offer()`/`.poll()` directly,
the same two methods any real caller (currently `main()`, in principle a
future Phase 8 trigger) would call. This is a genuine invariant test: it
would catch a regression where some *future* third call site into `speak()`
forgot to route through the dispatcher, which is exactly the class of bug
Phase 6's own reasoning ("a test mirroring `main()`'s wiring would duplicate
it, not test it") correctly said a naive test wouldn't catch. My Phase 6
objection is answered, not just addressed.

**Was it re-verified live, and does the record show it honestly?** Yes, and
the record is explicit about the gap between "tests pass" and "verified on
camera" the whole way through — worth naming because this project has a
specific, earned reason to be paranoid about exactly this gap: the Phase 6
bug this refactor touches passed 93/93 the entire time it was silently
dropping half a session's alerts. The 2026-08-28 "Phase 7 follow-up" entry
states plainly that the concurrency live test ran against `b768510`, *before*
this refactor, so 103/26/16 green tests were not sufficient on their own.
The "AlertDispatcher refactor live-re-verified" entry then diffs 6 real
`ALERT:` terminal lines against 6 DB rows for `run_started_at
=2026-08-28T11:07:33.821289+00:00`, id 27-32, in order, kind, hazard/person
id, and zone — the same method that caught the original bug, applied again
specifically because a clean suite didn't prove anything the first time. I
read the row dump given to me and it lines up exactly as claimed against
the log lines described. Two `session_local_id` gaps (5, 6) are explained as
signals opened but held-then-superseded by the arbiter, never voiced —
consistent with "only what reached `speak()` gets a row," which is the
entire point of the fix. This is real, and it's real specifically because
the team applied its own hard-earned lesson (a green suite is not proof for
this code) rather than assuming it this time.

## Claim 6 — what Phase 7 deliberately did not do

I checked each of these directly rather than trusting the decision log's
list:

- **`/events` does not synthesize `ended_at` from
  `last_seen_at + ALERT_HOLD_SECONDS`.** Confirmed by reading the route
  (`server.py:404-427`) — it's a thin wrapper over
  `get_recent_events()`, no post-processing, and the docstring states the
  reasoning explicitly (synthesizing a value would be the serving layer
  inventing data Phase 6 deliberately never captured). `API.md` repeats the
  same warning at the route level. Good — this is the kind of restraint
  that's easy to quietly violate under pressure to make a UI feel complete,
  and it wasn't violated here.
- **`PersonTracker` ID churn is untouched, and is now visible through
  `/events` to an HTTP client for the first time.** Confirmed the same
  caveat appears in three places (`db.py`, `API.md`, and `server.py`'s
  `/events` docstring) — consistent with how Phase 6's write-up praised this
  same discipline for the `ended_at` gap. Worth restating precisely what
  changes at Phase 7 specifically: this was always true of the *data*, but
  before Phase 7 the only way to see it was reading the DB directly; now any
  HTTP client (Phase 8's Flet app, or anyone else) hits it as a first-class
  API response, which raises the cost of not fixing it before Phase 8
  builds a timeline UI.
- **No authentication, `127.0.0.1` by default.** Confirmed in code
  (`start_server`'s `host: str = "127.0.0.1"` default) and in the
  reasoning (`server.py`'s module docstring, the kickoff entry). **Is the
  Phase 8 consequence adequately flagged where Phase 8 will actually hit
  it?** I think mostly yes: `PHASE_PLAN.md`'s own Phase 7 section states
  outright "Phase 8's done-when requires LAN binding, so Phase 8 must
  decide authentication before it points `--host` at the network," and
  `server.py`'s docstring repeats the same sentence almost verbatim. That's
  two independent places a Phase 8 developer is likely to read (the phase
  plan before starting, the file itself while working) both saying the same
  thing, which is a reasonable redundancy rather than relying on one
  memory. What I'd flag as genuinely unresolved, not unflagged: **flagged
  is not the same as decided.** Nothing in the record commits to *which*
  auth approach Phase 8 will use (the kickoff entry names "LAN with no
  auth" and "LAN plus a shared token" as the two options considered and
  rejected/deferred for Phase 7, but doesn't pre-select between them for
  Phase 8) — which is appropriate for now, but means Phase 8 opens with a
  real design decision still to make, not just a flag to acknowledge.

## Claim 7 — the "unreviewed hazard escalating to RED" path: not actually
still open, and a stale cross-reference found in the process of checking

The task briefing for this review described this as still open — "one
specific, safety-critical path never watched on camera... two live sessions
have now run since that was flagged without closing it" — and asked me to
say whether it should become a precondition for something. Checking it
against the record directly, rather than accepting the framing, turned up
something different: **it was closed at Phase 4, the same day Phase 4
closed (2026-08-22), by a targeted follow-up recording** —
`cv/captures/Screen Recording 2026-08-22 at 13.34.16.mov`, per both the
2026-08-22 "Unreviewed-approach path live-verified" decision-log entry and
`docs/phase-writeups/phase-4.md`'s own "Addendum" section, which I read in
full. The addendum is specific, not hand-wavy: a placed-but-never-confirmed
object correctly fired `RISK: RED person #1 vs object (normalized dist
0.05)` against a hazard map with **0 confirmed entries**, frame-pulled
directly via ffmpeg rather than taken from a summary — exactly the same
evidentiary standard this project applies elsewhere (Phase 4's original
audit re-pulled frames itself; Phase 6's audit distrusted a "landed
everywhere" claim until it grepped for it).

What actually happened is a **stale cross-reference**, not an unresolved
safety gap: `docs/phase-writeups/phase-4.md` was updated the same day with
the addendum recording the closure, but `PHASE_PLAN.md`'s own Phase 4 status
block — the file someone would read first, before opening the linked
write-up — was never edited to match. It still said, until I fixed it this
session, "code-and-test-verified but NOT yet watched happen on camera,"
directly contradicting its own linked write-up. This is the same shape of
finding as Phase 6's audit catching `backend/API.md`'s stale
2026-08-26 date after a "landed everywhere" claim — a correction that
happened in one place and was asserted, not verified, to have happened
everywhere. I fixed `PHASE_PLAN.md`'s Phase 4 bullet in place this session
(a mechanical correction, not a judgment call — the underlying fact was
already settled and documented elsewhere) rather than just reporting it,
consistent with how Phase 6's equivalent finding was handled.

**Practical consequence:** there is no live-safety-path precondition
outstanding for Phase 8 or Phase 9 stemming from this specific item. The
task's premise here should be treated as itself an instance of the same
"a claim needs checking, not just repeating" pattern this project has now
hit at Phase 4, Phase 6, and Phase 7's own review — worth noting for the
workflow record below, since this is the first time it's been the task
briefing itself, rather than a decision-log entry or a phase write-up, that
carried a stale claim forward.

## Test evidence, and the one disclosed flaky test

- `cd cv && ../.venv/bin/python3 test_risk_engine.py` → **103 passed** (93 at
  Phase 6 close, +10: 5 `AlertDispatcher` tests plus whatever else landed in
  the same window — I did not separately audit every non-Phase-7 test that
  moved the count, since the task scoped this review to Phase 7's own
  claims).
- `cd backend && ../.venv/bin/python3 test_server.py` → **26 passed**, new
  file this phase. I read all 26 and traced several by hand above
  (the `/review` regression test, the traversal-guard tests, the
  command-timeout test) rather than just counting them.
- `cd backend && ../.venv/bin/python3 test_persistence.py` → **16 passed**,
  unchanged from Phase 6 — expected, since this phase didn't touch
  `persistence.py`'s logic, only wrapped it in HTTP.

**One flaky test, disclosed rather than found by me and worth taking
seriously precisely because it was disclosed rather than hidden**: a
`write_clip`/`ClipRecorder` test (most likely one of
`test_write_clip_produces_a_real_playable_file` and its siblings in
`cv/test_risk_engine.py`, which do real file I/O against a
`tempfile.TemporaryDirectory` and a background write thread — I did not
independently reproduce the failure, so I'm not asserting which exact test
it was) failed once with `OSError: [Errno 66] Directory not empty` on one
run, then passed three consecutive re-runs. This has the shape of a real
race between `ClipRecorder`'s background write thread finishing a file
write and `tempfile.TemporaryDirectory`'s teardown racing to delete the
directory before that write is fully flushed — plausible, not confirmed. I
checked the one thing that matters for scoping this correctly: **Phase 7
touches zero lines of `ClipRecorder`/`write_clip`/`RollingBuffer`** — I
grepped for `AlertDispatcher`'s own docstring, which explicitly states clip
triggering "stays in `main()`," and nothing in `server.py` or the
`build_risk_status`/`hazard_counts`/`ReviewQueue.ids()` code I read touches
those classes either. This is a pre-existing test-infrastructure flake, not
a Phase 7 regression. It is **not currently recorded anywhere in the
decision log** — I agree with the suggestion that it should be, on the same
"log the finding even if it's small and not blocking" standard this project
applies to everything else (the API.md stale date, the ReviewQueue bug),
and I'm recording it here as a named gap below rather than letting it live
only in this write-up's prose.

## Named gaps, in priority order

1. **The one committed test suite cannot prove the phase's core concurrency
   claim; the test that can is uncommitted.** `test_server.py`'s 26 tests
   all run through Starlette's `TestClient`, which never touches real
   `uvicorn` or a real socket. The specific claim this phase's whole
   thread-model rests on — `/risk_status` still answers promptly while
   `/video_feed` is actively streaming — was checked by a real-`uvicorn`
   harness (`scratchpad/preflight.py`) that exists outside the repo. This is
   the same category of gap Phase 3's write-up flagged for
   scratchpad-only evidence: real, credible, reported honestly, but not
   independently re-runnable by a future reader of only this repository. A
   minimal, committed version of this check (even a slow, marked-skip-by-
   default integration test that spins up real `uvicorn` on a random port)
   would close this permanently rather than requiring it be re-proven by
   hand the next time the concurrency model is touched.
2. **The flaky `ClipRecorder`/write_clip test is real, disclosed, scoped
   correctly as pre-existing — and not yet in the decision log.** Low
   urgency (Phase 7 didn't cause it, doesn't touch the code), but this
   project's own standing rule is that findings get logged in the turn
   they're found, and this one hasn't been yet. Worth a one-line entry
   rather than living only in this write-up.
3. **`/review`'s original bug was correctly found but mis-graded as a
   documentation caveat rather than a defect** (Claim 4 above). Already
   fixed and tested; named here as a process finding for the team to watch
   for in future phases (calibrating severity, not just finding issues), not
   as an outstanding code gap.
4. **Phase 8 has "decide authentication before LAN binding" flagged in two
   places, but no candidate approach pre-selected.** Not a Phase 7 gap
   (deliberately out of scope, per Shaked's own decision), but worth
   surfacing here as a concrete thing Phase 8 needs to actually resolve,
   not just acknowledge.
5. **The stale `PHASE_PLAN.md` cross-reference on the unreviewed-hazard path
   (Claim 7) is fixed as of this session** — named here only so the fix is
   traceable to this review, matching how the Phase 6 API.md date fix was
   recorded in that phase's own write-up.

None of these five block closing the phase. #1 and #2 are the two I'd want
addressed soonest, in that order, before the concurrency model or the clip
pipeline get touched again by someone who doesn't have this session's
context.

## Verdict: is Phase 7 closeable?

**Yes.** Reading `PHASE_PLAN.md`'s own three-part gate:

- **(1) demoably works** — met on two independent kinds of evidence: a real
  browser/keyboard session against real hardware (`b768510`), and a
  real-socket concurrency check plus a comprehensive `TestClient` suite
  covering every route (26/26). The one thing not fully demoable *from the
  repo alone* is the concurrency proof specifically (named gap #1) — the
  routes and the state machine are fully demoable and verified.
- **(2) docs-agent reviewed it, code matches claim** — this write-up. Every
  specific mechanism named in the four 2026-08-28 decision-log entries
  (`SharedState`, `CommandQueue`, `build_risk_status`, `hazard_counts`,
  `ReviewQueue.ids()`, `AlertDispatcher`, the capture-point ordering) was
  traced directly against the current code, not summarized from the log —
  including one check (the Starlette `iterate_in_threadpool` behavior) that
  went past this project's own code into the pinned dependency's actual
  source.
- **(3) docs-agent produced the write-up** — this document.

Against Phase 7's own done-when line — "open the stream URL directly in a
browser and see live annotated video, and hit the API endpoints and get
correct JSON" — both halves are met, with the caveats named above attached
precisely rather than smoothed over: the browser half rests on one
first-person live report against one specific commit (consistent with how
this project has always treated a sensory human confirmation); the JSON half
rests on a strong committed test suite plus one uncommitted but credible
concurrency check. Phase 6's own gap #4 (no test coverage for the exact code
that shipped Phase 6's real bug) is closed, and closed correctly — the new
test exercises the actual invariant through real objects, not a
reimplementation of the wiring, which is precisely what Phase 6's write-up
said would be needed and didn't yet exist.

`PHASE_PLAN.md`'s Phase 7 status set to `[x]` in this same turn.

## For the write-up's teaching purpose: check your own understanding

1. `main()` keeps the main thread and `uvicorn` runs on a background daemon
   thread — the reverse of what every FastAPI tutorial shows. Explain, in
   one or two sentences and without re-reading "Claim 2" above, the actual
   measured reason this project needed the reverse arrangement, and name the
   one CLAUDE.md decision this reversal was specifically protecting.
2. `SharedState.publish()` stores a *reference* to the status dict rather
   than copying it, and the code comment calls this deliberate. What
   property of how `build_risk_status()` constructs that dict each frame is
   the reason storing a bare reference is safe here, rather than a bug
   waiting to surface as a "the JSON changed underneath the response I was
   building" race?
3. `hazard_map.dismiss()` needs the *current* camera frame to compute a
   fingerprint. Explain why this one fact is the entire reason
   `CommandQueue` exists at all, rather than HTTP handlers just calling
   `hazard_map.confirm()`/`.dismiss()` directly — and connect it back to a
   specific bug from an earlier phase that this design is protecting
   against a repeat of.
4. `/review`'s original bug derived `queue` as "every hazard whose `state`
   is `pending`." Explain, concretely, the exact moment (which action, in
   what order) that set diverges from the real `ReviewQueue`, and why a
   Phase 8 UI built on the wrong version would have been actively
   misleading to a parent rather than just slightly wrong.
5. This write-up distinguishes "a real defect, found and fixed correctly"
   from "a real defect, found, fixed correctly, but mis-graded in severity
   when first reported." Explain in your own words why that's a
   meaningfully different finding from either "the agent missed a bug" or
   "the agent did its job well" — and why a review process that only checks
   "were the bugs found" would not have caught it.
6. `encoded = rolling_buffer.append(annotated, now)` happens *before* the
   FPS/model/risk-readout/banner overlay is drawn onto `annotated`. Explain
   why this ordering makes CLAUDE.md decision 1's "diagnostics never reach
   the served frame" rule structurally true rather than just a rule someone
   has to remember to follow — and what would have to go wrong in a future
   edit for that guarantee to actually break.
7. If you had to defend to Tom, in one sentence, why this write-up says the
   task's own briefing was wrong about the "unreviewed hazard escalating to
   RED" gap still being open — what's the one-sentence version of how that
   was checked, rather than just re-asserted either way?
