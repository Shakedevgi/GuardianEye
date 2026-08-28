# GuardianEye — Phase Plan

Rule: a phase is complete when (1) it demoably works, (2) docs-agent has
reviewed it and confirmed the code matches the claim, and (3) docs-agent has
produced the phase write-up and walked the team through it. Do not start
phase N+1 before phase N is closed this way.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` complete & docs-agent signed off

---

### Phase 1 — Foundation
**Owner:** orchestrator + cv-agent
**Goal:** dev environment, git repo, camera reliably streaming raw frames via
OpenCV. No AI yet. No risk logic yet.
**Done when:** you can run one script and see a live raw video window from the
USB camera, with no crashes, for several minutes straight.
`[x]`

### Phase 2 — Baseline detection
**Owner:** cv-agent
**Goal:** stock YOLO (COCO classes) running on live frames, drawing boxes for
person / oven / knife / etc. No custom classes, no risk logic yet.
**Done when:** live feed shows correct bounding boxes + labels for whatever
COCO objects are actually in the room, in real time.
`[x]`

### Phase 3 — Custom hazard classes → reframed to "detect objects on
reachable surfaces"
**Owner:** cv-agent
**Goal (original):** dataset collection + fine-tuning for the classes stock
COCO doesn't handle well enough (small choking-hazard objects, stairs, etc.)
**Goal (as it actually evolved — see `docs/phase-writeups/phase-3.md` for the
full account):** the fine-tuning approach was tried three times, measured
properly against a held-out building each time, and did not generalise well
enough to alert on (best result: `sharp_object` recall 0.339 on an unseen
building; pooled recall never exceeded 0.216). Five further methods were
tried for making a model judge an unfamiliar frame's hazards alone (named
classification, colour clustering, texture objectness, a VLM, class-agnostic
segmentation) and all five failed at that job. The phase closed on a
different design instead: setup becomes a guided, parent-confirmed
walkthrough (CLAUDE.md decisions 3 & 4, amended 2026-08-11), and anything
appearing after setup is caught by frame-to-frame change detection
(CLAUDE.md decision 7). **Neither of those two mechanisms exists as
committed code yet** — this phase validated the architecture with
measurement; it did not build the pipeline. That is explicitly Phase 4's
job now, in addition to Phase 4's original scope.

**Step 0 — measure before collecting.** Phase 2 found that input resolution,
not model size, dominates small-object detection (see
`docs/phase-writeups/phase-2.md`). Every "COCO can't see this" observation
made before that discovery was taken at settings that were starving the
model, so the list of classes needing fine-tuning is currently *unknown*.
Re-test with `yolo26x --imgsz 1600` across the real testing areas and write
down which hazard classes stock COCO actually handles and which it doesn't.
That list defines this phase's scope — collecting data before producing it
risks labelling things the stock model could already detect. This also
settles whether CLAUDE.md decision 7 (fine-tune only what COCO lacks) still
holds or needs amending.

**Labelling stays local — no cloud upload of training images.** The earlier
"Roboflow or similar" note is superseded. Training images are photographs of
a real home and, unavoidably in this project, of a child. Uploading them to
a third-party service contradicts the spirit of the privacy requirement in
CLAUDE.md decision 8, even though that decision is written about event logs
and clips rather than training data. Use a locally-run labelling tool
(e.g. Label Studio, self-hosted CVAT); fine-tuning itself already runs
locally on the M5 Pro via MPS, so no stage of this phase needs the cloud.

**Dataset must span the real deployment variation.** There is no single
permanent camera position — the camera moves between testing areas at
varying heights and angles. Data collected from one fixed setup would train
a model that works in that setup and fails elsewhere, so deliberately vary
area, height, angle and lighting. Variety matters more than sheer volume.

**Done when (revised 2026-08-11 by docs-agent, per Shaked's task — see
`docs/phase-writeups/phase-3.md` for the full reasoning; this replaces the
original bar above, which described a deliverable the phase tried three
times and did not reach, and for good, measured reasons):** the fine-tuning
approach has been measured against a genuinely held-out location (not a
same-building split) enough times to know whether it generalises, AND — if
it doesn't generalise well enough to alert on — a replacement architecture
has been decided and written into CLAUDE.md with the measurement evidence
that justifies it. This phase does **not** require a working end-to-end
hazard-detection pipeline; that is Phase 4's job. It requires knowing, with
evidence, what Phase 4 should build.
`[x]` — closed 2026-08-11, second attempt. The scoping/architecture question
was closed by measurement in the first close attempt; that attempt also
found one of its own headline claims (change detection's "~1.4ms/frame, car
key and lighter" result) had zero supporting evidence in the repo. Two
re-measurement rounds since then produced real, committed, spot-checked
evidence for change detection — including a genuine, eyeballed catch of a
placed object on a purpose-shot clip — and an honest, much lower precision
number (~1 in 3) than the original claim implied. Shaked's explicit decision
closes the phase here rather than continuing to iterate on that detector in
isolation (see `docs/decision-log.md`, 2026-08-11, and
`docs/phase-writeups/phase-3.md` for the full reasoning): five rounds of
offline/photo-based fixing (three fine-tune rounds, two change-detection fix
rounds) each traded one metric's improvement for another's regression
without the system ever running as a whole, so Phase 4 now begins with
testing the real pipeline on live footage instead of a sixth isolated fix.
**Read "demoably works" here against the corrected scope this phase settled
on** (a validated direction plus reusable, runnable measurement tooling —
`measure_detection.py`, `evaluate_home_frames.py`, `measure_change_detection.py`,
`measure_segmentation.py`, `measure_openvocab.py` all genuinely run and
produce real numbers), **not against "a parent could use this today."** That
stronger bar was never re-litigated as Phase 3's requirement; it is
explicitly Phase 4's job below. Change detection specifically ships into
Phase 4 as a **known-weak, not a finished, mechanism** — good enough to test
integrated into a live system, not good enough to alert a parent on
unfiltered.

### Phase 4 — Risk engine
**Owner:** cv-agent

**Goal, restated 2026-08-13 after a deliberate scope reset** (see
`docs/decision-log.md`, 2026-08-13, and CLAUDE.md decisions 3/4/7 — this
replaces the earlier framing, which had grown to five simultaneous detection
mechanisms and was debugging its own complexity rather than the product):

> A camera detects objects. A human classifies which are dangerous. The system
> detects new objects appearing after that. It alerts.

Concretely, four things:
1. **Detect** — a periodic class-agnostic scan lists occupied spots on
   reachable surfaces. It proposes; it never decides.
2. **Human classifies** — each proposal is presented one at a time and
   confirmed as a hazard or dismissed. Confirmed hazards seed Layer A.
3. **Detect new** — each scan is compared against what is already known:
   present-and-new means something arrived; known-and-absent means something
   was removed, and it clears.
4. **Alert** — Layer B scores child-to-hazard proximity every frame and
   escalates yellow/orange/red. A newly arrived object alerts on appearance,
   and counts as dangerous until a human says otherwise.

**Done when:** with the system running live, (a) a parent can tap through the
room's objects and have the confirmed ones tracked, (b) an object introduced
afterwards is noticed and alerted on without anyone touching the app, and (c)
moving toward a hazard visibly and correctly escalates the risk level in real
time.

`[x]` — closed 2026-08-22 by docs-agent, after the 2026-08-13 scope reset
(five detection models down to two — per-frame YOLO for person/oven/
microwave/refrigerator, plus a periodic class-agnostic room scan — and one
rule: nothing auto-adds a hazard, every detector proposes and a human
disposes) and a live hardware test the same day the phase closed. Full
reasoning, including exactly what is and isn't proven, in
`docs/phase-writeups/phase-4.md`. Read "done when" here precisely, not as a
blanket pass:
- **(b) new object noticed/alerted without touching the app** and
  **(c) approach to a hazard escalates risk in real time** — both
  live-verified on real hardware (FPS restored 2→15, no freezes, a placed
  object caught by the scan and alerted on its own, approach to a confirmed
  hazard correctly firing RED, removal clearing a hazard's box, and the
  "spot changed since dismissal" re-raise rule firing for real) — checked by
  docs-agent directly against the actual recorded frames, not taken on the
  report alone.
- **(a) "a parent can tap through the room's objects"** is met at the logic
  layer only (the scan/confirm/dismiss/re-raise state machine is real,
  tested, and live-verified) — the *keyboard* h/n/s loop that exercises it
  is a developer stand-in, not a parent-usable interaction. That interaction
  is explicitly Phase 8's job.
- **One specific, safety-critical path is code-and-test-verified but NOT
  yet watched happen on camera**: approaching an *unreviewed* hazard
  (arrived after the first scan, not yet confirmed) escalating to RED —
  CLAUDE.md decision 4's "unreviewed means dangerous" row. Every RED event
  in the live test was against an already-confirmed hazard. Flagged as the
  first thing to check in a future recording, not hidden inside "Phase 4
  works."
- **Two forward pointers, not silently dropped**: debug telemetry (FPS,
  model config, raw distances) is currently drawn into the served frame's
  pixels rather than exposed as `/risk_status` JSON — a concrete note for
  Phase 7's design, proposed in `docs/decision-log.md`'s 2026-08-22 entry,
  awaiting Shaked/Yahli confirmation, not yet a CLAUDE.md change. Overlapping
  hazard-box labels when boxes cluster is cosmetic, filed for Phase 8.

### Phase 5 — Alerts + event capture
**Owner:** cv-agent (buffer/trigger logic) + backend-agent (clip storage hookup)
**Goal:** visual alerts, pre-recorded voice-clip playback, rolling video buffer
that saves 5–7s clips on critical alerts only.
**Done when:** a simulated critical event produces a correct saved clip file
and the right alert fires — visually and audibly.

`[x]` — closed 2026-08-26 by docs-agent. Built entirely inside cv-agent's
scope, as expected this early (no `backend/` directory exists yet; clip
storage hookup/SQLite/auto-delete stay Phase 6's job per this file's own
division of labor). Full reasoning in `docs/phase-writeups/phase-5.md`; read
"done when" here precisely, not as a blanket pass:
- **Visual alerts and saved clips** are both backed by real evidence: two
  live-camera recordings confirmed to exist on disk, a real saved clip file
  at the agreed `cv/clips/pending/<timestamp>_event<id>_<label>.mp4` path,
  and terminal-log text quoted in `docs/decision-log.md` that docs-agent
  cross-checked character-for-character against the actual banner-text
  format strings in `cv/risk_engine.py` — strong evidence the logged
  sessions are genuine runs of this code, not a paraphrase.
- **Audibility — the other half of this phase's own "visually and audibly"
  bar — is NOT confirmed on the record.** `AudioPlayer` is verified, by
  code reading, to launch `afplay` non-blockingly and to fail closed on a
  missing file; nobody has yet confirmed a parent can actually hear the
  result. Flagged as the first thing to check next, the same way Phase 4
  closed with its own one named, safety-relevant gap rather than hiding it.
- **Two fix-then-fail-then-fix-again cycles this session** (the alert
  cooldown, and especially the dismissal-re-raise fix, which failed once
  before the real root cause — comparing against a scan's own jittering
  bbox instead of a stable fingerprint bbox — was found) are read by
  docs-agent as the project's established "measure, don't assume"
  discipline working correctly across iterations, not a process failure —
  see the write-up for the reasoning.
- **A documentation gap, not an engineering one:** `docs/decision-log.md`'s
  own most recent Phase 5 entry ends with the dismissal-re-raise fix "not
  yet live-verified" against the object that had been failing — but no
  sixth decision-log entry exists recording that re-test happening. A
  clip file matching a successful outcome exists on disk, but docs-agent
  had no shell access this session to independently confirm the specific
  frame-count/duration/codec claims made about it. Recorded as an open
  item, not smoothed into "verified."
- **Two smaller forward pointers, not blockers:** `PersonTracker` ID churn
  observed in one live-test log (`PERSON_STALE_SECONDS = 1.0` is a
  candidate but unconfirmed explanation), and the 89-test suite was
  verified by direct code/test reading rather than actually executed this
  session (no Bash tool available) — someone should run
  `cd cv && python test_risk_engine.py` and confirm the pass count before
  fully trusting it.
- **Amendment, 2026-08-28 (docs-agent, at Phase 6 close):** the audibility
  gap named above is now closed. During Phase 6's live test, the
  orchestrator played `cv/audio/hazard_detected.wav` and asked Shaked
  directly; Shaked confirmed hearing it ("yes i did heard it works." —
  `docs/decision-log.md`, 2026-08-28 "Phase 6 live test" entry). This
  closes Phase 5's own done-when bar ("visually **and audibly**") on both
  halves. Phase 5's status and `[x]` above are not rewritten — this is a
  same-day-as-Phase-6 confirmation of a gap Phase 5 closed with, not a
  correction to Phase 5's own history.

### Phase 6 — Persistence
**Owner:** backend-agent
**Goal:** SQLite schema, event log writes/reads, clip file management incl.
auto-delete of undecided pending clips.
**Done when:** events and clip references are correctly stored, queryable, and
old undecided clips actually get cleaned up.

`[x]` — closed 2026-08-28 by docs-agent, after a live hardware test found and
fixed a real bug the same day. Full reasoning in
`docs/phase-writeups/phase-6.md`; read "done when" here precisely, not as a
blanket pass:
- **"Queryable" and "old undecided clips actually get cleaned up"** are both
  backed by tests that do real file I/O (real files created, backdated,
  deleted; real `kept`/`discarded`/`expired` status transitions checked
  against the actual filesystem, not mocked) — traced by docs-agent against
  the implementation, not just counted.
- **"Correctly stored" — the fix is now live-verified, closing this
  phase's one real asterisk.** A live session found that roughly half of
  all alerts actually voiced to the parent were never reaching the
  database — `speak()` had two call sites, and Phase 6 originally
  persisted at only one of them. Fixed by moving the persistence call
  inside `speak()` itself. A second live session (2026-08-28,
  `docs/decision-log.md`'s "Phase 6's top open item closed" entry) confirmed
  it holds: every voiced alert's underlying event is present in the
  database, either as its own row or correctly folded into an update of an
  already-open one (verified via `started_at`/`last_seen_at` per row, not a
  bare count — a bare row-count diff against the log's `ALERT:` lines is
  NOT the right test, since a legitimately re-voiced event collapses
  multiple alert lines into one updated row by design).
- **`PersonTracker` ID churn (a forward pointer since Phase 5) now visibly
  corrupts the persisted event log**: one continuous approach to a hazard
  was recorded as three unrelated event rows in this session's real data,
  because `AlertManager` keys events on `(person_id, hazard_id)` and a
  re-acquired person id starts a new event. Not Phase 6's mechanism to fix,
  but flagged as a precondition worth resolving before Phase 8 builds a
  parent-facing event history on top of this data.
- **No `ended_at` column on `events`, by design** (decision 3: only voiced
  signals are persisted, and `AlertManager`'s "closed" signal is never
  voiced) — well documented in `backend/db.py` and `backend/API.md`, but its
  Phase 8 consequence (a parent can't be told precisely when an event ended,
  only approximately) hasn't been explicitly weighed against what Phase 8
  needs to show.
- **`main()`'s alert wiring — exactly where this phase's bug lived — has no
  unit test coverage**, and the stated reason (a test mirroring `main()`'s
  own call sequence would duplicate the wiring, not test it) is judged
  correct but incomplete: an invariant-level test (every signal that reaches
  `speak()` also reaches `record()`, using the real classes) is possible and
  doesn't exist yet. The proposed fix — extract `main()`'s alert wiring into
  an injectable object in Phase 7 — is a credible plan (Phase 7 needs that
  extraction anyway to drive these paths from FastAPI) rather than empty
  deferral, but it's a plan, not yet a fact.
- **One process finding, not a code finding:** the decision log's own claim
  that a same-day date-typo correction (2026-08-26 → 2026-08-28) "landed
  everywhere" was checked directly and was not quite true — `backend/API.md`
  still had one stale reference, found and fixed during this close-out.

### Phase 7 — Serving layer
**Owner:** backend-agent
**Goal:** FastAPI exposing the MJPEG video stream + `/events`, `/risk_status`,
`/clips` (with keep/discard actions).

**Scope amended 2026-08-28 at kickoff, approved by Shaked (see
`docs/decision-log.md`, "Phase 7 kickoff") — three additions to the goal line
above, which is left in place rather than rewritten:**

1. **Hazard-review endpoints are in scope** (`/review`, plus confirm/dismiss/
   skip). Not in the original goal line, added deliberately: Phase 4's own
   closure named the keyboard `h`/`n`/`s` loop "a developer stand-in...
   explicitly Phase 8's job" to become a real parent interaction — and Phase 8
   cannot build that against an API that doesn't exist. Shipping Phase 7
   without it means Phase 8's first act is reopening Phase 7. The marginal cost
   is near zero, because the command queue these routes use has to exist for
   the concurrency model regardless.
2. **`/clips/{id}/video` serves actual video bytes**, while `/clips` itself
   stays metadata-only JSON. Clips are already encoded `avc1`/H.264
   specifically for this (`cv/risk_engine.py:497` says so explicitly); the
   alternative is Phase 8 reading the local filesystem directly, which would
   break CLAUDE.md decision 1's "Flet is a client of FastAPI."
3. **`/health`** — reports whether the camera loop is alive and how old the
   last published frame is. Cheap, and the only way to tell "the stream is
   black because the room is dark" from "the loop thread died."

**The real work of this phase is the concurrency model, not the routes.**
`main()` is a single-threaded blocking loop that owns the camera, both models,
and a `cv2.imshow` window; ASGI must serve HTTP concurrently with it. Settled
by measurement: the camera loop keeps the **main** thread (because `cv2.imshow`
from a background thread throws on macOS — Cocoa requires the main thread, and
CLAUDE.md decision 1 protects the debug window), and uvicorn runs on a
background daemon thread. Full reasoning in the decision-log entry.

**The server binds `127.0.0.1` by default — a privacy decision, not an
oversight** (Shaked, chosen over LAN-with-no-auth and LAN-plus-token). Note the
consequence for the phase below: **Phase 8's done-when requires LAN binding, so
Phase 8 must decide authentication before it points `--host` at the network.**

**Done when:** you can open the stream URL directly in a browser and see live
annotated video, and hit the API endpoints and get correct JSON.
`[~]` — in progress, opened 2026-08-28.

### Phase 8 — UI
**Owner:** ui-agent
**Goal:** Flet app shell (desktop + web from one codebase), consuming the
FastAPI backend only — video embed, risk banner, event log, settings,
keep/discard clip controls.
**Done when:** the same Flet code runs as a desktop window AND is reachable
from a tablet/phone browser on the same WiFi, both showing live data.
`[ ]`

### Phase 9 — Full integration + hardening
**Owner:** orchestrator (all agents contribute)
**Goal:** everything running together end-to-end; edge cases; demo prep.
**Done when:** full run-through works unattended for an extended session
without manual intervention.
`[ ]`

---

## Known future/optional extensions (not in scope unless explicitly promoted)
- Multi-room tracking
- Cloud architecture / remote access
- Dedicated mobile app
- Automatic new-hazard discovery mid-session beyond current detection loop
