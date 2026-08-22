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

`[~]` — Layers A and B, proximity scoring, zone thresholds and the
scan/confirm loop are built and live-verified across five recorded sessions.
The 2026-08-13 reset (simplifying five models to two, and making "unreviewed
means dangerous" real) is in progress.

### Phase 5 — Alerts + event capture
**Owner:** cv-agent (buffer/trigger logic) + backend-agent (clip storage hookup)
**Goal:** visual alerts, pre-recorded voice-clip playback, rolling video buffer
that saves 5–7s clips on critical alerts only.
**Done when:** a simulated critical event produces a correct saved clip file
and the right alert fires — visually and audibly.
`[ ]`

### Phase 6 — Persistence
**Owner:** backend-agent
**Goal:** SQLite schema, event log writes/reads, clip file management incl.
auto-delete of undecided pending clips.
**Done when:** events and clip references are correctly stored, queryable, and
old undecided clips actually get cleaned up.
`[ ]`

### Phase 7 — Serving layer
**Owner:** backend-agent
**Goal:** FastAPI exposing the MJPEG video stream + `/events`, `/risk_status`,
`/clips` (with keep/discard actions).
**Done when:** you can open the stream URL directly in a browser and see live
annotated video, and hit the API endpoints and get correct JSON.
`[ ]`

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
