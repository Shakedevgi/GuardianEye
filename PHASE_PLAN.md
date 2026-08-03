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
`[ ]`

### Phase 3 — Custom hazard classes
**Owner:** cv-agent
**Goal:** dataset collection (Roboflow or similar) + fine-tuning for the small
set of classes COCO doesn't cover (small choking-hazard objects, stairs, etc.)
**Done when:** the fine-tuned model reliably detects the new classes on test
footage, without badly regressing the original COCO classes.
`[ ]`

### Phase 4 — Risk engine
**Owner:** cv-agent
**Goal:** Layer A (hazard map) + Layer B (child-proximity scoring) + zone
thresholds (red/orange/yellow), running on top of detections.
**Done when:** moving a test object/person toward a flagged hazard visibly and
correctly escalates the risk level in real time.
`[ ]`

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
