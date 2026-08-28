# GuardianEye — Project Context

This file is read automatically by every Claude Code session and subagent in this
project. It is the single source of truth for architecture decisions. If code and
this file ever disagree, this file wins until it is deliberately updated — and any
update to this file should be logged in `docs/decision-log.md`.

**Nothing in this file is permanent.** These are engineering decisions, not
rules handed down — they were made with the information available at the time,
and measurement can show them to be wrong. When evidence contradicts a decision
here, say so and propose the change; do not quietly work around it, and do not
treat it as settled just because it is written down. What is required is only
this: **ask Shaked and Yahli before changing it, and log the change in
`docs/decision-log.md` in the same turn.** Changing a decision without asking is
the problem. Questioning one is not.

## What this project is

GuardianEye is a home-safety monitoring system for toddlers. A fixed USB camera
watches a room; an AI vision pipeline detects the child and household hazards in
real time; a risk engine tracks how close the child is getting to each hazard;
the system raises visual + pre-recorded voice alerts before an accident happens,
and saves short video clips of critical events for the parent to review.

Team: Shaked Ivgi, Yahli Mazri. Advisor: Tom Cohen.

## Core architecture decisions (changeable — but ask first, never silently)

1. **Video pipeline vs. control UI are decoupled.**
   OpenCV + YOLO run the detection loop and produce annotated frames. FastAPI
   serves those frames as an MJPEG stream (`/video_feed`) plus a small JSON API
   (`/events`, `/risk_status`, `/clips`, `/review`, `/health` — the last two
   added at Phase 7, see `docs/decision-log.md`'s 2026-08-28 "Phase 7 kickoff"
   entry: `/review` gives Phase 8 an API for the hazard confirm/dismiss/skip
   flow Phase 4 always meant to move off the keyboard, `/health` distinguishes
   a dark room from a dead camera loop). Flet is a *client* of FastAPI — it does
   NOT push frames through its own state system. Flet runs as both a Desktop app
   and a local Web App from the same codebase, both talking to the same FastAPI
   backend.

   **What may be drawn into the served frame, and what may not** (clarified
   2026-08-22, approved by Shaked; raised by docs-agent at Phase 4 close):

   - **Hazard and person boxes may be drawn onto the frame.** They are product,
     not diagnostics — a parent should see what the system has flagged.
   - **Diagnostics must NOT be drawn onto the frame.** FPS, model name/`imgsz`/
     `conf`, device, and raw normalized distances are `/risk_status` JSON
     fields. The UI decides whether to render them; a parent view hides them,
     a developer view shows them.

   The reason is that pixels are one-way. Anything painted into the frame is
   permanent by the time Flet receives it, so a decision made in the cv layer
   silently becomes a decision the UI layer cannot undo — which is exactly the
   coupling this decision exists to prevent.

   **This does not constrain the local OpenCV debug window.** `risk_engine.py`'s
   `cv2.imshow` view is a developer tool and should stay as verbose as it is
   useful. The rule applies to what Phase 7 *serves*: build `/video_feed` from
   a clean annotated frame plus `/risk_status`, rather than reusing the debug
   window's image.

2. **Single YOLO pass per frame**, detecting both the child/person class and all
   hazard classes together. There is no separate "person detector" and "hazard
   detector." (Reaffirmed and strengthened 2026-08-13: `oven`/`microwave`/
   `refrigerator` are read out of the *same* per-frame result as `person`, at no
   extra cost, rather than from their own model — see decision 7.)

3. **Two always-on logical layers, not two sequential stages:**
   - **Layer A — hazard map:** continuously updated for as long as the camera is
     on. No persistence between sessions/boots — every session starts with a
     fresh, empty map (camera angle/room/lighting cannot be assumed identical to
     last time).
   - **Layer B — child proximity risk:** runs the moment a child is detected,
     computed against whatever Layer A currently knows.
   - There is no hard "setup phase ends, monitoring phase begins" switch. Both
     layers run from frame 1, even if a child is already in frame at boot.

   **Layer A is maintained by one mechanism: a periodic room scan whose results
   a human judges** (rewritten 2026-08-13 — the scope reset; supersedes the
   2026-08-11 "setup walkthrough" amendment, which made this a setup-only
   event). Every few seconds the system lists the occupied spots on reachable
   surfaces (floor, low table, low shelf). Comparing that list against what it
   already knows yields three facts, with no extra machinery:
   - a spot in the new list that isn't in the old one → **something arrived**
   - a known spot missing from the new list → **something was removed**, clear it
   - everything else → unchanged, stay quiet

   **The scan proposes; a human decides.** The detector does not need to be
   right about what an object is, or whether it is dangerous. It only needs to
   flag "this spot is not empty." Over-flagging is acceptable by design — it
   costs one tap. This is the same review hand-off decision 4 has specified from
   the start, now applied continuously rather than only at startup.

   **Why a periodic re-scan rather than frame-to-frame pixel differencing**
   (which this replaced): pixel differencing compares raw brightness, so a
   shifting shadow, an auto-exposure adjustment, or a person walking past reads
   identically to an object appearing — measured at roughly 1 real detection in
   3, and observed live filling the screen with boxes on blank walls. A re-scan
   compares *objects*, so a wall is never "new" because a wall is never an
   object in either list. It also self-cancels the scan's own false positives:
   sofa texture and floor grout appear in both lists, at the same place, so they
   are never "new." Full history in `docs/phase-writeups/phase-3.md` and
   `docs/phase-4-detection-research.md`.

4. **Alert behavior.** Four states an object can be in, and what each does when
   the child approaches (rewritten 2026-08-13):

   | State | Child approaches it |
   |---|---|
   | Parent confirmed it is a hazard | **Full alert** |
   | Appeared after the first scan, not yet reviewed | **Full alert** — unreviewed means unknown, and unknown is treated as dangerous |
   | Found in the first scan, still awaiting review | No alert — this is the room's normal state and the parent is present reviewing it |
   | Parent marked it "not a hazard" | No alert |

   - **A newly appeared object raises an alert on appearance, not only on
     approach** — the parent should not have to be watching the screen to learn
     that something arrived. It simultaneously joins the review queue.
   - **"Not a hazard" genuinely dismisses.** This corrects a line that survived
     here until 2026-08-13 ("a prior 'seen it, ignore' acknowledgment never
     silences a real-time proximity alert"), which was written before an
     explicit dismiss action existed and, read literally against the current
     design, would have made the dismiss button meaningless — a dismissed sofa
     cushion would alarm every time the child climbed on the sofa.
   - **Dismissals are per-spot, and are re-raised when that spot changes.** A
     dismissal records what the spot looked like. If a later scan finds that
     same spot looking materially different, it returns to the review queue
     rather than staying silent forever — deliberately erring toward asking
     twice (Shaked, 2026-08-13: "better safe than sorry"), since the failure
     this guards against is a real hazard placed exactly where something
     harmless was dismissed.
   - Voice alerts are **pre-recorded audio clips**, not live TTS. A small fixed
     set: e.g. "Hazard detected," "Baby getting close," "Immediate danger."

5. **Proximity/risk scoring** uses bounding-box centers (child vs. hazard),
   Euclidean pixel distance, normalized against frame diagonal and/or hazard
   bbox size (NOT raw pixels — must be resolution-independent). Smoothed over a
   rolling window (~5–10 frames) to avoid flicker. This is a deliberately
   approximate "closer/farther within frame" signal, not true real-world
   distance (no depth sensing, no stereo, no calibration ritual) — this
   approximation is a conscious, documented tradeoff, not an oversight.

6. **Critical (red) alerts trigger a rolling video buffer save:** a circular
   buffer holds the last ~5 seconds of frames at all times; on a critical alert,
   buffer contents + ~2 more seconds get stitched into a clip, saved pending,
   and shown to the parent to keep/discard. Undecided clips auto-delete after a
   timeout. Only critical alerts trigger this — not every yellow/orange event.

7. **Object detection model** (rewritten 2026-08-13, the scope reset).

   **The goal is to find anything on a surface a child can reach — floor, low
   table, low shelf — whether or not we can name it.** A knife, a coin, a bottle
   cap, a lighter, or something nobody has thought of are the same signal: there
   is an object where there should not be one. Naming it is a bonus, never the
   requirement.

   **Exactly two models run, and each has one job:**

   - **The per-frame pass (`yolo26l`, imgsz 640)** — finds the child. Also
     yields `oven`/`microwave`/`refrigerator` from the same result at no extra
     cost, per decision 2. This is the only thing on Layer B's critical path and
     the only thing that must keep up with the camera.
   - **The periodic scan (class-agnostic segmentation, ~120ms, every few
     seconds)** — lists occupied spots on reachable surfaces, without naming
     them. Feeds decision 3's arrived/removed comparison and decision 4's review
     queue. Never on Layer B's critical path, so its cost is irrelevant to alert
     latency.

   Optionally, one more, and only because it covers a gap the scan structurally
   cannot: **wall sockets via open-vocabulary prompting** (measured 0.48–0.90).
   A socket is flush with a wall, so it is never an "object sitting on a
   surface." It runs on a slow cadence and its hits enter the review queue like
   any other proposal — it is not permitted to add a hazard on its own.

   **Nothing auto-adds a hazard. Every detector proposes; the parent disposes.**
   This is the rule that replaced five hazard categories with their own colours,
   matching rules, expiry rules and alert behaviours. One kind of hazard, one
   rule.

   **What was deliberately removed on 2026-08-13, and why** — all of it worked
   in isolation and none of it survived contact with the whole system running at
   once (five models competing in one loop):
   - **Frame-to-frame pixel change detection** — replaced by decision 3's
     re-scan comparison, which does the same job in the object domain instead of
     the pixel domain. Reasoning in full there. The offline measurement tool
     (`cv/measure_change_detection.py`) is kept as a research artifact; the live
     mechanism is gone.
   - **The Grounding DINO crop-confirmation step** — existed only to filter the
     above's false positives before a human saw them. With the human as the
     filter by design, it had no job left, and at ~500ms per call it was the
     prime suspect for the live loop collapsing to ~2fps.
   - **The `knife`/`scissors` (`sharp_object`) open-vocab detector** — the scan
     proposes these objects anyway and the parent confirms them, reaching the
     same outcome with one fewer model. It was also measured at precision 0.382
     / recall 0.210 on an unseen building — worse than simply showing the parent
     a candidate box.

   **Fine-tuning is not the move**, and this is settled by measurement rather
   than preference: three rounds against a held-out building reached at best
   precision 0.179 / recall 0.216. **The standing test for any detector is
   measured performance in a building the model has not seen** — a leak-free
   train/val split is necessary and not sufficient; only a held-out *location*
   catches a model that memorised one room.

   **Some hazards are not object detection at all.** "Fan or heater running" is
   motion. An open window, once found, is a fixed region. Do not force these
   into a class list.

   **Measured facts worth keeping** (detail in
   `docs/phase-3-step0-findings.md`, `docs/phase-4-detection-research.md`):
   stock COCO's `knife` scores 0.48–0.885 held in a hand but 0.01–0.05 resting
   unattended on a surface — it fails precisely in the case this project exists
   to catch. `scissors` swings 0.03–0.80 across near-identical frames, and white
   scissors on a white surface top out around 0.045 at any resolution. Named
   detection of small hazards is unreliable in exactly the conditions that
   matter, which is the evidence behind not depending on it. `sink` remains
   unverified — no test area contains one.

8. **Storage:** SQLite for event/log metadata, local disk for video clips. No
   cloud, ever, for this data — privacy requirement, not just a convenience
   choice.

## Team structure (this is deliberate, not incidental)

This project is being built using a small simulated engineering team of Claude
Code subagents, on purpose — partly to build the system well, partly as a real
skill/portfolio exercise in orchestrating multi-agent AI workflows. That
meta-goal is itself part of what gets documented (see docs-agent below) — it is
not a side note.

- **Orchestrator** — Shaked/Yahli talking to the main Claude Code session.
  Breaks work into tasks, delegates to the right subagent, makes final calls.
- **cv-agent** — camera capture, YOLO (baseline + fine-tuning), hazard map,
  proximity/risk scoring, rolling video buffer.
- **backend-agent** — FastAPI server, MJPEG streaming, events/status/clips API,
  SQLite schema and queries.
- **ui-agent** — Flet desktop + web shell, wired to the FastAPI backend only.
- **docs-agent** — full read access to the whole repo. Not just a scribe: it
  audits whether a phase's actual code matches what the phase claimed to
  deliver, teaches the phase back to the team in plain language, keeps a
  running decision log, and separately documents the multi-agent workflow
  process itself as its own narrative thread for the final paper. A phase is
  not "done" until docs-agent has reviewed it.

## Phase plan

See `PHASE_PLAN.md`. Do not start work on a phase whose dependencies aren't
marked complete in that file.

## Decision log

See `docs/decision-log.md` — one line per real architectural decision, dated,
with a one-sentence reason. Update it in the same turn a decision is made, not
retroactively.
