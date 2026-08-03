# GuardianEye — Project Context

This file is read automatically by every Claude Code session and subagent in this
project. It is the single source of truth for architecture decisions. If code and
this file ever disagree, this file wins until it is deliberately updated — and any
update to this file should be logged in `docs/decision-log.md`.

## What this project is

GuardianEye is a home-safety monitoring system for toddlers. A fixed USB camera
watches a room; an AI vision pipeline detects the child and household hazards in
real time; a risk engine tracks how close the child is getting to each hazard;
the system raises visual + pre-recorded voice alerts before an accident happens,
and saves short video clips of critical events for the parent to review.

Team: Shaked Ivgi, Yahli Mazri. Advisor: Tom Cohen.

## Core architecture decisions (locked — do not silently change)

1. **Video pipeline vs. control UI are decoupled.**
   OpenCV + YOLO run the detection loop and produce annotated frames. FastAPI
   serves those frames as an MJPEG stream (`/video_feed`) plus a small JSON API
   (`/events`, `/risk_status`, `/clips`). Flet is a *client* of FastAPI — it does
   NOT push frames through its own state system. Flet runs as both a Desktop app
   and a local Web App from the same codebase, both talking to the same FastAPI
   backend.

2. **Single YOLO pass per frame**, detecting both the child/person class and all
   hazard classes together. There is no separate "person detector" and "hazard
   detector."

3. **Two always-on logical layers, not two sequential stages:**
   - **Layer A — hazard map:** continuously updated for as long as the camera is
     on. No persistence between sessions/boots — every session starts with a
     fresh, empty map (camera angle/room/lighting cannot be assumed identical to
     last time).
   - **Layer B — child proximity risk:** runs the moment a child is detected,
     computed against whatever Layer A currently knows.
   - There is no hard "setup phase ends, monitoring phase begins" switch. Both
     layers run from frame 1, even if a child is already in frame at boot.

4. **Alert behavior:**
   - Hazards found during initial room settling ("setup-phase") are logged
     silently — no sound, just visible in the log for the parent to review
     whenever.
   - Anything detected *after* that initial settling — a genuinely new hazard
     appearing, OR the child approaching any known hazard — triggers a full
     visual + pre-recorded voice alert. No exceptions: a prior "seen it, ignore"
     acknowledgment during setup never silences a real-time proximity alert.
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

7. **Object detection model:** hybrid approach — use stock YOLO/COCO classes
   where they already fit (person, oven, knife, scissors...), fine-tune a small
   number of additional custom classes only for what COCO is missing (small
   choking-hazard objects, stairs, etc.). No training from scratch.

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
