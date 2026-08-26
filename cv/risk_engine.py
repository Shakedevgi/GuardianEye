"""
risk_engine.py - Phase 4 main script: live risk overlay on the camera feed.

Implements CLAUDE.md's two always-on layers plus the object-review workflow
decisions 3/4/7 describe (rewritten 2026-08-13 - see docs/decision-log.md,
"Scope reset", for why this file looks the way it does and what it replaced;
this docstring describes current behaviour only):

  A camera detects objects. A human classifies which are dangerous. The
  system detects new objects appearing after that. It alerts.

Concretely:

  Layer A - hazard map (HazardMap below). In-memory only, empty at the start
  of every run (CLAUDE.md decision 3 - no persistence between sessions). Two
  detectors feed it, and NEITHER is allowed to add a hazard on its own -
  every detection becomes a HazardEntry in PENDING state, awaiting a human
  decision (CLAUDE.md decision 7: "nothing auto-adds a hazard; every detector
  proposes, the parent disposes"):

    - The per-frame pass: the same single YOLO26 inference this file already
      runs for `person` also yields `oven`/`microwave`/`refrigerator` at no
      extra cost (CLAUDE.md decision 2). A second, cadenced, open-vocabulary
      pass adds `wall_socket` the same way, since a socket is flush with a
      wall and therefore invisible to the surface scan below. Both are
      matched against existing entries by position (find_match) so the same
      physical fridge doesn't spawn a new entry every frame.
    - The periodic scan: every `--scan-interval` seconds (a few, by default -
      this is Layer A, not latency-critical), a class-agnostic segmentation
      pass (FastSAM, via measure_segmentation.py, imported not reimplemented)
      lists occupied spots on reachable surfaces. Comparing that list against
      what is already known is Layer A's entire "detect new" mechanism
      (CLAUDE.md decision 3):
        - a spot with no match in the map -> something arrived
        - a known spot with no match in the new scan -> something was
          removed, and it clears
        - everything else -> unchanged, stay quiet
      A spot must appear in two CONSECUTIVE scans before it counts as
      "arrived," and a known spot must be absent from two consecutive scans
      before it's cleared - guards against a single noisy scan reading as
      churn (see HazardMap.apply_scan_candidates and _Provisional below).

  Three states a HazardEntry can be in (CLAUDE.md decision 7's rule replaced
  five source-specific hazard categories with this one):
    - PENDING   - proposed by a detector, no human judgment yet.
    - CONFIRMED - a human pressed 'h'. This is a hazard.
    - DISMISSED - a human pressed 'n'. Not a hazard, remembered - but a
      dismissal is re-raised (back to PENDING) if that exact spot reads as
      materially changed (region_change_frac, imported from
      measure_change_detection.py) for DISMISS_REAPPEAR_CONSECUTIVE_SCANS_
      REQUIRED scans IN A ROW, per Shaked's "better safe than sorry" ruling:
      a real hazard placed where something harmless was dismissed must not
      silently inherit that dismissal. The consecutive-scan requirement
      (added 2026-08-26) is a jitter guard on the re-raise trigger itself,
      not a loosening of it - see DISMISS_REAPPEAR_CONSECUTIVE_SCANS_
      REQUIRED's comment for why a single noisy scan was forcing a fresh
      dismiss every scan cycle for at least one visually ambiguous object.

  Every HazardEntry also carries `is_first_scan`: True if it was proposed
  before the very first periodic scan finished looking at the room (i.e. it
  is part of the room's starting state, reviewed with a human presumably
  still standing in front of the camera), False if it showed up later.

  Layer B - proximity scoring (score_frame, PersonTracker, unchanged in
  spirit since Phase 4 Slice 1): for every tracked person, bbox-center
  Euclidean distance to every ALERT-ELIGIBLE hazard-map entry, normalized by
  frame diagonal (CLAUDE.md decision 5 - resolution-independent, not real-
  world distance - no depth sensing, no calibration ritual, a deliberate,
  documented approximation), smoothed over a rolling window per (person,
  hazard) pair. Which entries are alert-eligible is CLAUDE.md decision 4's
  table, implemented directly in hazard_alerts_on_approach():

      state                              | alerts on approach?
      -----------------------------------|--------------------
      CONFIRMED                          | yes
      PENDING, arrived after first scan  | yes - unreviewed means unknown,
                                          |   and unknown is treated as
                                          |   dangerous
      PENDING, from the first scan       | no - this is the room's normal
                                          |   state with a human present
                                          |   reviewing it
      DISMISSED                          | no

  A newly arrived object (not from the first scan) also raises an alert the
  moment it's noticed, not only on approach (CLAUDE.md decision 4) - the
  parent should not have to be watching the screen to learn something showed
  up. See ALERT_BANNER_SECONDS and main()'s scan/propose call sites.

  The human review queue (ReviewQueue below) is a live FIFO of PENDING
  entries awaiting an 'h'/'n' decision, drawn one at a time on the live feed
  ('h' = confirm hazard, 'n' = not a hazard, 's' = skip everything currently
  queued - new arrivals still get queued afterward). It receives new items
  continuously from either detector, for as long as the camera runs - there
  is no separate "setup phase" that ends (CLAUDE.md decision 3).

Usage:
    python risk_engine.py                                # default index, mps, yolo26l
    python risk_engine.py --name Arducam
    python risk_engine.py --seed-hazard 200,400,300,200,test_stove
    python risk_engine.py --scan-interval 4               # faster room re-scan
    python risk_engine.py --disable-socket-detect          # skip the wall-socket pass
    python risk_engine.py --disable-scan                   # skip the periodic room scan
                                                            # (named-class detection only)
    python risk_engine.py --disable-audio                   # visual alerts + clips, no sound

Phase 5 adds the alert lifecycle, rolling video buffer, saved clips, and
voice-clip playback (CLAUDE.md decision 6; PHASE_PLAN.md Phase 5) on top of
everything above, unchanged. See AlertManager's docstring for the full
design (an AlertEvent replaces the old single overwritable alert-banner
slot, with real identity, exit-hysteresis cooldown measured against a real
recording, and per-(person,hazard) dedup); RollingBuffer/ClipRecorder for
the JPEG-buffered, background-thread-written clip mechanism; AudioPlayer for
the non-blocking `afplay` playback of the three pre-recorded voice clips in
--audio-dir (default: cv/audio/hazard_detected.wav / baby_getting_close.wav
/ immediate_danger.wav - missing files log a warning once and are skipped,
not a hard failure). Saved clips land in --clips-dir (default:
cv/clips/pending/) as one file per critical (red) alert event; keep/discard
and auto-delete of undecided clips are explicitly Phase 6's job, not this
one's.

--seed-hazard is repeatable, takes pixel coordinates of the ACTUAL capture
resolution (not display resolution), and is added directly as a CONFIRMED
hazard - it exists purely for deterministic testing of Layer B's zone
escalation without needing a real hazard in frame. x,y is the top-left
corner, w,h is the box's width/height; label is free text (everything after
the fourth comma, so a label can't itself contain a comma).

Keys (window must be focused):
    q   quit
    h   confirm the entry currently at the front of the review queue as a
        hazard (state -> CONFIRMED)
    n   dismiss it - "not a hazard" (state -> DISMISSED; re-raised later if
        that spot's appearance changes materially)
    s   clear everything currently queued for review without deciding (they
        stay PENDING and keep alerting per the table above if they arrived
        after the first scan) - future arrivals still get queued normally
"""

import argparse
import math
import os
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Must be set before torch is imported anywhere (including transitively via
# ultralytics) - see detect_stream.py's identical comment.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2

from camera import (
    DEFAULT_CAMERA_INDEX,
    CameraCapture,
    CameraSelectionError,
    startup_failure_message,
)
from detect_stream import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL,
    FPS_SMOOTHING_ALPHA,
    OVERLAY_COLOR,
    OVERLAY_OUTLINE,
    draw_overlay_line,
    load_model,
    prepare_for_display,
    resolve_device,
)

# Wall-socket open-vocabulary detection - reused, not reimplemented (see
# measure_openvocab.py's own docstring for the YOLOWorld/YOLOE set_classes()
# binding this wraps).
from measure_openvocab import load_open_vocab_model

# Only region_change_frac survives from measure_change_detection.py - it is
# reused here for the dismissal re-raise check (Step 4: does a dismissed
# spot look materially different now?), not for the frame-to-frame change
# detector this module used to run live. That detector (and the rest of this
# module's live wiring) is gone from the live pipeline as of the 2026-08-13
# scope reset - docs/decision-log.md has the full reasoning. The offline
# measurement script itself (cv/measure_change_detection.py) is kept intact
# as a research artifact.
from measure_change_detection import region_change_frac

# Class-agnostic segmentation candidate generation for the periodic room
# scan - reused from measure_segmentation.py exactly as tuned there
# (MAX_AREA_FRAC tightened in Phase 3 Part 5, zero measured recall cost).
# `--min-aspect-ratio` is deliberately never enabled - Phase 3 measured it
# costing 46-50% of recall on elongated true positives (knives/scissors),
# this project's own most safety-critical shape.
from ultralytics import FastSAM
from measure_segmentation import (
    DEFAULT_SEG_IMGSZ,
    DEFAULT_SEG_MODEL,
    MAX_AREA_FRAC,
    MIN_AREA_FRAC,
    MIN_EXTENT,
    PERSON_OVERLAP_THRESHOLD,
    apply_filter as seg_apply_filter,
    load_weights as load_seg_weights,
    segment_frame,
)

WINDOW_NAME = "GuardianEye - risk engine"

# --- Layer A/B tuning constants -------------------------------------------

# Hazard-map matching: an incoming detection updates an existing entry
# instead of spawning a new one if it clears EITHER threshold - high IoU
# (same box, roughly), or a close center even if the box size changed a bit
# frame to frame (partial occlusion, angle change). Matching is scoped to a
# candidate pool the caller chooses (same label for named-class entries;
# same origin for scan entries - see find_match's `label` parameter and
# HazardMap below) so, e.g., an oven detection can never merge into a
# refrigerator entry just because they happen to be adjacent.
HAZARD_MATCH_IOU_THRESHOLD = 0.3
HAZARD_MATCH_CENTER_DIST_FRAC = 0.08

# Two overlapping segmentation candidates from the SAME scan pass, above
# this IoU, are treated as one physical object and deduplicated before
# anything is queued for review (Step 3's "duplicate-candidate bug" fix) -
# looser than HAZARD_MATCH_IOU_THRESHOLD on purpose: two boxes from one
# segmentation pass describing the same object can disagree on exact edges
# more than two independent detections of an already-tracked entry would.
SCAN_DUPLICATE_IOU_THRESHOLD = 0.4

# A dismissed spot is a re-raise CANDIDATE once a later scan finds it changed
# by more than this fraction of pixels (region_change_frac, imported from
# measure_change_detection.py). Deliberately LOWER (more sensitive) than
# that module's own STABILITY_MAX_CHANGE_FRAC (0.3, tuned to ask "did this
# stay the same" when CONFIRMING persistence) - here the goal is the
# opposite: erring toward re-asking rather than missing a hazard placed
# where something harmless was previously dismissed (Shaked, 2026-08-13:
# "keep it as double and even triple mark - better safe than sorry"). This
# sensitivity is deliberately UNCHANGED as of 2026-08-26 - live testing
# confirmed it's sensitive enough to false-trigger on scan-to-scan noise for
# at least one visually ambiguous object (see
# DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED below, which is the fix that
# was applied instead of loosening this threshold), not that it's wrong.
DISMISS_REAPPEAR_CHANGE_FRAC = 0.15

# A single scan crossing DISMISS_REAPPEAR_CHANGE_FRAC is no longer enough to
# re-raise a dismissed entry on its own - it must read as changed this many
# CONSECUTIVE scans in a row (reset to 0 the instant a scan reads
# unchanged). Added 2026-08-26 after a live test: one ambiguous object kept
# tripping the single-scan check on noise alone (lighting, a shifted
# segmentation box, JPEG artifacts - nothing about the physical spot
# actually changed), forcing a fresh dismiss every ~5s scan cycle. This is
# the SAME jitter-guard shape as SCAN_CONSECUTIVE_SCANS_REQUIRED below,
# applied to the re-raise path instead of the arrival path - deliberately
# reusing a proven mechanism rather than loosening DISMISS_REAPPEAR_
# CHANGE_FRAC itself, which would also make a genuinely smaller real hazard
# swap easier to miss. Cost: a REAL hazard swapped into a dismissed spot
# now takes one extra scan cycle (~5s) to be caught, since it must persist
# across two scans instead of one - a deliberate, named tradeoff, not an
# oversight.
DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED = 2

# A scan-origin spot must appear in this many CONSECUTIVE scans before it's
# added to the map as "arrived," and a known scan-origin entry must be
# absent from this many consecutive scans before it's cleared as "removed."
# 2 is the smallest value that is a guard at all (1 would mean no guard).
# Not measured against real scan-to-scan jitter yet (no camera in this
# session) - see the report for what that means is still unverified.
SCAN_CONSECUTIVE_SCANS_REQUIRED = 2

# If tracked people cover more than this fraction of the frame when a scan
# is due, the scan is deferred to the next cadence tick instead of run this
# tick. Reasoning: person-overlap suppression (reused from
# measure_segmentation.py's own apply_filter) already drops segments
# overlapping a person, but if a person fills most of the frame there is
# too little of the room actually visible for "removed" to mean anything -
# without this guard, a person standing in front of most of a shelf could
# read as several hazards disappearing at once. A first-guess threshold,
# not measured live.
SCAN_PERSON_COVERAGE_SKIP_FRAC = 0.35

# A tracked person not matched by any detection for this long (wall-clock
# seconds) is dropped, along with their rolling-window history - otherwise
# someone who walks out of frame and back in would incorrectly inherit a
# stale smoothed distance instead of the risk score correctly resetting.
PERSON_MATCH_IOU_THRESHOLD = 0.2
PERSON_MATCH_CENTER_DIST_FRAC = 0.15
PERSON_STALE_SECONDS = 1.0

# Rolling window length in FRAMES (not seconds) per (person, hazard) pair,
# per CLAUDE.md decision 5 ("~5-10 frames"). Unmeasured/untuned midpoint.
ROLLING_WINDOW_SIZE = 8

# Zone thresholds on NORMALIZED distance (raw pixel distance / frame
# diagonal - decision 5 requires resolution-independence, not raw pixels).
# THESE ARE A FIRST GUESS, not a measured result - unchanged since Phase 4
# kickoff (docs/decision-log.md, 2026-08-12), still awaiting live tuning.
RISK_ZONE_RED_MAX = 0.15
RISK_ZONE_ORANGE_MAX = 0.30
RISK_ZONE_YELLOW_MAX = 0.50
RISK_ZONE_NONE = "none"
RISK_ZONE_ORDER = ("none", "yellow", "orange", "red")

# --- Hazard entry state / origin --------------------------------------------

HAZARD_STATE_PENDING = "pending"
HAZARD_STATE_CONFIRMED = "confirmed"
HAZARD_STATE_DISMISSED = "dismissed"

# `origin` records WHICH detector proposed an entry, purely to decide which
# mechanism is responsible for keeping it current - it has no bearing on
# alerting (that's `state` + `is_first_scan`, see hazard_alerts_on_approach).
#   NAMED - the per-frame YOLO pass (oven/microwave/refrigerator) or the
#     cadenced wall-socket pass. Re-checked every time its own detector
#     runs, so its bbox stays fresh on a match; nothing here ever removes it
#     (a fridge does not legitimately vanish mid-session the way a small
#     object left on a scanned surface can).
#   SCAN - the periodic class-agnostic room scan. Subject to
#     HazardMap.apply_scan_candidates' arrived/removed diffing and the
#     dismissal re-raise check - the only origin either of those touches.
#   SEED - a manually seeded entry (--seed-hazard), added directly as
#     CONFIRMED for deterministic zone-escalation testing. Untouched by
#     either detector.
HAZARD_ORIGIN_NAMED = "named"
HAZARD_ORIGIN_SCAN = "scan"
HAZARD_ORIGIN_SEED = "seed"

# Named-class -> hazard-map label grouping. Only classes Phase 3 measured as
# reliable without training (docs/decision-log.md, 2026-08-12): oven and
# microwave fold into one "oven_microwave" concept because both classes fire
# on the same physical countertop object, not two different objects. "chair"
# is deliberately excluded - a live test showed it repeatedly false-firing
# on a coffee table/shelf structure, and a chair was never really the
# hazard anyway (the hazard is whatever ends up placed on a reachable
# surface, which the room scan is meant to catch).
HAZARD_LABEL_BY_CLASS_NAME = {
    "oven": "oven_microwave",
    "microwave": "oven_microwave",
    "refrigerator": "refrigerator",
}

# --- Wall-socket open-vocabulary detection ----------------------------------
#
# A socket is flush with a wall, so it is never "an object sitting on a
# surface" and is structurally invisible to the room scan below - CLAUDE.md
# decision 7 keeps this as the one extra named detector earning its keep.
# Model/imgsz/conf choices are the measured values from Phase 3/4 (see
# cv/measurements/openvocab.csv): yolov8s-worldv2.pt reached 0.90 confidence
# at imgsz 1600 and 0.85 at imgsz 1280 - imgsz 1280 is the default since it's
# within 0.05 of 1600's ceiling at near-identical inference cost. conf 0.4
# sits below the real-socket range's floor (0.48-0.90) with margin, well
# above the measured noise median (0.02-0.08).
DEFAULT_SOCKET_MODEL = "yolov8s-worldv2.pt"
DEFAULT_SOCKET_PROMPTS = ["wall socket"]
DEFAULT_SOCKET_IMGSZ = 1280
DEFAULT_SOCKET_CONF = 0.4
# Wall-clock seconds between socket-detection passes - sockets are static
# once found and this is Layer A, not Layer B, so a cadence (not every
# frame) is deliberate, not an oversight. Unmeasured first guess.
DEFAULT_SOCKET_SCAN_INTERVAL_SECONDS = 2.0
SOCKET_HAZARD_LABEL = "wall_socket"

# --- Periodic room scan (class-agnostic segmentation) -----------------------
#
# CLAUDE.md decision 3: "every few seconds," explicitly not latency-critical
# since this is Layer A. Unmeasured first guess, cheap to retune via
# --scan-interval once someone is watching this live.
DEFAULT_SCAN_INTERVAL_SECONDS = 5.0
SCAN_HAZARD_LABEL = "object"

# --- Overlay colors ----------------------------------------------------------
#
# Distinct from person boxes (green) and from each risk zone's connector-
# line color, so "hazard-map entry state" is never confused with "risk
# level right now" at a glance - those are two different axes (see the
# module docstring's Layer A/B split).
PERSON_BOX_COLOR = (0, 255, 0)  # green (BGR)
HAZARD_STATE_BOX_COLOR = {
    HAZARD_STATE_CONFIRMED: (0, 0, 255),  # red - a human confirmed this is a hazard
    HAZARD_STATE_DISMISSED: (140, 140, 140),  # grey - a human said "not a hazard"
}
# PENDING is split into two colors because its ALERTING behaviour differs
# (see hazard_alerts_on_approach) and the overlay should make that visible,
# not just the state name:
PENDING_FIRST_SCAN_BOX_COLOR = (255, 255, 0)  # cyan - awaiting review, room's
# normal starting state, does not alert yet.
PENDING_ARRIVED_BOX_COLOR = (0, 140, 255)  # orange - awaiting review, but
# arrived after the first scan, so it alerts on approach same as a
# confirmed hazard ("unreviewed means unknown, and unknown is dangerous").
# The entry currently sitting at the front of the review queue (about to be
# judged) also gets a white outline on top of its state color, so "this box
# is the one awaiting your h/n/s keypress right now" is never confused with
# "this box is just pending in general."
REVIEW_CANDIDATE_OUTLINE_COLOR = (255, 255, 255)  # white (BGR)
ZONE_COLORS = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 255, 255),
    RISK_ZONE_NONE: (140, 140, 140),
}

# How long a "new object" banner stays on screen after arrival/re-raise.
ALERT_BANNER_SECONDS = 4.0

# --- Phase 5: alert lifecycle, rolling buffer, clips, audio -----------------
#
# CLAUDE.md decision 6's ~5-7s clip and decision 4's alert behaviour need a
# real event to hang state off - the pre-Phase-5 code had none, only a single
# overwritable alert_text/alert_until slot with no identity, no cooldown, no
# dedup, which would re-fire continuously for a hazard sitting in RED. See
# AlertManager below for the replacement.
#
# ALERT_HOLD_SECONDS is exit hysteresis, not entry debounce: a proximity
# event stays open (no re-alert, no re-clip) until the (person, hazard) pair
# has gone unseen in a scored zone for this long. Measured directly against
# the 2026-08-22 unreviewed-approach recording
# (cv/captures/Screen Recording 2026-08-22 at 13.34.16.mov): one continuous
# walk toward one object flickered RED->ORANGE->RED twice, with dips up to
# 0.75s, DESPITE the existing 8-frame rolling window already smoothing the
# raw distance - more smoothing alone would not have prevented it. 2.0s
# gives ~2.7x margin over the worst dip actually observed. A pure
# edge-triggered design (alert only on the none->red transition) was
# considered and rejected: it would have re-armed after each dip and fired 3
# separate alerts/clips for what a parent would experience as one approach.
ALERT_HOLD_SECONDS = 2.0

# Global (not per-pair) pacing on actually VOICING an alert - added
# 2026-08-26 after a live test found two things stacking into what felt
# like six alarms at once: (1) HazardMap's dismissal re-raise (deliberately
# sensitive per Shaked's 2026-08-13 "better safe than sorry" ruling) flipped
# the SAME physical spot between DISMISSED and PENDING four times in a row,
# and every single re-raise fired its own independent, unthrottled
# new-object alert; (2) that burst landed in the same few seconds as a real
# proximity escalation into RED. ALERT_HOLD_SECONDS above already prevents
# ONE (person, hazard) pair from re-alerting on its own flicker; nothing
# previously stopped DIFFERENT signals - or repeated new-object pulses for
# a single re-raising hazard, which have no per-pair hysteresis at all -
# from all voicing independently. See AlertArbiter: this does not change
# WHETHER AlertManager considers something worth an alert (that logic is
# unchanged), only whether THIS PARTICULAR MOMENT is when the parent
# actually gets interrupted about it.
GLOBAL_ALERT_MIN_INTERVAL_SECONDS = 2.5

# CLAUDE.md decision 6: a critical (red) alert stitches the rolling buffer's
# preceding ~5s plus ~2 more seconds of live tail into a saved clip.
ROLLING_BUFFER_SECONDS = 5.0
CLIP_TAIL_SECONDS = 2.0

# Nominal frame rate used to size the rolling buffer (in FRAMES, not
# seconds) and as the fixed playback fps written into a saved clip's header.
# NOT read from the live smoothed_fps - that fluctuates frame to frame, and
# a clip's playback rate has to be fixed once at write time. 15.0 is the
# steady-state FPS actually measured live post-Phase-4-reset
# (docs/decision-log.md, 2026-08-22), not a guess.
DEFAULT_TARGET_FPS = 15.0

# Global disk-safety backstop, deliberately NOT per-hazard: a parent camped
# at the edge of RED for minutes should not fill the disk with clips no
# matter how the per-event hysteresis above behaves. Separate from
# ALERT_HOLD_SECONDS on purpose - hold controls when an EVENT ends, this
# controls how often a NEW clip file can start, even across different
# events/hazards.
CLIP_MIN_INTERVAL_SECONDS = 30.0

# JPEG quality for the rolling buffer's frames. Measured
# (docs/decision-log.md, 2026-08-22 Phase 5 entry): q75 at the FULL capture
# resolution (no downscaling) averages ~200KB/frame across 10 real captures
# from cv/captures/ (111-245KB range) - a 5s/75-frame buffer is ~15MB, a 32x
# reduction from holding raw 1080p frames (~467MB for the same 5s), at
# ~2.7ms/frame encode cost (~4% of one 15-FPS frame's 66.7ms budget).
# Downscaling was considered and rejected: it saves another ~10MB against an
# already-negligible number, in exchange for a lower-resolution saved clip.
ROLLING_BUFFER_JPEG_QUALITY = 75

# H.264 in an mp4 container, falling back to mp4v if the platform's OpenCV
# build lacks an avc1 encoder. Measured on 105 real (non-synthetic) frames
# at 960x540: avc1 wrote a 0.75MB file in ~194ms, mp4v wrote a 2.28MB file
# in ~130ms - avc1 is chosen for the smaller file and because Phase 7/8 will
# want these playable in a browser, and ~200ms either way is why clip
# writing runs on a background thread (see ClipRecorder) rather than inline
# in the frame loop.
CLIP_FOURCC_PRIMARY = "avc1"
CLIP_FOURCC_FALLBACK = "mp4v"

DEFAULT_CLIPS_DIR = "clips/pending"
# Relative to CWD (this project's own convention is to run from cv/, e.g.
# "cd cv && python risk_engine.py" - see the module docstring's Usage
# section) - "pending" because CLAUDE.md decision 6 says a saved clip is
# shown to the parent to keep/discard; the keep/discard mechanism itself and
# the auto-delete-if-undecided timeout are explicitly Phase 6's job, not
# this one.

DEFAULT_AUDIO_DIR = "audio"
# Pre-recorded voice clips, per CLAUDE.md decision 4 ("pre-recorded audio
# clips, not live TTS"). Someone has to actually record these; AudioPlayer
# below logs a warning once per missing file and keeps running rather than
# blocking Phase 5 on that dependency.
AUDIO_HAZARD_DETECTED = "hazard_detected.wav"
AUDIO_BABY_GETTING_CLOSE = "baby_getting_close.wav"
AUDIO_IMMEDIATE_DANGER = "immediate_danger.wav"

ALERT_KIND_PROXIMITY = "proximity"
ALERT_KIND_NEW_OBJECT = "new_object"


# --- geometry helpers -------------------------------------------------------


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def normalized_center_distance(box_a, box_b, frame_diagonal: float) -> float:
    """Euclidean distance between two bbox centers, normalized by frame
    diagonal - CLAUDE.md decision 5's exact model. Deliberately NOT also
    normalized by hazard bbox size (decision 5 says "and/or"); frame
    diagonal alone is simpler to reason about and is revisited only if live
    testing shows it isn't resolution-independent enough in practice.
    """
    (ax, ay), (bx, by) = bbox_center(box_a), bbox_center(box_b)
    dist = math.hypot(ax - bx, ay - by)
    return dist / frame_diagonal if frame_diagonal > 0 else 0.0


def classify_zone(normalized_distance: float) -> str:
    if normalized_distance < RISK_ZONE_RED_MAX:
        return "red"
    if normalized_distance < RISK_ZONE_ORANGE_MAX:
        return "orange"
    if normalized_distance < RISK_ZONE_YELLOW_MAX:
        return "yellow"
    return RISK_ZONE_NONE


def zone_rank(zone: str) -> int:
    return RISK_ZONE_ORDER.index(zone)


def find_match(new_box, entries, frame_diagonal, iou_threshold, center_dist_frac, label=None):
    """Return the best-matching entry for `new_box` among `entries` (any
    object exposing a `.bbox` attribute), or None if nothing clears either
    threshold.

    An entry matches if its IoU with `new_box` clears `iou_threshold` OR its
    normalized center distance is within `center_dist_frac` - either
    condition alone can miss real matches (a box that shrank/grew a bit
    still has close centers; a box that shifted sideways at a fixed size
    still has decent IoU), so either is accepted. Ties are broken by
    preferring the highest IoU. If `label` is given, only entries whose
    `.label` equals it are considered.
    """
    best = None
    best_iou = -1.0
    for entry in entries:
        if label is not None and getattr(entry, "label", None) != label:
            continue
        iou = bbox_iou(new_box, entry.bbox)
        center_dist = normalized_center_distance(new_box, entry.bbox, frame_diagonal)
        if iou >= iou_threshold or center_dist <= center_dist_frac:
            if iou > best_iou:
                best = entry
                best_iou = iou
    return best


def dedupe_candidates(candidates: list, iou_threshold: float = SCAN_DUPLICATE_IOU_THRESHOLD) -> list:
    """Step 3's duplicate-candidate fix: segmentation sometimes proposes two
    overlapping boxes for the same physical object. Keep the first of any
    group of candidates whose IoU with an already-kept candidate clears
    `iou_threshold`, drop the rest. Order-preserving, pure geometry - no
    frame/model dependency, so it's directly testable.
    """
    kept: list = []
    for candidate in candidates:
        if any(bbox_iou(candidate, k) >= iou_threshold for k in kept):
            continue
        kept.append(candidate)
    return kept


def crop_bbox(frame, bbox):
    """Clamp `bbox` to `frame`'s bounds and return the cropped region (a
    copy, so it survives after `frame` itself is overwritten next
    iteration), or None if the clamped region is empty.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return None
    return frame[y1c:y2c, x1c:x2c].copy()


def fingerprint_changed(
    fingerprint, frame, bbox, threshold: float = DISMISS_REAPPEAR_CHANGE_FRAC, label: str = None
) -> bool:
    """Step 4: has the spot behind a DISMISSED entry changed materially
    since it was dismissed? `fingerprint` is the crop captured at dismissal
    time (see HazardMap.dismiss). Reuses region_change_frac's pixel-diff
    core (measure_change_detection.py) rather than inventing a new
    comparison.

    `bbox` MUST be the entry's STABLE fingerprint_bbox (the region recorded
    at dismiss time), NOT that scan's own fresh candidate bbox - fixed
    2026-08-26 after live testing showed one object re-raising on nearly
    every scan. The room scan's segmentation boundary is not pixel-
    identical run to run even for a completely static object (worse for a
    thin/irregular shape), so comparing against each scan's own wobbling
    box was measuring "we sampled different pixels this time" as "the scene
    changed" - not a real signal. Cropping the SAME region every time (this
    function still resizes if the two crops' shapes differ slightly, since
    crop_bbox can clip differently right at a frame edge) removes that
    noise source without touching what counts as a real change.

    Missing/degenerate input (no fingerprint recorded, or the current frame
    can't produce a matching crop) returns True - err toward re-asking
    rather than silently trusting a comparison that couldn't actually run,
    per Shaked's "better safe than sorry" ruling.

    `label`, if given, prints the computed change fraction - a lightweight,
    opt-in diagnostic (added 2026-08-26 alongside this fix) so a live run's
    terminal output shows the REAL measured numbers if this still trips
    unexpectedly, instead of guessing at DISMISS_REAPPEAR_CHANGE_FRAC blind.
    None (the default) prints nothing; existing callers/tests are
    unaffected.
    """
    if fingerprint is None or fingerprint.size == 0:
        if label:
            print(f"  [fingerprint] {label}: no fingerprint recorded - treating as changed")
        return True
    current_crop = crop_bbox(frame, bbox)
    if current_crop is None:
        if label:
            print(f"  [fingerprint] {label}: current crop unavailable - treating as changed")
        return True
    if current_crop.shape != fingerprint.shape:
        current_crop = cv2.resize(current_crop, (fingerprint.shape[1], fingerprint.shape[0]))
    local_bbox = (0, 0, fingerprint.shape[1], fingerprint.shape[0])
    frac = region_change_frac(fingerprint, current_crop, local_bbox)
    if label:
        print(f"  [fingerprint] {label}: change_frac={frac:.3f} (threshold={threshold})")
    return frac > threshold


def person_coverage_frac(person_boxes, frame_width: int, frame_height: int) -> float:
    """Rough fraction of the frame covered by tracked people, used to decide
    whether to defer a scan (SCAN_PERSON_COVERAGE_SKIP_FRAC). Sums bbox
    areas without correcting for overlap between multiple people - a
    deliberate simplification (this project targets a single-child home,
    not a crowd), fine for a threshold check that only needs to be roughly
    right.
    """
    frame_area = float(frame_width * frame_height)
    if frame_area <= 0:
        return 0.0
    total = sum(max(0.0, (x2 - x1) * (y2 - y1)) for x1, y1, x2, y2 in person_boxes)
    return min(1.0, total / frame_area)


# --- Layer A: hazard map ----------------------------------------------------


@dataclass
class HazardEntry:
    id: int
    label: str
    bbox: tuple[float, float, float, float]
    origin: str  # HAZARD_ORIGIN_NAMED / _SCAN / _SEED
    state: str = HAZARD_STATE_PENDING
    is_first_scan: bool = False
    last_seen: float = 0.0
    hits: int = 1
    # Only meaningful for origin == HAZARD_ORIGIN_SCAN: consecutive periodic
    # scans this entry went unmatched. Reset to 0 on every match, entry is
    # removed once this reaches SCAN_CONSECUTIVE_SCANS_REQUIRED.
    absent_scans: int = 0
    # Only set once state == HAZARD_STATE_DISMISSED (see HazardMap.dismiss):
    # a small crop of the frame at the moment of dismissal, compared against
    # later scans via fingerprint_changed() to decide whether to re-raise.
    fingerprint: object = None
    # The STABLE bbox `fingerprint` was cropped from - reused for cropping
    # every later scan's comparison frame too, instead of that scan's own
    # fresh candidate bbox. Added 2026-08-26 after live testing showed one
    # object re-raising on nearly every single scan even with the
    # consecutive-scan guard above: the comparison was cropping a
    # DIFFERENT bbox each scan (the room scan's segmentation boundary isn't
    # pixel-identical run to run even for a completely static object -
    # especially a thin/irregular shape), so it was measuring "different
    # pixels sampled" as "the scene changed," not real content change. This
    # field keeps the sampled region fixed so the comparison actually
    # answers "does this exact patch of the frame look different," matching
    # Shaked's own framing: "if nothing came or moved in the frame, nothing
    # should be alarted."
    fingerprint_bbox: tuple = None
    # Only meaningful while state == HAZARD_STATE_DISMISSED: consecutive
    # scans in a row fingerprint_changed() has read as "different from the
    # dismissal-time crop." Reset to 0 the moment a scan reads unchanged -
    # this is a jitter guard, the same shape as absent_scans/
    # SCAN_CONSECUTIVE_SCANS_REQUIRED above, added 2026-08-26 after a live
    # test found one visually ambiguous object re-raising and requiring a
    # fresh dismiss every scan cycle (~5s) because a single noisy pixel-diff
    # comparison was enough to trip DISMISS_REAPPEAR_CHANGE_FRAC. See
    # DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED.
    changed_scans: int = 0


@dataclass
class _Provisional:
    """A scan-origin candidate seen exactly once, not yet matched a second
    consecutive time - see SCAN_CONSECUTIVE_SCANS_REQUIRED and
    HazardMap.apply_scan_candidates. Not yet a HazardEntry: it has no id, no
    state, and is never drawn or scored.
    """

    bbox: tuple[float, float, float, float]
    is_first_scan: bool


@dataclass
class ScanDiffResult:
    """What one HazardMap.apply_scan_candidates() call did, for the caller
    (main()) to react to - enqueue newly-visible entries for review, alert
    on the ones that mean "something new is here right now," and drop
    removed entries from the review queue if they happened to be sitting in
    it.
    """

    arrived: list = field(default_factory=list)
    reraised: list = field(default_factory=list)
    removed: list = field(default_factory=list)


class HazardMap:
    """Layer A per CLAUDE.md decision 3: in-memory only, empty at
    construction, never written to or read from disk - a fresh HazardMap()
    every run is the whole point, since a new session cannot assume the
    camera angle/room/lighting match a previous one.
    """

    def __init__(self):
        self.entries: list[HazardEntry] = []
        self._next_id = 1
        self._provisional: list[_Provisional] = []

    def get(self, entry_id: int):
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def _new_entry(self, label, bbox, origin, is_first_scan, state=HAZARD_STATE_PENDING) -> HazardEntry:
        entry = HazardEntry(
            id=self._next_id,
            label=label,
            bbox=bbox,
            origin=origin,
            state=state,
            is_first_scan=is_first_scan,
            last_seen=time.monotonic(),
        )
        self._next_id += 1
        self.entries.append(entry)
        return entry

    def add_seed(self, bbox, label: str) -> HazardEntry:
        """--seed-hazard: added directly as CONFIRMED, for deterministic
        Layer B testing without a real hazard in frame. Not touched by
        either detector (origin is neither NAMED nor SCAN).
        """
        return self._new_entry(
            label, bbox, HAZARD_ORIGIN_SEED, is_first_scan=True, state=HAZARD_STATE_CONFIRMED
        )

    def propose_named(
        self, bbox, label: str, frame_diagonal: float, is_first_scan: bool
    ) -> tuple[HazardEntry, bool]:
        """The per-frame/named-class and wall-socket detectors' single entry
        point. Matches against existing NAMED-origin entries sharing `label`
        (so an oven detection can never merge into a refrigerator entry);
        refreshes position on a match, otherwise creates a new PENDING
        entry. Returns (entry, created) so the caller can enqueue/alert only
        on genuinely new entries.
        """
        candidates = [e for e in self.entries if e.label == label and e.origin == HAZARD_ORIGIN_NAMED]
        match = find_match(bbox, candidates, frame_diagonal, HAZARD_MATCH_IOU_THRESHOLD, HAZARD_MATCH_CENTER_DIST_FRAC)
        if match is not None:
            match.bbox = bbox
            match.last_seen = time.monotonic()
            match.hits += 1
            return match, False
        entry = self._new_entry(label, bbox, HAZARD_ORIGIN_NAMED, is_first_scan)
        return entry, True

    def confirm(self, entry_id: int):
        """'h' - a human says this is a hazard."""
        entry = self.get(entry_id)
        if entry is not None:
            entry.state = HAZARD_STATE_CONFIRMED
        return entry

    def dismiss(self, entry_id: int, frame):
        """'n' - a human says this is not a hazard. Records a fingerprint
        crop of the spot so a later scan can tell if it's since changed
        (Step 4) - see fingerprint_changed().
        """
        entry = self.get(entry_id)
        if entry is None:
            return None
        entry.state = HAZARD_STATE_DISMISSED
        entry.fingerprint = crop_bbox(frame, entry.bbox)
        entry.fingerprint_bbox = entry.bbox
        entry.absent_scans = 0
        entry.changed_scans = 0
        return entry

    def apply_scan_candidates(
        self, raw_candidates: list, frame, frame_diagonal: float, is_first_scan_cycle: bool
    ) -> ScanDiffResult:
        """The periodic room scan's entire "detect new"/"detect removed"
        mechanism (CLAUDE.md decision 3), run once per scan cadence tick.

        `raw_candidates` is this scan's occupied-spot bboxes, already
        person-suppressed (see generate_scan_candidates), NOT yet
        deduplicated - deduplication happens here first (Step 3's
        duplicate-candidate fix).

        Sequence:
          1. Dedupe overlapping candidates from this one scan pass.
          2. Match each candidate against known SCAN-origin entries
             (any state, including DISMISSED - a dismissed spot is still a
             "known spot" for matching purposes). A match refreshes
             position and resets the absence counter; a match against a
             DISMISSED entry additionally checks fingerprint_changed(), and
             re-raises to PENDING once the spot has looked materially
             different for DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED
             scans IN A ROW - a single noisy scan bumps the counter but
             does not itself re-raise; a scan that reads unchanged resets
             the counter to 0.
          3. Every known SCAN-origin entry NOT matched this cycle gets its
             absence counter bumped; once that counter reaches
             SCAN_CONSECUTIVE_SCANS_REQUIRED, the entry is removed.
          4. Every candidate NOT matched to a known entry is checked against
             last cycle's provisional (seen-once) list. A second consecutive
             sighting promotes it to a real PENDING entry ("arrived");
             anything else becomes this cycle's new provisional list.
        """
        candidates = dedupe_candidates(raw_candidates)
        scan_entries = [e for e in self.entries if e.origin == HAZARD_ORIGIN_SCAN]

        matched_ids: set[int] = set()
        reraised: list[HazardEntry] = []
        unmatched_candidates: list = []

        for candidate in candidates:
            pool = [e for e in scan_entries if e.id not in matched_ids]
            match = find_match(candidate, pool, frame_diagonal, HAZARD_MATCH_IOU_THRESHOLD, HAZARD_MATCH_CENTER_DIST_FRAC)
            if match is None:
                unmatched_candidates.append(candidate)
                continue
            matched_ids.add(match.id)
            match.bbox = candidate
            match.last_seen = time.monotonic()
            match.hits += 1
            match.absent_scans = 0
            if match.state == HAZARD_STATE_DISMISSED:
                # Compare against the STABLE fingerprint_bbox, NOT `candidate`
                # (this scan's own fresh, possibly-jittering segmentation
                # box) - see fingerprint_changed's docstring for why using
                # the wobbling candidate box was the actual root cause of
                # near-constant false re-raises, not just occasional noise.
                if fingerprint_changed(match.fingerprint, frame, match.fingerprint_bbox, label=f"entry #{match.id}"):
                    match.changed_scans += 1
                else:
                    # Read as unchanged this scan - whatever tripped the
                    # comparison before (noise, a lighting blip) didn't
                    # persist, so it's not evidence of a real swap.
                    match.changed_scans = 0
                if match.changed_scans >= DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED:
                    match.state = HAZARD_STATE_PENDING
                    # A re-raise is, by definition, not part of the room's
                    # original starting state - the parent already judged
                    # this spot once, so it alerts like any other later
                    # arrival.
                    match.is_first_scan = False
                    match.fingerprint = None
                    match.fingerprint_bbox = None
                    match.changed_scans = 0
                    reraised.append(match)

        removed: list[HazardEntry] = []
        for entry in scan_entries:
            if entry.id in matched_ids:
                continue
            entry.absent_scans += 1
            if entry.absent_scans >= SCAN_CONSECUTIVE_SCANS_REQUIRED:
                removed.append(entry)
        if removed:
            removed_ids = {e.id for e in removed}
            self.entries = [e for e in self.entries if e.id not in removed_ids]

        arrived: list[HazardEntry] = []
        remaining_provisional = list(self._provisional)
        new_provisional: list[_Provisional] = []
        for candidate in unmatched_candidates:
            match = find_match(
                candidate, remaining_provisional, frame_diagonal,
                HAZARD_MATCH_IOU_THRESHOLD, HAZARD_MATCH_CENTER_DIST_FRAC,
            )
            if match is not None:
                remaining_provisional = [p for p in remaining_provisional if p is not match]
                entry = self._new_entry(SCAN_HAZARD_LABEL, candidate, HAZARD_ORIGIN_SCAN, match.is_first_scan)
                arrived.append(entry)
            else:
                new_provisional.append(_Provisional(bbox=candidate, is_first_scan=is_first_scan_cycle))
        self._provisional = new_provisional

        return ScanDiffResult(arrived=arrived, reraised=reraised, removed=removed)


def hazard_alerts_on_approach(entry: HazardEntry) -> bool:
    """CLAUDE.md decision 4's table, as code:

        CONFIRMED                            -> alert
        PENDING, arrived after the first scan -> alert (unreviewed = unknown
                                                  = treated as dangerous)
        PENDING, from the first scan          -> no alert (room's normal
                                                  starting state, human
                                                  present reviewing it)
        DISMISSED                             -> never alert
    """
    if entry.state == HAZARD_STATE_CONFIRMED:
        return True
    if entry.state == HAZARD_STATE_PENDING:
        return not entry.is_first_scan
    return False


# --- Human review queue ------------------------------------------------------


class ReviewQueue:
    """A live FIFO of HazardEntry ids awaiting an 'h'/'n' decision. Can
    receive new items at any time (from either detector, for as long as the
    camera runs) - there is no fixed startup list and no "review is closed"
    state, per CLAUDE.md decision 3.
    """

    def __init__(self):
        self._ids: deque[int] = deque()

    def __len__(self) -> int:
        return len(self._ids)

    def enqueue(self, entry_id: int) -> None:
        self._ids.append(entry_id)

    def current_id(self):
        return self._ids[0] if self._ids else None

    def advance(self) -> None:
        """'h'/'n' - the current item has been decided, move to the next."""
        if self._ids:
            self._ids.popleft()

    def skip_remaining(self) -> None:
        """'s' - clear everything currently queued. The underlying
        HazardEntry objects are untouched (still PENDING, still alert-
        eligible per hazard_alerts_on_approach if they arrived after the
        first scan) - only their place in the review queue is dropped.
        Future arrivals still enqueue normally afterward.
        """
        self._ids.clear()

    def discard(self, entry_id: int) -> None:
        """Remove one specific id from wherever it sits in the queue - used
        when the scan's own removal logic clears a HazardEntry that was
        still awaiting review.
        """
        if entry_id in self._ids:
            self._ids = deque(i for i in self._ids if i != entry_id)


# --- Periodic room scan: candidate generation -------------------------------


def generate_scan_candidates(frame, seg_model, person_boxes, imgsz: int, device: str) -> list:
    """One scan cycle's occupied-spot candidates: runs
    measure_segmentation.py's segment_frame()/apply_filter() (imported
    unmodified) against `frame`, person-suppressed using THIS script's own
    already-computed person boxes (no second person-detection model). Called
    on a cadence from main(), not once at startup - CLAUDE.md decision 3.

    Returns a plain list of (x1, y1, x2, y2) bboxes in `frame`'s own pixel
    coordinates.
    """
    segments = segment_frame(seg_model, frame, imgsz, device)
    person_dicts = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in person_boxes]
    filtered = seg_apply_filter(
        segments, person_dicts, MIN_AREA_FRAC, MAX_AREA_FRAC, PERSON_OVERLAP_THRESHOLD,
        MIN_EXTENT, y_containment_frac=None, min_aspect_ratio=None,
    )
    return [(seg["x1"], seg["y1"], seg["x2"], seg["y2"]) for seg in filtered]


# --- Layer B: person tracking + proximity scoring ---------------------------


@dataclass
class PersonEntry:
    id: int
    bbox: tuple[float, float, float, float]
    last_seen: float


class PersonTracker:
    """A minimal nearest-match tracker, NOT a real multi-object tracker (no
    motion model, no optimal assignment, no re-identification after
    occlusion) - adequate for a single-camera proximity signal, flagged as a
    known simplification rather than presented as more than it is. Exists so
    Layer B's rolling window can be keyed per (person, hazard) pair across
    frames instead of resetting every frame's smoothing.
    """

    def __init__(self):
        self.entries: list[PersonEntry] = []
        self._next_id = 1

    def update(self, detection_boxes, frame_diagonal: float) -> list[PersonEntry]:
        now = time.monotonic()
        claimed_ids: set[int] = set()
        live: list[PersonEntry] = []

        for box in detection_boxes:
            match = find_match(
                box, [e for e in self.entries if e.id not in claimed_ids], frame_diagonal,
                PERSON_MATCH_IOU_THRESHOLD, PERSON_MATCH_CENTER_DIST_FRAC,
            )
            if match is not None:
                match.bbox = box
                match.last_seen = now
                claimed_ids.add(match.id)
                live.append(match)
            else:
                entry = PersonEntry(id=self._next_id, bbox=box, last_seen=now)
                self._next_id += 1
                self.entries.append(entry)
                claimed_ids.add(entry.id)
                live.append(entry)

        self.entries = [e for e in self.entries if now - e.last_seen <= PERSON_STALE_SECONDS]
        return live


def score_frame(live_persons, hazard_entries, frame_diagonal: float, rolling_windows: dict):
    """Layer B core: for every live person x every ALERT-ELIGIBLE hazard-map
    entry (caller filters by hazard_alerts_on_approach before calling this),
    compute the normalized center distance, push it into that pair's rolling
    window, and classify the smoothed distance into a zone.

    Returns (per_person, frame_risk):
      per_person: {person_id: (HazardEntry, smoothed_distance, zone)} - each
        live person's nearest hazard by smoothed distance.
      frame_risk: the single highest-ranked zone this frame across every
        pair, for the corner-text readout and for `/risk_status` (Phase 7) -
        carries hazard_id/hazard_bbox so a client can identify and highlight
        the specific hazard without drawing its own connector line.
    """
    per_person: dict[int, tuple] = {}
    frame_risk = {
        "zone": RISK_ZONE_NONE, "value": None, "person_id": None, "hazard_label": None,
        "hazard_id": None, "hazard_bbox": None,
    }

    for person in live_persons:
        best = None
        for hazard in hazard_entries:
            raw_distance = normalized_center_distance(person.bbox, hazard.bbox, frame_diagonal)
            key = (person.id, hazard.id)
            window = rolling_windows[key]
            window.append(raw_distance)
            smoothed = sum(window) / len(window)
            zone = classify_zone(smoothed)

            if best is None or smoothed < best[1]:
                best = (hazard, smoothed, zone)
            if zone_rank(zone) > zone_rank(frame_risk["zone"]):
                frame_risk = {
                    "zone": zone, "value": smoothed, "person_id": person.id, "hazard_label": hazard.label,
                    "hazard_id": hazard.id, "hazard_bbox": hazard.bbox,
                }

        if best is not None:
            per_person[person.id] = best

    return per_person, frame_risk


# --- Phase 5: alert lifecycle -----------------------------------------------


@dataclass
class AlertEvent:
    """One thing a parent needs to be told about, with an identity that
    survives across frames - the replacement for the pre-Phase-5 single
    overwritable alert_text/alert_until slot. Two kinds share this shape:

      PROXIMITY  - a (person, hazard) pair sitting in a scored zone.
        Opened the first time a pair enters yellow/orange/red, held open by
        ALERT_HOLD_SECONDS of hysteresis (see AlertManager.update_proximity),
        closed after that long unseen. `person_id`/`peak_zone` are
        meaningful; `reason` is not.
      NEW_OBJECT - a hazard-map entry that just appeared/re-raised
        (arrival, wall-socket, dismissal re-raise - CLAUDE.md decision 4).
        A one-shot pulse, not held open (there is no "approaching" phase to
        hold on to). `reason` names which mechanism raised it (e.g. "room
        scan", "spot changed since dismissal") for the banner text;
        `person_id`/`peak_zone` are not meaningful.
    """

    id: int
    kind: str  # ALERT_KIND_PROXIMITY / ALERT_KIND_NEW_OBJECT
    person_id: object  # None for NEW_OBJECT
    hazard_id: int
    hazard_label: str
    hazard_bbox: tuple
    zone: str  # current zone (PROXIMITY only; RISK_ZONE_NONE for NEW_OBJECT)
    peak_zone: str  # highest zone rank reached during this event's life
    started_at: float
    last_seen_at: float
    ended_at: float = None
    clip_triggered: bool = False
    reason: str = ""  # only meaningful for NEW_OBJECT


@dataclass
class AlertSignal:
    """One state transition an AlertEvent just made, for the caller to react
    to (play a sound, show a banner, maybe start a clip). "opened" and
    "escalated" both mean "tell the parent, right now"; "closed" means the
    event's hysteresis window has elapsed with no re-sighting - it exists so
    a caller COULD react to it (e.g. clearing a banner early) but nothing
    currently does, since the banner already times out on its own via
    ALERT_BANNER_SECONDS.
    """

    event: AlertEvent
    kind: str  # "opened" / "escalated" / "closed"


class AlertManager:
    """Turns score_frame's per-frame per_person dict, plus one-shot
    new-object proposals from HazardMap, into AlertEvents with real
    identity, cooldown, and dedup - see the ALERT_HOLD_SECONDS comment above
    for why a naive "alert every frame a pair is in a zone" or "alert only
    on the none->red edge" design were both rejected, with the measured
    evidence.

    Pure state machine: no I/O, no camera, no audio, no file writes - the
    caller (main()) is responsible for turning returned AlertSignals into an
    actual sound/banner/clip. This split is what makes it testable the same
    way HazardMap is (synthetic in-memory objects, no camera required).
    """

    def __init__(self):
        self._proximity_events: dict[tuple, AlertEvent] = {}
        self._next_id = 1
        self._last_clip_time = float("-inf")

    def update_proximity(self, per_person: dict, now: float) -> list[AlertSignal]:
        """Call once per frame with score_frame's per_person dict
        ({person_id: (HazardEntry, smoothed_distance, zone)} - already
        narrowed to each live person's NEAREST hazard, same simplification
        the debug connector line already relies on, per the 2026-08-22
        decision log entry: the nearest hazard is always the highest-risk
        one by construction, so no signal is lost by only tracking pairs
        that appear here).

        Returns every AlertSignal this frame produced, in no particular
        order: "opened" for a pair entering a scored zone for the first
        time, "escalated" for a pair reaching a HIGHER zone than it has
        reached so far this event (yellow->orange, orange->red, or opening
        straight into a high zone), "closed" for any event whose pair has
        gone unseen in a scored zone for more than ALERT_HOLD_SECONDS.
        De-escalation (red->orange while still within the hold window) and
        an unchanged/lower re-sighting update the event's bookkeeping
        (last_seen_at, current bbox/zone) silently, with no signal - this is
        the hysteresis itself, not a missing case.
        """
        signals: list[AlertSignal] = []

        for person_id, (hazard, _smoothed, zone) in per_person.items():
            if zone == RISK_ZONE_NONE:
                continue
            key = (person_id, hazard.id)
            event = self._proximity_events.get(key)
            if event is None:
                event = AlertEvent(
                    id=self._next_id, kind=ALERT_KIND_PROXIMITY, person_id=person_id,
                    hazard_id=hazard.id, hazard_label=hazard.label, hazard_bbox=hazard.bbox,
                    zone=zone, peak_zone=zone, started_at=now, last_seen_at=now,
                )
                self._next_id += 1
                self._proximity_events[key] = event
                signals.append(AlertSignal(event=event, kind="opened"))
            else:
                event.hazard_bbox = hazard.bbox
                event.last_seen_at = now
                event.zone = zone
                if zone_rank(zone) > zone_rank(event.peak_zone):
                    event.peak_zone = zone
                    signals.append(AlertSignal(event=event, kind="escalated"))

        for key in list(self._proximity_events.keys()):
            event = self._proximity_events[key]
            if now - event.last_seen_at > ALERT_HOLD_SECONDS:
                event.ended_at = now
                del self._proximity_events[key]
                signals.append(AlertSignal(event=event, kind="closed"))

        return signals

    def open_new_object(self, hazard_entry: HazardEntry, reason: str, now: float) -> AlertSignal:
        """A hazard-map arrival/re-raise - always a fresh, one-shot event
        (never matched against a previous one; there is nothing to
        de-duplicate against since HazardMap itself is the source of truth
        for "is this the same physical object").
        """
        event = AlertEvent(
            id=self._next_id, kind=ALERT_KIND_NEW_OBJECT, person_id=None,
            hazard_id=hazard_entry.id, hazard_label=hazard_entry.label, hazard_bbox=hazard_entry.bbox,
            zone=RISK_ZONE_NONE, peak_zone=RISK_ZONE_NONE, reason=reason,
            started_at=now, last_seen_at=now, ended_at=now,
        )
        self._next_id += 1
        return AlertSignal(event=event, kind="opened")

    def should_trigger_clip(self, signal: AlertSignal, now: float) -> bool:
        """CLAUDE.md decision 6: ONLY critical (red) alerts trigger a clip
        save, and only once per event (an event that flickers red/orange
        within its hold window - the exact behaviour ALERT_HOLD_SECONDS was
        measured against - must not produce a second clip). Decide-and-
        commit, like HazardMap.confirm(): calling this marks the event as
        having used its one clip attempt, whether or not
        CLIP_MIN_INTERVAL_SECONDS' global cooldown actually allows the save
        this time. A blocked attempt is deliberately not retried later in
        the same event - the cooldown is a disk-safety backstop, not a
        queue, and an event already sitting in RED has already been
        clip-recorded by definition once this method has run for it.

        `now` is an explicit parameter, like every other method on this
        class and on HazardMap/PersonTracker - kept out of the wall clock so
        the cooldown is exercisable with synthetic time in a test.
        """
        event = signal.event
        if event.kind != ALERT_KIND_PROXIMITY:
            return False
        if signal.kind not in ("opened", "escalated"):
            return False
        if event.peak_zone != "red":
            return False
        if event.clip_triggered:
            return False
        event.clip_triggered = True
        if now - self._last_clip_time < CLIP_MIN_INTERVAL_SECONDS:
            return False
        self._last_clip_time = now
        return True


# Cross-signal severity ranking used by AlertArbiter to pick the most
# important of several signals competing inside one pacing window (Shaked,
# 2026-08-26). Higher number = more urgent. Exactly the three tiers from
# CLAUDE.md decision 4/6's own alert vocabulary - immediate danger, getting
# close, a newly noticed hazard - not a made-up scale.
ALERT_PRIORITY_RED = 3
ALERT_PRIORITY_GETTING_CLOSE = 2
ALERT_PRIORITY_NEW_OBJECT = 1


def alert_priority(signal: AlertSignal) -> int:
    event = signal.event
    if event.kind == ALERT_KIND_PROXIMITY:
        return ALERT_PRIORITY_RED if event.zone == "red" else ALERT_PRIORITY_GETTING_CLOSE
    return ALERT_PRIORITY_NEW_OBJECT


class AlertArbiter:
    """Global pacing gate in front of the actual voice/banner output -
    AlertManager still decides WHETHER a (person, hazard) pair's state
    genuinely changed (opened/escalated/closed, with its own per-pair
    hysteresis); this decides whether THIS PARTICULAR MOMENT is when the
    parent should actually be interrupted about it. See
    GLOBAL_ALERT_MIN_INTERVAL_SECONDS' comment for the live-test evidence
    this was built from.

    RED is exempt from the pacing entirely (Shaked, 2026-08-26: immediate
    danger must never wait its turn, and it also always clears out anything
    currently held - a stale "getting close" is not worth surprise-firing
    right after a RED interrupt). Everything else is capped at one spoken
    alert per GLOBAL_ALERT_MIN_INTERVAL_SECONDS: whichever candidate is
    HIGHEST PRIORITY (ties broken toward the most recent) during a blocked
    window wins once the window reopens, rather than whichever happened to
    arrive first chronologically - lower-priority alternatives seen during
    that window are dropped, not queued for later.

    Pure state, no I/O - testable the same way AlertManager is.
    """

    def __init__(self):
        self._last_voiced_at = float("-inf")
        self._held: AlertSignal = None

    def offer(self, signal: AlertSignal, now: float):
        """Call for every 'opened'/'escalated' signal ('closed' signals
        never reach here - callers already skip those before voicing
        anything). Returns the AlertSignal to voice right now (which may be
        THIS signal, or a higher-priority one that was already being held),
        or None if this signal was held back to respect the pacing window -
        in which case the caller must still call poll() every frame so a
        held signal isn't lost forever if nothing newer arrives to trigger
        this method again.
        """
        if alert_priority(signal) == ALERT_PRIORITY_RED:
            self._held = None
            self._last_voiced_at = now
            return signal

        if now - self._last_voiced_at >= GLOBAL_ALERT_MIN_INTERVAL_SECONDS:
            winner = signal
            if self._held is not None and alert_priority(self._held) > alert_priority(signal):
                winner = self._held
            self._held = None
            self._last_voiced_at = now
            return winner

        if self._held is None or alert_priority(signal) >= alert_priority(self._held):
            self._held = signal
        return None

    def poll(self, now: float):
        """Call once per frame. Releases a held-back signal once the pacing
        window has elapsed, if nothing has already claimed it via offer().
        """
        if self._held is not None and now - self._last_voiced_at >= GLOBAL_ALERT_MIN_INTERVAL_SECONDS:
            winner = self._held
            self._held = None
            self._last_voiced_at = now
            return winner
        return None


def audio_for_signal(signal: AlertSignal):
    """Which pre-recorded voice clip (if any) a signal should play, per
    CLAUDE.md decision 4's three-clip set. "closed" signals never speak -
    hysteresis ending is bookkeeping, not something worth interrupting a
    parent for.
    """
    event = signal.event
    if signal.kind == "closed":
        return None
    if event.kind == ALERT_KIND_NEW_OBJECT:
        return AUDIO_HAZARD_DETECTED
    if event.zone == "red":
        return AUDIO_IMMEDIATE_DANGER
    if event.zone in ("yellow", "orange"):
        return AUDIO_BABY_GETTING_CLOSE
    return None


def banner_text_for_signal(signal: AlertSignal) -> str:
    event = signal.event
    if event.kind == ALERT_KIND_NEW_OBJECT:
        return f"New object detected ({event.reason}): {event.hazard_label}"
    return f"RISK {event.zone.upper()}: {event.hazard_label} approaching (person #{event.person_id})"


class RollingBuffer:
    """CLAUDE.md decision 6's "last ~5 seconds always in memory," JPEG-
    encoded rather than raw - see ROLLING_BUFFER_JPEG_QUALITY's comment for
    the measured memory tradeoff (~15MB vs. ~467MB for 5s at full 1080p).
    Frames are appended AFTER hazard/person boxes are drawn but BEFORE any
    diagnostic overlay (FPS, model config, risk readout, connector line) -
    CLAUDE.md decision 1: boxes are product, diagnostics are pixels that
    must never reach a served frame. main() enforces the ordering; this
    class just stores whatever numpy array it's given.
    """

    def __init__(self, fps: float, seconds: float = ROLLING_BUFFER_SECONDS, quality: int = ROLLING_BUFFER_JPEG_QUALITY):
        maxlen = max(1, round(fps * seconds))
        self._frames: deque[tuple[float, "np.ndarray"]] = deque(maxlen=maxlen)
        self._quality = quality

    def __len__(self) -> int:
        return len(self._frames)

    def append(self, frame, timestamp: float):
        """Encode and store `frame`. Returns the encoded JPEG bytes (so the
        caller can hand the same encoding straight to ClipRecorder's live
        tail instead of re-encoding), or None if encoding failed.
        """
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return None
        self._frames.append((timestamp, encoded))
        return encoded

    def snapshot(self) -> list[tuple[float, "np.ndarray"]]:
        """A plain-list copy of what's currently buffered. A copy, not a
        live view, because ClipRecorder keeps appending its own post-trigger
        tail onto the snapshot while this deque keeps rolling forward for
        the NEXT possible trigger - the two must not alias.
        """
        return list(self._frames)


class ClipRecorder:
    """CLAUDE.md decision 6: on a critical alert, stitch the rolling
    buffer's preceding ~5s plus ~2s of live tail into a saved clip file.

    Encoding+writing measured at ~130-200ms for a 7s clip
    (docs/decision-log.md, 2026-08-22 Phase 5 entry) - done inline in the
    frame loop that is 2-3 whole frame budgets at 15 FPS, a visible hitch
    exactly like the thing Phase 4's crisis was about. Writing therefore
    runs on a background thread (see poll()); this class's own bookkeeping
    (which clips are mid-tail, when their tail completes) is plain
    synchronous state, deliberately kept separate from the threaded part so
    it stays testable without spawning real threads.
    """

    def __init__(self, output_dir: str, fps: float, tail_seconds: float = CLIP_TAIL_SECONDS):
        self._output_dir = output_dir
        self._fps = fps
        self._tail_seconds = tail_seconds
        self._pending: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def trigger(self, buffer_snapshot: list, event: AlertEvent, now: float) -> None:
        """Start a tail capture: `buffer_snapshot` is the pre-trigger ~5s
        already pulled from RollingBuffer.snapshot(). The caller keeps
        calling add_tail_frame() every subsequent frame until poll() reports
        the tail window has elapsed.
        """
        self._pending.append({
            "frames": list(buffer_snapshot),
            "event": event,
            "deadline": now + self._tail_seconds,
        })

    def add_tail_frame(self, encoded_jpeg, timestamp: float) -> None:
        for record in self._pending:
            if timestamp <= record["deadline"]:
                record["frames"].append((timestamp, encoded_jpeg))

    def poll(self, now: float) -> list[str]:
        """Call once per frame. Finalizes (spawns a background write thread
        for) any pending recording whose tail window has elapsed. Returns
        the destination paths of clips just started this call - the path is
        known immediately even though the file itself is written
        asynchronously.
        """
        ready = [r for r in self._pending if now >= r["deadline"]]
        self._pending = [r for r in self._pending if now < r["deadline"]]
        paths = []
        for record in ready:
            paths.append(self._start_write(record))
        return paths

    def _start_write(self, record: dict) -> str:
        event = record["event"]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_event{event.id}_{event.hazard_label}.mp4"
        path = os.path.join(self._output_dir, filename)
        jpeg_frames = [buf for _, buf in record["frames"]]
        thread = threading.Thread(target=write_clip, args=(path, jpeg_frames, self._fps), daemon=True)
        thread.start()
        return path


def write_clip(path: str, jpeg_frames: list, fps: float) -> bool:
    """Decode a list of JPEG-encoded frames and write them out as one mp4.
    A module-level function (not a ClipRecorder method) specifically so it
    can be called directly and synchronously in a test - ClipRecorder always
    calls it on a background thread, but the encode/decode/write logic
    itself has no thread-only behaviour worth hiding from a test.

    Tries CLIP_FOURCC_PRIMARY (avc1/H.264) first, falls back to
    CLIP_FOURCC_FALLBACK (mp4v) if the platform's OpenCV build can't open an
    avc1 writer - see CLIP_FOURCC_PRIMARY's comment for the measured
    size/cost tradeoff. Returns True if a clip was actually written.
    """
    if not jpeg_frames:
        print(f"Warning: no frames to write for {path} - skipping.")
        return False
    first = cv2.imdecode(jpeg_frames[0], cv2.IMREAD_COLOR)
    if first is None:
        print(f"Warning: could not decode first frame for {path} - skipping.")
        return False
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*CLIP_FOURCC_PRIMARY), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*CLIP_FOURCC_FALLBACK), fps, (width, height))
    if not writer.isOpened():
        print(f"Warning: VideoWriter failed to open for {path} (tried {CLIP_FOURCC_PRIMARY} and {CLIP_FOURCC_FALLBACK}).")
        return False
    written = 0
    for buf in jpeg_frames:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            writer.write(img)
            written += 1
    writer.release()
    print(f"Clip saved: {path} ({written}/{len(jpeg_frames)} frames)")
    return True


class AudioPlayer:
    """Fire-and-forget voice-clip playback via macOS's built-in `afplay`,
    launched with subprocess.Popen (NOT subprocess.run/.call - measured,
    docs/decision-log.md 2026-08-22 Phase 5 entry: run() blocked for ~1.9s
    on a 1.0s clip, which at 15 FPS is ~29 dropped frames and would
    reproduce Phase 4's FPS-collapse crisis for every single alert. Popen's
    OWN call cost was measured at 2-5ms, ~4-7% of one frame's 66.7ms budget,
    with zero frames over budget in a simulated loop firing sounds
    mid-frame). AppKit's NSSound was also measured and rejected: its
    .play() call itself cost up to 112ms, over one full frame budget on its
    own.

    This class's non-blocking property was verified by direct measurement
    (a real, reproducible script), not by an automated regression test in
    test_risk_engine.py - deliberately, so the test suite stays fast and
    silent rather than launching real audio playback on every run. This
    mirrors how ROLLING_WINDOW_SIZE/SCAN_CONSECUTIVE_SCANS_REQUIRED are
    flagged in-code as measured-or-not-yet rather than silently assumed;
    what IS unit-tested here is the missing-file path, which is pure logic.
    """

    def __init__(self, audio_dir: str):
        self._audio_dir = audio_dir
        self._warned: set[str] = set()

    def play(self, filename: str) -> None:
        path = os.path.join(self._audio_dir, filename)
        if not os.path.isfile(path):
            if filename not in self._warned:
                print(f"Warning: audio clip not found, skipping playback: {path}")
                self._warned.add(filename)
            return
        try:
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            if filename not in self._warned:
                print(f"Warning: could not launch afplay for {path}: {exc}")
                self._warned.add(filename)


# --- CLI parsing -------------------------------------------------------------


def parse_seed_hazard(raw: str) -> tuple[float, float, float, float, str]:
    """Parse one `--seed-hazard x,y,w,h,label` value into an (x1, y1, x2,
    y2, label) tuple in pixel coordinates of the capture resolution. `label`
    is everything after the fourth comma, taken verbatim.
    """
    parts = raw.split(",", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--seed-hazard must be 'x,y,w,h,label' (5 comma-separated values), got {raw!r}"
        )
    x_str, y_str, w_str, h_str, label = parts
    try:
        x, y, w, h = (float(x_str), float(y_str), float(w_str), float(h_str))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--seed-hazard x,y,w,h must all be numbers, got {raw!r} ({exc})")
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"--seed-hazard label is empty in {raw!r}")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"--seed-hazard w,h must both be positive, got {raw!r}")
    return (x, y, x + w, y + h, label)


# --- overlay drawing ---------------------------------------------------------


def draw_label(image, text: str, origin: tuple[int, int], color) -> None:
    for c, thickness in ((OVERLAY_OUTLINE, 3), (color, 1)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, thickness)


# cv2.putText's origin is the text BASELINE, not a bounding-box corner -
# glyphs are drawn extending UPWARD from it. A label positioned "N px above"
# an anchor near the top of the frame (e.g. a hazard box's top edge, or a
# person box's) therefore has an origin - and so nearly all of its own
# pixels - off-canvas, making it invisible. Reported live, 2026-08-26 (a
# follow-up to the same day's white-outline fix): "still can't see some of
# the white marks... make sure it doesn't disappear above or below the
# screen." Confirmed directly against the recording: multiple hazard boxes
# near the top of frame had visibly missing top borders AND missing state
# labels in the same frame.
LABEL_TOP_CLEARANCE = 14


def label_anchor_y(anchor_y: int, frame_height: int, above_offset: int, below_offset: int) -> int:
    """Pick a y-coordinate for a label normally drawn `above_offset` px
    above `anchor_y`, flipped to `below_offset` px below it instead if
    there isn't enough room above (see LABEL_TOP_CLEARANCE) - rather than
    clamping to y=0, which is where the invisible-label bug came from in
    the first place. The flipped position is also clamped against the
    frame's bottom edge, so a label is never placed off-canvas on either
    side.
    """
    if anchor_y - above_offset >= LABEL_TOP_CLEARANCE:
        return anchor_y - above_offset
    return min(frame_height - 1, anchor_y + below_offset)


def hazard_box_color(entry: HazardEntry):
    if entry.state == HAZARD_STATE_PENDING:
        return PENDING_FIRST_SCAN_BOX_COLOR if entry.is_first_scan else PENDING_ARRIVED_BOX_COLOR
    return HAZARD_STATE_BOX_COLOR[entry.state]


def hazard_state_tag(entry: HazardEntry) -> str:
    if entry.state == HAZARD_STATE_CONFIRMED:
        return "HAZARD"
    if entry.state == HAZARD_STATE_DISMISSED:
        return "DISMISSED"
    return "PENDING" if entry.is_first_scan else "NEW-UNREVIEWED"


def draw_hazard_box(image, entry: HazardEntry, is_review_candidate: bool) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in entry.bbox)
    frame_height, frame_width = image.shape[:2]
    color = hazard_box_color(entry)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if is_review_candidate:
        # An extra white outline so "this is the one awaiting your h/n/s
        # keypress right now" is unmistakable from any other pending box.
        # Drawn with a dark halo first (same two-pass technique draw_label
        # already uses for text) - a bare 1px white line with nothing behind
        # it disappears against a light background (the water heater, the
        # tile floor) and against video compression, which was reported
        # live 2026-08-22 ("couldn't see the white lines") and confirmed by
        # reading this exact code: unlike every text label in this file,
        # this rectangle had no halo.
        #
        # Coordinates are clamped to the frame's own bounds (not just
        # offset by -2/+2) - a box near any edge would otherwise have this
        # OUTWARD-expanding outline push part of itself off-canvas, drawing
        # nothing for that side (reported live 2026-08-26: boxes near the
        # top of frame had visibly missing top borders).
        ox1, oy1 = max(0, x1 - 2), max(0, y1 - 2)
        ox2, oy2 = min(frame_width - 1, x2 + 2), min(frame_height - 1, y2 + 2)
        cv2.rectangle(image, (ox1, oy1), (ox2, oy2), OVERLAY_OUTLINE, 3)
        cv2.rectangle(image, (ox1, oy1), (ox2, oy2), REVIEW_CANDIDATE_OUTLINE_COLOR, 1)
    label_y = label_anchor_y(y1, frame_height, above_offset=8, below_offset=16)
    draw_label(image, f"[{hazard_state_tag(entry)}] {entry.label}", (x1, label_y), color)


def draw_person_box(image, person: PersonEntry) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in person.bbox)
    frame_height = image.shape[0]
    cv2.rectangle(image, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 2)
    label_y = label_anchor_y(y1, frame_height, above_offset=8, below_offset=16)
    draw_label(image, f"person #{person.id}", (x1, label_y), PERSON_BOX_COLOR)


def draw_connector(image, person: PersonEntry, hazard: HazardEntry, zone: str) -> None:
    p_center = tuple(int(round(v)) for v in bbox_center(person.bbox))
    h_center = tuple(int(round(v)) for v in bbox_center(hazard.bbox))
    cv2.line(image, p_center, h_center, ZONE_COLORS.get(zone, ZONE_COLORS[RISK_ZONE_NONE]), 2)


def review_candidate_label(position: int, total: int) -> str:
    """Text shown ON the box currently awaiting a decision - 1-based, since
    that's how a parent would count. `total` is the current queue length
    (the queue is live and can grow while a decision is pending, so this is
    NOT a fixed "N of a startup batch" count).
    """
    return f"REVIEW {position}/{total} - h=hazard n=not s=skip queued"


def draw_risk_readout(image, frame_risk: dict, hazard_map: HazardMap, review_queue: ReviewQueue) -> None:
    zone = frame_risk["zone"]
    if zone == RISK_ZONE_NONE or frame_risk["person_id"] is None:
        risk_text = "RISK: none (no person-hazard pair in a scored zone)"
    else:
        risk_text = (
            f"RISK: {zone.upper()}  person #{frame_risk['person_id']} vs "
            f"{frame_risk['hazard_label']}  (normalized dist {frame_risk['value']:.2f})"
        )
    draw_overlay_line(image, risk_text, 3)

    n_confirmed = sum(1 for e in hazard_map.entries if e.state == HAZARD_STATE_CONFIRMED)
    n_pending_first = sum(
        1 for e in hazard_map.entries if e.state == HAZARD_STATE_PENDING and e.is_first_scan
    )
    n_pending_new = sum(
        1 for e in hazard_map.entries if e.state == HAZARD_STATE_PENDING and not e.is_first_scan
    )
    n_dismissed = sum(1 for e in hazard_map.entries if e.state == HAZARD_STATE_DISMISSED)
    draw_overlay_line(
        image,
        f"Hazard map: {len(hazard_map.entries)} entries ({n_confirmed} confirmed, "
        f"{n_pending_first} pending/first-scan, {n_pending_new} pending/NEW [alerts], "
        f"{n_dismissed} dismissed)  |  review queue: {len(review_queue)}",
        4,
    )


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: live risk overlay - a per-frame YOLO pass plus a "
        "periodic class-agnostic room scan propose hazard candidates; a human "
        "confirms/dismisses them ('h'/'n'/'s'); Layer B scores child proximity "
        "against confirmed hazards and unreviewed new arrivals."
    )
    parser.add_argument("--index", type=int, default=None, help=f"OpenCV camera device index (default: {DEFAULT_CAMERA_INDEX} if neither --index nor --name is given). Ignored if --name is also given.")
    parser.add_argument("--name", type=str, default=None, help="Select the camera by a substring of its device name (e.g. 'Arducam') instead of a numeric index. Wins over --index if both are given.")
    parser.add_argument("--device", type=str, default="mps", help="Inference device: 'mps' (default, Apple GPU) or 'cpu'.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help=f"Minimum detection confidence to consider a box (default: {DEFAULT_CONF_THRESHOLD}).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"YOLO26 weights to run (default: {DEFAULT_MODEL}).")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"Inference resolution (default: {DEFAULT_IMGSZ}).")
    parser.add_argument(
        "--seed-hazard", type=parse_seed_hazard, action="append", default=[], metavar="x,y,w,h,label",
        dest="seed_hazards",
        help="Add a fixed, CONFIRMED hazard-map entry at startup, in pixel coordinates of the ACTUAL "
        "capture resolution. Repeatable. E.g. '--seed-hazard 200,400,300,200,stove'. For deterministic "
        "testing of zone escalation without a real hazard in frame.",
    )
    parser.add_argument(
        "--disable-socket-detect", dest="socket_detect_enabled", action="store_false", default=True,
        help="Turn off the open-vocabulary wall-socket pass entirely (no second model is even loaded).",
    )
    parser.add_argument("--socket-model", type=str, default=DEFAULT_SOCKET_MODEL, help=f"Open-vocabulary weights for wall-socket detection (default: {DEFAULT_SOCKET_MODEL}).")
    parser.add_argument("--socket-prompts", type=str, default=",".join(DEFAULT_SOCKET_PROMPTS), help=f"Comma-separated text prompt(s) for the socket pass (default: {','.join(DEFAULT_SOCKET_PROMPTS)}).")
    parser.add_argument("--socket-conf", type=float, default=DEFAULT_SOCKET_CONF, help=f"Minimum confidence for a socket detection (default: {DEFAULT_SOCKET_CONF}).")
    parser.add_argument("--socket-imgsz", type=int, default=DEFAULT_SOCKET_IMGSZ, help=f"Inference resolution for the socket pass (default: {DEFAULT_SOCKET_IMGSZ}).")
    parser.add_argument("--socket-scan-interval", type=float, default=DEFAULT_SOCKET_SCAN_INTERVAL_SECONDS, help=f"Wall-clock seconds between socket-detection passes (default: {DEFAULT_SOCKET_SCAN_INTERVAL_SECONDS}).")
    parser.add_argument(
        "--disable-scan", dest="scan_enabled", action="store_false", default=True,
        help="Turn off the periodic class-agnostic room scan entirely (no segmentation model is even "
        "loaded) - named-class/socket detection only. CLAUDE.md decision 3 expects this to run "
        "continuously by default; this flag exists mainly for isolating the named-class path.",
    )
    parser.add_argument("--scan-interval", type=float, default=DEFAULT_SCAN_INTERVAL_SECONDS, help=f"Wall-clock seconds between periodic room scans (default: {DEFAULT_SCAN_INTERVAL_SECONDS}).")
    parser.add_argument("--scan-seg-model", type=str, default=DEFAULT_SEG_MODEL, help=f"Class-agnostic segmentation weights for the room scan (default: {DEFAULT_SEG_MODEL}).")
    parser.add_argument("--scan-imgsz", type=int, default=DEFAULT_SEG_IMGSZ, help=f"Inference resolution for the room scan (default: {DEFAULT_SEG_IMGSZ}).")
    parser.add_argument("--clips-dir", type=str, default=DEFAULT_CLIPS_DIR, help=f"Directory saved critical-alert clips are written to, relative to CWD unless absolute (default: {DEFAULT_CLIPS_DIR}).")
    parser.add_argument("--audio-dir", type=str, default=DEFAULT_AUDIO_DIR, help=f"Directory containing the pre-recorded alert voice clips (default: {DEFAULT_AUDIO_DIR}).")
    parser.add_argument(
        "--disable-audio", dest="audio_enabled", action="store_false", default=True,
        help="Turn off voice-clip playback entirely (visual alerts and clip-saving still run).",
    )
    args = parser.parse_args()

    socket_prompts = [p.strip() for p in args.socket_prompts.split(",") if p.strip()]
    if args.socket_detect_enabled and not socket_prompts:
        print("Error: --socket-prompts produced no usable prompts.")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading {args.model} (imgsz={args.imgsz}) ...")
    model = load_model(args.model, device)
    print("Model loaded.")

    class_name_to_id = {name: idx for idx, name in model.names.items()}
    person_class_id = class_name_to_id.get("person")
    if person_class_id is None:
        print(
            "Error: the loaded model has no 'person' class - CLAUDE.md decision 2 requires person "
            "detection to come from the same single pass as hazards. Refusing to run without it."
        )
        return

    hazard_class_id_to_label: dict[int, str] = {}
    for class_name, hazard_label in HAZARD_LABEL_BY_CLASS_NAME.items():
        class_id = class_name_to_id.get(class_name)
        if class_id is None:
            print(f"Warning: model has no '{class_name}' class - the '{hazard_label}' hazard group will be missing that source class this run.")
            continue
        hazard_class_id_to_label[class_id] = hazard_label

    socket_model = None
    if args.socket_detect_enabled:
        print(f"Loading {args.socket_model} for wall-socket detection (prompts={socket_prompts}) ...")
        socket_model, socket_model_type = load_open_vocab_model(args.socket_model, device, socket_prompts)
        print(f"{args.socket_model} loaded as {socket_model_type}.")
    else:
        print("Socket detection disabled (--disable-socket-detect).")

    scan_model = None
    if args.scan_enabled:
        print(f"Loading {args.scan_seg_model} for the periodic room scan (imgsz={args.scan_imgsz}) ...")
        scan_model = load_seg_weights(FastSAM, args.scan_seg_model, device)
        print(f"{args.scan_seg_model} loaded.")
    else:
        print("Periodic room scan disabled (--disable-scan).")

    try:
        camera = CameraCapture(index=args.index, name=args.name)
    except CameraSelectionError as exc:
        print(str(exc))
        return

    ok, first_frame = camera.verify_startup()
    if not ok:
        print(startup_failure_message(camera.index))
        camera.release()
        return

    height, width = first_frame.shape[:2]
    frame_diagonal = math.hypot(width, height)
    device_label = camera.resolved_name or f"index {camera.index}"
    print(f"Streaming from camera {camera.index} ({device_label}) at {width}x{height}. Press 'q' to quit.")

    hazard_map = HazardMap()
    for x1, y1, x2, y2, label in args.seed_hazards:
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"Warning: --seed-hazard '{label}' bbox extends outside the {width}x{height} capture frame - added anyway.")
        hazard_map.add_seed((x1, y1, x2, y2), label)
        print(f"Seeded hazard '{label}' (CONFIRMED) at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}).")

    review_queue = ReviewQueue()
    person_tracker = PersonTracker()
    rolling_windows: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW_SIZE))

    last_socket_scan_time = time.monotonic() - args.socket_scan_interval
    last_room_scan_time = time.monotonic() - args.scan_interval
    room_scan_index = 0
    # Flips True the moment the first periodic room scan completes - used to
    # tag NAMED-origin proposals as is_first_scan (see the module docstring
    # and HazardEntry.is_first_scan). A coarse proxy for named-class
    # entries specifically (they're evaluated every frame, not on the scan's
    # own cadence) - flagged explicitly, not silently assumed exact.
    first_scan_done = not args.scan_enabled

    alert_text = None
    alert_until = 0.0

    def raise_alert(text: str) -> None:
        nonlocal alert_text, alert_until
        print(f"ALERT: {text}")
        alert_text = text
        alert_until = time.monotonic() + ALERT_BANNER_SECONDS

    # Phase 5: alert lifecycle, rolling buffer, clips, audio - see the
    # AlertManager/RollingBuffer/ClipRecorder/AudioPlayer docstrings for the
    # measured reasoning behind each. rolling_buffer/clip_recorder use
    # DEFAULT_TARGET_FPS (a fixed, measured steady-state number), not the
    # live smoothed_fps, for the reason given on that constant.
    alert_manager = AlertManager()
    alert_arbiter = AlertArbiter()
    audio_player = AudioPlayer(args.audio_dir) if args.audio_enabled else None
    rolling_buffer = RollingBuffer(fps=DEFAULT_TARGET_FPS)
    clip_recorder = ClipRecorder(args.clips_dir, fps=DEFAULT_TARGET_FPS)

    def speak(signal: AlertSignal) -> None:
        audio_name = audio_for_signal(signal)
        if audio_name is not None and audio_player is not None:
            audio_player.play(audio_name)
        raise_alert(banner_text_for_signal(signal))

    def handle_alert_signal(signal: AlertSignal, now: float) -> None:
        if signal.kind == "closed":
            return
        # Clip-saving is intentionally NOT gated by the arbiter below: a
        # critical moment is worth recording even on a frame where we chose
        # not to re-announce it audibly because something else just spoke
        # (GLOBAL_ALERT_MIN_INTERVAL_SECONDS) - should_trigger_clip has its
        # own independent once-per-event/30s-cooldown gate already.
        if alert_manager.should_trigger_clip(signal, now):
            clip_recorder.trigger(rolling_buffer.snapshot(), signal.event, now)
            print(f"Critical alert - recording clip for event #{signal.event.id} ({signal.event.hazard_label}).")
        voiced = alert_arbiter.offer(signal, now)
        if voiced is not None:
            speak(voiced)

    def enqueue_and_maybe_alert(entry: HazardEntry, alert_reason: str) -> None:
        review_queue.enqueue(entry.id)
        if not entry.is_first_scan:
            now = time.monotonic()
            handle_alert_signal(alert_manager.open_new_object(entry, alert_reason, now), now)

    smoothed_fps = None
    last_frame_time = time.monotonic()
    # Used by the 'n' (dismiss) key handler below so a dismissal on a frame
    # where the camera happened to drop a read still gets a real fingerprint
    # crop, instead of silently no-op'ing until the next successful read.
    last_valid_frame = first_frame

    cv2.imshow(WINDOW_NAME, prepare_for_display(first_frame))
    cv2.waitKey(1)

    try:
        for frame in camera.frames():
            if frame is not None:
                last_valid_frame = frame
                now = time.monotonic()

                results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)

                person_boxes: list[tuple[float, float, float, float]] = []
                hazard_detections: list[tuple[str, tuple]] = []
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        bbox = tuple(float(v) for v in box.xyxy[0])
                        if cls_id == person_class_id:
                            person_boxes.append(bbox)
                        elif cls_id in hazard_class_id_to_label:
                            hazard_detections.append((hazard_class_id_to_label[cls_id], bbox))

                for label, bbox in hazard_detections:
                    entry, created = hazard_map.propose_named(bbox, label, frame_diagonal, is_first_scan=not first_scan_done)
                    if created:
                        enqueue_and_maybe_alert(entry, "named detection")

                if socket_model is not None and now - last_socket_scan_time >= args.socket_scan_interval:
                    socket_results = socket_model.predict(frame, conf=args.socket_conf, imgsz=args.socket_imgsz, device=device, verbose=False)
                    socket_boxes = socket_results[0].boxes
                    if socket_boxes is not None:
                        for box in socket_boxes:
                            bbox = tuple(float(v) for v in box.xyxy[0])
                            entry, created = hazard_map.propose_named(bbox, SOCKET_HAZARD_LABEL, frame_diagonal, is_first_scan=not first_scan_done)
                            if created:
                                enqueue_and_maybe_alert(entry, "wall socket")
                    last_socket_scan_time = now

                if scan_model is not None and now - last_room_scan_time >= args.scan_interval:
                    coverage = person_coverage_frac(person_boxes, width, height)
                    if coverage > SCAN_PERSON_COVERAGE_SKIP_FRAC:
                        # Deferred, not skipped for good: retried on the next
                        # regular cadence tick (this is Layer A, not
                        # latency-critical - waiting one more interval for a
                        # person to clear the shot is an acceptable, simpler
                        # tradeoff over busy-polling every frame).
                        last_room_scan_time = now
                    else:
                        raw_candidates = generate_scan_candidates(frame, scan_model, person_boxes, args.scan_imgsz, device)
                        diff = hazard_map.apply_scan_candidates(
                            raw_candidates, frame, frame_diagonal, is_first_scan_cycle=(room_scan_index == 0)
                        )
                        for entry in diff.arrived:
                            enqueue_and_maybe_alert(entry, "room scan")
                        for entry in diff.reraised:
                            enqueue_and_maybe_alert(entry, "spot changed since dismissal")
                        for entry in diff.removed:
                            review_queue.discard(entry.id)
                        room_scan_index += 1
                        if room_scan_index == 1:
                            first_scan_done = True
                        last_room_scan_time = now

                live_persons = person_tracker.update(person_boxes, frame_diagonal)

                live_person_ids = {p.id for p in live_persons}
                for key in list(rolling_windows.keys()):
                    if key[0] not in live_person_ids:
                        del rolling_windows[key]

                alert_eligible_hazards = [e for e in hazard_map.entries if hazard_alerts_on_approach(e)]
                per_person, frame_risk = score_frame(live_persons, alert_eligible_hazards, frame_diagonal, rolling_windows)

                for signal in alert_manager.update_proximity(per_person, now):
                    handle_alert_signal(signal, now)

                annotated = frame.copy()

                review_candidate_id = review_queue.current_id()
                for hazard in hazard_map.entries:
                    draw_hazard_box(annotated, hazard, is_review_candidate=(hazard.id == review_candidate_id))
                for person in live_persons:
                    draw_person_box(annotated, person)

                # Phase 5 buffer/clip capture point: boxes only, no
                # diagnostics, no connector line yet - CLAUDE.md decision 1
                # (boxes are product, everything drawn below this point is
                # developer-only debug overlay that must never reach a
                # served frame). This is also exactly the "clean annotated
                # frame" Phase 7's /video_feed is meant to be built from,
                # one phase early, for free.
                encoded = rolling_buffer.append(annotated, now)
                if encoded is not None:
                    clip_recorder.add_tail_frame(encoded, now)
                for path in clip_recorder.poll(now):
                    print(f"Clip write started: {path}")
                held = alert_arbiter.poll(now)
                if held is not None:
                    speak(held)

                # --- everything below is the LOCAL cv2.imshow debug window
                # only (connector line + diagnostics) - exempt from decision
                # 1's pixels-vs-JSON split; see CLAUDE.md decision 1.
                for person in live_persons:
                    nearest = per_person.get(person.id)
                    if nearest is not None:
                        hazard, _smoothed, zone = nearest
                        draw_connector(annotated, person, hazard, zone)

                now = time.monotonic()
                instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-6)
                last_frame_time = now
                smoothed_fps = (
                    instantaneous_fps if smoothed_fps is None
                    else FPS_SMOOTHING_ALPHA * instantaneous_fps + (1 - FPS_SMOOTHING_ALPHA) * smoothed_fps
                )

                draw_overlay_line(annotated, f"FPS: {smoothed_fps:.1f}", 1)
                draw_overlay_line(annotated, f"{args.model}  imgsz={args.imgsz}  conf={args.conf}  {device}", 2)
                draw_risk_readout(annotated, frame_risk, hazard_map, review_queue)

                if review_candidate_id is not None:
                    candidate_entry = hazard_map.get(review_candidate_id)
                    if candidate_entry is not None:
                        cx1, cy1, cx2, cy2 = (int(round(v)) for v in candidate_entry.bbox)
                        # above_offset=24 clears the state label (drawn 8px
                        # above the same box top); below_offset=40 keeps the
                        # same clearance when both flip below near the top
                        # edge, so the two labels never land on each other.
                        review_label_y = label_anchor_y(cy1, height, above_offset=24, below_offset=40)
                        draw_label(
                            annotated, review_candidate_label(1, len(review_queue)),
                            (cx1, review_label_y), REVIEW_CANDIDATE_OUTLINE_COLOR,
                        )

                if alert_text is not None and now < alert_until:
                    draw_overlay_line(annotated, f"ALERT: {alert_text}", 5)

                cv2.imshow(WINDOW_NAME, prepare_for_display(annotated))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                entry_id = review_queue.current_id()
                if entry_id is not None:
                    hazard_map.confirm(entry_id)
                    review_queue.advance()
                    print(f"Confirmed hazard entry #{entry_id}.")
            elif key == ord("n"):
                entry_id = review_queue.current_id()
                if entry_id is not None:
                    hazard_map.dismiss(entry_id, last_valid_frame)
                    review_queue.advance()
                    print(f"Dismissed entry #{entry_id} (not a hazard).")
            elif key == ord("s"):
                if len(review_queue) > 0:
                    print(f"Skipping {len(review_queue)} currently-queued candidate(s).")
                review_queue.skip_remaining()

            if key == ord("q"):
                print("Quit key pressed - exiting.")
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed - exiting.")
                break

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C) - exiting.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
