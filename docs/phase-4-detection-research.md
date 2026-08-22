# Phase 4 detection research — supervision, broadened open-vocab prompts, and Grounding DINO

**Status: research/measurement only, delivered separately from the concurrent
Phase 4 Slice 1 integration work in `cv/risk_engine.py`. Nothing here has been
integrated into any running pipeline.** This document exists to answer three
questions Shaked raised after seeing Phase 4 Slice 1 running with only
`person`/`refrigerator`/`oven`/`microwave` in scope:

1. Is `github.com/roboflow/supervision` worth adopting, and for what?
2. Does broadening the open-vocabulary prompt list (already proven to work for
   `wall socket`, per CLAUDE.md decision 7) help on the hazard concepts Shaked
   actually cares about, measured honestly against the existing held-out set?
3. Do heavier zero-shot detectors (Grounding DINO, generally regarded as
   stronger than YOLO-World/YOLOE at grounding) do meaningfully better on the
   same test, and is running one locally even feasible on this machine?

Same discipline as Phase 3: measured against the **32 held-out
`cv/captures/*home*_raw.jpg` frames** (a building the system was never trained
or tuned on — see `docs/phase-3-step0-findings.md`), not a training-adjacent
split, and every number below is reproducible from a script and a CSV, not
asserted from memory. Where a script or CSV lives outside this repo (see
"Where the evidence actually lives," below), that's stated plainly, following
the same evidentiary-tier honesty the Phase 3 write-up insisted on.

---

## 1. `github.com/roboflow/supervision` — what it actually is, and the verdict

**Confirmed by reading its README and the actual installed source (not just
skimming marketing copy):** supervision is a **tracking/annotation/zone-logic
toolkit that wraps the output of a detection model — it is not a detector
itself.** Its own README states this directly: *"Supervision was designed to
be model agnostic... plug in any classification, detection, or segmentation
model."* It never runs inference; it consumes an existing model's boxes
(`sv.Detections.from_ultralytics(results)`) and does things to them
afterward — filter, track, annotate, count in a zone. Installed into a scratch
venv (`pip install supervision==0.30.0`, not this repo's own `.venv` or
`cv/requirements.txt`) and both classes below were loaded and instantiated to
confirm the API actually works on this machine, not just read from source.

### (a) ByteTrack as a replacement for `risk_engine.py`'s `PersonTracker`

Read `cv/risk_engine.py`'s `PersonTracker` class in full for comparison. As
its own docstring says: **"a minimal nearest-match tracker, NOT a real
multi-object tracker (no motion model, no Hungarian/optimal assignment, no
re-identification after occlusion)."** Concretely, each frame it does one
greedy IoU-or-center-distance match per detection against existing tracks
(`find_match()`), and drops any track not matched for `PERSON_STALE_SECONDS`
(1.0s wall-clock) — there is no prediction step, so a track that goes fully
unmatched for even a few frames (a hand passing in front of the child, brief
occlusion behind a hazard) either survives on luck (the next detection
happens to land within `PERSON_MATCH_CENTER_DIST_FRAC` of the last known box)
or is dropped and reappears as a **new** person ID, silently resetting every
rolling-window smoothed distance keyed to the old ID (per the module's own
"risk should reset when they leave and return" comment — which is correct
behavior for someone leaving the room, but wrong behavior for someone
transiently occluded for half a second).

`supervision.ByteTrack` (read `supervision/tracker/byte_tracker/core.py` in
full, both from GitHub main and the installed package) is a real
implementation of the published ByteTrack algorithm:

- A **Kalman filter** predicts each track's next-frame position, so a track
  that goes undetected for a frame or two still has a sensible predicted box
  to match against, instead of just sitting at its last known position.
- **Two-stage association** — high-confidence detections are matched first,
  then low-confidence detections are given a second chance against whatever's
  left unmatched, which is ByteTrack's specific contribution over vanilla SORT
  (recovers detections a naive confidence-threshold would have discarded).
- A **`lost_track_buffer`** (default 30 frames) keeps a track alive as "lost"
  rather than deleting it immediately on the first missed frame, and
  `remove_duplicate_tracks()` merges a track re-found after a brief gap back
  into its original ID via a real Hungarian-style `linear_assignment`, not a
  single nearest-neighbor check.

**This directly addresses the exact gap `risk_engine.py`'s own docstring
names** — "no re-identification after occlusion" — with a real, published,
widely-used mechanism instead of hand-rolled nearest-match. For Layer B
specifically (the moment-to-moment child-proximity signal that actually fires
alerts), a child briefly occluded behind furniture or a parent's body and then
reappearing would currently get a fresh ID and a reset rolling window; with
ByteTrack it would very likely keep the same ID and continuum of smoothed
distance, which is a real behavioral improvement, not a marginal one.

**One important, non-obvious finding: `sv.ByteTrack` is itself deprecated as
of `supervision==0.28.0`** (confirmed by the loaded class's own docstring,
which fires a `FutureWarning` on instantiation) **and scheduled for removal in
0.31.0, in favor of a separate `trackers` package (`pip install trackers`,
`ByteTrackTracker`, roughly the same algorithm, moved out of the main
library).** Anyone adopting this needs to target the `trackers` package
directly, not `supervision.ByteTrack`, or land on a pinned pre-0.31 supervision
version — this is exactly the kind of "pin deliberately, don't silently
follow latest" situation `cv/requirements.txt`'s own header comment already
warns about for this project's other dependencies.

**Recommendation: adopt later, not now, and target `trackers.ByteTrackTracker`
specifically when it happens.** Two reasons to defer rather than do it
immediately:

1. Phase 4 Slice 1 (per `docs/decision-log.md`, 2026-08-12) is deliberately
   scoped to "ship pure proximity first, watch real zone thresholds live,
   revisit after" — swapping the tracker is a second, independent change and
   mixing it into the same slice makes it harder to attribute any live-testing
   surprise to the right cause.
2. `PersonTracker`'s current occlusion weakness hasn't actually been observed
   causing a problem yet (Slice 1 hasn't had a live occlusion test run) — this
   is a plausible, well-reasoned prediction of a future gap, not yet a
   measured one. Per CLAUDE.md's own preamble ("ask before changing, evidence
   over habit"), swapping a working (if simple) mechanism before a real
   failure is observed would be doing exactly what Phase 3's postmortem
   criticized fine-tuning round 3 for: fixing a problem nobody had measured
   yet, in isolation, off of a hunch rather than a live-system finding.

**When it's worth doing:** the moment Phase 4's live testing (already the
team's stated next move per Phase 3's closure) shows a child's tracked ID
resetting on brief occlusion in a way that visibly breaks the rolling-window
smoothing — at that point, swap `PersonTracker` for `trackers.ByteTrackTracker`
directly, not `sv.ByteTrack`.

### (b) `PolygonZone` for a future "reachable surface" region

Read `supervision/detection/tools/polygon_zone.py` in full. It's a small,
well-scoped utility: define a polygon (`np.ndarray` of `(x, y)` points), and
`.trigger(detections)` returns a boolean array of which detections have their
chosen anchor point (default: `Position.BOTTOM_CENTER`, i.e. where an object
actually touches the floor/surface — the right anchor for "is this on the
counter" rather than a box's raw center) inside that polygon. It does the
point-in-polygon test via a cached boolean mask, no fancy geometry library
needed.

This maps cleanly onto decision 3's "reachable surface" concept (floor, low
table, low shelf) once someone actually defines those regions — likely during
the guided setup walkthrough itself (a parent could plausibly trace the
polygon on the video feed as part of that flow). **Not relevant to build now**
— Phase 4 Slice 1 has no walkthrough yet, and there's nothing to trigger a
zone against — but worth keeping in mind as the tool of choice when that
Slice happens, rather than hand-rolling point-in-polygon logic from scratch.

### Overall supervision verdict

| Piece | Verdict |
|---|---|
| `ByteTrack` / `trackers.ByteTrackTracker` | Adopt **later** — a real, meaningfully better occlusion-handling tracker than `PersonTracker`, but defer until live testing shows the current tracker's known weakness actually causing a problem, and target the `trackers` package (not the deprecated `sv.ByteTrack`) when it happens. |
| `PolygonZone` | Adopt **later** — the right tool for "reachable surface" regions, relevant once the guided walkthrough (not yet built) needs to define them. |
| supervision as a detector | **Not applicable** — it has none; it's not a competitor to YOLO/YOLO-World/Grounding DINO, it's plumbing around whichever of those a project already uses. |

---

## 2. Broadened open-vocabulary prompts, re-measured on the 32 held-out frames

### Method

Reused `cv/measure_openvocab.py` unchanged (per the task's constraint — no
modification to that script), against `cv/captures/*home*_raw.jpg` only (the
32-frame held-out set, **not** the full 215-image `cv/captures/` directory —
an earlier run without `--images` accidentally swept all captures, including
non-`home` frames from other sessions, and was discarded; the corrected
command is below). Both already-available local weights were used —
`yoloe-26s-seg.pt` and `yolov8s-worldv2.pt` — at imgsz {640, 1280}, conf floor
0.01 (records everything, thresholds applied afterward, same convention as
every other `measure_*.py` script in this repo).

```
python cv/measure_openvocab.py \
  --images "cv/captures/*home*_raw.jpg" \
  --models yoloe-26s-seg.pt,yolov8s-worldv2.pt \
  --prompts "small object,small item,tiny object,small object on the floor,sharp object,blade,knife,kitchen knife,scissors,cable,power cord,electrical cord,charging cable,wire,bottle,plastic bottle,medicine bottle,pill bottle,medication bottle,toy,small toy,toy block,cabinet handle,drawer handle,cabinet knob,handle,person" \
  --imgsz 640,1280 --conf 0.01 --device mps \
  --output cv/measurements/openvocab_phase4_homeframes.csv
```

Output: `cv/measurements/openvocab_phase4_homeframes.csv` (5,205 rows,
gitignored per this project's existing measurement-CSV convention — same as
every other file in `cv/measurements/`).

**Ground truth limitation, stated up front:** the 32 home frames' label files
only cover the two classes labelled during Phase 3 — `sharp_object` (knives,
scissors) and `small_swallowable` (keys, coins, batteries, bottle caps,
lighters, small toy parts) — per `docs/labelling-guide.md`. There is **no
ground truth at all** for cable, bottle, or cabinet/drawer-handle concepts in
this test set, because nobody has ever drawn those boxes. Scoring precision/
recall for those prompts is therefore not possible without new labelling,
which the task explicitly said not to do. What follows is genuine
precision/recall for the two prompt families that map onto existing ground
truth, and honest, named, checkable qualitative frame examples (not invented
numbers) for the ones that don't.

Scoring used the identical IoU≥0.5 greedy-match methodology as
`cv/evaluate_home_frames.py` (same repo, same rule, reimplemented in a scratch
scoring script rather than modifying that file — see "Where the evidence
actually lives"), pooling multiple synonym prompts against the one ground-
truth concept they're meant to represent:

- **`sharp_object` prompt family:** `sharp object`, `blade`, `knife`,
  `kitchen knife`, `scissors`
- **`small_swallowable` prompt family:** `small object`, `small item`,
  `tiny object`, `small object on the floor`, `toy`, `small toy`,
  `toy block`

### Results — pooled prompt family vs. ground truth, at conf ≥ 0.25 (the
standard operating threshold used throughout this project)

| concept | model | imgsz | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| sharp_object | yoloe-26s-seg | 640 | 13 | 37 | 49 | 0.260 | 0.210 |
| sharp_object | yoloe-26s-seg | 1280 | 24 | 106 | 38 | 0.185 | 0.387 |
| sharp_object | yolov8s-worldv2 | 640 | 13 | 21 | 49 | **0.382** | 0.210 |
| sharp_object | yolov8s-worldv2 | 1280 | 12 | 30 | 50 | 0.286 | 0.194 |
| small_swallowable | yoloe-26s-seg | 640 | 3 | 43 | 51 | 0.065 | 0.056 |
| small_swallowable | yoloe-26s-seg | 1280 | 4 | 82 | 50 | 0.047 | 0.074 |
| small_swallowable | yolov8s-worldv2 | 640 | 0 | 4 | 54 | 0.000 | 0.000 |
| small_swallowable | yolov8s-worldv2 | 1280 | 0 | 1 | 54 | 0.000 | 0.000 |

At looser thresholds (conf ≥ 0.05), sharp_object recall reaches **0.500–0.532**
(yolov8s-worldv2 imgsz 640: precision 0.142, recall 0.532), trading precision
down into the low teens — the same shape of tradeoff Phase 3's fine-tuning
rounds hit repeatedly. Full per-threshold table (conf 0.25/0.10/0.05 × both
models × both imgsz) is in the scoring script's printed output, reproducible
from the CSV.

**Bottom line for these two concepts, compared honestly against Phase 3's
fine-tuning results (round 2, the best fine-tune: precision 0.179 / recall
0.216 on `sharp_object`):** broadening the open-vocab prompt list to a
*family* of synonyms for `sharp_object`, at conf 0.25, gets **comparable
recall (0.21) at meaningfully better precision (0.382 vs 0.179)**, with **zero
training** — this is a genuine, measured improvement over the fine-tuning
approach for this one concept, using a mechanism this project can run today.
`small_swallowable`, on the other hand, **did not improve** — broadening the
prompt list produced worse numbers than fine-tuning round 1 (0.204 recall) on
every configuration tested. Open-vocab prompting is not a uniform win; it
depends on the concept.

### Which individual prompt phrasing actually did the work

Per-prompt breakdown (same scoring, single prompt vs. its concept's ground
truth, conf ≥ 0.25) shows the family-level number is carried almost entirely
by the two named-object prompts already known to work from Step 0, **not**
by the generic descriptive prompts added this session:

| prompt | best config | TP | FP | precision | recall |
|---|---|---|---|---|---|
| `knife` | yolov8s-worldv2, 1280 | 11 | 21 | 0.344 | 0.177 |
| `scissors` | yoloe-26s-seg, 1280 | 13 | 44 | 0.228 | 0.210 |
| `sharp object` | any config | 0 | 0–21 | 0.000 | 0.000 |
| `blade` | any config | 0 | 0 | 0.000 | 0.000 |
| `kitchen knife` | any config | 0 | 0 | 0.000 | 0.000 |
| `small object on the floor` | yoloe-26s-seg, 1280 | 4 | 78 | 0.049 | 0.074 |
| `small object` / `small item` / `tiny object` / `toy` / `small toy` / `toy block` | every config | 0 | 0–4 | 0.000 | 0.000 |

**This matches Phase 3's Step 0 finding exactly, and generalizes it:**
specific, concrete, named-object prompts (`knife`, `scissors`) work; abstract
descriptive prompts (`sharp object`, `small object`, `tiny object`, `blade`)
essentially don't — they either detect nothing at all, or (`sharp object` at
imgsz 1280) fire 21 times with zero true positives. Trying phrasing variants,
as the task asked, is what surfaced this: the fix for `sharp_object`'s
recall gap was never going to be a better abstract phrase for "sharp" — it
was always going to be enumerating concrete object names, exactly the way
`knife`/`scissors`/`wall socket` already worked before this session. The
concrete lesson for anyone extending this prompt list further: **spend effort
on more named nouns, not on more adjectives** (already-tried adjectives here
all failed: "sharp," "small," "tiny").

### Qualitative check on the ungrounded-truth prompts (cable/bottle/handle)

Since there's no ground truth to score against, three of the top-confidence
hits per prompt were pulled and visually inspected (crops saved with the
predicted box drawn on, in the scratch directory — see "Where the evidence
actually lives"). The most informative result:

**The exact same real object — a detached cabinet/drawer handle sitting flat
on a coffee table (frame `20260810-143042_home-60cm...`, also seen in frame
`20260810-143205_...`)** — was independently found and tightly boxed by
**three different prompts**: `charging cable` (conf 0.46), `drawer handle`
(conf 0.36), and `cabinet handle` (conf 0.26), plus a fourth prompt, `bottle`
(conf 0.20, frame `20260810-143036_...`), also boxing the same object. Two of
those four names (`drawer handle`, `cabinet handle`) are in fact
**correct** — it genuinely is a piece of cabinet hardware, just not attached
to a cabinet; the other two (`charging cable`, `bottle`) are wrong names for a
real object.

This is a directly useful, concrete confirmation of CLAUDE.md decision 7's
core reframe: **the model doesn't need to get the name right to be useful for
this project's actual job.** Four different text prompts converged on one
real "something is sitting on this reachable surface that shouldn't be
empty" signal — exactly what the guided walkthrough needs a model to
surface, independent of whether the label attached to the box is correct.
`power_cord`'s top hit, by contrast (0.09 confidence, the lowest of any
prompt checked), was a crop of nothing — visually indistinguishable rug/floor
texture, no real object in the box at all — a reminder that a lucky object-
agnostic hit at one prompt doesn't mean every prompt in that family is
reliable; the low-confidence end is still mostly noise, consistent with
everything else measured in this project so far.

---

## 3. Grounding DINO — a heavier zero-shot detector, measured the same way

### Why, and the feasibility question first

Grounding DINO (IDEA-Research, via HuggingFace `transformers`) is a phrase-
grounding transformer generally regarded in the field as stronger than
CLIP-based detectors like YOLO-World at open-vocabulary localization,
specifically because it fuses text and image features throughout the network
rather than only at a final classification step. The concrete question worth
answering before anything else: **does it even run on this machine, locally,
on MPS, per CLAUDE.md decision 8's no-cloud requirement?**

**Yes — confirmed by actually running it, not assumed.** `transformers==5.15.0`
+ `torch==2.13.0` were installed into a **separate scratch venv**
(`gdino_venv`, not this repo's `.venv` or `cv/requirements.txt` — per the
task's explicit instruction). Both `IDEA-Research/grounding-dino-tiny`
(~170M params) and `IDEA-Research/grounding-dino-base` (~230M params) loaded
and ran real inference on Apple MPS with no unsupported-op crash and no CPU
fallback needed. Weights download once from the HuggingFace Hub (a public
model repository, not a live API call sending frames anywhere) and all
inference after that is local — consistent with decision 8's "no cloud" rule
in the same way the Open Images public-data pull and the local Florence-2 VLM
test already reasoned about in Phase 3.

**One real API surprise worth flagging for anyone reusing this:**
`transformers`' `post_process_grounded_object_detection()` renamed its
`box_threshold` argument to `threshold` at some point between versions —
the first attempt at this script crashed on that exact mismatch. Worth
knowing before assuming any Grounding DINO code snippet found online will
run unmodified against the currently-pinned `transformers` version.

### Method

Both models were run against the same 32 held-out frames, with the same
broadened prompt list as Part 2 (minus `medication bottle`/`medicine bottle`,
merged for brevity — see the script), formatted as Grounding DINO's documented
period-separated phrase-grounding query (`"small object. small item. ...
person."`), `threshold=0.20`, `text_threshold=0.15`.

**A real, load-bearing quirk of this model, discovered while scoring, not
assumed going in:** Grounding DINO does **not** return one label per input
phrase the way YOLO-World's `set_classes()` does. When many overlapping short
phrases are fed in one query, its phrase-grounding output frequently returns
**compound labels** — multiple adjacent prompt phrases fused into a single
predicted string, e.g. one box labelled literally `"blade knife kitchen
knife"` (all three matched the same detected span). This is a genuine
practical drawback of reusing one long prompt list with this specific model
family: unlike YOLO-World/YOLOE's clean one-class-per-box output, a caller
has to do substring/keyword matching against the returned text rather than an
exact class-name lookup. Scoring below accounts for this explicitly (a
predicted box counts toward `sharp_object` if **any** of `knife`, `scissors`,
`blade`, `sharp object` appears anywhere in its — possibly compound —
label), and this is flagged as a looser matching rule than every other
scorer in this project, not silently normalized away.

### Results — precision/recall vs. the same ground truth, `grounding-dino-tiny`

| concept | conf threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| sharp_object | ≥0.30 | 31 | 90 | 31 | 0.256 | 0.500 |
| sharp_object | ≥0.20 | 49 | 398 | 13 | 0.110 | **0.790** |
| small_swallowable | ≥0.30 | 2 | 59 | 52 | 0.033 | 0.037 |
| small_swallowable | ≥0.20 | 35 | 404 | 19 | 0.080 | 0.648 |

`grounding-dino-base` (the larger checkpoint):

| concept | conf threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| sharp_object | ≥0.30 | 22 | 67 | 40 | 0.247 | 0.355 |
| sharp_object | ≥0.20 | 34 | 163 | 28 | 0.173 | 0.548 |
| small_swallowable | ≥0.30 | 1 | 0 | 53 | 1.000 | 0.019 |
| small_swallowable | ≥0.20 | 21 | 88 | 33 | 0.193 | 0.389 |

**This is the highest recall measured anywhere in this project on either
concept, by a wide margin** — 0.790 on `sharp_object` (tiny model, conf≥0.20)
against the previous best of 0.387 (broadened YOLO-World/YOLOE, Part 2) and
0.339 (fine-tuning round 2). 0.648 on `small_swallowable` against a previous
best of 0.204 (fine-tuning round 1). **The cost is precision, and it is
severe** — 0.110/0.080 at that same threshold means roughly 9 in 10 boxes are
wrong, which is worse than change detection's already-judged-insufficient
~33% (1-in-3). At the stricter conf≥0.30, precision recovers to a usable
0.256 range (comparable to the best fine-tuning/open-vocab results) while
recall stays genuinely higher (0.500 vs 0.387/0.210) than everything else
measured. **Grounding DINO's real advantage here is a strictly better
precision/recall tradeoff curve, not a magic fix for the precision problem
itself** — the same fundamental issue Phase 3 identified (this is a genuinely
hard single-frame judgment task) still holds; this model is simply better
positioned on that curve than what's been tried before.

**Interesting, worth stating plainly rather than assuming "bigger is
better":** the smaller `tiny` checkpoint **outperformed** the larger `base`
checkpoint on this specific test at matched thresholds (0.790 vs 0.548 recall
at conf≥0.20 on `sharp_object`; 0.648 vs 0.389 on `small_swallowable`) while
running measurably faster. This is plausibly an artifact of this particular
32-frame test set and a single fixed threshold pairing rather than a general
claim about tiny-vs-base Grounding DINO — but it's the honest result of the
actual measurement, not the expected one, and is reported as such rather than
assumed away or unreported because it doesn't match "bigger should be
better."

### Frame-by-frame: does it find what YOLO-World/YOLOE missed?

Checked directly, by name, per the task's instruction not to just eyeball
averages: every frame where broadened YOLO-World/YOLOE (Part 2, conf≥0.25)
correctly matched a `sharp_object` ground-truth box, Grounding DINO
(conf≥0.30) **also** matched it — 11 frames, zero misses in that direction.
**One additional frame, `20260810-143143_home-60cm_yolo26l_imgsz640_raw.jpg`,
was caught by Grounding DINO (label `"blade knife kitchen knife"`, confidence
0.5545) and missed by both YOLO-World and YOLOE at conf≥0.25.** The crop (see
"Where the evidence actually lives") shows a real knife, held in a hand mid-
motion — the exact "held in hand" scenario CLAUDE.md decision 7 already
documents stock COCO's `knife` handling well (0.48–0.885) and open-vocab
struggling with more, since it's competing against many other prompt phrases
in the same query rather than being the model's one dedicated class. No
frame was found where YOLO-World/YOLOE caught a true positive that Grounding
DINO missed — a strict, if narrow (32-frame), win.

### Speed — and why it's still a legitimate Layer A candidate despite being slow

| model | mean inference time / frame (MPS) |
|---|---|
| `grounding-dino-tiny` | ~496 ms |
| `grounding-dino-base` | ~637 ms |
| (comparison) `yoloe-26s-seg`, imgsz 640 | ~22 ms |
| (comparison) `yolov8s-worldv2`, imgsz 640 | ~13 ms |

Roughly **20–35× slower per frame** than either open-vocab option already in
use, and completely impractical for a per-frame Layer B pass. **This is not
disqualifying** — it's exactly the situation the Phase 3 write-up's "why the
alarm stays instant" section already establishes: anything serving Layer A
(the hazard map, checked once when something is placed, not every frame) can
legitimately be slow, because Layer A and Layer B never share a critical
path. Half a second to identify what just got placed on the counter, followed
by 32 minutes of the hazard map already knowing about it, is the identical
shape as that section's worked example. Grounding DINO is explicitly **not**
a candidate to replace the live YOLO26 pass that Layer B depends on every
frame; it's a candidate for whatever eventually confirms/labels a spot the
change-detection mechanism (or the guided walkthrough) has already flagged as
occupied.

---

## Recommendations

1. **`supervision`: don't adopt now.** Revisit `trackers.ByteTrackTracker`
   (not the deprecated `sv.ByteTrack`) specifically once live testing shows
   `PersonTracker`'s occlusion weakness actually causing a problem, per the
   team's own stated post-Phase-3 discipline of finding out what to fix by
   running the real system rather than pre-emptively fixing an unmeasured
   gap. Keep `PolygonZone` in mind for whenever the guided walkthrough needs
   to define reachable-surface regions.
2. **Broadened open-vocab prompts: adopt the `sharp_object` family
   (`knife` + `scissors`, the two prompts that actually carry the result —
   the added abstract prompts contributed nothing) as a genuine, no-training
   upgrade over the current class list**, with the explicit caveat that
   precision at usable thresholds is still in the 0.26–0.38 range, not
   "reliable." Do not adopt the `small_swallowable` broadening — it measured
   worse than what fine-tuning already produced and worse than doing
   nothing.
3. **Grounding DINO is the strongest single result across this whole
   research pass** — real, meaningfully higher recall than everything tried
   in Phase 3 or Part 2 of this document, at a precision/recall tradeoff that
   is at minimum competitive with the best prior result at matched
   precision, confirmed to run locally on this machine without a cloud call.
   **Recommend Phase 4 (or a dedicated follow-up) build a small object-
   agnostic "confirm what's in this spot" step using `grounding-dino-tiny`
   (the faster, and on this test, higher-recall checkpoint) that runs only
   when Layer A's change-detection mechanism has already flagged a candidate
   region — never per-frame, never in Layer B's path** — mirroring exactly
   how the abandoned Florence-2 VLM detour was scoped, but with a model that
   (unlike Florence-2's ungrounded mode) actually produced tight, correct
   boxes on the real object in the frames checked here, not confident boxes
   over empty space. This is a recommendation to build and measure further
   with live data, not a claim that the precision problem is solved — Phase
   3's whole closing argument (static single-method testing has hit its
   limit; the next real answer comes from running the integrated system) still
   applies here as much as it did to change detection.

## Blockers / things not done

- **No fine-tuned-vs-Grounding-DINO head-to-head on identical prompts** was
  run — fine-tuning round 2's classes (`sharp_object`, `small_swallowable`)
  are compared to Grounding DINO's raw text-prompt families above, which is
  the right comparison for "should we keep fine-tuning or use this instead,"
  but not a controlled ablation isolating prompt wording from model choice.
- **OWLv2 was not measured** — time was spent getting Grounding DINO's API
  quirks (the `threshold` rename, compound-label matching) right and running
  both its checkpoints properly rather than spreading effort across a third
  model shallowly. Worth a follow-up session if Grounding DINO is pursued
  further, specifically to check whether OWLv2's per-box single-label output
  (no compound-label issue, per its documented API) makes controlled scoring
  cleaner.
- **No new photography, no new labelling** — per the task's explicit
  constraint. The `small_swallowable`/cable/bottle/handle numbers above are
  therefore weaker evidence in exactly the way the write-up states: either
  genuinely poor (small_swallowable, measured against real ground truth) or
  unmeasurable against ground truth that doesn't exist yet (cable/bottle/
  handle, qualitative only).

---

---

## Part 4 (2026-08-12) — the guided-walkthrough question specifically:
which mechanism finds "every occupied spot," measured fresh

**Status: research/measurement only, same as every other part of this
document. No walkthrough code exists anywhere in this repo. `cv/risk_engine.py`
and `cv/test_risk_engine.py` were not touched for this section.**

CLAUDE.md decisions 3/4 need a mechanism that flags "this reachable-surface
spot is not empty" for a parent to confirm — not a mechanism that names or
judges the object, per decision 3's explicit reframe. Two candidates already
exist in this repo and neither had ever been measured for that specific job:
class-agnostic segmentation (`cv/measure_segmentation.py`, FastSAM) and
Grounding DINO with genuinely generic prompts (as opposed to the named-noun
prompts Part 3 above already measured). Both are re-measured here, fresh,
against a spread that deliberately includes frames Part 2/3 above never
touched (`*room2*_raw.jpg`, a different, darker area of the training
building — not the 32 `*home*_raw.jpg` frames alone), to avoid a rosy number
from the same heavily-reused test set.

### Method

Segmentation: `cv/measure_segmentation.py`, unmodified, default filter
(`MIN_AREA_FRAC=0.0003`, `MAX_AREA_FRAC=0.05`, `MIN_EXTENT=0.35`,
`PERSON_OVERLAP_THRESHOLD=0.3`), `--room2-sample 30` (up from the script's
own default 15, for a bigger qualitative sample):

```
python cv/measure_segmentation.py --room2-sample 30 \
  --output cv/measurements/segmentation_phase4.csv
```

Output: `cv/measurements/segmentation_phase4.csv` (62 rows: 32 home + 30
room2, gitignored, same convention as every `cv/measurements/*.csv`).

Grounding DINO, broad prompts: `cv/measure_grounddino.py`, unmodified,
`grounding-dino-tiny` (Part 3's own finding: faster AND higher-recall than
`-base` on this project's held-out set, so it is the one worth spending a
second measurement round on), with a prompt list containing **only** generic/
abstract phrasing — no named nouns at all, unlike every prompt list measured
anywhere earlier in this document:

```
python cv/measure_grounddino.py --model grounding-dino-tiny \
  --images "cv/captures/*home*_raw.jpg" \
  --prompts "object,item,thing,small item on a surface,object on the floor,object on a table,small object sitting on a surface,anything on the floor,person" \
  --conf 0.20 --text-threshold 0.15 --score-conf 0.30,0.20,0.10 \
  --output cv/measurements/gdino_broadprompts_home.csv

python cv/measure_grounddino.py --model grounding-dino-tiny \
  --images "cv/captures/*room2-60cm*_raw.jpg" \
  --prompts "object,item,thing,small item on a surface,object on the floor,object on a table,small object sitting on a surface,anything on the floor,person" \
  --conf 0.20 --text-threshold 0.15 --no-score \
  --output cv/measurements/gdino_broadprompts_room2_sample.csv
```

Outputs: `cv/measurements/gdino_broadprompts_home.csv` (932 rows, 32 frames),
`cv/measurements/gdino_broadprompts_room2_sample.csv` (858 rows, all 38
`room2-60cm` frames — not a sub-sample, every frame in that tag was run).
Both gitignored, same convention.

**A scoring caveat that matters and is stated up front, not buried:**
`measure_grounddino.py`'s built-in `score_against_ground_truth()` matches a
prediction to a concept (`sharp_object`/`small_swallowable`) by checking
whether the concept's keyword family (`knife`, `scissors`, `sharp object`,
etc.) appears as a literal substring of the predicted label. None of this
section's broad prompts contain those words, so that built-in scorer
structurally returns 0/0 for every threshold on every concept — not because
the model failed to localize the object, but because a label like `"object
thing object on the floor object on a table"` never contains the token
`"knife"`. That 0.000/0.000 table is real script output (see the raw run
above) but it is **not evidence of anything** about broad-prompt performance
— it is an artifact of a scorer built for a different, keyword-based
question. To get an honest number, a separate, class-agnostic scorer was
written (same rule as `measure_segmentation.py`'s own `match_recall()`: does
ANY predicted box, regardless of its label text, cover a ground-truth box at
IoU≥0.5) and run against `gdino_broadprompts_home.csv`. That is the number
reported below.

### Quantitative results — 32-frame held-out home set (116 ground-truth
boxes, `sharp_object` + `small_swallowable` pooled, class-agnostic — this is
the "partial proxy" signal the task anticipated existing ground truth could
give, not a full test of "any object" since the labels only cover two
concepts)

| mechanism | config | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| segmentation (FastSAM-s, imgsz 1024) | raw, unfiltered | 57 | — | 59 | — | **0.491** |
| segmentation (FastSAM-s, imgsz 1024) | filtered (default filter) | 28 | — | 88 | — | 0.241 |
| Grounding DINO tiny, broad prompts | conf≥0.30, class-agnostic | 25 | 111 | 91 | 0.184 | 0.216 |
| Grounding DINO tiny, broad prompts | conf≥0.20, class-agnostic | 105 | 827 | 11 | 0.113 | **0.905** |
| (for reference) Grounding DINO tiny, named nouns (Part 3) | conf≥0.20 | 49 | 398 | 13 | 0.110 | 0.790 |

(`measure_segmentation.py` does not compute a class-agnostic FP count in its
own summary output — its script docstring explains why: "scoring a hit on an
unlabelled real object as a false positive would be wrong," since the 32
home frames aren't exhaustively labelled for every real object in them, only
for the two Phase 3 concepts. The `FP` column is left blank for segmentation
rather than invented; the room2 per-frame box counts below are the real
substitute signal for "how much noise does a parent have to wade through.")

**The headline number that actually matters for this task, and it is not
recall:** broad-prompt Grounding DINO at conf≥0.20 produces the single
highest raw recall measured anywhere in this entire project (0.905, beating
even Part 3's named-noun result of 0.790) — but at a candidate volume that
makes that recall useless for a walkthrough, established next.

### Qualitative — does either produce a walkable candidate list?

**Segmentation, per-frame filtered candidate counts** (from
`segmentation_phase4.csv`, all 62 frames, not a subsample):

| split | frames | filtered candidates/frame | range |
|---|---|---|---|
| home (32 frames) | 32 | mean 33.8 | 10–79 |
| room2 (30 sampled) | 30 | mean 64.6 | 23–97 |

Named example: `20260810-143033_home-60cm_..._raw.jpg` has 10 real
ground-truth objects in it and the filter still leaves **79** candidate
boxes standing. `20260810-214630_room2-60cm_..._raw.jpg` (no ground truth,
but a real, ordinary living-room frame) produces **29** filtered candidates.
Three crops pulled from that exact frame and eyeballed directly: two are
sofa-arm/cushion fabric texture (`seg000`, `seg002`, both compact,
high-extent 0.62/0.72 blobs that pass the filter cleanly — the extent filter
was specifically designed to reject "long thin" structural noise, and these
are neither long nor thin, they're just furniture texture, a failure mode
the filter was never designed to catch), one is a wall-corner edge
(`seg001`). Crops from the home frame's own top of the filtered list include
one genuine true positive (`seg000`, a pair of scissors, tightly boxed) next
to a baseboard/wall-panel edge (`seg001`) and an ambiguous dark blob
(`seg002`). **This is not a walkable list.** A parent looking at 79 (or even
29) boxes per surface sweep, most of them furniture texture rather than
discrete objects, cannot tap through that "in seconds" the way decision 4
describes the review needing to feel — this directly reproduces, with fresh
frames and a real reproducible CSV, the same qualitative character the old
unreproduced Tier-2 number (recall 0.32) implied but never showed with
actual box counts.

**Grounding DINO, broad prompts, per-frame candidate counts:**

| split | conf floor | boxes/frame | range |
|---|---|---|---|
| home (32 frames) | ≥0.20 | mean 29.1 | 20–36 |
| home (32 frames) | ≥0.30 | mean 4.25 | 1–8 |
| room2-60cm (38 frames, full tag, not sampled) | ≥0.30 | mean 1.9 | 1–3 |

At conf≥0.20 the candidate count (mean 29/frame) is in the same unusable
range as unfiltered segmentation — expected, since that's also where the
0.905 recall number above comes from (a much denser, more granular sweep of
small boxes). At conf≥0.30 the count drops into what would, on its own, look
like a walkable range (1–8 boxes/frame) — but inspecting what those boxes
actually *are* changes the read entirely. Named, checkable examples from
`20260810-143046_home-60cm_..._raw.jpg` at conf≥0.30 (8 boxes total on this
frame): two are duplicate `person` boxes (expected, harmless — `person` was
kept in the prompt list as Part 2's control), but the rest include a box
labelled `"surface object on the object on a table"` covering **44.3%** of
the entire frame, and a second labelled `"anything on the floor"` covering
**56.8%** of the frame — both crops inspected directly, and both are
genuinely the whole coffee table and the whole floor/rug respectively, not a
specific item on either. Only 2 of the 8 boxes on that frame are
tightly-boxed, plausible single-item candidates. The room2 sample at the
same threshold is worse in this specific way, not better: **every single
frame's small surviving candidate set (1–3 boxes) consisted only of
duplicate `person` boxes and one giant `"anything on the floor"` box
covering 60–61% of the frame** — inspected across all three named frames
above (`20260810-214618`, `-214621`, `-214624`) — **zero** per-item
candidates survived at conf≥0.30 in that sample. One genuine tight true
positive was found elsewhere in the home set at this threshold — a knife on
`20260810-143033_home-60cm_..._raw.jpg`, labelled `"object thing object on
the floor object on a table"`, area 1.4% of frame, a real correct box — so
broad prompts are not *incapable* of a tight per-item hit, but they are
inconsistent about it in a way named-noun prompts (Part 3) were not: no
frame in this session's broad-prompt sample produced a walkable, mostly-real
candidate list the way decision 3 needs one to.

**The shared, load-bearing finding across both mechanisms:** there is no
single confidence/filter threshold, for either mechanism, where "candidate
count is small enough to tap through" and "candidates are mostly real
discrete objects rather than furniture texture or whole-surface catch-all
regions" are both true at once. Turning either mechanism's threshold up
trades the unusable-volume problem for an unusable-content problem (giant
regions or near-total misses), not for a clean middle ground — measured
directly here, not assumed.

### Speed

| mechanism | mean time / frame (MPS) |
|---|---|
| segmentation (person-detect + FastSAM combined) | 20.7ms + 95.5ms ≈ **116ms** |
| Grounding DINO tiny, broad prompts | **~545–565ms** (544.7ms room2 sample, 564.5ms home sample) |

Segmentation is roughly **5× faster**. Per this document's Part 3 reasoning
(reaffirmed, not repeated blindly): both are legitimate Layer-A/setup-time
candidates regardless of this gap, since neither runs in Layer B's per-frame
path — the walkthrough is a once-per-session, parent-present flow. But the
gap is not irrelevant either, exactly as the task flagged: decision 4
describes a 2–3 minute *total* walkthrough, which per decision 3 likely
means sweeping several reachable-surface regions (floor, low table, low
shelf) one at a time, not one single frame. At ~550ms/frame, sweeping even
5–6 distinct region-frames with Grounding DINO alone already costs 3+
seconds before any candidate is even shown to a parent — tolerable once, not
free, and materially worse than segmentation's ~120ms if either mechanism
needs several passes per region (e.g. one FastSAM "everything" pass plus a
per-candidate confirmation call).

### Recommendation

**Neither mechanism, as measured here, is ready to be the walkthrough's
sole detector — this section does not have a clean winner to hand off,
and says so plainly rather than picking one to seem decisive.** Both fail
the same test (a candidate list a parent can actually tap through in
seconds) in different ways:

- **Segmentation over-produces uniformly** — it never collapses to a small
  list at any measured setting; the "occupied spot" candidates are
  contaminated with furniture texture and structural edges the current
  extent/area filter does not catch (the two sofa-texture false positives
  above passed the filter cleanly — they are compact, high-extent blobs,
  which is exactly what the filter is designed to *keep*).
- **Broad-prompt Grounding DINO doesn't fix this — it changes shape,
  not size.** At the confidence floor needed to catch a comparable
  fraction of real objects (≥0.20), it produces the same 20–36
  boxes/frame problem segmentation has. At the floor where volume looks
  walkable (≥0.30), what survives is dominated by giant whole-frame region
  boxes ("anything on the floor," "surface... object on a table") that are
  actively *worse* for a per-item tap-to-confirm UI than too many small
  boxes would be — a parent cannot usefully confirm/deny "yes, this entire
  floor is a hazard."

**What the evidence does support, stated as plainly as the numbers allow:**
segmentation is the better base to keep iterating on, not Grounding DINO's
broad-prompt mode, for three specific, measured reasons: (1) its failure
mode is fixable by filter engineering — every false positive inspected here
is a real geometric region (furniture texture, an edge) that a better
area/extent/aspect-ratio filter could plausibly still separate from a
compact object, whereas Grounding DINO's giant-region failure mode is a
property of what the *model* considers "an object" under vague phrasing, not
something a downstream threshold can fix; (2) it is 5× faster, which
compounds across a multi-region sweep; (3) it needs no prompt engineering at
all, which matters because this task's own central finding (repeated a third
time now, after Phase 3's Step 0 and this document's Part 2) is that
descriptive/abstract text prompts are the weak link everywhere they've been
tried — "sharp object," "small object," "tiny object" failed for YOLO-World
(Part 2), and this session's broader "object"/"item"/"anything on the floor"
family failed differently but not less for Grounding DINO. A mechanism that
doesn't depend on getting a text phrase right removes an entire axis of
failure decision 3 doesn't need to accept.

**Concretely, for whoever picks this up next (explicitly not decided here —
this is a recommendation for a follow-up measurement round, not a shipped
architecture change to CLAUDE.md):** tighten `measure_segmentation.py`'s
filter past what this session tested — specifically an aspect-ratio/
elongation signal distinct from `extent` (the sofa-texture false positives
here are compact *and* high-extent, so `extent` alone cannot separate them;
they are large relative to a real handheld hazard, so a tighter
`MAX_AREA_FRAC` than the current 0.05 is the more promising first lever,
untested in this session) and re-measure candidate-count-per-frame as the
primary metric, not recall alone — recall was never segmentation's problem,
volume was. Grounding DINO's already-established role from Part 3 (a
per-candidate, named-noun confirmation step, never the primary generator,
never per-frame) stands unchanged by this session; broad/generic prompting
specifically is now a measured dead end for Grounding DINO, the same way it
already was for YOLO-World, just via a different failure shape (giant
regions instead of zero hits).

### Where this evidence lives

**Tier 1 (script + CSV both in this repo, unmodified scripts, rerunnable
as-is):**
- `cv/measure_segmentation.py` (unmodified) → `cv/measurements/segmentation_phase4.csv`
  (62 rows, gitignored, exact command above).
- `cv/measure_grounddino.py` (unmodified) → `cv/measurements/gdino_broadprompts_home.csv`
  (932 rows) and `cv/measurements/gdino_broadprompts_room2_sample.csv` (858
  rows), both gitignored, exact commands above.

**Tier 2 (a small scratch scorer, not committed to this repo, reimplementing
the same class-agnostic IoU≥0.5 rule `measure_segmentation.py`'s own
`match_recall()` already uses in this repo — described here in enough
detail, and using only this repo's already-committed ground-truth loader
`evaluate_home_frames.load_ground_truth`/`iou`, that anyone can reproduce it
exactly): the class-agnostic precision/recall table for broad-prompt
Grounding DINO above (TP/FP/FN at conf≥0.30 and ≥0.20). This was necessary
because `measure_grounddino.py`'s own built-in scorer is keyword-based and
structurally cannot score prompts that don't contain the ground-truth
concept words — see the scoring caveat above. Not promoted to Tier 1 because
committing a second, narrower scorer into the repo for a family of prompts
this section is recommending against felt like the wrong permanent
artifact to leave behind; the class-agnostic matching logic itself is not
novel (it is `measure_segmentation.py`'s existing `match_recall()` function,
read and reused, not reinvented).

**Tier 3 (described, not reproducible by anyone but this session, same
convention as Part 3's own Tier 3 entries):** the crop inspections — the
scissors/sofa-texture/wall-corner crops from segmentation, the knife/
whole-table/whole-floor crops from broad-prompt Grounding DINO. Every crop
is named by exact source frame filename and (for Grounding DINO) exact box
coordinates and confidence, both reported above and both present in the
committed Tier 1 CSVs, so anyone with `cv/captures/` on disk can regenerate
the identical crop from the identical committed row.

---

## Where the evidence actually lives

Following Phase 3's own stated concern about evidence tiers (a claim with no
traceable script/CSV is weaker evidence than one with a script anyone can
rerun) — stated here plainly rather than smoothed over:

- **Tier 1 (script + CSV both in this repo, rerunnable as-is):**
  - Part 2's open-vocab broadening. `cv/measure_openvocab.py` is unmodified,
    existing repo code; `cv/measurements/openvocab_phase4_homeframes.csv` is
    the real output, gitignored per this project's existing measurement-CSV
    convention (same as every other file already in that directory).
  - **Part 3's Grounding DINO numbers — promoted from Tier 2 to Tier 1,
    2026-08-12 (Phase 4 Part 1).** `cv/measure_grounddino.py` is now a real,
    committed, rerunnable tool (`transformers==5.15.0` pinned in
    `cv/requirements.txt` against this repo's existing `torch==2.13.0` pin —
    verified in THIS repo's own `.venv`, not just copied from the research
    session's scratch venv, that Grounding DINO actually loads and runs
    real inference on MPS with that exact pin combination before it was
    committed). It reuses `evaluate_home_frames.py`'s ground-truth loading
    and IoU matching unchanged, adding only the label-substring matching
    Grounding DINO's compound-label output requires (see that script's
    docstring for both quirks — the `threshold`/`box_threshold` rename and
    compound labels — worked out here exactly as this doc originally
    described them). Regenerated CSVs:
    `cv/measurements/gdino_tiny_phase4.csv` and
    `cv/measurements/gdino_base_phase4.csv` (gitignored, same convention).
    **Reproduction result, checked directly rather than assumed:**
    `sharp_object` — this doc's actual recommendation — reproduces closely
    for both checkpoints (tiny, conf≥0.20: TP49/FP395/FN13/precision
    0.110/recall 0.790 here vs. this doc's TP49/FP398/FN13/precision
    0.110/recall 0.790 above — functionally identical; base, conf≥0.20:
    TP34/FP165/FN28/precision 0.171/recall 0.548 here vs. TP34/FP163/FN28/
    precision 0.173/recall 0.548 above — same result). Inference speed
    reproduces in the same range too (tiny 569.8ms/base 671.4ms mean here vs.
    ~496ms/~637ms above — same order of magnitude, not the same machine-load
    conditions). **`small_swallowable` on `grounding-dino-base` specifically
    does NOT reproduce in the same range** (conf≥0.30: TP=12/FP=43/recall
    0.222 here vs. this doc's TP=1/FP=0/recall 0.019 above — a real,
    roughly-10x discrepancy, confirmed deterministic on rerun in this repo,
    not measurement noise). Traced to a real, previously-unreported
    property of `grounding-dino-base`'s output on this prompt list: it
    frequently emits several near-duplicate, heavily self-repeating compound
    labels for the same real small object (e.g. `"small object tiny object
    small object"`), which the base HuggingFace post-processing does not
    deduplicate — whether the original scratch scoring script filtered
    these differently is unknown and unverifiable now that it no longer
    exists. Flagged here rather than silently accepted, per this file's own
    evidentiary standard. This does not change any recommendation in this
    document: `small_swallowable` broadening was already rejected in Part 2
    regardless of model, and `sharp_object` (the one concept actually
    recommended for adoption) reproduces closely on both checkpoints.
- **Tier 3 (described, not reproducible by anyone but this session):** the
  `supervision` API confirmation (loading `ByteTrack`/`PolygonZone` in a
  scratch venv) and the qualitative crop inspections (the cabinet-handle
  object found by four different prompts, the knife-in-hand frame Grounding
  DINO caught and open-vocab missed, the empty power-cord false positive).
  These are real — each crop was generated from actual pixels at the
  reported coordinates and visually inspected, not estimated — but the crop
  images themselves live only in this session's scratch directory, not this
  repository, and are described here in enough specific, checkable detail
  (exact frame filenames, exact coordinates, exact confidence values, all
  independently verifiable from the CSVs that are committed) that anyone
  with this repo's `cv/captures/` frames on disk can reproduce every crop
  exactly.

---

## Part 5 (2026-08-12) — tightening `measure_segmentation.py`'s filter:
does `MAX_AREA_FRAC` or a new aspect-ratio signal get segmentation to a
walkable candidate count?

**Status: measurement/tuning only, same discipline as every other part of
this document. `cv/risk_engine.py` and `cv/test_risk_engine.py` were not
touched.** This section picks up Part 4's own stated next step exactly:
segmentation's problem was never recall (raw 0.491, filtered 0.241 — plenty
of true positives are found), it is candidate *volume* (mean 33.8/frame home,
64.6/frame room2) — and Part 4's two named false positives (a sofa-arm/
cushion fabric texture blob, a wall-corner edge, both from frame
`20260810-214630_room2-60cm_..._raw.jpg`) were both **compact and
high-extent**, meaning `MIN_EXTENT` (a fill-ratio filter) structurally cannot
reject them. Two levers were tried, exactly as Part 4 recommended: a tighter
`MAX_AREA_FRAC` (currently 0.05), and a genuinely new elongation signal
(`aspect_ratio = bbox min-side/max-side`, distinct from `extent`, which
measures fill ratio not shape).

### What changed in `cv/measure_segmentation.py`

- Added an `aspect_ratio` field to every segment (`segment_frame()`), and a
  `--min-aspect-ratio` CLI flag (default `None`, **not** part of the default
  filter — same convention as the existing `--y-containment-frac`), following
  this repo's `measure_*.py` convention of exposing filter constants as
  flags. `MIN_AREA_FRAC`/`MAX_AREA_FRAC`/`--min-extent` were already flags
  before this session; only `aspect_ratio` is new.
- `save_crop()` filenames now also encode `aspect_ratio` (`_asp0.XX`), so a
  saved crop's shape signal is checkable without re-running the script.
- **`MAX_AREA_FRAC`'s default was changed from `0.05` to `0.01`** — the one
  setting measured below with a real, free improvement (see "Recommendation"
  for why `--min-aspect-ratio` was left off by default rather than baked in).

### Method — same spread as Part 4, for a fair before/after

```
python cv/measure_segmentation.py --room2-sample 30 --seed 0 \
  --output cv/measurements/segmentation_phase5_default.csv
```

(32 `*home*_raw.jpg` frames + a `--room2-sample 30` sample, `--seed 0` fixed
so the room2 sample is identical across every run in this section — Part 4
did not fix a seed, but its own reported per-frame ranges reproduce closely
under seed 0, confirmed by rerunning Part 4's exact unmodified filter first
and checking against its own numbers, below.)

A parameter sweep was then run over `--max-area-frac` (0.05 down to 0.003)
crossed with `--min-aspect-ratio` (none, 0.15, 0.3, 0.45), each writing its
own CSV to a scratch directory (not committed — same gitignored-CSV
convention as every other `measure_*.py` output; the sweep is fully
reproducible by rerunning the one-line command above with each flag
combination). Ground truth, matching rule (IoU≥0.5, class-agnostic), and the
32-frame set are all unchanged from Part 4.

### Reproducing Part 4's own numbers first, as a sanity check

Rerunning the **unmodified pre-session filter** (`--max-area-frac 0.05`, no
aspect-ratio flag, which did not exist yet) against this exact spread:
recall (raw) 57/116 = 0.491, recall (filtered) 28/116 = 0.241, home mean
33.8 candidates/frame (range 10–79), room2 mean 64.6/frame (range 23–97) —
**identical to Part 4's own reported numbers**, confirming this session's
spread and scoring match Part 4's exactly before anything was changed.

### Results — `MAX_AREA_FRAC` sweep (no aspect-ratio filter)

| `MAX_AREA_FRAC` | recall (filtered) | home candidates/frame (mean, range) | room2 candidates/frame (mean, range) |
|---|---|---|---|
| 0.05 (old default) | 28/116 = 0.241 | 33.8 (10–79) | 64.6 (23–97) |
| 0.03 | 0.241 | 33.8 (unchanged — no home segment removed at this level) | 61.5 |
| 0.02 | 0.241 | — | 58.5 |
| **0.015** | 0.241 | — | 55.6 |
| **0.01 (new default)** | **0.241 (unchanged)** | **27.9 (9–54)** | **52.0 (22–80)** |
| 0.007 | 0.181 (recall starts dropping) | 24.4 (9–50) | 47.3 |
| 0.005 | 0.172 | 21.1 (9–39) | 44.1 |
| 0.003 | 0.129 | — | — |

**Why `0.01` and not lower, stated plainly with the number behind it:** the
32 home frames' 116 ground-truth boxes range from `area_frac` 0.00016 to
0.019 (median 0.0034, p90 0.013) — i.e. **real hazard sizes in this test set
already overlap the noise-blob size range almost entirely** (the room2
cushion-texture blob is `area_frac` 0.0053, well inside the real-object
range). `MAX_AREA_FRAC=0.01` is the largest value below which recall starts
being cut, and it happens to sit just below the real-object p90 — pushing it
lower starts discarding genuine large hazards (a knife photographed close to
the 60cm camera, etc.), not noise. This directly falsifies the hypothesis
Part 4 stated as its untested next lever ("the false positives are large
relative to a real handheld hazard") — checked directly here, not assumed:
**the false positives inspected are actually small (`area_frac` 0.0038–0.016
across the three named ones below), the same size range as real objects, not
larger.** `MAX_AREA_FRAC` alone was never going to solve this because size
does not separate the two populations in this data.

**Net effect of the area change alone: a real, free win, but a small one** —
zero recall lost, home candidates down ~17% (33.8→27.9), room2 down ~19%
(64.6→52.0). Nowhere near a walkable count on its own.

### Results — adding `--min-aspect-ratio` on top of `MAX_AREA_FRAC=0.01`

| `--min-aspect-ratio` | recall (filtered) | home candidates/frame (mean) | room2 candidates/frame (mean) |
|---|---|---|---|
| none | 0.241 | 27.9 | 52.0 |
| 0.15 | **0.241 (unchanged)** | 27.6 | 50.4 |
| 0.30 | 0.129 (−46% of TPs found) | 20.0 | 43.0 |
| 0.45 | 0.121 (−50% of TPs found) | 16.3 | 34.4 |

**Aspect ratio does cut volume further at 0.30/0.45, but at a severe and
specific recall cost, checked by name, not just by count:** the 40 true
positives matched at `MAX_AREA_FRAC=0.01` (no aspect filter) have
`aspect_ratio` values spanning **0.16 to 0.98** — real hazard detections are
just as often elongated as compact. Two concrete, checkable examples: on
`20260810-143033_home-60cm_..._raw.jpg`, a true positive at `aspect_ratio
0.161` (`area_frac 0.0010`) and another at `0.188` (`area_frac 0.0012`) both
match ground-truth `sharp_object` boxes — i.e. **real knife/scissors
detections sit in the exact same aspect-ratio range (≈0.16–0.3) as the
wall-corner false positive this filter was designed to catch.** This is not
a coincidence — CLAUDE.md decision 7 already documents that `sharp_object`
merges knives and scissors specifically because both are inherently
thin/elongated objects, and that same shape property is what any
elongation-based filter penalises. **A signal engineered to reject structural
edges structurally cannot avoid rejecting the sharp objects this project
cares about most, because they share the same bbox shape.** At
`--min-aspect-ratio 0.15` — low enough to only reject the most extreme
edges — the cost drops to zero (recall unchanged, 0.241) but so does almost
all of the benefit (candidates barely move, 27.9→27.6 home).

**And critically, aspect ratio does not even fix the false positive it was
originally motivated by.** Re-checking the exact room2 cushion-texture blob
named in Part 4 (`20260810-214630_room2-60cm_..._raw.jpg`, box
`(538, 251, 721, 489)`, `area_frac 0.0053`, `extent 0.72`) at
`--min-aspect-ratio 0.45`: **it survives** (`aspect_ratio 0.77`, above the
0.45 threshold) — a crop pulled at this exact box is still visibly ribbed
cushion fabric, not a real object (same crop as Part 4 inspected). The
elongation signal only removed the *other* named false positive, the
wall-corner edge (`aspect_ratio 0.161`, correctly gone at every
`--min-aspect-ratio ≥ 0.2` tested) — a genuinely elongated structural false
positive is fixable this way; a **round, compact texture blob is not**,
because it has no elongation to detect. This confirms Part 4's own framing
exactly: extent and aspect ratio are both real, different shape signals, and
between them they still cannot describe "this is furniture fabric, not an
object" — that distinction is about *texture*, not geometry, and neither
signal measures texture.

### Named crop check — did anything else get newly lost?

Spot-checked `20260810-143033_home-60cm_..._raw.jpg` directly (10
ground-truth boxes, the densest frame in the set): at the old default
(`MAX_AREA_FRAC=0.05`, no aspect filter) it produced 79 candidates, 5 found;
at the new default (`MAX_AREA_FRAC=0.01`, no aspect filter) it produces 54
candidates, **5 found — unchanged**, confirming the area tightening cost
nothing on the single most crowded frame in the set. Adding
`--min-aspect-ratio 0.30` on top drops that same frame to 34 candidates but
**only 2 found** — losing 3 real detections on this one frame alone, which
is the aspect-ratio recall cost concretely, not just as an aggregate
percentage.

### Recommendation

**Adopted as the new default: `MAX_AREA_FRAC = 0.01`.** Measured, free
(zero recall cost on this test set), and the honest, stated reason it can't
go lower (real hazard sizes already overlap the noise-blob size range in
this data) is recorded in the constant's own comment in
`cv/measure_segmentation.py`.

**Not adopted as a default: `--min-aspect-ratio`.** Kept as an opt-in flag
(same pattern as the existing `--y-containment-frac`) rather than a default,
because the measurement is genuinely mixed rather than a clear win: it can
cut volume further (16–20 candidates/frame at 0.30–0.45 vs. 28/frame with
none) but costs 46–50% of the true positives this test set can find, and the
specific mechanism of that cost — penalising elongation — directly conflicts
with `sharp_object`'s own defining shape property per CLAUDE.md decision 7.
A project whose most safety-critical hazard class is inherently elongated
should not default to a filter that structurally fights that shape. The flag
is left available for anyone who wants to trade recall for volume on a
different object mix, with this finding attached so the tradeoff isn't
rediscovered blind.

**Honest bottom line, stated plainly rather than declared solved:** the best
setting found here that costs **zero** recall (`MAX_AREA_FRAC=0.01`, no
aspect filter) still leaves **27.9 candidates/frame on the home set and 52.0
on room2** — nowhere near "a parent can tap through this in seconds" (decision
4's own bar). The best setting found that trades recall away
(`MAX_AREA_FRAC=0.01` + `--min-aspect-ratio=0.45`) still leaves **16.3/frame
home, 34.4/frame room2**, while losing half the true positives this test set
can find — worse on the walkability axis than it looks, because losing real
detections on a "flag everything, let the parent judge" mechanism (CLAUDE.md
decision 3) is a direct, not just incidental, cost. **Neither area nor
aspect-ratio tightening gets segmentation to walkable. This is not a "close,
just needs one more tweak" result — it is a real, measured, and still
substantial gap**, and the reason is now specific rather than vague: the
dominant remaining false-positive character (per Part 4's own crop
inspection, reconfirmed here) is compact, roughly-square furniture texture
that is genuinely indistinguishable from a real compact object using either
size or shape geometry alone — it would need an actual texture or
appearance-based signal (not a bbox/mask geometry signal) to separate, which
neither `extent`, `MAX_AREA_FRAC`, nor `aspect_ratio` can provide by
construction. This narrows what the next lever needs to be, if this
mechanism is pursued further: something that looks at pixel content
(texture, colour uniformity, edge density inside the mask), not more
geometry filters on a shape that has already been measured to not carry the
needed information.

### Where this evidence lives

**Tier 1 (script + CSV both in this repo, unmodified-going-forward script,
rerunnable as-is):** `cv/measure_segmentation.py` (this session's real,
committed changes: `aspect_ratio` field, `--min-aspect-ratio` flag,
`MAX_AREA_FRAC` default `0.05`→`0.01`) against the exact command above,
`cv/measurements/segmentation_phase5_default.csv` (regenerate with the
command in "Method" above; gitignored, same convention as every other file
in that directory).

**Tier 2 (a parameter sweep, not committed as separate CSVs, but each cell
in the sweep tables above is reproducible by rerunning the same one-line
command with the stated `--max-area-frac`/`--min-aspect-ratio` values and
`--seed 0`):** all rows in both sweep tables above.

**Tier 3 (described, not reproducible by anyone but this session): the named
crop inspections** — the cushion-texture blob at `20260810-214630_room2-60cm
_..._raw.jpg` box `(538, 251, 721, 489)` surviving every aspect threshold
tested, and the two named low-aspect-ratio true positives on
`20260810-143033_home-60cm_..._raw.jpg`. Every crop is named by exact source
frame filename and exact box coordinates (both reported above and derivable
from `segment_frame()`'s own output at the stated settings), so anyone with
this repo's `cv/captures/` frames on disk can regenerate the identical crop.

---

## Part 6 (2026-08-12) — segmentation candidates + an isolated-crop Grounding
DINO "is this a real object" confirm step: does the change-detection-confirm
pattern generalize to the walkthrough's candidate-volume problem?

**Status: measurement only, same discipline as every other part of this
document. `cv/risk_engine.py` and `cv/test_risk_engine.py` were read (for
`compute_crop_region`, reused unmodified) but not touched.** Part 4/5 left the
walkthrough with no clean winner: segmentation finds real objects (raw recall
0.491) but never collapses to a walkable candidate count at any geometry-only
filter setting (best zero-recall-cost setting: 27.9/frame home, 52.0/frame
room2), and broad-prompt whole-frame Grounding DINO fails differently (giant
whole-surface boxes instead of per-item ones). This section tests a specific,
different idea neither of those tried: reuse this project's own proven
pattern from live change-detection-confirmation (`HAZARD_SOURCE_CHANGE_CONFIRMED`
in `cv/risk_engine.py`, Part 3 above) — don't ask Grounding DINO to search a
whole busy scene, **crop the candidate region first, pad it, and ask a
narrow presence question about the isolated crop alone** — but apply it to
segmentation's candidate boxes instead of change-detection's diff blobs.

### Method

Two new scratch scripts (not committed to `cv/` — same Tier 2/3 convention
Part 4's own class-agnostic Grounding DINO scorer already used in this
document; described here in enough detail, using only unmodified,
already-committed repo functions, that anyone can reproduce them exactly),
both importing unmodified repo code rather than reimplementing anything:
`cv/measure_segmentation.py`'s `segment_frame`/`apply_filter`/
`read_yolo_labels`/`iou`/`load_weights`/`resolve_device` (candidate
generation + the Part 5 geometry filter + ground-truth loading + IoU
matching), `cv/measure_grounddino.py`'s `load_grounding_dino_model`/
`run_grounding_dino` (the Grounding DINO call itself, including its two
already-documented API quirks), and `cv/risk_engine.py`'s
`compute_crop_region` (the same padded-bbox-to-crop-region math the live
change-confirm step already uses, imported unchanged, not reimplemented).

**Step 1 — sanity check first, per the task's own instruction not to run a
full sweep blind.** The exact named hard cases Part 4/5 already identified —
the room2 cushion-texture blob (`20260810-214630_room2-60cm_..._raw.jpg`,
box ≈`(538, 251, 721, 489)`, `area_frac 0.0053`, `extent 0.72`), the
wall-corner edge from the same frame (`aspect_ratio 0.161`), and, as the
positive controls, the scissors true positive and a second named
low-aspect-ratio true positive from `20260810-143033_home-60cm_..._raw.jpg`
(Part 5's own named examples, `area_frac 0.0010`/`aspect_ratio 0.161`) —
were regenerated by rerunning segmentation's default (Part 5) filter on
exactly those two frames, matched back to Part 4/5's own reported
area_frac/extent/aspect_ratio values to identify the exact same candidate
box, then cropped (via `compute_crop_region`) and run through
`grounding-dino-tiny` in isolation, at `conf=0.05` (record everything) with
several prompt variants.

**A real, load-bearing finding from this step alone: padding amount matters
a lot, and the live change-confirm step's existing default is the wrong
choice for this new use case.** At `pad_frac=0.5` (`CHANGE_CONFIRM_CROP_PAD_FRAC`,
the value `cv/risk_engine.py` already uses for change-detection blobs), the
separation between the named false positives and true positives was weak:
`object` prompt scored the cushion FP 0.417, the wall-corner FP 0.369, the
scissors TP 0.594, the low-aspect TP 0.562 — a bare ~0.15 gap, not safely
separable at any one threshold. At `pad_frac=0.15` (tighter — and a
principled choice, not an arbitrary retry: a change-detection blob is a
pixel-diff bounding rect that may only catch an edge of the real object, so
it needs generous padding for context, but a segmentation candidate box is
already FastSAM's own estimate of the object's full extent, so heavy padding
mostly dilutes the crop with surrounding clutter instead of adding needed
context), the same four crops separated far more cleanly: cushion FP 0.510,
wall-corner FP 0.356, scissors TP 0.783, low-aspect TP 0.786 — a ~0.27 gap.
A four-way contrastive prompt (`"a solid physical object"` vs. `"fabric
texture"` vs. `"a wall or floor edge"` vs. `"empty surface"`) was also
tried and was **worse**, not better — it reproduced this project's
already-documented compound-label problem (Part 3) and, on the scissors
crop specifically, the model's own top-scoring label was `"a wall floor
edge"` — the wrong side of the contrast, on the one crop that most needs to
be called an object. The plain single-word prompt `object` at `pad_frac=0.15`
was therefore the configuration carried into the full sweep below.

**On these four named examples alone, this looked genuinely promising** —
which is exactly why the task's own sanity-check gate said to proceed to a
full sweep rather than stop. **The full sweep below shows this promise does
not survive contact with the actual candidate population.**

Full sweep command shape (scratch script, unmodified repo functions only):
segment every frame with FastSAM (`segment_frame`, default settings),
compute the Part 5 area/extent filter's pass/fail per candidate
(`apply_filter`, `MAX_AREA_FRAC=0.01`), crop every RAW candidate (padded,
`pad_frac=0.15`, via `compute_crop_region`) and run `grounding-dino-tiny`
with `prompts=["object"]`, `conf=0.05` (record everything, threshold
applied post-hoc, this project's standing convention), `text_threshold=0.10`,
over the identical spread as Part 4/5: all 32 `*home*_raw.jpg` frames + a
`--room2-sample 30 --seed 0` sample (identical seed, reproduced the exact
same 30 room2 filenames Part 5 used). One Grounding DINO call per candidate
— **5,887 calls total** (62 frames, mean ~95 raw candidates/frame) — with
confidence recorded raw, so both option (a) (raw candidates + crop-confirm
as the SOLE filter) and option (b) (Part-5-filtered candidates +
crop-confirm ON TOP) could be evaluated at every threshold from the same
one run, without re-running inference per threshold.

### Results — candidate volume and recall, both options, several confirm thresholds

Box-level recall uses the identical rule as every other table in this
document (`match_recall`'s own rule, imported unchanged): a ground-truth box
counts as found if ANY surviving candidate covers it at IoU≥0.5, pooled
across both concepts, against the same 116 ground-truth boxes on the 32 home
frames. "0.05" is the pre-confirm baseline (nothing filtered out by
Grounding DINO yet) and reproduces Part 4/5's own raw (0.491) and filtered
(0.241) recall numbers exactly, confirming this sweep's candidate generation
matches Part 4/5's before anything new was applied.

**Option (a): segmentation's RAW candidates + crop-confirm as the sole filter**

| confirm conf ≥ | home candidates/frame | room2 candidates/frame | box-level recall |
|---|---|---|---|
| 0.05 (no filter) | 80.4 | 110.4 | 57/116 = 0.491 |
| 0.30 | 79.1 | 109.7 | 57/116 = 0.491 |
| 0.40 | 61.4 | 81.9 | 51/116 = 0.440 |
| 0.50 | 35.5 | 39.7 | 44/116 = 0.379 |
| 0.55 | 28.1 | 27.1 | 39/116 = 0.336 |
| 0.60 | 20.6 | 18.6 | 29/116 = 0.250 |
| 0.65 | 15.0 | 12.8 | 25/116 = 0.216 |
| 0.70 | 10.5 | 7.6 | 20/116 = 0.172 |

**Option (b): Part 5's geometry-FILTERED candidates + crop-confirm on top**

| confirm conf ≥ | home candidates/frame | room2 candidates/frame | box-level recall |
|---|---|---|---|
| 0.05 (no filter) | 27.9 | 52.0 | 28/116 = 0.241 |
| 0.30 | 27.2 | 51.5 | 28/116 = 0.241 |
| 0.40 | 21.7 | 35.8 | 26/116 = 0.224 |
| 0.50 | 15.2 | 16.0 | 23/116 = 0.198 |
| 0.55 | 13.0 | 11.3 | 20/116 = 0.172 |
| 0.60 | 10.4 | 8.4 | 15/116 = 0.129 |
| 0.65 | 8.0 | 6.1 | 14/116 = 0.121 |
| 0.70 | 5.5 | 4.2 | 12/116 = 0.103 |

**The headline finding: this is a real filter — candidate volume genuinely
drops as the threshold rises — but recall drops in close proportion, not
independently of it.** There is no threshold on either option where
candidate count reaches a walkable range (single digits to low teens per
region, decision 4's own implicit bar) while recall stays anywhere near
Part 4/5's already-modest baseline. Option (b) reaches single digits
(5.5–10.4/frame home) only at conf≥0.60–0.70, where recall has already
fallen to 0.103–0.129 — **losing 85–90% of the ground truth this test set
can find**, a materially worse trade than Part 5's own rejected
`--min-aspect-ratio=0.45` (which cost "only" 46–50% of recall for a
similar-magnitude volume cut). Option (a) never reaches a walkable count at
all in this sweep's range — even at conf≥0.70 it is still 7.6–10.5/frame,
comparable to option (b)'s single-digit range, but arrived at from a much
higher starting volume and a correspondingly worse recall (0.172) at that
point.

### Why the promising 4-example sanity check didn't generalize — checked directly, not assumed

The four named crops (Part 5's cushion FP, wall-corner FP, scissors TP,
low-aspect TP) reproduced their sanity-check scores exactly in the full
sweep (cushion 0.510, wall-corner 0.356, scissors 0.783, low-aspect 0.786 —
same numbers, confirming no drift between the two runs). **But checked
against the full population of 79 raw home candidates that actually match a
ground-truth box (IoU≥0.5), that four-example gap does not hold:**

- **36.7% of real true-positive candidates (29/79) score BELOW the named
  cushion false positive's own 0.510** — meaning any threshold high enough
  to reject that one named piece of furniture texture also rejects over a
  third of the real objects this test set contains.
- **36.6% of ALL non-ground-truth-matching candidates (2,128/5,808) score
  AT OR ABOVE that same 0.510** — meaning a threshold that keeps the
  cushion FP out lets through a near-identical *fraction* of everything
  else that isn't a labelled hazard.

TP-candidate confidence distribution, for the same honesty: min 0.323, p10
0.383, median 0.585, p90 0.800, max 0.872 — a wide spread that overlaps
almost entirely with the non-TP population's own spread. **The four named
examples were not wrong, they were just an unrepresentatively easy subset**
— a real, useful, checkable finding in its own right (this is exactly the
kind of thing a 4-crop eyeball check cannot catch and only a full-population
sweep reveals), and the honest reason this section doesn't get to claim the
crop-confirm idea "worked."

### Latency — measured plainly, as the task required

| quantity | measured value |
|---|---|
| Grounding DINO tiny, isolated crop, mean call time | **432.1 ms** (5,887 calls, this session's actual run — slightly faster than Part 3's whole-frame 496ms and the live change-confirm step's 500–700ms estimate, plausibly because these crops are smaller than a full frame) |
| Total wall time, full sweep (62 frames, both options measured from one pass) | **2,610 s ≈ 43.5 minutes** |
| Confirm-only cost, ONE frame/region, option (a) raw candidates | home **~34.7 s** (80.4 × 432ms), room2 **~47.7 s** (110.4 × 432ms) |
| Confirm-only cost, ONE frame/region, option (b) filtered candidates | home **~12.1 s** (27.9 × 432ms), room2 **~22.5 s** (52.0 × 432ms) |
| Segmentation (person-detect + FastSAM) per frame, for reference (Part 4) | ~116 ms — negligible next to the confirm cost |

**This is the number that matters most for whoever builds the walkthrough
next, stated exactly as the task asked, not folded into a precision/recall
table where it's easy to miss:** even the cheaper of the two options
(Part-5-filtered candidates, confirm on top) costs on the order of **12–23
seconds of Grounding DINO calls for a single surface-region frame**, before
a parent sees a single candidate to tap through. CLAUDE.md decision 3
anticipates the guided walkthrough sweeping multiple reachable-surface
regions (floor, low table, low shelf) in one 2–3 minute session. At
**12–23 s/region** (option b) that is 36–115 s across 3–5 regions — a real
fraction of the stated total budget, and that is BEFORE accounting for any
parent-facing UI time (showing candidates, waiting for taps) at all. At
**35–48 s/region** (option a, raw candidates) the same 3–5-region sweep
alone costs roughly 1.75–4 minutes of confirm-call time — **exceeds
decision 4's entire stated 2–3 minute budget on its own**, independent of
segmentation time or any parent interaction. Raw candidates + crop-confirm (option a) is strictly worse than option (b)
on latency (roughly 3x the confirm-call time per region, per the table
above), and does not make up for that with better recall at a matched
candidate count: at conf≥0.55, option (a) leaves 28.1 candidates/frame
(home) at recall 0.336, while option (b) at conf≥0.55 leaves only 13.0
candidates/frame at recall 0.172 — a meaningfully smaller, more walkable
list, but at meaningfully worse recall too. The two options are not simply
rescaled versions of each other and neither dominates the other cleanly at
every point on the curve — but option (a)'s extra volume and extra latency
never buys a recall advantage large enough to justify either cost, at any
threshold checked in this sweep.

### Recommendation

**No — this is not ready to be the walkthrough's filter, and the reason is
different from, and in one sense more informative than, Part 4/5's own
verdict.** Part 4/5 found geometry-only filtering hits a hard wall because
size and shape do not separate furniture texture from real objects in this
data. This section found something more specific: **an isolated-crop
semantic "is this an object" question from Grounding DINO IS a real,
measurably different signal from geometry — it is not simply re-deriving
extent/aspect-ratio in a different vocabulary — but the true-positive and
non-true-positive confidence distributions overlap too heavily, at the
scale of the full candidate population, for any single threshold to trade
volume for recall favorably.** The four hand-picked hard cases that
motivated this whole section looked cleanly separable and were not
representative of that overlap — a concrete, checkable illustration of why
this project's own standing rule (measure on the full held-out set, not a
handful of eyeballed examples) exists.

**What would need to be true for this idea to work, stated for whoever
picks it up next (not attempted here — out of scope for a measurement
session per the task):**

- A confirm signal with genuinely better separation than a single
  `"object"` prompt's raw confidence — e.g. averaging or voting across
  several independently-worded presence prompts on the SAME crop (tried
  here only as one combined multi-phrase query, which Part 3 already showed
  produces confusable compound labels, not as independent votes averaged
  post-hoc, which is a different and untested idea), or combining the
  Grounding DINO confidence with segmentation's own geometry signals
  (extent, aspect_ratio) as a joint two-signal filter rather than treating
  crop-confirm as a wholesale replacement for geometry.
- Something to reduce candidate volume BEFORE the expensive confirm step
  runs at all, since even the correctly-separated cases here don't fix the
  cost problem shown in the latency section — option (b)'s already-filtered
  27.9–52.0 candidates/frame still cost 12–23 seconds of Grounding DINO
  calls each region, independent of whether the eventual threshold chosen
  is a good one.
- If pursued, it should build on option (b) (filtered candidates +
  crop-confirm), never option (a) (raw candidates) — option (a) is worse on
  every axis measured here: more candidates before confirm, no better
  recall at matched confidence thresholds, and roughly 3x the latency cost
  per region.

**One genuinely positive, reusable result from this section, worth keeping
regardless of the overall verdict:** the change-detection-confirm crop
pattern (`compute_crop_region` + isolated-crop Grounding DINO) generalizes
cleanly to a second use case (segmentation candidates, not just
change-detection blobs) with zero code changes needed beyond a different
padding fraction — a real point in favor of that pattern's design, even
though the padding fraction tuned for one use case (change-detection blobs,
`pad_frac=0.5`) turned out to be measurably wrong for the other
(segmentation candidates, where `pad_frac=0.15` performed much better on the
four named examples) — **this padding-fraction finding is scoped to this
measurement session only and has NOT been applied to
`CHANGE_CONFIRM_CROP_PAD_FRAC` in `cv/risk_engine.py`**, since that constant
serves change-detection blobs specifically, which this session did not
re-measure, and changing a live constant based on a different use case's
tuning would be exactly the kind of unmeasured, silently-applied change this
document's own conventions exist to prevent.

### Where this evidence lives

**Tier 1 (unmodified, already-committed repo functions, reused not
reimplemented):** `cv/measure_segmentation.py`'s `segment_frame`/
`apply_filter`/`read_yolo_labels`/`iou`/`load_weights`/`resolve_device`,
`cv/measure_grounddino.py`'s `load_grounding_dino_model`/`run_grounding_dino`,
`cv/risk_engine.py`'s `compute_crop_region` — all read and imported exactly
as committed, none modified for this section.

**Tier 2 (scratch scripts, not committed to this repo, described here in
enough detail to reproduce exactly): the sanity-check script (four named
crops, both padding fractions, five prompt variants) and the full-sweep
script (5,887-row CSV: every raw segmentation candidate across the 32 home +
30-room2-sampled `--seed 0` frames, its Part 5 filter pass/fail, its
ground-truth-match flag, and its isolated-crop Grounding DINO `"object"`
confidence) — both existed only in this session's scratchpad directory, not
this repository, following the exact same Tier 2 convention Part 4's own
class-agnostic Grounding DINO scorer used (a small scorer built entirely
from this repo's own already-committed, unmodified functions, not
reimplementing IoU matching or ground-truth loading from scratch).

**Tier 3 (described, not reproducible by anyone but this session): the named
crop confidence values** — cushion FP 0.510, wall-corner FP 0.356, scissors
TP 0.783, low-aspect TP 0.786, all at `pad_frac=0.15`, prompt `"object"`,
reproduced identically between the standalone sanity check and the full
sweep (a real internal consistency check, not assumed) — and the aggregate
TP-confidence-distribution numbers (36.7% of TPs below the cushion FP's own
score, 36.6% of non-TPs at/above it), both derivable from the Tier 2 CSV's
own `gdino_conf`/`matches_gt` columns by anyone who reruns the full-sweep
script as described.
