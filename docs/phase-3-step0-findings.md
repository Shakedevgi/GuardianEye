# Phase 3 Step 0 — measured detection baseline

**Status: partial. Measured from 12 existing raw captures (one room, two
sessions). The live capture session that closes the remaining gaps has not
been run yet — see "Still unmeasured" at the end.**

This supersedes `docs/phase-3-scoping-notes.md` wherever the two disagree.
Those notes were eyeballed from annotated JPEGs at a single configuration;
this document is measured with `cv/measure_detection.py` across a
model × resolution × confidence sweep. Where the notes were wrong, they were
wrong in a specific and instructive way — recorded below rather than quietly
corrected.

## How this was measured

`cv/measure_detection.py` (new, Step 0 deliverable) runs stock YOLO26 offline
over already-saved **raw** frames and emits one CSV row per detection. Two
choices matter:

- **Confidence floor 0.01, not 0.25.** Every capture before this was taken at
  conf 0.25, which cannot distinguish "the model emitted nothing" from "the
  model emitted a 0.04 box we discarded." Those two have completely different
  remedies — the second is a threshold/stability problem, the first is a
  genuine domain gap — so Step 0 had to re-measure at a floor.
- **Offline sweep on raw frames, not live re-shooting.** `--imgsz` only
  affects inference; the frame written to disk is always full-resolution.
  So one live capture can be re-measured at any model/resolution/confidence
  afterwards, as many times as needed. This is what makes a 20–30 minute
  live session sufficient.

Sweep run: `yolo26x.pt` × imgsz {640, 960, 1280, 1600} × conf 0.01 over all
12 raw captures. 528 detections.

## The class list

### Detects reliably — stock COCO is sufficient

| Class | Evidence | Caveat |
|---|---|---|
| `refrigerator` | 0.88–0.93 in every frame it appears, at every resolution | none |
| `chair` | found in every frame; 0.30–0.90 | confidence varies with occlusion, but presence is never lost |
| `person` | **0.90–0.98 at imgsz 640** across 10 frames | **only at imgsz 640–960 — see the inversion finding below** |

### Detects unreliably — COCO has the class, the output is not trustworthy

| Class | Evidence | Failure mode |
|---|---|---|
| `scissors` | detected in most frames, but confidence is chaotic and **non-monotonic in resolution** | no configuration clears 0.5 on all five near-identical frames of the same object |
| `knife` | 0.48–0.885 when **held in a hand**; 0.011–0.049 when resting on a surface or shelf | context-dependent in exactly the wrong direction for this project |

### Misses entirely — no COCO class exists

| Class | Evidence |
|---|---|
| power strips, wall sockets, plug adapters, trailing cables | never detected at any resolution. Checked for a substitute misclassification: exactly one spurious box across the entire sweep (`remote`, 0.012). That is noise, not a stable mislabel — so this is a true absence, not a renaming problem. |

## Finding 1 — `person` is degraded by high resolution, not improved by it

This is the most consequential result of Step 0 and it reverses the
scoping notes.

Max `person` confidence per frame, conf floor 0.01:

| frame | 640 | 960 | 1280 | 1600 |
|---|---|---|---|---|
| 20260805-012930 | 0.905 | 0.923 | 0.833 | 0.793 |
| 20260805-012947 | 0.323 | 0.509 | 0.057 | 0.063 |
| 20260805-012953 | 0.932 | 0.913 | 0.148 | 0.083 |
| 20260805-012954 | 0.942 | 0.912 | 0.247 | 0.259 |
| 20260809-082705 | 0.960 | 0.949 | 0.913 | 0.829 |
| 20260809-082759 | 0.905 | 0.742 | 0.913 | 0.857 |
| 20260809-082904 | 0.976 | 0.968 | 0.876 | 0.141 |
| 20260809-082938 | 0.976 | 0.957 | 0.880 | **0.029** |
| 20260809-083010 | 0.530 | 0.899 | 0.878 | 0.483 |
| 20260809-083249 | 0.651 | 0.436 | 0.476 | 0.368 |

The scoping notes concluded "person detection is weaker than expected — this
may be the bigger risk," from frames where `person` scored 0.38–0.42 or
vanished. That was correct as an observation and wrong as a diagnosis: those
frames were all shot at imgsz 1600. **At imgsz 640 the same frames score
0.90–0.98.** Person detection on partial bodies is not weak; it was being
measured at the one setting that breaks it.

Why: as `imgsz` rises the box tightens onto the closest, most-cropped part of
the limb (bbox area fraction falls 0.48 → 0.22 on 082938), which is further
outside COCO's whole-body distribution, not closer to it. Raising resolution
helps a small distant object gain pixels; it does not help a large partial
object that is already unambiguous, and here it actively hurts.

**Consequence for CLAUDE.md decision 2.** Phase 2's write-up raised the
single-pass question as a possibility ("the cheap configuration and the
thorough configuration are not the same pass"). This measurement makes it
concrete and forced: `person` wants 640–960, small hazards want 1280–1600,
and the gap is not a preference — at 1600, `person` drops to 0.03–0.26 on
half the frames tested, which would break Layer B outright. A single pass at
either setting sacrifices one of the two layers. **This is a Phase 4 design
decision and is not resolved here**, but Phase 4 can no longer treat it as
hypothetical.

## Finding 2 — `knife` fails in exactly the direction that matters

| frame | knife position | 640 | 960 | 1280 | 1600 |
|---|---|---|---|---|---|
| 20260805-012930 | held in hand | 0.487 | 0.172 | 0.844 | 0.665 |
| 20260805-012947 | held in hand | 0.476 | 0.885 | 0.734 | 0.276 |
| 20260805-012953 | held in hand | 0.488 | 0.411 | 0.201 | 0.107 |
| 20260805-012954 | held in hand | 0.702 | 0.173 | 0.716 | 0.317 |
| 20260805-012906 | **resting on chair, blade up** | — | — | 0.019 | 0.043 |
| 20260809-082653 | **resting beside chair** | — | — | — | 0.016 |
| 20260809-082759 | **on a shelf** | — | — | — | 0.049 |

COCO's `knife` images are overwhelmingly knives being used — held, cutting,
near food. An unattended knife on a surface is barely in the distribution.

For GuardianEye this is the worst possible orientation of the failure. A
knife in an adult's hand is a supervised knife. The hazard this system exists
to catch is the knife **left within a toddler's reach on a counter** — which
is precisely the case that scores 0.02–0.05. Detection quality is
anti-correlated with danger.

## Finding 3 — `scissors` is unstable, not absent

The scoping notes called scissors a "complete miss… not a resolution
problem" and concluded decision 7 was wrong on that basis. The stronger and
more accurate statement is that scissors are *detectable but not dependable*:

| frame (same scissors, same chair) | 640 | 960 | 1280 | 1600 |
|---|---|---|---|---|
| 20260805-012906 | 0.026 | 0.029 | 0.795 | 0.783 |
| 20260805-012930 | 0.557 | 0.765 | 0.276 | 0.805 |
| 20260805-012947 | 0.478 | 0.128 | 0.034 | 0.339 |
| 20260805-012953 | 0.382 | 0.165 | 0.029 | 0.689 |
| 20260805-012954 | 0.391 | 0.169 | 0.066 | 0.474 |

Five frames a human cannot tell apart, and at imgsz 1280 the same object
scores anywhere from 0.029 to 0.795. There is no configuration that stays
above a usable alerting threshold across all five. The instability is the
defect, more than any single miss.

Separately, there is one genuine hard failure: white scissors hanging on a
white wall (`20260809-083249`) top out at **0.045** — 0.041/0.028/0.045 at
960/1280/1600 and literally nothing at 640. Low contrast plus a mounting
position far outside COCO's distribution. That one is a real gap, not a
threshold artifact.

The notes' second scissors "miss" (`20260809-083010`) should be discounted as
evidence — the scissors there are clipped by the bottom frame edge with only
the blade tips visible. Missing that is unsurprising and proves little.

## What this means for CLAUDE.md decision 7

Decision 7 currently sorts classes into two bins: COCO already fits it
(`person`, `oven`, `knife`, `scissors`), or COCO lacks it entirely (choking
hazards, stairs) and it needs fine-tuning.

The measured reality has three bins, and `knife` and `scissors` are in the
one that doesn't exist in decision 7: **COCO has the class, and detects it
too unreliably to alert on.** They are named in decision 7 as examples of
things that already fit. They do not.

**Recommended amendment — narrow, not a rewrite.** The hybrid principle
survives intact, as does "no training from scratch." What changes is the
sorting criterion: from *"does COCO have this class?"* to *"does COCO detect
this class reliably in this deployment?"* `knife` and `scissors` move into
the fine-tuning set. This makes Phase 3 larger than planned, which is a real
cost and should be an explicit choice rather than an inference.

Fine-tuning an existing-but-weak class is cheaper than a new class from
nothing — the model already has the concept and needs domain adaptation, not
teaching — so this is a smaller expansion than the class count suggests.

## Still unmeasured — do not assume these

- **`oven` and `sink`.** Decision 7 names `oven` explicitly as an
  already-fits class. No capture contains either, and neither exists in the
  currently accessible testing areas. **Decision 7's claim about `oven` is
  untested, not confirmed.** It should not be written up as validated.
- **Stairs.** No stairs available in the accessible areas.
- **Small choking hazards** (coins, batteries, bottle caps, marbles, small
  toy parts) — available but not yet shot.
- **Chemicals / medicine bottles** — available but not yet shot. COCO has
  `bottle`; whether it fires on these, and how specifically, is unknown.
- **A toddler-sized subject at room distance.** All `person` measurements
  above are an adult, mostly partial bodies at close range. Finding 1 says
  person detection is strong at 640 *for adults*; it does not establish that
  for a small body at 3–4m, or crawling, or mostly occluded by furniture.
  This remains the single largest unvalidated assumption in the risk engine's
  input.
- **Variation.** One room, one camera position range, one lighting condition.
  `PHASE_PLAN.md` requires the dataset to span varied areas, heights, angles
  and lighting; the measurement should span them too.

---

# Session 2 — live capture, 2026-08-09 (21 frames)

Blocks A (sharp objects), B (small objects), C (person) shot. Block D
(camera-position variation) deferred — the USB cable is too short to move the
camera; waiting on an extender. **All 21 frames are 1920x1080: the `--4k` flag
did not take.** Not fatal — they are directly comparable to the 12 earlier
frames — but the 4K question is still open.

Measured: `yolo26x` x imgsz {640, 1280, 1600} x conf 0.01, 530 detections
(`cv/measurements/session2.csv`).

## Block A — `knife` is confirmed dead, across two independent sessions

Max `knife` confidence anywhere in the entire session: **0.244**, on a frame
where it is being reached for by a person. On the deliberately staged
resting-knife shots:

| shot | placement | 640 | 1280 | 1600 |
|---|---|---|---|---|
| 102050 | red knife on dark green chair, blade up, unobstructed | — | — | 0.018 |
| 102311 | knife on surface | — | 0.012 | 0.186 |
| 102720 | knife on surface | — | 0.011 | 0.015 |
| 102954 | knife on ledge, person reaching for it | 0.244 | 0.087 | 0.075 |

Nothing clears the default 0.25 threshold. Session 1 found the same thing on
different objects in different placements. **Two independent sessions agree:
stock COCO does not detect an unattended knife in this environment.** This is
no longer a provisional finding.

`scissors` continues to be unstable rather than absent — 102255 reaches 0.894,
102227 (white scissors on a white chair) goes 0.531 / 0.067 / 0.329 across
640/1280/1600. Same non-monotonic behaviour as session 1, now reproduced.

## Block B — the finding that changes the phase

The operator reported this block as a failure because "it didn't even detect
anything." The measurement says something considerably worse, and much more
useful: **the small objects are not invisible. They are confidently
misclassified as harmless things.**

| shot | object actually present | what COCO said |
|---|---|---|
| 102753 | **car key** held in an open palm | `cell phone` **0.92** |
| 102728 | small object held | `cell phone` 0.60, `toothbrush` 0.41 |
| 102739 | cigarette lighter held | `cell phone` 0.47, `bottle` 0.24, `banana` 0.18 |
| 102734 | small objects on floor | `sports ball` **0.74** |
| 102623 | objects scattered on speckled floor | `sports ball` 0.43 |

A car key reading as `cell phone` at 0.92 is a categorically different problem
from a miss. A miss leaves a gap in the hazard map. A high-confidence wrong
label **fills that gap with a benign object** — the risk engine would see a
phone lying on the floor, conclude there is nothing to flag, and stay silent
while a toddler crawls toward a swallowable metal object.

This also corrects a session-1 conclusion. For **electrical** hazards the
earlier finding holds: no class, and no substitute mislabel either (one stray
`remote` at 0.012). For **small choking hazards** it does not: there is a
stable, high-confidence substitute mislabel, and `cell phone` / `sports ball`
are it.

**Consequence for Phase 3's fine-tuning:** it is not enough to add new classes.
The fine-tune has to actively *suppress* `cell phone` and `sports ball` on
these objects, which means the training set needs those objects labelled as
the new hazard class in frames where stock COCO currently fires the wrong
label confidently. Adding a class without correcting the competing one leaves
the false positive in place.

The speckled terrazzo floor in these frames is a genuinely hard background —
the tile pattern has the same scale and contrast as the objects. That is
realistic for this building and good training material, but it should not be
the *only* floor in the dataset.

## Block C — `person` confirmed, at 640

`person` reaches 0.94–0.977 across the block, including crouching, reaching
over a ledge, and partially occluded behind a doorway. Session 1's finding
holds: person detection is strong at imgsz 640 and the earlier "person is
weak" reading was an artefact of measuring at 1600.

Note 102954 specifically — a person reaching toward a knife on a ledge, which
is close to the exact scenario the risk engine exists for. `person` 0.977,
`knife` 0.244. The system would see the child perfectly and the hazard not at
all.

## Still open after session 2

- **Block D — camera position variation.** Not shot; needs a USB extender.
  `PHASE_PLAN.md` requires the dataset to span heights and angles, so this is
  a real gap, not an optional extra.
- **4K.** The flag did not take; cause unknown.
- **`oven`, `sink`, stairs** — still no test area contains them.
- **A toddler-sized subject.** All person data remains adult.

---

# Session 3 — 4K, two camera heights, 2026-08-09 (12 frames)

All 12 confirmed **3840x2160** — the `--4k` path works on real hardware. The
previous session's 1080p result was a forgotten flag, not a bug.

Two camera positions: ~60cm (toddler eye level) and ~1.9m (corner-mount), the
second in a **new area** — a kitchenette containing a toaster oven, microwave,
kettle and wall sockets. Measured `yolo26x` x imgsz {640, 1280, 1600, 2560} x
conf 0.01 (`cv/measurements/session3.csv`).

## Finding 4 — imgsz 2560 is harmful. 4K capture does not mean 4K inference.

Shooting at 4K tempted an obvious next step — raise `imgsz` to match. It makes
things sharply worse, near-universally:

| class (best frame) | 640 | 1280 | 1600 | **2560** |
|---|---|---|---|---|
| `microwave` | 0.943 | 0.950 | 0.890 | **0.126** |
| `oven` | 0.449 | 0.751 | 0.872 | **0.261** |
| `scissors` (104604) | 0.975 | 0.967 | 0.952 | **0.603** |
| `person` (104824) | 0.959 | 0.185 | 0.461 | **0.075** |

This is the same mechanism as Finding 1, pushed further: YOLO26 is trained at
640, and the further the inference scale departs from that, the more every
object is rendered at a size the model never saw. Small objects gain from
1280–1600 because they start below the useful scale; everything else is
already fine and only loses.

**Practical rule: capture at 4K, infer at 640–1600.** The extra pixels are
worth having in the dataset — they are detail available to label and to train
on, and they cannot be recovered later — but they are not an inference
setting. Nothing in this project should run above 1600.

## Finding 5 — `knife` is driven by scene context, not size, distance or pixels

The kitchen frames finally produced a strong knife score, and comparing them
against the floor frames explains every earlier result:

| shot | knife placement | best conf |
|---|---|---|
| 105154 | on a **kitchen counter**, beside a toaster oven and kettle | **0.827** |
| 105134 | on a kitchen counter | 0.628 |
| 104604 | on the **floor**, 60cm camera, large in frame, beside a reaching hand | 0.214 |
| 104758 | on the floor | 0.287 |
| 105219 | on the floor of the adjacent room | 0.100 |
| 105255 / 105256 | on the floor | 0.057 / 0.112 |

**Frame 104604 is the decisive one.** 4K capture, camera at 60cm, a red-handled
knife large and unoccluded right beside a reaching hand, good light. `knife`
tops out at **0.214**. In the *same frame*, at the *same resolution and
distance*, `scissors` reaches **0.975**.

So the variable is not size, not resolution, not lighting, not occlusion — all
of those are controlled by that single frame. It is **context**. COCO's knife
images are knives in kitchens: on counters, near food, near appliances, in
hands. Put the identical knife on a floor and it stops being recognisable to
the model.

This is the worst possible shape for this project. A knife on a kitchen counter
is a knife in its normal place. A knife on the floor where a crawling child is
about to reach it is the emergency — and it is the case that scores 0.06–0.21.
Finding 2 said detection was anti-correlated with danger; session 3 shows *why*,
and confirms it survives 4K capture, a toddler-height camera, and a knife
occupying a large share of the frame.

## Finding 6 — `oven` measured for the first time; decision 7's claim holds here

The kitchenette settles a question open since session 1. The toaster oven is
detected as `oven` up to **0.872** and as `microwave` up to **0.965**,
consistently across all five kitchen frames at 640–1600.

Two caveats worth keeping:
- **It fires as both classes at once.** For a hazard map that is probably
  acceptable — both mean "hot appliance" — but any code keyed on a single
  class name will behave unpredictably. Phase 4 should treat these as a group,
  not as distinct hazards.
- This is a countertop toaster oven, not a full range/stove. A built-in oven at
  floor level, which is the actual toddler hazard, is still unmeasured.

`sink` remains unverified — max 0.085 across every frame; no real sink is
visible in any capture.

## Camera height

`person` at the 60cm toddler-height position holds up well (0.956 crawling,
0.913 crouching), so the low mounting angle costs nothing for Layer B. Both
heights are usable; neither is disqualified.

## Updated class list

| verdict | classes |
|---|---|
| **Detects reliably** | `person` (at 640–1600), `refrigerator`, `chair`, `oven`/`microwave` as a pair |
| **Detects unreliably** | `scissors` — 0.975 in one frame, 0.02 in another |
| **Effectively misses** | `knife` outside kitchen contexts — the case that matters |
| **Misses entirely, no class** | sockets, power strips, cables; small choking hazards (and these are *confidently mislabelled*, see session 2) |

---

# Round 1 fine-tune: measured generalisation failure (2026-08-10)

A first fine-tune (`yolo26l`, 92 labelled office frames, 79 train / 13 val,
leak-free split by duplicate group) was evaluated against **32 labelled frames
from a second building** the model had never seen.

| | precision | recall |
|---|---|---|
| validation split (same building) | 0.902 | 0.607 |
| **home frames (unseen building)** | **0.246** | **0.147** |

At the default 0.25 threshold the model finds **15% of hazards** and **three of
every four detections are wrong**. Lowering the threshold to 0.05 lifts recall
to 0.29 but leaves precision at 0.23 — no threshold rescues it.

**Correction to an earlier claim in this document.** The decisive-frame results
reported for the fine-tuned model (floor knife 0.214 → 0.861, car key → 0.930)
were all measured on frames from the **training building**. They are real
evidence that the model learned those objects; they are not evidence that the
approach generalises, and they were initially presented as though they were.

**What this establishes:**

- 92 images from one building is not enough — now quantified rather than
  assumed.
- A leak-free train/val split is necessary but **not sufficient** to detect
  this. 13 validation images from the same building cannot distinguish learning
  from memorisation. Only a second location can.
- The 32 home frames are now a real held-out test set. Every future change is
  measurable against precision 0.246 / recall 0.147.

**Consequence for the plan:** the home frames move into training (they are
labelled); a **third** location becomes the new test set. Public data (Open
Images) is now justified — the measured weakness is visual diversity, which is
exactly what one more building cannot supply and a public corpus can.

# Local VLM as Layer A safety net: rejected (2026-08-10)

Tested Florence-2 (base and large) locally on MPS — the only candidate that
installed cleanly under Python 3.14. **Verdict: cannot serve as Layer A's
safety net.**

**The disqualifying finding: prompted grounding has no "not found" case.**
`<CAPTION_TO_PHRASE_GROUNDING>` returns a confident, tightly-drawn box for
whatever text you give it, whether or not the object is present:

| frame | asked for | returned |
|---|---|---|
| car key in palm | `"knife"` | confident box **on the car key** |
| car key in palm | `"medicine bottle"` | same box, relabelled |
| floor knife | `"cigarette lighter"` | confident box **on the knife** |
| wall socket | `"banana"` | confident box **on the socket** |

It snaps to the most salient region and applies whatever label was requested.
"Did grounding return a box for knife" is therefore not a usable presence
signal — for a safety system this fails silently and convincingly, which is
worse than a miss.

**What did work:** unprompted discovery (`<OD>` / `<DENSE_REGION_CAPTION>`) does
not appear to invent boxes over empty space, produced usable pixel-space boxes
suitable for decision 5's centre-distance maths, and scored one genuine win — it
found and correctly named a **floor-level power outlet** unprompted, a class
stock COCO structurally cannot produce. Speed is fine: 0.28–0.55s per call on
MPS, well inside a change-triggered budget.

**But it is sparse, inconsistent across near-identical frames, and mislabels
hazards as other hazards** — Florence-2-large described the floor knife as
*"person cutting tile floor with red scissors."* That is the exact
knife/scissors confusion Findings 3 and 5 document, now with more confidence
behind it.

**Not tested:** Moondream and Qwen2.5-VL (stopped once Florence-2 gave a
decisive answer); whether a yes/no VQA-style prompt would avoid the grounding
hallucination; behaviour at native 4K.

**Consequence:** Phase 4 builds on fine-tuned YOLO alone. If revisited later,
the useful direction is not a bigger model but requiring independent
confirmation — accepting a grounded box only where unprompted discovery also
flags something in that location. That is real engineering, and a decision to
take deliberately rather than drift into.

---

# Round 3: mixed training with oversampling — worse, and why (2026-08-11)

Round 3 trained on public + own data **simultaneously**, with our 79 office
frames repeated 14× to reach parity with the 1,078 public images (2,184 train
entries). Hypothesis: keeping class 1 supervised throughout would retain round
2's `sharp_object` gain while restoring `small_swallowable`.

**It did the opposite.** All figures on the same 32 held-out home frames, conf
0.25, same verified scorer:

| | R1 (office only) | R2 (sequential) | R3 (mixed, 14× oversample) |
|---|---|---|---|
| `sharp_object` recall | 0.097 | **0.339** | 0.032 |
| `small_swallowable` recall | 0.204 | 0.074 | **0.000** |
| pooled precision | 0.246 | 0.179 | 0.133 |
| pooled recall | 0.147 | **0.216** | 0.017 |

`small_swallowable` produced **zero** true positives at every threshold.

**This is not a training failure.** The run was healthy on its own validation
split — mAP50 0.73 at its best epoch (26), precision 0.912 by epoch 50. It
learned the office frames extremely well and transferred almost nothing.

**Diagnosis: oversampling by duplication caused memorisation.** With a 14×
repeat factor, the model saw each of 79 office frames **364 times** (14 per
epoch × 26 epochs) before its best checkpoint. That is enough to memorise them
outright. Round 2's sequential schedule accidentally protected against this —
95 epochs on 1,078 *distinct* public images built general features first, and
the short fine-tune afterwards could not erase them.

**The lesson, which matters more than the ranking:** what generalises is the
number of **distinct scenes**, not the number of training *instances*.
Duplicating 79 images into 1,106 entries adds no information and actively
encourages memorisation. No scheduling trick converts 79 distinct frames into
enough diversity for a new building.

**Consequences:**
- **Round 2 (sequential) remains the best model** — `sharp_object` recall 0.339
  on an unseen building.
- Oversampling by duplication is ruled out as a lever. If class balance needs
  fixing later, use loss weighting rather than repeated file entries.
- This reinforces the 2026-08-10 reframe rather than contradicting it: three
  training schedules on the same 92 frames produced 0.147, 0.216 and 0.017
  recall. The variable that never changed is the one that matters — the number
  of distinct rooms the model has seen.

---

# Change detection: re-measured with real person-suppression (2026-08-11)

A docs-agent audit found CLAUDE.md decision 7's claim for classical frame
differencing — **"~1.4ms/frame, and it correctly boxed a car key and a
lighter that stock COCO called `cell phone` at 0.92"** — has zero supporting
evidence in the repo: no script, no CSV, nothing. Person-overlap suppression,
named at the time as the fix for the dominant false-positive source, was
never actually implemented either. This section replaces both gaps with
`cv/measure_change_detection.py` (committed) and
`cv/measurements/change_detection.csv` (gitignored, same policy as every
other measurement CSV in this repo — see "Evidentiary status" below for what
that means in practice).

**Method:** grayscale + blur + `absdiff` + threshold(25) + dilate + contours,
filtered to bbox-area-fraction >= 0.0003. Every candidate blob is checked
against the union of `yolo26l.pt`-detected person boxes (imgsz 640, conf
0.25, both frames of the pair) and suppressed if the person-box overlap
covers >= 50% of the blob's own area. Full rationale for every threshold is
in the script's module docstring.

**Bursts measured** (auto-grouped by filename session tag, consecutive pairs
kept only if <= 15s apart): `home-60cm` (31 pairs), `room1-60cm` (23),
`room1-190cm` (22), `room2-60cm` (37), `room2-190cm` (30), and an `untagged`
group covering the pre-tagging captures (44 adjacent frames, only 10 pairs
kept — the rest were tens of minutes to over 4 days apart and correctly
excluded as meaningless for change detection).

## Recall against labelled ground truth (IoU >= 0.5, unsuppressed blobs only)

| session | gt_total | gt_found | recall |
|---|---|---|---|
| `home-60cm` | 106 | 2 | 0.019 |
| `room1-190cm` | 79 | 1 | 0.013 |
| `room1-60cm` | 70 | 3 | 0.043 |
| `untagged` | 16 | 2 | 0.125 |
| **pooled** | **271** | **8** | **0.030** |
| `room2-60cm` / `room2-190cm` | — | — | not labelled yet, blob counts only |

This is low, and the reason is structural, not a detector defect: `home-60cm`,
`room1-60cm` and `room1-190cm` are static-scene labelling bursts — frames of
the **same already-present objects**, ~3s apart, shot to capture them from
slightly different moments. Frame differencing cannot find an object that is
present and unmoving in both frames of a pair; there is nothing to threshold.
This is exactly CLAUDE.md decision 3's own split (change detection solves
"appears while running," not "already there at startup") — this measurement
confirms it rather than contradicting it. Pooled recall 0.030 is the honest
number for what these three bursts actually test, which is close to nothing
relevant to this method.

## Precision by eye (real sample, not vibes)

Stratified random sample of 36 unsuppressed blobs (6 per session, seed 7),
cropped and inspected directly (`cv/measurements/precision_sample_sheet.jpg`,
gitignored — home photographs). **Roughly 10–12 of 36 (≈28–33%) are a
distinct real object** (a cable on the floor, a warning sticker, a small
bottle/jar, what looks like a wall socket, a marker). The rest — the clear
majority — are floor-tile speckle/grout seams, wall/furniture edges, and
shelf-corner lines: the same terrazzo-floor problem Session 2's Block A
findings already flagged as a hard background for this building, now shown to
also defeat pixel-level differencing, not just single-frame classifiers.

## Timing (measured on this machine, MPS)

| stage | mean | median | notes |
|---|---|---|---|
| frame-diff only | 5.825ms | 5.715ms | n=153 pairs, mixed 1080p/4K |
| frame-diff only, 1080p pairs specifically | **1.37–1.43ms** | — | matches the old claim almost exactly |
| person-suppression (2x YOLO calls) | 37.043ms | 35.812ms | not part of the original claim |
| **combined** | **42.868ms** | — | the real per-pair cost once suppression exists |

## The specific claim: car key and lighter

Frames `20260809-102739` (lighter) and `20260809-102753` (car key), and their
in-burst predecessors, were run through the pipeline and an annotated
before/after crop was saved for each pair
(`cv/measurements/verify_20260809-102734_..._20260809-102739_....jpg` and
`...102739_..._102753_....jpg` — gitignored, real home photos, viewed
directly for this write-up).

**Verdict: the claim does not hold up under direct measurement.**

- Every blob produced by both pairs is suppressed as person-overlap (the
  object is held in an open palm, so its motion region is inside the arm/hand
  silhouette) — after suppression, the detector emits **zero boxes** for
  either frame.
- Independent of suppression, none of the raw (pre-suppression) blobs are a
  tight box on the key or lighter either. The largest blobs are a
  half-frame-tall region and a full-height vertical strip (door-frame
  lighting/exposure shift between shots), not an object-sized box. Checked
  by eye against both saved crops: no blob boundary tracks the object's
  actual outline.
- Checked quantitatively too: neither frame's labelled `small_swallowable`
  ground-truth box (both frames are labelled) is matched by any blob at
  IoU >= 0.5, suppressed or not.

**Most likely explanation for the original claim:** the frame-and-metric it
actually describes — "car key/lighter read as `cell phone` 0.92 by stock
COCO" — is real, measured, and already correctly documented, but it is
**stock YOLO's own classifier output** (Session 2, Block B, above), not a
change-detection result. CLAUDE.md decision 7's sentence attributes it to
change detection, contrasting it against "what stock COCO called it," which
reads as a conflation of two different methods measured in the same session
rather than a second, independent confirmation. This document does not have
direct evidence for which happened (no script existed to check), but the
newly-measured data actively contradicts change detection being the source
of that specific result.

**What is confirmed:** the ~1.4ms/frame figure, for pure differencing on
1080p frames specifically. **What changes:** person-suppression, once
actually implemented, costs roughly 30x more than the diff step itself
(~37ms vs ~1.4ms) — the "cheap" framing was true only for the part that
wasn't doing the job identified as necessary. **What is refuted:** the car
key / lighter detection, specifically, as an achievement of change detection.

CLAUDE.md decision 7 and docs/decision-log.md should be corrected to reflect
this — flagged here rather than changed unilaterally, per this repo's rule
that CLAUDE.md changes need Shaked/Yahli's sign-off.

## Evidentiary status of the other four single-frame methods (checked, not assumed)

- **Class-agnostic segmentation** (`cv/measure_segmentation.py`): the script
  is real and committed. Its output CSV (`cv/measurements/segmentation.csv`)
  **is present on disk** (48 rows, last generated 2026-08-10) — it was not
  lost. However, a repo-wide check confirms its numbers **never made it into
  any findings write-up**: no mention of "segmentation" or "FastSAM" appears
  anywhere in this file or in `docs/phase-writeups/phase-3-step0.md`. The
  code and a real CSV exist; the citable prose does not. Writing that section
  is out of this task's scope (this task is about change detection) but is
  now a concretely flagged, narrow open item rather than an assumed gap.
- **Colour clustering and texture-based objectness:** confirmed, again, to
  have **no committed code anywhere in the repository** (no matching
  filenames, no matches for characteristic terms in a `cv/*.py` grep). Their
  numbers (recall 0.09; recall 0.79 / ~33% eyeballed precision) remain bare
  assertions in CLAUDE.md and the decision log with no way to check them from
  this repository alone. Per the task that requested this section, rebuilding
  them is explicitly out of scope here — this is a plain statement of their
  evidentiary state, left as an open item.

---

# Change detection, take 2: persistence tracking instead of single-pair
# person-suppression (2026-08-11)

The section above ("Change detection: re-measured with real
person-suppression") is kept as-is below this one — it is the OLD, now
superseded method, not deleted, per this document's own convention for
tracking what changed and why. **This section replaces its conclusions,**
using a rewritten `cv/measure_change_detection.py` (same file, in place) and
a fresh `cv/measurements/change_detection.csv`.

## What was wrong with the old method (confirmed, not assumed)

The previous measurement suppressed any change-blob overlapping a person box
above 50% of the blob's own area, on a single frame pair. Two problems were
already suspected going in and are now confirmed directly:

1. **It deletes real hazards, not just the person.** An object in an open
   palm moves as part of the hand/arm silhouette, so the blob that captures
   it gets discarded along with the person — which is exactly why the old
   section above reports the car-key/lighter claim as refuted: not because
   the object didn't move, but because the one fix in place erased anything
   touching the person, including the object being placed.
2. **The old test bursts (`home-60cm`, `room1-60cm`, `room1-190cm`) cannot
   exercise "object appears and is left."** They are static-scene labelling
   bursts, ~3s apart, of the same already-present objects. There is
   structurally no diff to threshold for something unmoving in both frames
   of a pair.

## The fix: persistence tracking, with a hard lesson along the way

The new method: for a blob found by diffing frame[N-1] against frame[N],
look ahead to frame[N-1+K] for K in {2, 3} and re-diff frame[N-1] (the
pre-change reference) against that later frame. If a blob still shows up in
roughly the same place (centroid within 4% of the frame diagonal,
`POSITION_TOLERANCE_FRAC`) and it is not covered by a person box at that
later frame, the detection survives — matching the task's literal
instruction to check "at frame N+2 or N+3."

**The first version of this, location-only, does not work — measured
directly, not assumed.** Run against `changetest`'s first pair
(`20260811-021847` → `021848`), a true negative control (nobody in frame,
nothing placed, camera static), location-only persistence kept **57 of 69**
raw blobs as "real." Every one checked by eye
(`cv/measurements/verify_20260811-021847_..._021848_....jpg`) is floor
grout, a whiteboard diagram line, or glare on the firefighter jacket's
plastic wrap — not one is an object. The mechanism: those are the
highest-local-contrast edges in the frame, so any tiny lighting/exposure
difference between two exposures lights them up as a diff blob **at the
same location every time**, because the location is fixed by the physical
grout pattern, not by anything moving. Location persistence alone cannot
distinguish "the same object is still there" from "the same edge relights
slightly differently on every comparison."

**Traced further, this specific pair turned out to be a camera-settling
transient, not steady-state floor noise** — a second, separate finding worth
keeping. Per-pair raw blob counts for the first six pairs of `changetest`:
69, 104, then **zero, zero, zero**, then small counts (2–14) once a person
enters frame. The camera's auto-exposure/white-balance was still converging
for the first two captures after this recording session started; once
settled, frame-to-frame floor diff genuinely drops to nothing until real
motion (a person) re-enters. A deployed system should simply discard the
first few frames after camera init, same as any auto-exposure device — but
it means roughly half of `changetest`'s 332 raw blobs (173 of them) come
from a two-frame startup artifact specific to this recording, not from an
ongoing floor-noise problem. All precision/recall numbers below are reported
**with and without** this pair so the startup transient doesn't quietly
inflate or deflate the real number either way.

**The actual fix that reduced the false-positive rate:** require the
candidate's own bounding box to also be *visually stable* between frame N
and frame N+K — crop both frames to the candidate's bbox (valid because the
camera is static) and re-run the same diff pipeline restricted to that crop;
if more than 30% of the crop's pixels still differ (`STABILITY_MAX_CHANGE_FRAC`),
the region is still changing (a relighting edge, a hand still moving) and
the check fails. A blob must pass BOTH the location check and the stability
check at K=2 or K=3 to be kept. This is a real, necessary second gate, not a
redundant one — see the numbers below for what it did and didn't fix.

## Does it catch a hazard placed and left? Yes — checked by eye, real timestamps

`changetest` (22 frames, ~1s apart, 20260811-021847 through 021909):

- **021847–021852:** camera settling, then a firefighter jacket on a chair,
  empty tiled floor, nobody in frame — static.
- **021853:** a person enters frame from the right.
- **021855:** the person's hand is mid-air, dropping a pair of white-handled
  scissors toward the floor.
- **021857–021858:** the scissors are resting on the floor tile, the person
  still standing nearby (foot close to but not on the scissors).
- **021900–021903:** scissors still on the floor, person standing still a
  short distance away.
- **021904–021906:** the person bends down, picks the scissors back up, and
  carries them off toward the cabinet.
- **021907–021909:** scissors gone from frame, floor empty again.

**The detector catches it.** In the primary pair `021857`→`021858`, one of
the kept (persistence-confirmed) blobs is bbox `(2027, 2061)–(2098, 2151)` in
full 4K pixel space — cropped and viewed directly
(`/tmp/scissors_crop_check.jpg` during this session; reproducible by cropping
`cv/captures/20260811-021858_..._raw.jpg` at that box), it is a tight,
correctly-shaped box on the scissors' handles, nothing else. This is the
scenario CLAUDE.md decision 3's guided-walkthrough model and decision 6's
critical-event trigger both depend on: something appears on the floor while
the person who placed it steps back, and it is still flagged after they've
moved away. **On this one real example, it works.**

It is not caught immediately — no blob matches the scissors location until
the primary pair that includes the settled floor shot, and the confirming
lookahead (K=2/K=3) adds a further ~2 frames (~2s in this burst) of latency
before the detection is trusted. For a ~5s rolling-buffer critical-alert
design (CLAUDE.md decision 6) this latency is likely fine; it has not been
measured against that specific budget.

## Precision: still bad, and the persistence fix does not rescue it

Same pair (`021857`→`021858`) also kept 46 of 53 raw blobs — one of those
46 is the scissors; the rest, sampled and checked by eye (stratified sample,
seed 7, `/tmp/sample_crops/`), are floor tile, a whiteboard/cabinet edge, and
plastic-wrap glare near where the person had just been standing — i.e. the
same failure mode as before, just narrower. Across the whole `changetest`
burst, excluding the two-frame startup transient, only **1 of 51** kept
blobs sampled/inspected was a real object (the scissors) — the remainder
are edges near the person's recent position that happened to pass both the
location and stability gates once the person moved on. **Person-adjacent
noise that settles into a stable-looking artifact after the person leaves is
a real, unsolved failure mode** — the stability check catches literal
relighting flicker (see the startup-transient case) but not motion residue
(shadow, plastic-wrap crease shift, blur) that happens to stabilize once the
person is gone.

On the three previously-labelled static bursts plus the untagged group
(same IoU>=0.5 scoring, reused from `evaluate_home_frames.py`):

| session | gt_total | gt_found | recall (old, single-pair) | recall (new, persistence) |
|---|---|---|---|---|
| `home-60cm` | 106 | 1 | 0.019 | 0.009 |
| `room1-190cm` | 79 | 0 | 0.013 | 0.000 |
| `room1-60cm` | 70 | 1 | 0.043 | 0.014 |
| `untagged` | 16 | 1 | 0.125 | 0.062 |
| **pooled** | **271** | **3** | **0.030** | **0.011** |

**Recall got worse on the labelled bursts, not better.** This is the
expected, honest cost of adding two more confirmation gates on top of an
already-low-recall method on data that structurally can't exercise
"appears and is left" — the persistence requirement asks for more evidence
than these bursts can usually supply, so fewer of the (already rare)
correct detections survive to be counted. This is a real regression on
this data, not noise — stated plainly rather than only reporting the
`changetest` win.

**Precision, stratified sample (6 per tag, seed 7, 24 total, checked by
eye):** roughly 8 of 24 (≈33%) are a real, distinct object — a marker/pen,
a cable, cables looped on a chair, scissors on a chair, a wire on a kitchen
floor. This is in the same ballpark as the old method's eyeballed 28–33%,
**not an improvement** — the persistence fix changed *which* false positives
get through (fewer transient-lighting artifacts, more person-adjacent
settling artifacts) without moving the overall precision number. Where it
did work well: the `untagged` group (frames that happen to already contain a
genuinely static, high-contrast object like scissors on a chair or a coiled
cable) — 5 of 6 sampled kept blobs there were real objects, because those
objects were trivially stable across every comparison from the start.

## Honest bottom line

- **The specific claim this task asked to re-check — does the detector catch
  a hazard placed and then left — now holds, with a real example, checked by
  eye, timestamps included.** That part of the original CLAUDE.md decision 7
  claim's *spirit* is no longer refuted.
- **Precision is still not good enough to alert on, and recall on
  non-purpose-built data got worse.** Persistence tracking fixed the specific
  failure mode it targeted (fast-moving hand overlap, and pure lighting
  flicker) but did not fix — and cannot, by itself, fix — noise that happens
  to be spatially near a person and settles into something stable once they
  leave, or the fact that requiring multiple future frames of confirmation
  structurally lowers recall on sparse data. **This is a second honest
  negative result**, not a success dressed up: change detection with
  persistence tracking is a better-understood, better-instrumented method
  than before, and it is still not something to alert a parent on
  unfiltered. Any future use of it should pair it with something else (e.g.
  requiring the confirmed region to also fall inside a Layer-A "reachable
  surface" mask, or a minimum absolute size) rather than being trusted alone.

## Timing (real, measured on this machine, MPS, full sweep across all seven
## session groups — 174 primary pairs)

| stage | mean | notes |
|---|---|---|
| primary-pair frame-diff | 5.8ms | same metric as before, unchanged method |
| lookahead frame-diff (K=2/K=3) | 6.2ms | 306 unique lookahead diffs; 4,766 cache hits avoided recompute within the run |
| person-detection (YOLO) | 20.6ms | 187 unique frames detected; 2,553 cache hits avoided recompute |
| **amortized cost per primary pair** | **38.8ms** | (all diff work + all person-detection work) / number of primary pairs — the realistic incremental cost in a sequential live stream where each frame's person boxes and each unique frame-pair diff are computed once and reused |

**This is not still "~43ms/pair."** It is close to that number by
coincidence on this particular sweep, but it is now a different and more
honest quantity — it includes the lookahead diffs and reflects heavy caching
that only makes sense in a system processing frames in order (a frame's
person boxes, once computed, are reused both as "frame b of pair N" and
"lookahead frame for pair N-2"). Caching is a legitimate real-system
design, not a trick to shrink the number — but it should be named
explicitly, which this table now does, rather than left implicit.
