# Phase 3 — dataset collection plan

Follows Step 0 (`docs/phase-writeups/phase-3-step0.md`). Step 0's measurements
define this plan; do not treat any target here as a guess.

## Classes

Two classes to label. Everything else is either handled without training or
deferred.

| class | contents | why |
|---|---|---|
| `sharp_object` | knives, scissors | Step 0 measured that both stock COCO and open-vocab models **confuse knives with scissors on floors** — on frame 104604 the model labelled the knife `scissors` at 0.78 with `knife` explicitly available. Merging deletes a discrimination we have measured we cannot win, and doubles the examples per class. The parent-facing alert is identical either way. |
| `small_swallowable` | keys, coins, batteries, bottle caps, lighters, small toy parts | No COCO class, and stock COCO **confidently mislabels** these (`cell phone` 0.92 on a car key). Open-vocab did not help (`car key` 0.20, `cigarette lighter` 0.017). |

**Not collected:**
- **Electrical** — solved by YOLO-World's `wall socket` prompt at 0.48–0.90, no
  training data required. Do not label sockets.
- **Stairs** — no accessible test area contains any. Deferred, not cancelled.
- **`person`, `oven`/`microwave`, `refrigerator`, `chair`** — stock COCO already
  handles these. Labelling them risks *regressing* what already works.

## How many images

**Round 1 target: 150 labelled images per class (~300 total), then train and
measure before collecting more.**

This is deliberately not the 500–1000/class figure often quoted. Two reasons:
we are fine-tuning a model that already has related concepts rather than
teaching from scratch, and Step 0's whole lesson is that collecting on
assumption wastes effort. Train at 150, measure with `cv/measure_detection.py`
against the stock baseline we already have, and let the failures say what to
collect next. A second targeted round aimed at measured weaknesses beats a
first round three times the size.

Counting rule: an image counts once per class it contains. A frame with a
knife and a coin serves both.

**Roughly 40% of frames should contain a person as well as a hazard** — the
deployed system sees a child near a hazard, so the training set must too.

## Negatives — this is not optional

**~20% of the set (≈60 frames) must contain NO hazard at all**, and the most
valuable of those are **hard negatives**: objects that stock COCO already
confuses with our classes.

- For `small_swallowable`: an actual phone, a TV remote, a ball, a bottle cap
  *next to* a real phone. Step 0 measured `cell phone` at 0.92 and
  `sports ball` at 0.74 firing on our hazards — the fine-tune has to learn the
  boundary, and it can only do that if both sides are present.
- For `sharp_object`: pens, spoons, forks, rulers, cutlery on a counter.

A model trained only on positives learns "there is always a hazard somewhere"
and will invent one. In a child-safety system that means false alarms, which
train the parent to ignore it — a worse failure than a miss.

## Variation — the part that decides whether this generalises

`PHASE_PLAN.md` requires the dataset to span real deployment variation. Step 0
proved why: `knife` scored 0.827 on a kitchen counter and 0.21 on a floor.
**Context is what the model keys on**, so context is what must vary.

Do not shoot the full cross-product. Make sure each dimension is well covered
across the whole set:

| dimension | must include |
|---|---|
| **area** | 4–6 distinct rooms/areas. Currently we have 2, both in one building. This is the largest gap. |
| **camera height** | ~60cm (toddler eye level), ~1.2m, ~1.9m (corner mount) |
| **angle** | straight-on and oblique; looking down and looking across |
| **lighting** | daylight, artificial, mixed, and one dim/evening set |
| **surface** | dark (fabric chair), light (counter), **patterned** (the speckled terrazzo — genuinely hard, and realistic), reflective |
| **distance** | ~0.5m, ~1.5m, ~3m |
| **object state** | single, several together, partly occluded, at frame edge, varied orientation, overlapping each other |

**Different buildings matter more than more frames in this one.** If you can
shoot in a home as well as the office, do — that is worth more than another
100 frames of the same corridor.

## Session structure

One session = **one area at one camera height**. Within it, vary surface,
distance, orientation and object count. ~30–40 frames per session, 8–10
sessions total.

Use interval capture and keep rearranging between saves:

```bash
source .venv/bin/activate && python cv/detect_stream.py --name Arducam \
  --model yolo26l.pt --imgsz 640 --4k \
  --interval 3 --session-tag kitchen-60cm-daylight
```

Rules that matter:

- **`--4k` every time.** Step 0 confirmed it works; a forgotten flag already
  cost one session. Check the startup line says `3840x2160` before starting.
- **Keep moving between saves.** Interval capture rewards constant
  rearrangement and punishes standing still — 40 frames of a static scene is
  one data point recorded 40 times.
- **Do not chase green boxes.** Compose for realism. Frames where detection
  fails are the point of the exercise.
- **Run `cv/triage_captures.py` after every session**, before labelling. It
  finds near-duplicates and blurred frames. Duplicates are the main way an
  interval-captured set quietly becomes worthless.

## Order of work

1. Shoot 2 sessions, triage them, and label ~30 frames.
2. **Stop and check the labelling workflow end to end before shooting the rest.**
   Discovering a labelling problem after 300 frames is expensive; after 30 it
   is nothing.
3. Shoot the remaining sessions.
4. Triage, label, train, then measure against the stock baseline with
   `cv/measure_detection.py`.
5. Let the measured failures define round 2.

## What must not happen

- **No cloud.** No Roboflow, no hosted labelling, no uploads. Local only
  (`PHASE_PLAN.md`, and the 2026-08-05 decision-log entry).
- **Do not label sockets** — YOLO-World already handles them.
- **Do not label `person`** — stock COCO is at 0.90–0.98 and fine-tuning over
  it risks regression on the one class Layer B depends on.
</content>
