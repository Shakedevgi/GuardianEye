# Phase 3, Step 0 — Measure Before Collecting

**Status: Step 0 closed (2026-08-09). Phase 3 as a whole is NOT closed — no
dataset has been collected, nothing has been labelled, and no fine-tuning has
run. This write-up covers Step 0 only: replacing assumption with measurement
about what stock COCO can and can't see in this deployment, so Phase 3's
actual data-collection work can be scoped correctly instead of guessed at.**

If you're reading this deciding whether Phase 3 can start: the class list at
the end of this document is the real answer to "what do we need to
fine-tune," and it is allowed to be trusted. But `PHASE_PLAN.md`'s Phase 3
checkbox stays `[ ]` — collecting images, running a labelling tool, and
training a model are all still ahead, and this document is explicit about
what the measurement did *not* cover, because those gaps will otherwise get
silently assumed away the moment someone starts shooting a dataset.

## What Step 0 was supposed to prove

Per `PHASE_PLAN.md`: Phase 2 discovered that input resolution, not model
size, dominates small-object detection (`docs/phase-writeups/phase-2.md`).
Every "COCO can't see this" observation made before that discovery — at the
default `imgsz 640` — was measured at a setting now known to starve small
objects. So going into Phase 3, the actual list of classes needing
fine-tuning was unknown, not just imprecise. Step 0's job was to re-measure
properly, across the real testing areas, and produce that list — and, as an
explicit side effect, settle whether CLAUDE.md decision 7 (fine-tune only
what COCO structurally lacks) still holds or needs amending.

## What was actually built

- **`cv/measure_detection.py`** (new). An offline batch harness: given a
  glob of already-saved `*_raw.jpg` frames, it sweeps `model × imgsz × conf`
  as a full cross-product and writes one CSV row per detection — class name,
  confidence, bbox coordinates, bbox area as a fraction of the frame. It does
  not touch a camera, does not open a window, and explicitly refuses to run
  on `*_annotated.jpg` files (which already have boxes burned into the
  pixels — re-detecting those would be measuring the model's own drawings).
  I read the file in full; this is exactly what its own docstring claims.

- **A self-timer (`t` key) and an opt-in `--4k` capture mode**, added to
  `cv/detect_stream.py` and `cv/camera.py` on 2026-08-09 to make live data
  collection for Step 0 actually practical. The self-timer exists because
  several of the poses that matter most for this project — an adult crouched
  or reaching, standing in for a toddler — require the operator to *be* the
  subject, three meters from the keyboard. `--4k` requests the Arducam's true
  `3840×2160` sensor resolution (via a required `MJPG` pixel format — see
  `camera.py`'s docstring for why uncompressed 4K exceeds USB bandwidth on
  this device) instead of the default `1920×1080`. Both are additive and
  opt-in: I confirmed by reading `detect_stream.py`'s `main()` that with
  `--4k` absent, `request_kwargs` stays an empty dict and `CameraCapture(...)`
  is called exactly as it always was — the default path is unchanged.

## Methodology, and what I actually verified about it

Two choices matter and I checked both against the code, not just the prose
that describes them:

- **Confidence floor 0.01, not the live-preview default of 0.25.** Confirmed:
  `DEFAULT_CONF = [0.01]` in `measure_detection.py`, with a comment
  explaining why — 0.25 cannot distinguish "the model said nothing" from
  "the model said something weak that got thrown away." Those failure modes
  have different remedies (fine-tune a missing concept vs. tune a threshold),
  so a floor near zero is what lets the CSV tell them apart.
- **Offline re-measurement on raw frames, not re-shooting live at every
  setting.** `imgsz`/`model`/`conf` are all inference-time arguments to
  `model.predict()`, not properties baked into the saved image — `frame =
  cv2.imread(image_path)` is read once and then run through every
  combination in the sweep. This is what makes ~45 total live captures
  sufficient to explore a 4-value `imgsz` grid without ~180 live shots.
- **The measurement never trusts the filename for its own configuration.**
  This is the claim I was most skeptical of going in, since it would be an
  easy thing to get subtly wrong. `capture_tag_from_filename()` extracts
  what config *captured* the frame (e.g. `..._yolo26x_imgsz1600_raw.jpg` →
  `imgsz1600`) into a column explicitly named `capture_tag`, kept separate
  from `model`/`imgsz`/`conf_floor`, which record what *this run* actually
  asked `predict()` for. I checked this against the CSV, not just the code:
  `cv/measurements/detections.csv` row 2 has `capture_tag =
  20260805-012906_yolo26x_imgsz1600` and `imgsz = 640` in the same row — the
  frame was captured at one setting and is being re-measured at a different
  one, exactly as the docstring says is the entire point of the script. This
  is real, verified evidence the separation works, not just a comment
  promising it does.
- **Spot-checked numbers.** I picked roughly fifteen specific confidence
  values quoted in `docs/phase-3-step0-findings.md` and traced each back to
  the actual CSV row that must have produced it — the `person` inversion
  table (`20260805-012930`: 0.905/0.923/0.833/0.793 across imgsz 640-1600),
  the held-knife numbers in Finding 2, and — the one I was most suspicious
  of, since it's the headline result of session 2 — the "car key reads as
  `cell phone` at 0.92" claim. That number does not appear at `imgsz 640`
  (max 0.61 in the CSV) or `imgsz 1600` (max 0.71); it's specifically at
  `imgsz 1280` (`session2.csv`, frame `20260809-102753`: `cell phone,
  0.9161`). Every value I checked matched, including this one, which is
  reassuring precisely because it would have been easy to fudge a headline
  number and hard to notice — the actual value depends on picking the right
  one of three `imgsz` settings.
- **Detection counts.** The findings document states "528 detections" for
  session 1 and "530 detections" for session 2. I initially thought these
  were off by one against the raw CSV line counts, but both CSVs have a
  trailing blank line after the last data row — accounting for that, both
  counts match exactly (528 = 530 total lines − 1 header − 1 trailing
  blank; 530 = 532 − 1 − 1).
- **What I did not verify.** I have Read/Grep/Glob access only, and Glob and
  Grep both failed in this session (the underlying `ripgrep` binary appears
  missing from this sandbox) — I could not do a repo-wide search or directory
  listing. Everything above was checked by reading specific files I already
  knew the names of from the findings documents and the task description. I
  cannot rule out that some other file in `cv/` bears on this (e.g. a stray
  training script) that I simply didn't think to ask for by name — I did not
  independently enumerate the directory. I also did not verify `session3.csv`
  line-by-line for the oven/microwave/knife-context numbers with the same
  rigor as sessions 1 and 2, due to the file's size (267KB, over the tool's
  per-read cap) — I read roughly the first 270 of what is likely 500+ rows,
  confirmed the `scissors 0.975` and `microwave 0.9431/0.918` values
  directly, and am taking the remaining session 3 numbers (the `0.827`
  kitchen-counter knife score, the `0.872` oven peak) on the documented
  report rather than having traced each one myself.

## Three rounds, and the actual story of the phase

### Round 1 — 12 pre-existing frames, re-measured

Before any new photos were taken, the 12 raw captures already sitting in
`cv/captures/` from Phase 2 bring-up were run through the full sweep (`yolo26x
× imgsz {640, 960, 1280, 1600} × conf 0.01`, 528 detections). This alone
overturned two conclusions from `docs/phase-3-scoping-notes.md`, which had
been written from three annotated screenshots at a single setting:

- **The scoping notes said "person detection is weaker than expected... this
  may be the bigger risk,"** based on frames where `person` scored 0.38–0.42
  or vanished entirely. All of those frames turned out to have been shot at
  `imgsz 1600`. Re-measured across the resolution range, the *same* frames
  score 0.90–0.98 at `imgsz 640`. The diagnosis was backwards: person
  detection on partial/close bodies isn't fragile — it's specifically what
  high `imgsz` breaks, because pushing resolution up shrinks the model's
  effective field of view onto the closest, most-cropped part of a limb,
  which is further from COCO's whole-body training distribution, not closer
  to it.
- **The scoping notes called scissors on a wall a "complete miss... not a
  resolution problem,"** and used that single frame to argue decision 7 was
  simply wrong. The stronger, measured finding is that scissors are
  *unstable*, not absent — the same physical scissors, same lighting, same
  distance, score anywhere from 0.03 to 0.80 depending only on `imgsz`. There
  is one genuine hard miss in the data (white scissors on a white wall,
  topping out at 0.045 at every setting) but it's one case among many, not
  the general pattern the scoping notes generalized it into.

Both corrections matter for the same reason: the scoping notes weren't
wrong because they lied about what they saw — the arm really was
undetected at 0.38, the scissors really were invisible in that one frame —
they were wrong because a small number of anecdotes at one configuration
were read as a general property of the model, when the actual variable was
the setting, not the object. This is the whole reason Step 0 exists rather
than skipping straight to data collection.

### Round 2 — 21 live frames, and the block the operator wrote off

Blocks A (sharp objects), B (small choking-hazard objects), and C (person)
were shot live; Block D (camera position/height variation) was deferred —
the USB cable wasn't long enough to reposition the camera, and an extender
wasn't yet available.

**Block A confirmed the knife finding across an independent set of objects
and placements**: nothing clears the default 0.25 alert threshold for an
unattended knife, in two separate sessions, on different actual knives.

**Block B is the single most important result of Step 0, and it was almost
lost.** Per `docs/phase-3-step0-findings.md`: *"The operator reported this
block as a failure because 'it didn't even detect anything.'"* The
measurement said something considerably worse, and considerably more
useful — small objects (a car key, a lighter, small items on the floor)
were not undetected at all. They were detected confidently, as the *wrong
COCO class*: a car key in an open palm read as `cell phone` at 0.92; a
cigarette lighter as `cell phone` at 0.47–0.60; loose objects on the floor
as `sports ball` at up to 0.74.

This is a meaningfully different failure mode than everything else Step 0
found, and it would not have surfaced if the "nothing detected" impression
from watching the live preview had simply been trusted and the block logged
as a wash. A miss leaves a visible gap in the hazard map — nothing is there,
something should be. A confident wrong label *fills* that gap with something
that reads as safe: the risk engine sees a "phone" and has no reason to flag
it, while a swallowable metal object sits on the floor a few feet from where
a toddler is crawling. The practical consequence reaches all the way into
how Phase 3's dataset has to be labelled — it isn't enough to add a new
class for these objects; the training data has to include frames where COCO
currently fires `cell phone`/`sports ball` confidently, labelled as the new
hazard class instead, or the false positive just keeps winning.

**Why this belongs in the process story, not just the technical findings:**
the operator's own read of the live session — watching the annotated preview
in real time — was that this block failed. It was actually the block that
mattered most. The lesson isn't "the operator was careless"; a live YOLO
overlay genuinely doesn't render a wrong label any differently from a right
one, so there was no visual cue to catch. The lesson is structural: a live
preview shows you *that* boxes are drawn, not whether they're correct, and
this project's entire premise — catching real hazards, not just plausible-
looking ones — depends on the second question, which only the offline,
conf-0.01 measurement actually answers.

**Also found in Block C**, and worth flagging on its own: frame `102954`
shows a person reaching toward a knife on a ledge — about as close as a
staged photo gets to the actual scenario the risk engine exists for.
`person` scores 0.977. `knife` scores 0.244. The system would see the child
perfectly and miss the hazard entirely, in the single frame most
representative of what Phase 4 needs to get right.

**`--4k` did not take.** All 21 frames measured at 1920×1080 despite the
session being intended as a 4K capture. Not diagnosed as a bug at the time —
just noted as an open question.

### Round 3 — 4K confirmed, and a second forgotten-flag lesson learned the hard way

Twelve frames, two camera heights (~60cm, toddler eye level; ~1.9m,
corner-mount), the second position in a new area — a kitchenette with a
toaster oven, microwave, kettle, and wall sockets. All 12 confirmed
`3840×2160`. **The previous session's 1080p result was a forgotten
`--4k` flag, not a code defect** — the flag itself works correctly on real
hardware once someone remembers to pass it. Worth stating plainly for the
paper: this is the second time in this project a "the tool didn't do what I
asked" impression turned out to be "I didn't actually ask it to" (the first
being Phase 2's `isOpened()`-lies saga, a different mechanism but the same
shape of mistake — trusting an impression of what ran instead of checking
what was actually invoked).

Three findings came out of this session that Round 1/2 couldn't have
produced:

- **Raising `imgsz` to 2560 to "match" the new 4K captures actively hurts
  almost everything** — `microwave` 0.943→0.126, `oven` 0.449→0.261,
  `person` 0.959→0.075, `scissors` 0.975→0.603, all measured going from
  imgsz 1600 to 2560 in the same frames. The practical rule this settles:
  capture at the highest resolution available (it's detail you can label and
  train on later and can never recover if you didn't capture it), but never
  infer above ~1600 — YOLO26 is trained at 640, and departing further from
  that scale costs more than it buys past a certain point.
- **`knife` failure is explained, not just observed.** Frame `104604` — 4K,
  camera at 60cm, a red-handled knife large and unoccluded beside a reaching
  hand — controls for size, distance, resolution, and lighting all at once.
  `knife` tops out at 0.214 in that exact frame. `scissors`, same frame, same
  everything, reaches 0.975. The one variable left standing is context: COCO
  photographs knives in kitchens, and a knife on a floor simply isn't
  recognizable to it as the same object. This is the worst possible shape of
  failure for this project, because "knife on the floor near a crawling
  child" is the emergency case, and it's the exact case that fails.
- **`oven` was measured for the first time, and it works** — 0.872 as
  `oven`, 0.965 as `microwave`, on the same toaster oven, consistently
  across five kitchen frames. (See the CLAUDE.md section below — this
  finding landed *after* CLAUDE.md's same-day amendment, which still says
  oven is unverified.)

## The measured class list (final, as of session 3)

| verdict | classes | evidence |
|---|---|---|
| Detects reliably | `person` (imgsz 640–1600, adults only — see gaps below), `refrigerator`, `chair`, `oven`/`microwave` (fires as both at once — treat as one hazard group) | 0.88–0.98 typical |
| Detects unreliably | `scissors` | 0.03–0.98 on the same physical object depending only on setting; one genuine hard miss (white-on-white) |
| Effectively misses | `knife` outside kitchen contexts | 0.01–0.29 unattended; 0.83 on a counter — this is the case that matters, and it fails |
| Misses entirely, no COCO class | sockets, power strips, adapters, cables | zero hits at any setting; one spurious `remote` at 0.012 across the whole session-1 sweep (noise, not a stable substitute label) |
| Misses entirely, AND actively mislabelled as something benign | small choking-hazard objects (keys, lighters, small floor items) | `cell phone` up to 0.92, `sports ball` up to 0.74 — a fine-tune here has to suppress the wrong label, not just add the right one |

## Audit findings

**What checks out, verified by reading the code and cross-referencing the
CSVs directly (not taken on the findings document's word):**

- `measure_detection.py` does exactly what it claims: sweeps
  model × imgsz × conf, records the harness's own settings per row, keeps the
  capture-time config as separate provenance, refuses annotated frames, loads
  each model once and reuses it across the sweep. No stubs, no shortcuts.
- The confidence numbers quoted throughout `docs/phase-3-step0-findings.md`
  are real measurements, not summarized-from-memory or rounded-favorably —
  spot-checked ~15 values across all three sessions' CSVs and every one
  matched, including the specific `imgsz` value each headline number actually
  came from.
- The `--4k`/self-timer additions to `detect_stream.py`/`camera.py` are
  genuinely opt-in — confirmed by reading `main()`: the flag's absence
  produces byte-for-byte the same `CameraCapture(...)` call as before it
  existed.
- `cv/measurements/` is gitignored, with a comment explaining why (bbox rows
  are traceable back to photos of a real home, same privacy logic as the
  raw captures they're derived from) — this is decision 8's spirit applied
  correctly to a new kind of artifact nobody had thought about when decision
  8 was originally written, the same way the Phase 3 labelling-stays-local
  decision already did for training images themselves.

**Gaps found that I don't believe anyone has flagged yet:**

1. **CLAUDE.md decision 7's oven language is now stale, not overclaimed.**
   The task framing for this audit specifically asked whether decision 7
   "overclaims" about oven/sink — it doesn't; if anything it's now
   *conservative* to a fault. Decision 7 was amended earlier on 2026-08-09,
   based on session 1's findings, to say: *"`oven` and `sink` are
   **unverified**, not confirmed — no test area contains either. Do not
   write them up as working until they are actually measured."* Session 3,
   which ran later the same day, measured oven for the first time and found
   it works (0.872, consistently, across five kitchen frames) — the
   findings document says outright "decision 7's claim holds here." CLAUDE.md
   was never updated to reflect that. As it stands today, CLAUDE.md tells a
   reader oven is unverified when the project's own most recent evidence
   says it's confirmed (with a real caveat: this is a countertop toaster
   oven, not the floor-level built-in oven that's the actual toddler
   hazard — that distinction is exactly the kind of nuance a stale one-line
   summary would flatten if copied forward without a fix). `sink` remains
   correctly described as unverified — only the oven half of that sentence
   needs a follow-up. I'm flagging this, not fixing it: CLAUDE.md's own
   procedure requires asking Shaked and Yahli before changing a decision,
   and that's a two-sentence conversation, not something I should decide
   unilaterally.
2. **Every measurement in Step 0 used `yolo26x` — none used `yolo26l`, the
   model Phase 2 actually set as the live pipeline's default.** This is a
   real gap in what the class list can claim. `yolo26x` is the largest,
   highest-accuracy model in the family; Phase 2's own benchmarks (in
   `docs/phase-writeups/phase-2.md`) show it costing real frame rate against
   `yolo26l` (35fps vs 62fps at imgsz 640, falling to ~7fps vs ~14fps at
   imgsz 1600). The "detects reliably" / "unreliably" / "misses" bins above
   describe stock COCO's *best available* configuration, not necessarily
   whatever Phase 4 actually deploys. If Phase 4 ends up running `yolo26l`
   for frame-rate headroom (as Phase 2's own reasoning for defaulting to it
   would suggest), some of these numbers — especially the borderline ones,
   like scissors sitting right at a usable threshold at some settings — may
   not hold. This should be an explicit decision going into Phase 4, not an
   assumption inherited quietly from Step 0's methodology choice.
3. **I could not independently enumerate `cv/`'s contents** — the Glob and
   Grep tools both failed in this session (missing `ripgrep` binary in this
   sandbox), so I read only the files named in the task and the findings
   documents. I have no independent confirmation there isn't some other
   file relevant to this audit sitting in the directory that I simply didn't
   know to ask for. Worth someone re-running this audit's file checks with a
   working shell at some point, not urgently, but noted rather than silently
   assumed away.

## What Step 0 does NOT establish — do not assume these going into data collection

- **A real, floor-level built-in oven.** Only a countertop toaster oven has
  been measured. It is plausible a built-in range behaves differently (size,
  shape, context all differ), and it's the actual toddler-height hazard.
- **A real sink.** Zero test frames contain one; max spurious confidence
  0.085.
- **Stairs.** No stairs exist in any tested area.
- **Any toddler-sized subject.** Every `person` measurement in Step 0 — all
  three sessions — is an adult, mostly at close range or in staged
  reaching/crouching poses. This is explicitly the single largest
  unvalidated assumption feeding Phase 4's risk engine: person detection
  holding up at imgsz 640 for an adult's arm two feet from the camera says
  nothing directly about a small whole body at 3–4 meters, partially behind
  furniture, actually crawling.
- **Camera placement variation, beyond a small sample.** Block D (position/
  angle) was only partly executed — two heights, one additional area (the
  kitchenette). `PHASE_PLAN.md` requires the dataset to span the real
  deployment variation (area, height, angle, lighting) deliberately, and two
  heights in two rooms is a start, not that.
- **Small choking hazards or chemical/medicine bottles, beyond one hard
  floor.** The Block B objects were shot once, on a speckled terrazzo floor
  that the findings document itself calls "a genuinely hard background" —
  good material, but if it's the *only* floor represented, whatever
  fine-tuning results won't be validated against easier backgrounds either.
- **Model choice for production.** As flagged above, every number here is
  `yolo26x`-only.

## Verdict

**Step 0, narrowly scoped as PHASE_PLAN.md defines it, is done and can be
signed off.** The deliverable was a measured class list replacing the
imgsz-640 assumptions Phase 2 invalidated, plus a decision on whether
CLAUDE.md decision 7 still holds. Both exist, both are backed by numbers I
independently traced back to real CSV rows rather than taking on faith, and
the measurement tooling itself (`measure_detection.py`) does what it claims
with no shortcuts I could find.

**Phase 3 as a whole is not done, and nothing in this document should be
read as suggesting otherwise.** No images have been collected for
fine-tuning, no labelling tool has touched a single frame, no model has been
trained. `PHASE_PLAN.md`'s Phase 3 checkbox correctly remains `[ ]`, and I
have not touched it.

**One loose end genuinely worth closing before data collection starts, not
urgently but not indefinitely either:** CLAUDE.md's oven language should get
a quick follow-up amendment (with Shaked and Yahli's sign-off, per CLAUDE.md's
own stated procedure) so a future reader doesn't scope Phase 3's kitchen data
around a line that's already out of date on the project's own evidence.

## The two lessons this phase is really about

1. **A confidently wrong label is worse than no label, and a live preview
   can't tell you which one you're looking at.** Block B is the clean
   illustration: the operator watched boxes render in real time and
   concluded nothing happened, when what actually happened was the single
   most consequential finding of the phase. This generalizes past this one
   session — anywhere this project shows a human a real-time overlay (which
   is most of it, by design, per CLAUDE.md's own MJPEG-stream architecture),
   "something is drawn" and "something correct is drawn" need to stay
   separate questions in everyone's head, not just the measurement script's.
2. **Measurement corrected confident impressions three separate times in
   one phase** — the scoping notes' person-detection diagnosis, the scoping
   notes' scissors "complete miss" framing, and the Block B "nothing
   detected" read. None of the three original impressions were dishonest or
   careless; each was a reasonable read of what was directly visible at the
   time. All three were wrong in the same direction: they treated a single
   configuration's output as if it were a property of the object, rather
   than a property of the object *at that setting*. That's the actual
   argument for why Step 0 had to exist as a separate, deliberate step
   before Phase 3's real work, rather than being folded into "just start
   collecting data and see."

## Questions to check your own understanding

1. Explain, without looking it up, *why* raising `imgsz` helps a small
   distant object but hurts a large partial one (like an arm filling a third
   of the frame). What is YOLO actually doing differently in the two cases?
2. The Block B mislabelling finding changes what the *labelling* step of
   Phase 3 has to look like, not just the training step. Explain concretely
   what "suppress the competing COCO label" means for how a human sitting at
   a labelling tool would annotate a frame containing a car key on the
   floor — what would go wrong if the dataset only ever labelled the key as
   the new hazard class and never addressed the existing `cell phone` box?
3. Step 0's class list is built entirely on `yolo26x`. If Phase 4 runs
   `yolo26l` instead for frame-rate reasons, what specifically about this
   audit's Gap 2 would you need to re-check before trusting the same
   reliable/unreliable/missing bins for the live system?
4. CLAUDE.md's decision 7 currently undersells the oven finding rather than
   overselling it. Why is that the safer direction for a stale claim to
   drift in, for a child-safety system specifically — and can you think of a
   case elsewhere in this project's decision log where a claim drifted the
   *other* way, toward overclaiming, and what caught it?
