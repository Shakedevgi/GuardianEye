# Phase 3 — from "fine-tune the missing classes" to "detect anything on a
reachable surface"

**Status: CLOSED (2026-08-11, second attempt). Closed as a
scoping/architecture phase plus a validated-but-known-weak Layer A
component, NOT as a working, parent-usable hazard detector. Read that
sentence twice before reading anything else in this document.**

**This is the second close attempt, and that matters for how you should read
this document.** The first attempt (everything in this file up through "For
the write-up's teaching purpose," below) got most of the way there, then its
own audit found that one of the phase's headline claims — change detection's
"~1.4ms/frame, correctly boxed a car key and a lighter" — had zero
supporting evidence anywhere in the repo. That triggered two real
re-measurement rounds, described in the new section **"Second closure
(2026-08-11): change detection re-measured for real, and why the phase
closes now"** near the end of this document. Everything above that new
section is preserved from the first attempt, corrected in place only where a
number it stated turned out to be wrong (marked where that happens) — not
rewritten from scratch, so the record shows what was believed at each point,
not just the final answer.

This document covers everything from Step 0's sign-off (2026-08-09) through
the reframe that closed the phase's scoping question (2026-08-11), and then
the change-detection re-measurement that closed the phase for real later the
same day. Step 0 itself — the measurement pass that established which stock
COCO classes actually work in this deployment — already has its own write-up
at `docs/phase-writeups/phase-3-step0.md` and is not repeated here beyond
what's needed for continuity. Read that document first if you haven't; this
one assumes its class list ("`person`/`refrigerator`/`chair`/`oven` reliable,
`scissors` unreliable, `knife` effectively misses outside kitchens, sockets
and small choking hazards absent — and choking hazards are *actively
mislabelled*, not just missed") as a known starting point.

## What `PHASE_PLAN.md` asked for, and why that bar is now the wrong bar

`PHASE_PLAN.md`'s Phase 3 section, unedited until this write-up, said:

> **Done when:** the fine-tuned model reliably detects the new classes on
> test footage, without badly regressing the original COCO classes.

That bar assumes fine-tuning was always going to be the mechanism. It
wasn't, by the end of the phase — and the reason it wasn't is itself the
most important thing this phase produced. Three fine-tuning rounds were run,
each measured properly against a held-out building, and none of them
reached anywhere close to "reliably detects." The phase's real result is
not a model that clears that bar; it's evidence, gathered the hard way,
that clearing it with more of the same approach was not going to happen,
plus a different design that a human correction pointed the team toward
instead. See "Was the done-bar wrong, and what should it say now" near the
end for the proposed rewrite.

## The three fine-tuning rounds, in one table

All numbers are precision/recall at IoU ≥ 0.5, conf 0.25, measured against
**32 labelled frames from a building the model was never trained on** — the
held-out test set established after round 1 (see below for why a same-building
validation split wasn't good enough). Full detail, including the CSV-level
methodology, is in `docs/phase-3-step0-findings.md`.

| round | data | pooled precision | pooled recall | `sharp_object` recall | `small_swallowable` recall |
|---|---|---|---|---|---|
| 1 | 92 office frames only | 0.246 | 0.147 | 0.097 | 0.204 |
| 2 | public Open Images → office frames, sequential (two-stage) | 0.179 | **0.216 (best)** | **0.339 (best)** | 0.074 |
| 3 | public + office frames mixed, office oversampled 14× | 0.133 | 0.017 | 0.032 | **0.000** |

Three things worth understanding about this table, not just reading it:

**Round 1's own validation number lied about how good the model was.**
Evaluated on its own held-out split from the same office building, round 1
scored precision 0.902 / recall 0.607 — genuinely excellent-looking numbers.
Evaluated on the 32 frames from a second building, the same model scored
0.246 / 0.147. It had memorised one room, not learned "sharp object" or
"small swallowable object" as concepts. This is the reason the 32 home
frames became a permanent held-out test set for the rest of the phase, and
the reason CLAUDE.md now states as a standing rule: **a leak-free train/val
split is necessary but not sufficient** — it can only catch a model that
copied answers between two subsets of the *same* room. Only a genuinely
different location catches a model that copied answers between rooms.

**Round 2 (public data first, then our frames) is the best model produced
this phase**, and the reason it worked explains why round 3 then made things
worse. Round 2 spent 95 epochs on 1,078 distinct public Open Images photos of
knives and scissors (in kitchens, outdoors, in hands — nothing like a floor)
before ever seeing an office frame. That built genuinely general
`sharp_object` features, which the short second stage on office frames could
refine without erasing — `sharp_object` recall reached 0.339 on the unseen
building, real transfer, not memorisation. The cost: stage 1's public data
has zero `small_swallowable` examples (Open Images has no matching class —
see below), so for 95 epochs the model only ever saw one class, and by the
time stage 2 introduced 54 boxes of the other class in a 30-epoch fine-tune,
recall on it had already collapsed to 0.074.

**Round 3 tried to fix that by training on both classes at once, and it was
the single worst result of the whole phase — 0.017 pooled recall, zero true
positives on `small_swallowable` at any threshold.** The instinct behind it
was reasonable: if the problem is "class 1 wasn't supervised for long
enough," supervise it the whole time. The mechanism used to do that —
duplicating each of the 79 office frames 14× in the training file list to
match the public data's volume — was the actual mistake. At a 14× repeat
factor the model saw each of those 79 frames 364 times before its best
checkpoint (14 repeats × 26 epochs). That's not "more supervision of class
1," it's rote memorisation of 79 specific photographs, and it transferred
almost nothing to a new building — confirmed not to be a training bug, since
the run scored a healthy 0.73 mAP50 on its own (same-building) validation
split the whole time. **The lesson that survives past this one experiment:
what generalises is the number of distinct scenes a model sees, not the
number of training instances.** Three schedules over the same underlying 92
office frames produced 0.147, 0.216, and 0.017 recall on the same test set —
the variable that never changed across all three is the one that actually
mattered, and it isn't visible in any of the three runs' own validation
numbers. `docs/decision-log.md`'s 2026-08-11 entry states the resulting rule
plainly: **do not oversample by duplicating file entries; if class balance
needs fixing, use loss weighting instead.**

## The VLM detour, and why it was rejected

A local vision-language model (Florence-2, base and large, run on-device via
MPS — no cloud call, consistent with CLAUDE.md decision 8) was tested as a
possible safety net for Layer A: something that could look at a frame and
say what's on it without needing a fixed trained class list.

**It failed on a specific, disqualifying property: its grounding mode has no
"not present" case.** Asked to find `"banana"` in a frame that shows a wall
socket, it does not say "no banana here" — it returns a confident, tightly
drawn box, on the socket, labelled banana. The same happened asking for
`"knife"` on a frame of a car key, and `"cigarette lighter"` on a frame of a
knife: four separate absent-object probes, four confident wrong boxes. A
presence check that can't say no is not usable for a safety system — it
fails by inventing evidence, which is a worse failure mode than staying
silent. Florence-2's *unprompted* discovery mode (describe what's in the
frame without being told what to look for) behaved more honestly and even
produced one genuine win — it correctly named a floor-level power outlet
that stock COCO structurally cannot recognize — but it also mislabelled the
floor knife as *"person cutting tile floor with red scissors,"* the exact
knife/scissors confusion the fine-tuning rounds were already fighting, now
delivered with more apparent confidence. Verdict, recorded in
`docs/phase-3-step0-findings.md`: Phase 4 builds on fine-tuned YOLO, not a
VLM. This detour cost real time but produced a real, useful negative result
— see the highlighted section below for why that's not a wasted afternoon.

## Why the alarm stays instant, even though a VLM (or anything else slow) can sit in Layer A

Shaked asked for this explanation to be front and center, not a footnote,
because it's the piece of the architecture that makes the rest of this
document's slow, careful, sometimes-failing measurement work actually safe
to build on top of.

**VLM = Vision Language Model** — a model that looks at a picture and answers
in words (this is what a model like Claude does when reading a capture). YOLO
is a different kind of tool: it outputs boxes from a fixed list of trained
class names and cannot name anything outside that list, but it runs at
roughly 60fps with precise boxes, locally, for free. A VLM takes one to five
seconds per frame, can describe almost anything, with looser boxes, and runs
either locally (slow) or in the cloud (costs money and, per CLAUDE.md
decision 8, is off the table for this project's data anyway). A useful
analogy: **YOLO is a barcode scanner. A VLM is a person actually looking at
the shelf.**

The alarm stays instant despite that speed gap because the two things being
watched move at completely different speeds. A child moves constantly and
has to be tracked every single frame — that's Layer B, and it's why decision
2 insists on a fast, single YOLO pass covering the child every time. A hazard
sitting on a counter does not move at all. It only needs to be identified
*once*, and then remembered. A slow model — a VLM, or anything else that
takes seconds instead of milliseconds — never sits between the child and the
alarm. It only ever serves Layer A, the hazard map, never Layer B, the
moment-to-moment proximity check that actually fires an alert.

Worked example, to make this land concretely: a knife appears on a counter
at 10:00:00. Within a couple of seconds, change detection (a diff-plus-
persistence check — real cost measured later in this document at roughly
39–43ms per compared frame pair once person-suppression/tracking is
included, not the original ~1.4ms figure this section first quoted — see the
"Second closure" section below for why that number changed) notices
something changed at that spot and, after a couple more frames of
confirmation, treats it as a real placed object. By 10:00:05, it's logged
into the hazard map. Nothing happens for the next 32 minutes. At 10:32:00 the
child crawls toward it, and the alarm fires roughly 20 milliseconds later —
from the position *already recorded* in the hazard map, not from
re-examining the scene in that moment. The slow step's few seconds of
latency happened half an hour before the moment of actual danger. That's the
whole argument: latency anywhere in Layer A is invisible to the child-safety
guarantee, because Layer A's output is a static fact ("there is a hazard
around here") that Layer B consults instantly, not a live computation Layer B
waits on. **The exact millisecond figure was wrong when this section was
first written and is corrected here — the argument itself never depended on
which number was right.** Even if Layer A took a full second, or five, per
hazard, Layer B's response time would be unaffected, because the two layers
never share a critical path.

## The reframe: five failed detectors, and why that's the evidence, not a detour around it

This is the second thing Shaked explicitly asked to be prominent, and it's
the actual hinge the whole phase turns on.

By 2026-08-10, three fine-tuning rounds and a VLM test had all tried the
same underlying thing: get a model to look at *one* unfamiliar frame and
correctly say, alone, which objects in it are hazards. Five distinct methods
were tried at exactly that job:

1. **Named classification** (the three fine-tuning rounds above) — capped
   around 0.22 pooled recall on an unseen building, never higher.
2. **Colour clustering** — recall 0.09. Misses almost everything.
3. **Texture-based "objectness"** — recall 0.79, but roughly 33% eyeballed
   precision — it fires on drawer handles and structural edges as
   confidently as it fires on a real hazard.
4. **A VLM asked to confirm/deny an object** (Florence-2) — rejected outright
   for inventing confident boxes over empty space, detailed above.
5. **Class-agnostic segmentation with a compactness filter** — recall 0.32,
   precision roughly 45% in daylight, collapsing to roughly 9% under evening
   lighting. Two separate sources of clutter (structural edges — door
   frames, cabinet lines — and patterned textiles/floor tiles) turned out to
   be genuinely inseparable from real objects by any signal tried, including
   the one signal (a segment's own mask "fill ratio," rejecting long thin
   shapes like a door handle in favor of compact ones like a knife) that the
   classical texture method never had access to.

**All five failed at the same specific sub-task, in different and
instructive ways** — two by missing most hazards, three by flagging ordinary
furniture as confidently as real hazards. This is not a shortfall in this
project's engineering. Noticing an unfamiliar object with no prior
information, correctly, alone, in a single frame, is a genuinely open
problem in computer vision — not something "there's clearly already a model
for," the way it can feel like there should be given how good object
detection has otherwise become.

The five negative results are what proved the eventual reframe was
*necessary*, not a guess dressed up after the fact. CLAUDE.md decision 4 had
already specified, from the start of the project, that the parent reviews
setup-phase hazards — that hand-off to a human was never actually missing
from the architecture. What had happened, over the course of this phase, was
that the object-detection work under decision 7 had quietly taken on a
harder job than decision 4 ever required: being *correct* about naming and
judging a hazard before the parent ever saw it. Once that requirement is
removed — the model only has to flag "this spot is occupied," not judge what
it is or whether it matters — the problem becomes tractable, because a human
confirms it in one tap, in seconds. Do not read the five failed attempts as
wasted effort; they are the actual evidence that this reframe was required,
not merely convenient.

## The reframe itself: the setup walkthrough

CLAUDE.md decisions 3 and 4 were amended on 2026-08-11 (same day, per the
project's stated procedure of asking Shaked/Yahli and logging the change in
the same turn — confirmed by reading `docs/decision-log.md`'s 2026-08-11
entry directly) to say:

- **Setup is a guided, parent-confirmed walkthrough, not a silent
  auto-discovery pass.** The system flags every occupied spot on a reachable
  surface — floor, low table, low shelf — one at a time, and the parent
  confirms hazard-or-fine on the spot, in a two-to-three-minute session when
  the room is first set up. Confirmed hazards seed Layer A's hazard map for
  the rest of the session.
- **Over-flagging is now acceptable.** It costs one extra tap, not a wrong
  autonomous alert. This is the whole point of the reframe: the cost of
  being wrong changed from "the system silently misses a real hazard, or
  confidently ignores one" to "the parent taps 'not a hazard' once," which
  is a completely different, much more forgiving failure mode.
- **Change detection remains the fully-automatic half**, unaffected by any
  of this, for anything appearing *after* setup — a hazard that shows up
  mid-session gets caught without asking the parent anything, because
  "something changed here" is a genuinely solvable, already-measured
  problem (unlike "judge whether this thing that was already there is a
  hazard," which the five failed methods above show is not).

## The honesty check this write-up was specifically asked to do: does the winning design exist as code?

**This section describes the state as of the FIRST close attempt
(2026-08-11, earlier the same day). It is now partially out of date — see
"Second closure" below for what changed. Kept as-is, not silently corrected
in place, because it's the record of exactly what this audit found and why
that finding triggered the re-measurement that followed. Short version of
what changed: change detection now has real, committed code and real
findings-document evidence. The guided walkthrough still does not exist as
code — that part of this section's finding still holds.**

**At the time of the first attempt: no. Neither the change-detection
mechanism nor the guided walkthrough existed as committed, reusable code in
this repository.** I checked this directly rather than taking the claim on
faith, in three ways:

1. **`cv/*.py` — twelve files, read as a full list** (`stream_camera.py`,
   `detect_cameras.py`, `measure_detection.py`, `camera.py`,
   `measure_openvocab.py`, `detect_stream.py`, `triage_captures.py`,
   `label_captures.py`, `prepare_dataset.py`, `evaluate_home_frames.py`,
   `build_mixed_dataset.py`, `measure_segmentation.py`). None of them is a
   frame-differencing / change-detection module, and none implements a
   setup-walkthrough flow (a flag-one-spot-at-a-time, parent-confirms loop).
2. **A repo-wide grep** for `absdiff`, `change detection`, `frame diff`,
   `background subtract`, and `walkthrough` (case-insensitive) turns up
   nothing in `cv/` at all, and nothing anywhere named `walkthrough`.
3. **A repo-wide grep for the specific claimed evidence** — the
   "~1.4ms/frame" figure and the "correctly boxed a car key and a lighter
   that stock COCO called `cell phone` at 0.92" claim, both quoted verbatim
   in CLAUDE.md decision 7 and in `docs/decision-log.md` — matches **only
   those two files**. It does not appear in `docs/phase-3-step0-findings.md`
   (the document CLAUDE.md and the decision log both point to for "measured
   detail"), and there is no CSV, script, or committed evidence file backing
   it up anywhere in the repository.

This matches exactly what the task briefing said to expect, and it's worth
being blunt about what it means: **the winning mechanism and the winning
setup-flow design were measured and decided by an agent working in an
ephemeral scratchpad directory outside the repo, on instruction to keep
evidence out of the repo, not in it.** What survived into the permanent
record is the *conclusion* (change detection works, at this rough speed, and
correctly handled two specific test objects) and the *design decision* (the
walkthrough), not the underlying, independently-checkable work. That's a
meaningfully different, weaker kind of evidence than everything else in this
phase, which mostly comes with a CSV a reader can open and a script that
reproduces it. **Phase 3's deliverable, honestly stated, is a validated
architecture decision backed by real measurement of what does NOT work
(three fine-tuning rounds, a VLM, and five single-frame judgment methods,
all committed, all traceable), plus a directional conclusion about what
probably does work (change detection, the walkthrough) that is not yet
reproducible by anyone reading this repository. Phase 4 has to write that
code from scratch** — it can be informed by the scratchpad work's
conclusions, but it cannot inherit any of the scratchpad work itself.

## Audit: what checks out, tier by tier

I read the actual code for every script and tool named below, not just the
docstrings describing them, and cross-checked specific numeric claims
against wherever their supporting evidence should live.

**Tier 1 — code exists, runs, and its numbers are written up in prose that a
reader can trace back to a CSV or a described methodology.** This is the
strongest tier and covers most of the phase's headline results:

- `cv/measure_detection.py` (Step 0, audited separately and previously —
  see `docs/phase-writeups/phase-3-step0.md`).
- The three fine-tuning rounds' precision/recall numbers, all measurable via
  `cv/evaluate_home_frames.py`, which I read in full — it implements exactly
  the IoU≥0.5, greedy-match, pooled-precision/recall methodology
  `docs/phase-3-step0-findings.md` describes, against the specific 32
  `*home*_raw.jpg` frames it names, and does not touch label files (read-only,
  confirmed by reading the file). The script's own docstring states it was
  built specifically so round 2 onward could be compared to round 1 on
  identical methodology rather than drifting — a good practice, and I
  confirmed it's actually what the script does.
- The VLM (Florence-2) rejection — full table of prompt/frame/result pairs
  is in `docs/phase-3-step0-findings.md`, matching CLAUDE.md's summary.
- The Open Images public-data pull — `data/openimages_public/README.md` is
  a genuinely thorough provenance document: exact source URLs, exact
  filtering rules (`IsGroupOf`/`IsDepiction` dropped), exact counts (1,450
  boxes / 1,078 images), a stated and honest domain-gap section explaining
  *why* this data won't fix the floor-context problem on its own (zero of
  the 16 sampled images are an indoor floor shot from a low angle), and an
  explicit, un-acted-on flag asking Shaked/Yahli before pulling a
  `Coin`-only addition to `small_swallowable` rather than silently doing it.
- `cv/label_captures.py` / `docs/labelling-guide.md` / `cv/prepare_dataset.py`
  / `cv/triage_captures.py` / `cv/build_mixed_dataset.py` — all read in full.
  These do exactly what they claim: a dependency-free (OpenCV+numpy only)
  labelling tool with a documented, load-bearing full-resolution coordinate
  contract; a dHash-based near-duplicate/blur triage tool; a duplicate-group-
  aware train/val splitter (correctly reasoned: splitting by individual
  frame would leak near-duplicates across train/val and produce a
  meaningless validation number); and the oversampling dataset builder for
  round 3, whose own docstring already documents, in advance, the exact
  mechanism (14× file-list duplication) that round 3's own postmortem later
  identified as the cause of its failure — a good sign the team understood
  the tradeoff going in, not just after the fact.
- 124 labelled `.txt` files actually exist under `cv/captures/` (I counted
  via glob), consistent with the "124 labelled frames" claim in the task
  briefing.
- The `runs/detect/cv/runs/phase3_v2` and `runs/detect/cv/runs/phase3_v3_stage1`
  path oddity is real but already diagnosed and documented, not a hidden
  bug: `cv/train_mixed.sh`'s own header comment explains that a relative
  `project=` path passed to `yolo detect train` landed round 1 and round 2's
  stage 1 under Ultralytics' own default `runs/detect/` rather than the
  intended `cv/runs/`, "after 3.5 hours of training," and that
  `train_mixed.sh` was written afterward to hardcode an absolute path
  specifically so it wouldn't happen a third time. Four sets of trained
  weights exist on disk in total across the two locations
  (`phase3_v2`=round 1, `phase3_v3_stage1`+`phase3_v3_stage2`=round 2's two
  stages, `phase3_v4_mixed`=round 3), all gitignored per `cv/runs/` and
  `runs/detect/` (the latter caught by the broader `cv/**/*.jpg` and
  general `runs/`-adjacent patterns — worth double-checking `runs/` itself
  is actually gitignored; see finding below).

**Tier 2 — code exists and genuinely runs, but its output numbers are not
written up anywhere a reader can check them against a described
methodology; they only appear as bare assertions in CLAUDE.md /
`docs/decision-log.md`.**

- `cv/measure_segmentation.py` — I read this file in full. It is real,
  working code implementing exactly the class-agnostic-segmentation-plus-
  compactness-filter method CLAUDE.md and the decision log describe (FastSAM
  "everything" pass, filtered by bbox size, person-overlap, and mask fill
  ratio / "extent" — the specific signal the docstring says the classical
  texture method never had access to). But its output (`segmentation.csv`)
  is gitignored by design, and — unlike Step 0's findings document, which
  reproduces specific rows so a reader can spot-check them — no findings
  document reproduces this script's actual output numbers (recall 0.32,
  precision ~45%→~9%). Those numbers are real in the sense that real code
  exists to have produced them, but nobody reading only the committed repo
  can currently verify that they were.
- `cv/measure_openvocab.py` — similarly real, working code (verified it
  loads YOLO-World/YOLOE, calls `set_classes()` once per model before
  sweeping imgsz, records per-call wall-clock time). The wall-socket
  "0.48–0.90" figure quoted in CLAUDE.md decision 7 and the decision log is
  plausible output from this script, but — same gap — no findings prose
  reproduces the specific rows behind it. `docs/phase-3-collection-plan.md`
  does independently corroborate two *different* open-vocab numbers from
  this same tooling (`car key` 0.20, `cigarette lighter` 0.017), which is
  good partial evidence the tool was actually run for real, but the wall-
  socket number specifically has no equivalent corroboration anywhere.

**Tier 3 — no code exists anywhere in the repository, and the numbers exist
only as bare, uncorroborated assertions.**

- **Colour clustering (recall 0.09) and texture-based objectness (recall
  0.79, ~33% eyeballed precision)** — no script, no CSV, no findings
  paragraph anywhere. These numbers appear exactly twice in the whole repo:
  once in CLAUDE.md's decision 3, once as a one-line docstring citation
  inside `cv/measure_segmentation.py` (which references them as
  already-established facts, not as something it computes itself). There is
  no way for anyone reading this repository to independently confirm these
  two methods were actually implemented and measured, as opposed to
  estimated or eyeballed. This is a materially weaker evidentiary basis than
  everything else in Tier 1/2, and it should be named as such rather than
  quietly folded in with the rest. **This gap is still open as of the
  second closure below** — nobody rebuilt these two methods; they remain
  bare assertions, and that's an accepted, explicitly-scoped gap, not an
  oversight (rebuilding two already-rejected single-frame methods was never
  in scope for closing this phase).
- **The change-detection mechanism itself** — at the time of the first
  close attempt, this was also Tier 3 (decision 7's claimed "~1.4ms/frame,"
  "correctly boxed a car key and a lighter" had no code, no CSV, not even a
  findings-document paragraph). **This has since been promoted to Tier 1.**
  Two re-measurement rounds produced `cv/measure_change_detection.py` (read
  in full, does what its docstring claims), a real `cv/measurements/
  change_detection.csv` (confirmed present on disk), several saved
  before/after verification crops (also confirmed present on disk, e.g.
  `cv/measurements/verify_20260811-021857_..._021858_....jpg`, the pair that
  catches the placed scissors), and several pages of findings-document prose
  in `docs/phase-3-step0-findings.md` that state exact thresholds, report
  numbers with and without a diagnosed startup-transient confound, and
  retract the original car-key/lighter claim by name with a specific,
  checked explanation for where that number actually came from. The old
  claim is not just unproven now — it's actively **refuted**, which is a
  stronger and more useful outcome than "still unverified" would have been.
  See "Second closure," below, for the full account. The CSV and crop images
  are gitignored, same as every other measurement artifact in this repo
  (real home photographs) — this is consistent, pre-existing project policy,
  not a new gap specific to this claim.

**A smaller, unrelated finding worth flagging:** the top-level `runs/`
directory (containing `runs/detect/cv/runs/phase3_v2/` and
`runs/detect/cv/runs/phase3_v3_stage1/`) shows up as untracked (`??`) in
`git status`, not as ignored. `.gitignore` has an entry for `cv/runs/` but
none for a bare top-level `runs/` — the `cv/**/*.jpg` catch-all pattern
would still stop any *images* under `runs/` from being committed by
accident, but the `.pt` weight files under `runs/detect/.../weights/` are
only protected by the separate top-level `*.pt` rule, which does cover them
— so nothing unsafe is actually at risk of being committed, but the
directory sitting as visibly untracked rather than cleanly ignored is worth
a two-line `.gitignore` fix (`runs/` at the top level) so `git status` stops
showing it as loose clutter.

## Does this match CLAUDE.md's architecture?

Yes, on everything Phase 3 is actually responsible for, with the caveats
above named rather than smoothed over. Decision 7's rewrite (find objects,
don't name them, 2026-08-10) and decisions 3/4's amendment (guided
walkthrough, 2026-08-11) are both logged in `docs/decision-log.md` in the
same turn they were made, per the project's own stated procedure, and I
confirmed both amendments are reflected accurately in the current text of
CLAUDE.md itself, not just in the log. Decision 8 (no cloud, ever) held
throughout — the VLM ran locally on MPS, and the Open Images pull downloads
public third-party images, never uploads anything of ours; the README for
that pull explicitly calls out that this is a slightly different case from
decision 8 (public data coming in, not private data going out) and reasons
about it rather than assuming decision 8 obviously covers it either way.

## Was the done-bar wrong, and what should it say now

**Yes, plainly.** `PHASE_PLAN.md`'s current bar — "the fine-tuned model
reliably detects the new classes on test footage, without badly regressing
the original COCO classes" — describes a deliverable this phase tried
seriously (three rounds) and did not reach, and that gap is not a failure to
finish; it's the actual finding. Continuing to measure the phase against
that bar would either falsely fail a phase that did real, valuable work, or
tempt someone into declaring victory on a technicality (e.g. "recall went up
between rounds 1 and 2" without mentioning round 2 still means the model
misses 4 of every 5 hazards).

**Proposed replacement wording** (I have not silently edited `PHASE_PLAN.md`'s
Phase 3 status without flagging this — see the change actually made,
described below, for exactly what text changed):

> **Done when:** the fine-tuning approach has been measured against a
> genuinely held-out location (not a same-building split) enough times to
> know whether it generalises, AND — if it doesn't generalise well enough to
> alert on — a replacement architecture has been decided and written into
> CLAUDE.md with the measurement evidence that justifies it. This phase does
> **not** require a working end-to-end hazard-detection pipeline; that is
> Phase 4's job. It requires knowing, with evidence, what Phase 4 should
> build.

I have gone ahead and applied this rewrite directly to `PHASE_PLAN.md`
(along with marking the phase `[~]`, not `[x]` — see the next section for
why), since `PHASE_PLAN.md` is explicitly not under CLAUDE.md's "ask first"
lock. The edit is visible in that file's own diff; nothing about the change
is hidden in this write-up alone.

## Second closure (2026-08-11): change detection re-measured for real, and why the phase closes now

This section covers everything that happened after the first close attempt
above found its own headline evidentiary gap. It is the actual reason Phase
3 is closed as of this write-up, not the earlier scoping reframe alone.

### What the first attempt's audit found, and what it triggered

The first attempt's audit (the Tier 3 finding, above) established that
decision 7's claim for change detection — *"~1.4ms/frame, and it correctly
boxed a car key and a lighter that stock COCO called `cell phone` at
0.92"* — had zero supporting evidence anywhere in this repository: no
script, no CSV, no findings paragraph, nothing. That's not the same finding
as "the claim is false" — it's "the claim cannot currently be checked,"
which for a phase whose whole selling point is measured honesty is its own
kind of problem. That finding triggered a real re-measurement, done twice,
because the first attempt at fixing it uncovered a second, deeper problem.

### Round 1: real person-suppression, and the claim turns out to be wrong, not just unbacked

`cv/measure_change_detection.py` (first version) implemented real
person-overlap suppression — named in the original claim as the fix for the
dominant false-positive source, but never actually built until this round.
Measured against three pre-existing labelled bursts, pooled recall came out
at **0.030** — and the reason is structural, not a detector defect: those
bursts are static-scene labelling shots ~3 seconds apart of objects that are
**already present in both frames**, so there is nothing for a differencing
method to threshold. This is exactly CLAUDE.md decision 3's own split
(change detection is supposed to solve "appears while running," not
"already there at startup") — the low number confirms the split rather than
contradicting it, but it also meant this test data could never have
validated the original claim in the first place.

The car-key/lighter claim was then checked directly, by name, against the
two actual named frames. **It does not hold up.** Every candidate blob from
those frames gets suppressed as person-overlap — the object is in an open
palm, so its own motion region sits inside the arm/hand silhouette that
person-suppression is designed to discard. Independent of suppression, none
of the raw blobs are even a tight box on the object either. Checked
quantitatively too: neither frame's labelled ground-truth box is matched by
any blob at IoU ≥ 0.5. **Most likely explanation, based on what the data
does show:** "car key read as `cell phone` at 0.92" is a real, correctly
measured number — but it belongs to stock YOLO's own classifier output
(Session 2, Block B, in `docs/phase-3-step0-findings.md`), not to change
detection. The two results were probably conflated when decision 7 was
originally written, both having been produced in the same measurement
session. This is worth sitting with for a second: **the underlying fact was
true, but attributed to the wrong mechanism** — a subtler and easier-to-make
error than an invented number, and a useful thing for the paper to name as
its own failure mode distinct from "just making something up."

What did hold up: pure frame-differencing on 1080p really is about
**1.4ms/frame** — that specific number was correct. What changed: real
person-suppression (two YOLO calls per pair) costs roughly **37ms more**, so
the honest combined figure is **~43ms/pair**, not ~1.4ms. The "cheap"
framing in the original claim was true only for the part of the pipeline
that, on its own, wasn't doing the job (suppressing the dominant false
positive) the claim said it was doing.

### Round 2: persistence tracking, and a genuine (if narrow) win

Round 1's fix had a real cost of its own, found while measuring it:
single-pair person-overlap suppression deletes a real hazard the moment it's
being placed, because the object's motion is inside the exact region being
suppressed. It also couldn't be properly evaluated at all on the existing
test bursts, which — as just established — structurally cannot contain an
"appears and is left" event.

The fix: persistence tracking. A candidate blob has to still be present, in
roughly the same location, two or three frames later, *and* pass a
visual-stability check (crop-to-crop re-diff of just that region) — not
location alone. Location-only was tried first and measured directly on a
true-negative control pair: it kept 57 of 69 raw blobs as "persisted," every
single one checked by eye and found to be floor grout, whiteboard glare, or
plastic-wrap sheen — the highest-contrast edges in the frame, which relight
slightly on every exposure and therefore reappear "in the same place" on
every comparison without anything having moved. Adding the stability gate
fixed that specific failure.

Tested on a new, purpose-shot continuous clip (`changetest`, 22 frames about
1 second apart, in which a person visibly places a pair of scissors on the
floor, steps back, and later picks them back up): **the detector genuinely
catches the placed object.** This was checked by eye, not just by score — a
crop of the kept, persistence-confirmed bounding box was pulled directly
from the actual pixels and it is a tight, correctly-shaped box on the
scissors' handles, nothing else, in the frame right after the person steps
back. This is the exact scenario the guided-walkthrough/change-detection
split (CLAUDE.md decisions 3 and 6) depends on: something is placed, the
person who placed it moves away, and it's still flagged.

**Precision is still bad, and got worse in exactly the places that
matter.** On the same `changetest` burst, only about 1 in 51 kept blobs
sampled and inspected was the real object; the rest are edges near where the
person had just been standing, now stabilized after they left — a distinct,
still-unsolved failure mode (motion residue that happens to settle, not
lighting flicker). On the older labelled bursts, pooled recall actually
**dropped**, from 0.030 to 0.011 — the expected, honest cost of requiring
more confirming evidence than sparsely-spaced data can supply. Precision
across a fresh stratified sample landed in the same ~33% range as before —
different false positives got through, not fewer of them overall.

### The honest bottom line, stated once, plainly

Change detection, as it exists after this phase, **works in the narrow
sense that matters most** — it can catch a real object placed and left,
checked against real pixels, not just a score — **and is not good enough to
alert a parent on unfiltered.** Roughly one in three of its detections is a
real object; the rest is floor texture, shadow, or motion residue near where
a person recently stood. That is the honest state of the "main path"
mechanism CLAUDE.md decision 7 currently names, and CLAUDE.md decision 7 has
been corrected in this same session to say exactly this instead of the
original, wrong claim — verified directly by reading the current file
against `docs/phase-3-step0-findings.md`'s numbers, not assumed.

### Why the phase closes here instead of continuing to iterate on change detection alone

This is a genuinely new methodological finding for the paper's process
narrative, not just a decision, and Shaked asked for it to be documented as
such rather than folded quietly into "measure honestly."

Look at the shape of what happened across this phase's attempts to make a
single Layer A mechanism reliable, tested offline, in isolation, against
static photos or short clips:

- Fine-tuning round 2 (public data first) fixed `sharp_object` recall and,
  as a side effect nobody had asked for, let `small_swallowable` recall
  collapse.
- Fine-tuning round 3, trying to fix exactly that side effect, made *both*
  numbers worse than round 1 — the single worst result of the phase.
- Change detection's first fix (person-suppression) fixed the dominant
  false-positive source and, as a side effect, deleted real hazards being
  placed by a person's hand.
- Change detection's second fix (persistence) fixed that specific problem
  and, as a side effect, made recall worse on every previously-usable test
  burst.

Five rounds, two different mechanisms, the same shape every time: fixing
what the last measurement flagged as broken reliably broke something else
the *previous* measurement had relied on, and each fix could only be judged
against whatever narrow slice of static test data happened to be on hand at
the time — never against the system actually running. **Shaked's correction
is to recognize this as an unproductive loop and stop running it, not to
declare any one of the five results "good enough."** The reasoning, stated
plainly: static, single-method, offline testing has been given five real
chances across this phase to converge on something usable and has not — not
because the team wasn't measuring carefully (the opposite: every round here
is more carefully measured than the round before it), but because isolated
testing structurally cannot tell you how a component behaves once it's
wired into the rest of the system it's meant to serve. The team's actual
next question — does change detection, run continuously on a real live
feed, integrated with the rest of Layer A and Layer B, produce a hazard map
a parent could plausibly trust — cannot be answered by a sixth static test,
however carefully constructed. It can only be answered by building the
integration and watching it run, which is explicitly Phase 4's job now.

**This is not a guarantee that Phase 4 will fix change detection's
precision.** It might not. What's being decided here is *how* the team finds
out what to fix next — by running the real system and observing real
failures, not by producing a sixth isolated offline number — not that the
current ~33% precision is an acceptable place to stop. Per CLAUDE.md's own
preamble, this is a revisable engineering call, made with the evidence
available now, not a final verdict.

## Verdict: is Phase 3 closeable?

**Yes — closed as of this write-up, second attempt.** The first attempt's
verdict, for the record (this paragraph is the one place in this document
where the earlier conclusion was fully replaced rather than left in place
with a correction note, so it's quoted here rather than silently dropped):
*"I am marking `PHASE_PLAN.md`'s Phase 3 status `[~]` (in progress), not
`[x]` (complete)... Condition (1) [demoably works] is not met under any
reading — there is nothing to demo. No change detection runs live... What
can honestly be said to be done, and done well, is the measurement and
decision-making work."* That was the correct call at the time it was made —
the change-detection evidence really was unverifiable then. The verdict
below replaces it now that the specific gap that conclusion was protecting
against has been closed with real, checked evidence.

Re-reading the project's own phase-gate rule with that in mind: *"a phase is
complete when (1) it demoably works, (2) docs-agent has reviewed it and
confirmed the code matches the claim, and (3) docs-agent has produced the
phase write-up."* Condition (1) — "demoably works" — has to be read against
the scope this phase actually settled on, which was explicitly *not*
re-litigated as "a parent could use this today": it's "a validated direction
plus real, runnable measurement tooling." Under that reading, there is now
something to demo that wasn't there at the first attempt: `measure_
detection.py` producing the class list; `evaluate_home_frames.py` producing
the three fine-tuning rounds' precision/recall on a real held-out building;
and `measure_change_detection.py`, run against the real `changetest` clip,
correctly catching a real placed object — not a hypothetical, an actual
pixel-checked result. That is a materially different, stronger position than
"there is nothing to demo," which was true at the first attempt and is no
longer true now.

What is **still** true, and should not be softened by marking the phase
`[x]`: the guided walkthrough does not exist as code (still Tier "no code" —
see above), change detection is not integrated into any running pipeline,
and its precision (~1 in 3) is not good enough to alert a parent on
unfiltered today. Marking the phase `[x]` is a statement that the scoping
and validation work this phase was actually responsible for is done and
honestly documented — it is explicitly **not** a statement that Layer A is
finished, safe to ship, or even good. Phase 4 inherits two concrete build
items that did not exist in its original scope (change detection
integration, the guided walkthrough), plus, now, an explicit instruction
from Shaked: use live testing on the real system to find out what to fix
next, rather than a seventh round of static offline measurement.

## For the write-up's teaching purpose: check your own understanding

1. Walk through the 10:00:00-to-10:32:00 worked example above from memory,
   without looking back at this document. Which layer (A or B) does the slow
   step touch, and why does that mean its multi-second latency is invisible
   to the alarm's response time?
2. Round 2 beat round 1 on `sharp_object` but round 3 (which trained on
   *more* data, including all of round 2's public images plus more epochs of
   our own frames) did dramatically worse on both classes. Explain, without
   re-reading the "oversample by duplication" section, why more training
   data and more epochs made the model worse rather than better here — what
   specifically was being memorised, and why didn't the model's own
   validation split catch it?
3. CLAUDE.md decision 4 already said, from the start of the project, that a
   parent reviews setup-phase hazards. What, concretely, changed between
   "decision 4 already covers this" being technically true throughout the
   phase and the team actually building around it? Put another way: what was
   decision 7's work actually doing differently from what decision 4
   assumed, for those first three fine-tuning rounds?
4. This document distinguishes three tiers of evidence (code+prose+traceable
   numbers; code+numbers-but-no-traceable-prose; no code at all). Pick one
   claim still in Tier 3 as of this write-up (colour clustering or texture
   objectness — change detection has since moved to Tier 1, see below) and
   explain what you would need to see committed to this repo before you'd be
   comfortable citing that number to your advisor as an established fact
   rather than "the team told me this."
5. If you had to defend, to Tom, why Phase 3 is marked `[x]` now but was
   correctly marked `[~]` at the first close attempt, despite the underlying
   architecture decisions (decisions 3, 4, 7) not having changed in between
   — what's the one-sentence version of what actually closed the gap?

### New questions, for the second closure specifically

6. Change detection's headline claim wasn't just unproven at the first
   attempt — round 1 of the re-measurement found it was actively wrong, and
   traced the real number to a different method entirely (stock YOLO
   classification, not change detection). Explain, in your own words, how a
   correct-but-misattributed number is a different kind of mistake than an
   invented one, and why that distinction matters for how carefully a team
   should cite results across sessions or methods.
7. Round 2 of the change-detection fix (persistence tracking) made recall
   *worse* on three of the four previously-labelled test bursts, while also
   producing the phase's one clean, eyeballed success (the placed scissors
   on the `changetest` clip). Without treating these as contradictory,
   explain why both can be true at once — what was actually different about
   the data in each case?
8. Five rounds (three fine-tuning, two change-detection fixes) each fixed
   one measured problem and introduced a new one. Shaked's response was not
   "try a sixth fix" but "stop fixing in isolation and test the whole system
   instead." Explain the difference between those two responses to a string
   of partial failures — what does the second one assume about *why* the
   first five didn't converge that the first response doesn't?
9. This write-up is explicit that closing Phase 3 does not mean change
   detection is fixed, or that Phase 4 is guaranteed to fix it. If Phase 4's
   live testing shows the same ~33% precision problem in a running system,
   would that mean Shaked's decision to close Phase 3 here was wrong? Why or
   why not?
