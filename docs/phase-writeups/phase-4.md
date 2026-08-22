# Phase 4 — the risk engine: detect, human classifies, detect new, alert

**Status: CLOSED (2026-08-22). Closed as "the logic is built, tested, and
live-verified on real hardware for the mechanisms that matter most, with one
specific safety-critical path still code-and-test-only, and the parent-facing
interaction explicitly deferred to Phase 8." Read that sentence before reading
anything else — it is more precise than "Phase 4 works," and the difference
matters.**

This write-up covers the whole Phase 4 session (2026-08-12 through
2026-08-22), reconstructed primarily from
`docs/session-logs/phase-4-session-transcript.md` — the raw session record,
not a secondhand summary — cross-checked directly against the committed code
(`cv/risk_engine.py`, `cv/test_risk_engine.py`) rather than taken on faith.
Where a claim below is "I checked this myself," it means I read the actual
function or ran the actual grep, not that I'm repeating what the transcript
says happened.

## What this phase was actually trying to build

`PHASE_PLAN.md`'s Phase 4 section states the requirement, after a mid-phase
rewrite, in four lines:

> A camera detects objects. A human classifies which are dangerous. The
> system detects new objects appearing after that. It alerts.

That's deceptively simple-sounding for what it took to get there. This phase
did not start from that sentence — it arrived at it, the hard way, after
building something much more complicated first and watching it fail.

## Act 1: five models in one loop, and why that collapsed

Phase 4 started (2026-08-12) with a reasonable, incremental plan: a live
overlay running person tracking plus whatever named hazards already worked
without training (oven/microwave, refrigerator), then layering in more
mechanisms slice by slice — wall sockets via open-vocabulary prompting, a
broader `sharp_object` detector, a Grounding DINO "double-checker" to filter
change-detection's known-weak false positives, and the change detector
itself, carried over from Phase 3 as a "known-weak, not finished" component.

Each piece worked in isolation when tested alone. Running together, by
2026-08-13, the system had accumulated **five models in one loop**: the
per-frame YOLO26 pass, two separate YOLO-World instances (sockets,
sharp objects), Grounding DINO for crop confirmation, and the change
detector itself. Live testing on real recordings surfaced three real,
user-visible bugs at once: the frame rate collapsed to 2–2.5 FPS (down from
13–15), roughly twenty blue "something changed" boxes appeared scattered
across blank walls with nothing actually there, and two distinct objects
placed near each other sometimes merged into one hazard-map entry.

Shaked stopped the session mid-debugging on 2026-08-13, not to ask for a
sixth fix, but to name what was actually happening: *"we've been doing that
for a few days... I think we got lost and stuck on specific things and we
need to rethink it."* He restated the requirement in its simplest form (the
four-line quote above) and delegated the technical mechanism entirely, with
three conditions: explain every decision in plain language, justify it
against that specific requirement (not "more robust" in the abstract), and
document the current architecture rather than layering more history on top
of it.

**The diagnosis that came back, and that held up:** nearly every live bug —
the freeze, the wall-spam, the duplicate merges, and (correctly suspected,
later confirmed by direct measurement) the FPS collapse — traced back to one
component: the automatic pixel-differencing change detector, known weak
since Phase 3 (~1 in 3 detections real) and the reason Grounding DINO existed
at all (to filter its mess before a human saw it). Days had gone into
patching that one component instead of asking whether the system needed it
to carry that much weight.

## Act 2: the reset — two models, one rule

The mechanism that replaced the pixel differencer is worth stating precisely,
because it's the actual technical core of this phase and it's genuinely
different from what it replaced, not just a smaller version of it:

**Layer A is now maintained by a periodic class-agnostic re-scan compared
against what is already known.** Every few seconds, a segmentation pass
(FastSAM, imported from `cv/measure_segmentation.py` — reused, not
reimplemented) lists occupied spots on reachable surfaces. A spot in the new
list that wasn't in the old one means something arrived; a known spot missing
from the new list means something was removed. Pixel differencing compares
*brightness* — a shadow, an auto-exposure adjustment, or a person walking by
all look identical to "an object appeared," which is exactly why the old
mechanism filled a blank wall with boxes. A re-scan compares *objects*, so a
wall is never "new" in either list, and the scan's own false positives (sofa
texture, floor grout) self-cancel because they show up in both lists, in the
same place.

**The rule that removed the most complexity: nothing auto-adds a hazard.
Every detector proposes; a human disposes.** This collapsed five
source-specific hazard categories (each with its own colour, matching logic,
staleness rule, and alert behaviour) into one `HazardEntry` with three states
— `PENDING`, `CONFIRMED`, `DISMISSED` — regardless of which detector proposed
it. I checked this directly: `HAZARD_STATE_PENDING`/`_CONFIRMED`/`_DISMISSED`
are the only three states defined in `risk_engine.py`, and both the per-frame
named-class pass and the socket pass now go through `HazardMap.propose_named`
into `PENDING`, same as the room scan — this is a genuine behaviour change
from earlier slices, where oven/fridge/socket detections went straight onto
the map as alert-eligible. It's a small extra tap per appliance at setup, in
exchange for the same "one rule, not five" simplification decision 7 now
states.

**Two rules Shaked added on review, both catching real holes in the first
draft, and both now written into CLAUDE.md decision 4:**

1. **An unreviewed new object is treated as dangerous until judged.** The
   first draft only alerted on parent-confirmed hazards — which meant an
   object appearing while nobody was watching the screen would sit silently
   as a child approached it, the exact accident the product exists to
   prevent. Exception: items from the *first* scan (the room's normal
   starting state, with the parent presumably standing right there) don't
   alarm while pending.
2. **A new object alerts on appearance, not only on approach** — a parent
   shouldn't have to be watching the screen to learn that something arrived.

**The dismissal re-raise rule ("better safe than sorry"):** dismissals are
recorded per-spot, so in principle a real hazard placed exactly where
something harmless was dismissed could inherit that dismissal. Shaked's call
was explicit: *"keep it as double and even triple mark — better safe than
sorry."* The fix: a dismissal records a crop of the spot at dismissal time
(`HazardMap.dismiss`), and a later scan finding that spot materially changed
(`fingerprint_changed`, reusing `region_change_frac` from
`measure_change_detection.py`) puts it back in `PENDING`, erring toward
re-asking on any missing/degenerate comparison rather than silently trusting
one that couldn't run.

## What was cut, and why cutting it was the fix, not a compromise

- **The live pixel-differencing change detector** — replaced by the re-scan
  comparison above, which does the same job in the object domain instead of
  the pixel domain. `cv/measure_change_detection.py` still exists on disk as
  a research artifact; only its live wiring is gone.
- **The Grounding DINO crop-confirmation step** — existed only to filter the
  change detector's false positives before a human saw them. With the human
  as the filter by design (decision 7's "every detector proposes, the parent
  disposes"), it had no job left, and at ~500ms/call it was the prime suspect
  for the FPS collapse — later confirmed correct by direct measurement (see
  below).
- **The `sharp_object` (knife/scissors) open-vocabulary detector** — the room
  scan proposes these objects anyway and the parent confirms them, same
  outcome with one fewer model. It also measured precision 0.382 / recall
  0.210 on an unseen building in Phase 3/4 research — worse than simply
  showing the parent a candidate box.

## Auditing the actual code against CLAUDE.md decisions 3, 4, and 7

I read `cv/risk_engine.py` (1,299 lines) and `cv/test_risk_engine.py` (604
lines) in full, not just the module docstrings, and checked specific claims
against the code rather than the comments describing it.

**Decision 3 (periodic re-scan is the sole "detect new/removed"
mechanism):** confirmed. `HazardMap.apply_scan_candidates` is the only place
that adds or removes `HAZARD_ORIGIN_SCAN` entries, and it does exactly what
decision 3 describes: dedupe within one scan pass
(`dedupe_candidates`, IoU ≥ 0.4), match against known scan-origin entries,
bump an absence counter on anything unmatched and remove it once that counter
reaches `SCAN_CONSECUTIVE_SCANS_REQUIRED` (2), and promote a candidate to a
real entry only after two *consecutive* sightings via the `_Provisional`
list. That two-scan guard is a real, deliberate jitter guard — a single
noisy scan cannot create or delete a hazard — and I confirmed it's exercised
by seven distinct tests (`test_apply_scan_candidates_*`), covering arrival,
non-arrival on a single sighting, removal after two misses, survival after
one miss, and the re-raise path.

**Decision 4 (the four-row alert-eligibility table):** the code matches the
table exactly. `hazard_alerts_on_approach` is an eight-line function that
implements precisely CONFIRMED→alert, PENDING+arrived-later→alert,
PENDING+first-scan→no alert, DISMISSED→never alert — I read it directly and
it has no branch not represented in the table. All four rows have a direct
unit test (`test_hazard_alerts_on_approach_confirmed_always_alerts`,
`_pending_first_scan_does_not_alert`, `_pending_arrived_later_does_alert`,
`_dismissed_never_alerts`). **What is not directly tested is the
composition** — an unreviewed, arrived-after-first-scan `PENDING` entry
actually producing a `"red"` zone once it's passed through `score_frame`.
This isn't a logic gap: `score_frame` never inspects `.state` at all (I
checked — it only touches `.bbox`, `.id`, `.label`), and `main()` filters the
hazard list through `hazard_alerts_on_approach` before calling `score_frame`,
so the composition is structurally sound. But there is no test — unit or
live — that exercises both halves together with a `PENDING` entry as input.
This matches exactly the gap the task briefing asked me to check for, stated
plainly rather than softened: **the rule is real code with a real, passing
unit test at the eligibility-check level; it has not been demonstrated, in
a test or on camera, actually escalating a live proximity score to red.**

**Decision 7 (two detectors, nothing auto-adds a hazard):** confirmed by
direct grep — `ChangeDetector`, `grounding_dino`/`gdino`, `sharp_object`, and
`remove_stale_auto` all return zero matches in `risk_engine.py`. The only two
detectors feeding the map are the per-frame YOLO pass (`oven`/`microwave`
folded into `oven_microwave`, `refrigerator` kept separate — `chair` is
deliberately excluded, per a code comment citing a live false-fire on a
coffee table) plus a cadenced open-vocabulary wall-socket pass. Both funnel
through `HazardMap.propose_named`, which only ever creates `PENDING` entries
— never `CONFIRMED` — matching decision 7's "every detector proposes"
literally, not just in spirit.

**Decision 2 (single YOLO pass, oven/microwave/fridge read from the same
result as person):** confirmed — `main()` runs exactly one `model.predict()`
call per frame and reads both `person` and the hazard classes out of the
same `results[0].boxes` loop.

**Decision 5 (resolution-independent proximity, rolling window):**
confirmed — `normalized_center_distance` divides by frame diagonal, and
`score_frame` maintains a `deque(maxlen=ROLLING_WINDOW_SIZE)` (8 frames) per
`(person, hazard)` pair. The zone thresholds (`RISK_ZONE_RED_MAX=0.15`,
`ORANGE_MAX=0.30`, `YELLOW_MAX=0.50`) and the window size are explicitly
commented in the code as unmeasured first guesses, unchanged since the
2026-08-12 kickoff — this is honestly stated in-code, not something I had to
dig for, and it's a real open item worth naming: nobody has yet watched
real approach behaviour and asked "does red actually fire at the distance
that feels dangerous."

## Live verification: what was actually watched happen, on real hardware

The transcript's final exchange (2026-08-22) is the payoff for the whole
reset. Shaked ran `python risk_engine.py --name Arducam`, recorded a 54.6-second
clip, and reported five things working. I did not take that report at face
value — I pulled every frame from the clip via `ffmpeg`, ran an automated
freeze check across the whole 54 seconds, and read the on-screen readouts
directly off multiple extracted frames myself, the same way the previous
phase write-ups' audits worked.

**Confirmed, independently, from the actual frames:**

- **FPS 15.0–15.2, steady for the whole clip**, versus 2–2.5 before the
  reset. This closes the leading hypothesis (Grounding DINO plus the two
  YOLO-World passes were the cost) with a real measurement, not just a
  plausible story.
- **No frozen frames anywhere** in the 54-second clip (automated pairwise
  frame-diff check, zero flagged).
- **A new object placed in frame was picked up by the scan on its own**,
  boxed orange (`NEW-UNREVIEWED`), with `ALERT: New object detected (room
  scan)` on screen — nobody touched a key.
- **Approaching a confirmed hazard produced `RISK: RED`**, with the red
  connector line drawn from the person to the hazard.
- **Removing a confirmed object made its box vanish on its own** (hazard
  count 17 → 11) — via the scan-comparison mechanism, not a stale-entry
  timer, matching the 2026-08-13 redesign's stated replacement for the old
  timer-based approach.
- **A dismissal re-raise fired live, unprompted**: the overlay read `ALERT:
  New object detected (spot changed since dismissal)` at one point in the
  clip. This is Shaked's "better safe than sorry" rule (decision 4) working
  against real pixels, not just passing its unit test.
- **Duplicate boxes on the same object are "mostly gone,"** per Shaked
  directly — not claimed as fully solved. `SCAN_DUPLICATE_IOU_THRESHOLD=0.4`
  is still an unmeasured first guess in the code; this is the first live
  evidence it's roughly in the right range, not a tuned, validated value.

**Not verified on this clip, stated plainly, not blurred with the above:**
every RED event in the recording was against an already-confirmed hazard,
because everything visible had already been tapped through by the time the
approach happened. The specific, safety-critical case — a child approaching
an object that arrived *after* setup and has not yet been reviewed —
was not exercised on camera this session. The code implements it and it has
a passing unit test (see the decision-4 audit above), but "implemented and
unit-tested" and "watched happen on real pixels" are two different claims,
and this write-up is treating them as two different claims deliberately, per
the task that produced it.

## Three items handed off, not silently dropped

**(a) Debug telemetry is drawn into the frame, not exposed as data.**
`risk_engine.py`, in `main()`, calls `draw_overlay_line` three times on the
same `annotated` image that also carries the hazard/person boxes: FPS
(`draw_overlay_line(annotated, f"FPS: {smoothed_fps:.1f}", 1)`), model
config (`f"{args.model} imgsz={args.imgsz} conf={args.conf} {device}"`), and
the risk readout including raw normalized distance
(`draw_risk_readout`). Per CLAUDE.md decision 1, Flet is meant to be a pure
client of FastAPI's `/video_feed` and JSON endpoints — it never touches frame
pixels directly. Bounding boxes on the frame are fine and intentional (a
parent should see what's flagged); the FPS/model/distance debug readout is
developer-only information baked into pixels the eventual UI can never strip
back out. This needs to become `/risk_status` JSON fields Phase 7 exposes
and the UI decides whether to render — a concrete Phase 7 architecture note,
not a cosmetic one. **I have not changed CLAUDE.md for this** — flagging it
here and in the decision log as a proposal for Shaked/Yahli to confirm before
Phase 7's design is locked in, per the project's own "ask first, log the
change" rule.

**(b) The h/n/s keypress loop is a developer stand-in, not the parent
feature.** `PHASE_PLAN.md`'s Phase 4 done-when (a) reads: "a parent can tap
through the room's objects and have the confirmed ones tracked." What
exists is a keyboard-driven review queue (`h`=confirm, `n`=dismiss,
`s`=skip) drawn onto an OpenCV window — genuinely the same underlying logic
a parent-facing UI would need (the `ReviewQueue`/`HazardMap.confirm`/`dismiss`
API is real and tested), but not something a parent could use: it requires a
keyboard, a focused OpenCV window, and single-letter keys with no visual
affordance beyond text. That interaction is explicitly Phase 8's job (the
Flet UI). Reading done-when (a) as "the logic exists and is provably
correct" rather than "a parent could use this today" mirrors exactly how
Phase 3 read its own "demoably works" bar — and it should be stated
explicitly here rather than left to sound like a finished parent feature
exists.

**(c) Overlapping labels when boxes cluster is cosmetic.** When several
hazard boxes sit close together (observed in the top-left corner during live
testing), their text labels overlap and become unreadable. Genuinely
cosmetic — no risk-logic or safety implication — filed for Phase 8, no
action needed now.

## Verdict: is Phase 4 closeable?

**Yes, closed as of this write-up (2026-08-22), with the specific gaps above
named rather than smoothed over.** Re-reading `PHASE_PLAN.md`'s own gate —
"(1) it demoably works, (2) docs-agent has reviewed it and confirmed the code
matches the claim, (3) docs-agent has produced the phase write-up" — against
Phase 4's own three-part done-when bar:

- **(a) "a parent can tap through the room's objects and have the confirmed
  ones tracked"** — met at the logic layer (real `HazardMap`/`ReviewQueue`
  state machine, live-verified to correctly track confirm/dismiss/re-raise),
  not met as a parent-usable feature (the keypress loop is a developer
  stand-in; Phase 8 owns the real interaction). This is a genuine, named
  partial-completion, not a clean pass — recorded as such rather than
  rounded up.
- **(b) "an object introduced afterwards is noticed and alerted on without
  anyone touching the app"** — live-verified on real hardware, independently
  re-checked against the actual video frames, not just a chat summary.
- **(c) "moving toward a hazard visibly and correctly escalates the risk
  level in real time"** — live-verified for the confirmed-hazard case; the
  unreviewed-hazard case (arguably the more important half of decision 4's
  table, since it's the "unreviewed means dangerous" safety rule) is code-
  and-test-verified only, not yet watched happen on camera.

The reason this still closes, rather than staying `[~]`: the actual
technical core of this phase — replacing five competing models with a
two-model, one-rule design, and getting it to run at a usable frame rate —
is now demonstrated, not theorized, and independently checked by pulling the
real frames rather than trusting the report. The three items above are named
gaps with a clear owner and a clear next step (a live camera test targeting
specifically the unreviewed-approach case; Phase 7's `/risk_status` design;
Phase 8's real review UI) — not things quietly left out of the record. Per
CLAUDE.md's own preamble, this is a revisable engineering call made with the
evidence available now, not a claim that every corner of decision 4's table
has been watched fire on a screen.

## For the write-up's teaching purpose: check your own understanding

1. Explain, without re-reading the "Act 1" section, why running five
   individually-correct detection models in one loop produced bugs that
   none of the five had on their own. What specifically made the pixel
   differencer the common root cause of the freeze, the wall-spam, *and*
   the duplicate-merge bug, when those look like three unrelated symptoms?
2. The periodic re-scan and the deleted pixel differencer both try to answer
   "did something new show up." Explain in your own words why comparing
   *object lists* self-cancels the scan's own false positives (sofa
   texture, floor grout) in a way that comparing *raw pixels* cannot.
3. CLAUDE.md decision 4's table has four rows. Which one is the "unreviewed
   means dangerous" rule, and why did Shaked call the first draft (which
   only alerted on confirmed hazards) a real hole rather than a reasonable
   simplification? What's the actual accident scenario that row exists to
   prevent?
4. This write-up distinguishes "implemented and unit-tested" from "watched
   happen on real pixels" for the unreviewed-hazard-approach case
   specifically. `score_frame` never even looks at `.state` — so why isn't
   the unit-test coverage of `hazard_alerts_on_approach` alone considered
   sufficient proof the full path works? What would a test (or a live
   clip) that actually closes this gap need to contain?
5. If you had to defend to Tom, in one sentence, why Phase 4 done-when (a)
   is marked satisfied despite there being no parent-usable interaction
   yet, what's the one-sentence version of the distinction this write-up is
   drawing?
6. Five models running together caused bugs the same five models didn't
   have running alone. Phase 3 also has an entry (Entry 6 in
   `docs/agent-workflow-notes.md`) about static, isolated testing failing to
   predict integrated behaviour. Are these the same underlying lesson
   applied twice, or two different lessons that happen to rhyme? Explain
   your answer with reference to what specifically was being tested in
   isolation in each case (a model's precision/recall, versus five models'
   combined runtime behaviour).
