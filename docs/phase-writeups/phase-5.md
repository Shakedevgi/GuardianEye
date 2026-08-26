# Phase 5 — alerts, rolling buffer, saved clips, voice playback

**Status: CLOSED (2026-08-26), with specific gaps named below rather than
smoothed over — the same pattern Phase 4 closed under.**

## How this write-up was produced, and a real limitation of this audit

Unlike Phase 4, there is no session transcript for this session
(`docs/session-logs/phase-5-session-transcript.md` does not exist, and the
task briefing said not to go looking for one). This write-up is built from
two things I checked independently rather than took on faith: **the five
dated 2026-08-22/2026-08-26 entries in `docs/decision-log.md`** (the primary
record of what was decided and why), and **a full read of the live
`cv/risk_engine.py` and `cv/test_risk_engine.py`** as they currently sit in
the working tree (both show as modified, uncommitted, in `git status` — I
read the actual files on disk, not a diff).

**A real tooling gap this session, stated plainly rather than hidden:** I did
not have shell/Bash access this session — the `Grep` and `Glob` tools both
failed with `ENOENT: rg not found` on every call, and there is no way for me
to run `python cv/test_risk_engine.py`, `ffprobe`, or `ffmpeg` directly.
Phase 4's audit ran the actual test suite and pulled real video frames with
`ffmpeg`; this one could not. What I could still do, and did: read every line
of both Python files with the `Read` tool, hand-trace several test assertions
against the implementation arithmetic to confirm they'd pass (not just that
they exist), count the test functions directly (89, matching the decision
log's own final count), and confirm specific binary artifacts exist on disk
(the `Read` tool errors with "cannot read binary file" rather than "file not
found" for a real file, which is a reliable existence check even without
being able to inspect the contents). Where a claim below rests on that
weaker form of verification, I say so explicitly rather than presenting it
with the same confidence as something I ran myself.

## What this phase was building

`PHASE_PLAN.md`'s Phase 5 section: visual alerts, pre-recorded voice-clip
playback, a rolling video buffer that saves 5–7s clips **on critical alerts
only**. Done when: "a simulated critical event produces a correct saved clip
file and the right alert fires — visually **and audibly**." Owner:
cv-agent (buffer/trigger logic) + backend-agent (clip storage hookup).

**Scope check, confirmed directly: this session stayed entirely inside
cv-agent's lane.** There is no `backend/` directory anywhere in the repo yet
(checked directly — Phase 6/7 haven't started), and every new symbol this
phase added lives in `cv/risk_engine.py`: no FastAPI, no SQLite, no HTTP.
Clips land on local disk at `cv/clips/pending/`; keep/discard and
auto-delete of undecided clips are explicitly named in the module docstring
as Phase 6's job, not this one's — matching `PHASE_PLAN.md`'s own division
of labor rather than quietly absorbing it.

## Three measurements before any code, and whether the numbers hold up

Per the project's established "measure before building" culture (the same
discipline that drove Phase 3's fine-tuning rounds and Phase 4's FPS
investigation), the 2026-08-22 kickoff entry records three measurements made
*before* the alert/buffer/clip code was written. I can't re-run these
measurements myself this session, but I checked that the constants actually
shipped in the code match what the decision log says was measured, which is
the cheapest available cross-check against a number being asserted without
being acted on:

1. **Rolling buffer memory.** Claimed: JPEG q75 at full capture resolution
   averages ~200KB/frame (~15MB for a 5s/75-frame buffer) versus ~467MB for
   raw 1080p frames over the same window. `ROLLING_BUFFER_JPEG_QUALITY = 75`
   is exactly what `RollingBuffer.append()` uses
   (`cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])`,
   `cv/risk_engine.py:1418`), and downscaling was explicitly considered and
   rejected in the same entry as "a bad trade" for a lower-resolution clip.
   No downscaling appears anywhere in `RollingBuffer`/`ClipRecorder` — the
   decision is real, not just written down.
2. **Non-blocking audio.** Claimed: `subprocess.run(afplay)` cost ~1.9s on a
   1.0s clip (~29 dropped frames at 15 FPS); `subprocess.Popen(afplay)` cost
   2–5ms; `AppKit.NSSound.play()` cost up to 112ms and was rejected.
   `AudioPlayer.play()` (`cv/risk_engine.py:1558`) uses exactly
   `subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL,
   stderr=subprocess.DEVNULL)` — no `.run()`, no `NSSound` import anywhere in
   the file. The decision matches the code.
3. **Alert cooldown policy.** Claimed: pulling the risk-zone readout at 4Hz
   from the real 2026-08-22 unreviewed-approach recording showed one
   continuous approach flicker RED→ORANGE→RED twice, with dips up to 0.75s,
   despite the existing 8-frame rolling window. `ALERT_HOLD_SECONDS = 2.0`
   ships with an in-code comment citing that exact recording and computing
   "~2.7x margin over the worst dip actually observed" — a number a reader
   can check by hand (2.0 / 0.75 ≈ 2.67), and it does.

I did not re-derive these three numbers from raw footage myself this
session (no shell access to `ffmpeg`/`afplay` timing scripts). What I *can*
say with confidence, from reading the code directly, is that the constants
actually shipped are the ones the decision log claims were measured, and the
reasoning attached to each is internally consistent and falsifiable (the
2.7x margin arithmetic, the JPEG-vs-raw byte math) rather than a vague
assertion.

## What was built (Phase 5's actual code, verified against decisions)

Everything below is under a section literally titled `# --- Phase 5: alert
lifecycle ---` in `cv/risk_engine.py` (line 1115 onward) plus a smaller
drawing-order change in `main()`. I read all of it, not just the class
docstrings.

**`AlertEvent` / `AlertSignal` / `AlertManager`** (`risk_engine.py:1118`–
`1284`). Replaces what the module's own comments describe as the
pre-Phase-5 single overwritable `alert_text`/`alert_until` slot with real
per-`(person_id, hazard_id)` event identity. `AlertManager.update_proximity`
implements **exit hysteresis, not entry debounce**: an event opens the first
time a pair enters a scored zone, re-signals only on escalating past its own
peak zone, and closes only after `ALERT_HOLD_SECONDS` (2.0s) of the pair
going unseen — a de-escalation within the hold window updates bookkeeping
silently, with no signal. `should_trigger_clip()` is a decide-and-commit
gate matching CLAUDE.md decision 6 exactly: `event.kind != PROXIMITY` → no
clip; `event.peak_zone != "red"` → no clip; `event.clip_triggered` (already
used) → no clip; otherwise marks `clip_triggered = True` and additionally
checks a **separate**, global `CLIP_MIN_INTERVAL_SECONDS = 30.0` disk-safety
backstop. I confirmed these are two independently-gated concerns by reading
the method body directly (`risk_engine.py:1254`–`1284`): a blocked clip
attempt is not retried later in the same event, by design, per its own
docstring.

**`RollingBuffer` / `ClipRecorder` / `write_clip()`** (`risk_engine.py:1394`–
`1530`). `RollingBuffer` is a bounded `deque` of `(timestamp, JPEG-bytes)`
tuples sized by `fps * seconds` (75 frames at the default 15.0 FPS / 5.0s).
`ClipRecorder.trigger()` takes a pre-trigger snapshot and appends live tail
frames via `add_tail_frame()` until `CLIP_TAIL_SECONDS` (2.0s) elapses, then
`poll()` spawns a **background thread** running `write_clip()` — a plain
module-level function (not a method) specifically so it's callable
synchronously in a test, per its own docstring, while `ClipRecorder` always
calls it threaded in production. `write_clip()` tries `avc1` (H.264) first,
falls back to `mp4v` if the platform's OpenCV build can't open an `avc1`
writer — I confirmed both `CLIP_FOURCC_PRIMARY = "avc1"` and
`CLIP_FOURCC_FALLBACK = "mp4v"` are used exactly as named, with the fallback
gated on `writer.isOpened()`, not merely attempted once. Filename convention
(`{timestamp}_event{id}_{hazard_label}.mp4`, `risk_engine.py:1488`) matches
the decision log's agreed convention character for character, and a real
file matching that shape (`20260826-204245_event8_object.mp4`) exists on
disk under `cv/clips/pending/` — I confirmed this with the `Read` tool
(it errors "cannot read binary .mp4 file" rather than "file does not
exist," which is a reliable existence check even though I can't inspect the
file's actual frame count/duration/codec without `ffprobe`).

**`AudioPlayer`** (`risk_engine.py:1533`–`1570`). Missing audio files log a
warning once per filename and are skipped, not a hard crash — confirmed by
reading `play()` directly and by the one unit test that exercises it
(`test_audio_player_missing_file_does_not_raise_and_warns_once`). The three
files it expects — `cv/audio/hazard_detected.wav`, `baby_getting_close.wav`,
`immediate_danger.wav` — all exist on disk (confirmed via the same
binary-file-error existence check as the clip above), matching CLAUDE.md
decision 4's three-clip set (`AUDIO_HAZARD_DETECTED`,
`AUDIO_BABY_GETTING_CLOSE`, `AUDIO_IMMEDIATE_DANGER`) by name.

**The drawing-order change** (`main()`, `risk_engine.py:1991`–`2017`).
Hazard/person boxes are drawn onto `annotated` first (`1994`–`1997`), the
buffer/clip capture point comes immediately after (`2006`–`2010`, with an
explicit comment citing CLAUDE.md decision 1: "boxes are product,
diagnostics are pixels"), and only *then* does the connector line and the
FPS/model/risk-readout diagnostic overlay get drawn, explicitly marked in a
comment as belonging only to the local `cv2.imshow` debug window
(`2015`–`2034`). This is a real, checkable ordering in the code, not an
assertion — I read the three drawing blocks in sequence and confirmed
`rolling_buffer.append(annotated, now)` (line 2006) runs strictly between
the box-drawing loop and the diagnostic-overlay calls.

## The live-test cycle: five rounds, two real bugs found and fixed twice each

This is the most substantial part of the session, and it's worth reading as
a single continuous debugging arc rather than five separate incidents,
because two of the five rounds are the *same* underlying failure fixed
twice — the first fix demonstrably didn't work, and the decision log says so
plainly rather than folding the correction into a clean success story.

**Round 1 (2026-08-26, first entry): "six alarms at once."** A 130s
recording (`cv/captures/Screen Recording 2026-08-26 at 19.26.36.mov` —
confirmed to exist on disk) showed hazard entry #15 dismissed and re-raised
four times in a few seconds, landing in the same window as a genuine
ORANGE→RED escalation. Root cause, as logged: `AlertManager.open_new_object`
had no memory across separate re-raises, so each one fired an independent,
unthrottled alert — the per-pair hysteresis in `AlertManager` was never the
gap, because a *new-object* alert isn't a `(person, hazard)` proximity pair
at all. **Fix: `AlertArbiter`** (`risk_engine.py:1304`–`1366`), a global
pacing gate sitting between `AlertManager` and the actual `speak()` call:
at most one alert voiced per `GLOBAL_ALERT_MIN_INTERVAL_SECONDS = 2.5`,
priority RED > getting-close > new-object, RED exempt from pacing entirely
and clearing anything held. I traced `AlertArbiter.offer()`'s logic by hand
against `test_alert_arbiter_repeated_same_hazard_bursts_collapse_to_one_
voiced` (the literal regression test modeling four re-raises in ~1.6s) and
confirmed the arithmetic collapses to exactly one voiced signal, matching
the assertion. Separately in the same round: the white review-candidate
outline was invisible against light backgrounds — fixed with the same
two-pass black-halo technique `draw_label` already used, confirmed present
at `risk_engine.py:1671`–`1672`.

**Round 2 (same day, follow-up): the outline fix wasn't the whole
picture.** Shaked reported boxes near the top of frame still had missing
borders. Two distinct, compounding geometric bugs, both confirmed directly
in the code: the review-outline rectangle's `-2/+2` offset could push part
of itself off-canvas near an edge (fixed by clamping to frame bounds,
`risk_engine.py:1669`–`1670`), and — separately — `cv2.putText`'s origin is
the text **baseline**, not a box corner, so a label meant to sit "above" a
box near `y=0` had its own glyphs entirely off-canvas even after clamping to
`y=0`. Fixed with `label_anchor_y()` (`risk_engine.py:1620`–`1631`), which
flips a label below its anchor instead of clamping to an invalid position.
I hand-traced three of its four unit tests against the actual arithmetic:
`label_anchor_y(100, 480, 8, 16)` → `100-8=92 >= 14` → returns `92` (matches
`test_label_anchor_y_places_label_above_when_room_exists`);
`label_anchor_y(5, 480, 8, 16)` → `5-8=-3 < 14` → `min(479, 5+16)=21`
(matches `test_label_anchor_y_flips_below_when_too_close_to_top`);
`label_anchor_y(2, 10, 8, 50)` → `min(9, 52)=9` (matches
`test_label_anchor_y_flipped_position_clamped_to_frame_bottom`). All three
traces match the test file's asserted values exactly, which is real
evidence the function does what the tests claim, independent of whether I
could execute the suite.

**Round 3 (follow-up #2): the same complaint, a different mechanism.**
Shaked: "we still have like the same thing i've marked as hazard or no
hazard showing up as a hazard detected 5 times every few seconds." This is
a *different* bug from Round 1's alert storm — the arbiter quieted the
*voiced* alert but did nothing about the underlying dismiss/re-raise
*rate*, which is the actual review burden being described. Three options
were presented with tradeoffs (require 2 consecutive changed scans; a grace
period post-dismiss; raise the sensitivity threshold itself) and Shaked
picked option A. **This fix did not work** — logged plainly as a failure in
the very next entry, not smoothed over.

**Round 4 (follow-up #3): the honest diagnosis of why Round 3 failed.**
"Requiring 2 consecutive scans of changed only helps if the noise is
occasional... this log shows it firing on very close to every scan." The
actual root cause: `fingerprint_changed()` was comparing the dismissed
entry's saved fingerprint crop against **that scan's own fresh,
independently-segmented candidate bbox** — and FastSAM's segmentation
boundary is not pixel-identical run to run even for a completely static
scene, so the comparison was measuring "we sampled slightly different
pixels" as "the scene changed." I confirmed this diagnosis directly against
the code: `HazardEntry` gained a `fingerprint_bbox` field
(`risk_engine.py:698`–`711`, with a docstring citing the exact bug), and
`apply_scan_candidates`'s dismissed-match branch was rewritten to pass
`match.fingerprint_bbox` (the stable bbox recorded at dismiss time) into
`fingerprint_changed()`, not `candidate` (that scan's fresh box) —
`risk_engine.py:883`–`888`, with an explicit comment naming this as the
actual fix. The reproduction method is worth noting on its own: a flat
color and a smooth gradient were tried first and found too shift-tolerant
after `region_change_frac`'s Gaussian blur to reproduce the bug at all — a
fine-striped synthetic pattern was needed to actually trip the old
candidate-bbox comparison on every 2–4px shift, matching the real live
symptom (near-every-scan re-raising, not occasional). I read
`_stripe_frame()` and both jitter-regression tests
(`test_apply_scan_candidates_dismissed_entry_box_jitter_alone_does_not_
false_trigger`, `..._still_reraises_on_real_change_with_jittering_boxes`)
and confirmed they exercise exactly this: six different pixel shifts that
must never re-raise, plus a genuine-change case (a solid frame swapped in)
that must still re-raise even while the candidate box also jitters.

**"Live test 5," as described in this task's briefing — a real gap in what
I could independently confirm.** The task narrative describes a fifth live
test: a pasted terminal log from `python risk_engine.py --name Arducam`
showing the previously-flapping entry's `change_frac` staying well under
threshold with zero false re-raises, a real RED escalation firing, and a
clip saved at `cv/clips/pending/20260826-204245_event8_object.mp4` —
claimed to have been independently checked by the orchestrator session with
`ffprobe` (h264, 1920x1080, 105 frames / 7.0s) and by extracting frames
(clean boxes, no diagnostic overlay baked in). **I want to be precise about
what I could and couldn't check here.** I confirmed the clip file physically
exists at that exact path (binary-file existence check via `Read`). I could
**not** independently re-verify the frame count, duration, resolution, or
codec claims, because I had no shell access this session to run `ffprobe` or
pull frames — a materially different evidentiary standing than Phase 4's
audit, which ran those checks itself. More importantly: **`docs/decision-
log.md`'s own Follow-up #3 entry — the most recent of the five Phase 5
entries, and the one that shipped the `fingerprint_bbox` fix — ends with
"Not yet live-verified against entry #13's actual object - that object is
the next thing to specifically re-test."** There is no sixth decision-log
entry recording that this re-test happened and succeeded. Per CLAUDE.md's
own decision-log discipline ("update it in the same turn a decision is
made, not retroactively"), a live-verification result this significant
(closing the loop on two failed fix attempts) should have its own logged
entry the same way every other live test this session did. Its absence is
either an oversight in this session's documentation discipline, or "Live
test 5" is something that happened in the orchestrator's own context after
this session's file-writing stopped — I can't tell which from the repo
alone, and I'm flagging the gap rather than guessing. **I am treating "Live
test 5" as reported, not independently confirmed**, and recommend the team
add the missing decision-log entry (or correct the record if it turns out
the re-test hasn't actually happened yet) in the same turn this write-up is
reviewed.

## Code audit against CLAUDE.md decisions 1, 4, 6

**Decision 1 (diagnostics are JSON, boxes are pixels).** Confirmed by
construction, not just by comment: the drawing-order change described above
means `RollingBuffer`/`ClipRecorder` physically cannot capture the connector
line or the FPS/model/risk-readout overlay, because those are drawn onto
`annotated` strictly *after* `rolling_buffer.append(annotated, now)` runs.
This also means Phase 5 has, as a side effect, already built the "clean
annotated frame" Phase 7's `/video_feed` needs — noted honestly in the code
as "one phase early, for free," which I can confirm structurally (the same
`annotated` array that feeds the buffer is the one with boxes and nothing
else) even though `/video_feed` itself doesn't exist yet.

**Decision 4 (alert behavior table, voice clips).** `audio_for_signal()` and
`banner_text_for_signal()` (`risk_engine.py:1369`–`1391`) implement the
three-voice-clip mapping and the eligibility table exactly: new-object
signals always map to `hazard_detected.wav`; RED maps to
`immediate_danger.wav`; yellow/orange map to `baby_getting_close.wav`;
`closed` signals never speak. I cross-checked the banner text format
directly against the decision log's quoted terminal output rather than just
reading the function in isolation: `banner_text_for_signal` for a
new-object signal produces `f"New object detected ({event.reason}):
{event.hazard_label}"`, and `main()` passes `"spot changed since dismissal"`
as `alert_reason` for a re-raise (`risk_engine.py:1970`) — concatenated,
this reproduces `"New object detected (spot changed since dismissal):
object"` **character for character** against the terminal log quoted in the
first 2026-08-26 decision-log entry. The same check holds for the proximity
format string against `"RISK ORANGE: object approaching (person #1)"`. This
is a genuinely useful cross-check: it means the decision log's quoted
terminal output is real output from this exact code, not a paraphrase or
something written up after the fact from memory.

**Decision 6 (5–7s clip, red-only, saved pending).** `ROLLING_BUFFER_SECONDS
= 5.0` + `CLIP_TAIL_SECONDS = 2.0` = 7s total, matching the "5-7s" spec.
`should_trigger_clip()` gates on `event.peak_zone != "red"` before anything
else, matching "only critical alerts trigger this — not every
yellow/orange event." Clips land at `cv/clips/pending/`, matching "saved
pending." Keep/discard and auto-delete are explicitly out of scope here and
named as Phase 6's job in the module docstring, matching the phase
boundary rather than silently building past it.

## Test suite: what I could verify, and the honest limit of that verification

`cv/test_risk_engine.py` currently contains **89 test functions** by direct
count (I read the whole file and enumerated every `def test_...`), matching
the decision log's own final count ("89/89 tests pass total," Follow-up #3).
I did **not** execute the suite this session — no Bash tool was available,
and both search tools (`Grep`, `Glob`) failed outright with `rg not found`,
which also ruled out even a lighter-weight sanity check. What I did instead:
read every test's assertions against the corresponding implementation code
and, for the specific new/changed logic this phase added (`label_anchor_y`,
`AlertArbiter`, the `fingerprint_bbox` jitter fix), hand-traced the
arithmetic by substituting the test's actual input values into the real
function body and confirming the result matches the asserted value (three
`label_anchor_y` cases shown above; the `AlertArbiter` priority-ranking and
RED-bypass logic against their respective tests). This is real verification,
not a rubber stamp — but it is a different, weaker claim than "I ran the
suite and it passed," and I want that distinction on the record the same
way Phase 4's write-up distinguished "unit-tested" from "watched happen on
real pixels." The next thing a human should do before fully trusting this
number: actually run `cd cv && python test_risk_engine.py` and confirm
"ALL TESTS PASSED (89 tests)" prints, since nobody in this specific
write-up's chain of verification has done that.

## Explicit open items

1. **Audio audibility was never independently confirmed, and this bears
   directly on Phase 5's own done-when bar.** `PHASE_PLAN.md` requires the
   alert to fire "visually **and audibly**." Every recording this session
   is a screen recording with an on-screen terminal log — none of it is
   evidence anyone actually *heard* a voice clip. `AudioPlayer.play()` is
   confirmed, by code reading, to launch `afplay` non-blockingly and to
   handle a missing file without crashing; that is a real but narrower claim
   than "the parent heard the alert." When asked directly, per this task's
   briefing, Shaked did not answer before asking to move to close-out. I am
   not aware of any evidence in the repo, this session or the prior one,
   that resolves this. Flagged precisely, not hidden inside "the alert
   pipeline works."
2. **`PersonTracker` ID churn, observed but not investigated.** The task
   briefing reports `person #1`, `#2`, and `#4` all appearing within a few
   seconds of one continuous RED escalation in the live-test-5 log. I did
   not independently see this log (no session transcript, and I can't
   re-read that specific terminal output myself), so I'm relaying the
   report rather than confirming it — but the mechanism it would implicate
   is real and checkable: `PERSON_STALE_SECONDS = 1.0`
   (`risk_engine.py:287`) drops a tracked person from `PersonTracker`'s
   state after one second unmatched, which would explain a genuine
   re-acquisition producing a new ID during any occlusion longer than that.
   Whether that's what happened, or the tracker is losing/reacquiring more
   aggressively than intended, is unresolved — a one-line forward pointer
   for whoever next works on Layer B, not a blocker for this phase.
3. **The missing sixth decision-log entry for "Live test 5,"** covered in
   detail above — worth restating here as its own open item rather than
   only as a footnote to the live-test narrative, since it's a process gap
   distinct from any engineering gap.

## Is the "fix didn't work, tried again" pattern a problem?

The task asked me to form my own view on this, not just restate the
framing it was handed in. Rounds 3 and 4 above are a real instance of a
fix not working and the team saying so plainly rather than quietly
replacing it with something that sounds like it always worked. I read this
as the same "measure, don't assume" discipline the project has shown at
least twice before (Phase 3's fine-tuning rounds, Phase 4's five-models
reset) working correctly here, not a process failure, for a specific
reason: **the failure was diagnosed by re-reading the code with the actual
symptom in mind, not by guessing at a second plausible cause.** Round 3
tried the mechanically obvious fix (make the trigger require repetition,
since the trigger was firing "too often") and it failed because the
symptom wasn't "occasional noise firing a false positive" but "the
comparison itself was structurally wrong" — a distinction that only became
visible once someone asked why a fix aimed at *frequency* didn't touch a
bug that turned out to be about *what was being compared*. That's a
narrower, more specific version of Entry 6 in `docs/agent-workflow-notes.md`
(isolated components composing badly) — here it's not five models
interacting, it's one function's implicit assumption (that a scan's own
candidate bbox is a stable reference) breaking only once the object being
tracked was visually irregular enough to expose FastSAM's own scan-to-scan
jitter. The fact that Round 4 explicitly built a new synthetic test pattern
(striped, not flat or gradient) *because* the first two patterns tried
weren't sensitive enough to reproduce the real bug is, to me, the strongest
evidence this was disciplined debugging rather than a lucky second guess —
reproducing the actual failure mode, not just writing a test that happens
to pass with the new code.

## Verdict: is Phase 5 closeable?

**Yes, closed as of this write-up, with the gaps above named rather than
rounded up.** Reading `PHASE_PLAN.md`'s own three-part gate against what
this session actually produced:

- **(1) demoably works** — met with real evidence, not just a report: two
  screen recordings confirmed to exist on disk
  (`cv/captures/Screen Recording 2026-08-26 at 19.26.36.mov` and
  `...19.58.26.mov`), a real saved clip file at the expected path matching
  the agreed naming convention, and terminal-log text quoted in the
  decision log that I cross-checked character-for-character against the
  actual format strings in the code — strong evidence the logged sessions
  are genuine runs of this exact code, not paraphrased or reconstructed
  after the fact.
- **(2) docs-agent reviewed it, code matches claim** — this write-up. Every
  constant, class, and drawing-order claim in the five decision-log entries
  was checked directly against the current `cv/risk_engine.py`, and all of
  it matched, with two named exceptions: I could not re-run the numeric
  measurements from the kickoff entry myself (no shell), and I could not
  execute the 89-test suite myself (no Bash tool this session — I hand-
  verified representative tests instead, a real but weaker form of
  verification).
- **(3) docs-agent produced the write-up** — this document.

Against Phase 5's own done-when line specifically — "a simulated critical
event produces a correct saved clip file and the right alert fires —
visually and audibly" — the **visual** half and the **saved-clip** half are
both backed by a real file on disk and code that structurally cannot
capture diagnostics into it (decision 1, enforced by construction). The
**audibly** half is the one piece of the phase's own stated bar that has
not been confirmed by anyone on the record, and I'm naming that precisely
rather than assuming `subprocess.Popen(["afplay", ...])` succeeding without
an exception is the same claim as a parent actually hearing the sound. This
mirrors exactly how Phase 4 closed with one specific, safety-critical path
(the unreviewed-hazard RED escalation) named as code-and-test-verified but
not yet watched on camera — a named, narrow gap with a clear next step
(literally: press play and listen), not a hidden one.

## For the write-up's teaching purpose: check your own understanding

1. `AlertManager` and `AlertArbiter` are two separate classes with two
   separate jobs. Explain, without re-reading the "What was built" section,
   what each one decides that the other doesn't — and why the alert-storm
   bug (Round 1) could not have been fixed by changing `AlertManager` alone.
2. `ALERT_HOLD_SECONDS` (2.0s, per-pair) and `GLOBAL_ALERT_MIN_INTERVAL_
   SECONDS` (2.5s, global) sound similar. What's the actual difference in
   what each one throttles, and why does RED bypass one of them but not
   need to bypass the other?
3. Rounds 3 and 4 both responded to the same live complaint ("still
   detected 5 times"). Explain why Round 3's fix (requiring 2 consecutive
   changed scans) was a reasonable first move, and specifically what
   evidence in Round 4's diagnosis proved it was solving the wrong layer of
   the problem.
4. CLAUDE.md decision 1 says diagnostics must never be drawn into a served
   frame. This phase enforces that for the rolling buffer/clips "by
   construction" rather than by convention. What does "by construction"
   mean here concretely — what would have to be true about the code for
   this claim to be false?
5. This write-up distinguishes "I read the code and the constants match
   what the decision log claims was measured" from "I re-ran the
   measurement myself." Why does that distinction matter for how much
   confidence a reader should place in the three Phase-5-kickoff
   measurements (buffer memory, non-blocking audio, alert cooldown)?
6. If you had to defend to Tom, in one sentence, why Phase 5 is marked
   closed despite nobody having confirmed a voice alert was actually heard,
   what's the one-sentence version of the distinction this write-up is
   drawing?
