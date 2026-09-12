# Session write-up — iPad video delay, the alert-sound feature (built, reverted, rebuilt), and the "hazard detected" bug's six-attempt saga (unsolved, deferred to Shaked)

**Dates covered:** 2026-09-11 to 2026-09-12, on top of commit `d58c2af`
("Phase 8: Flet UI (Console direction)...") on branch
`phase-8-ui-and-serving-hardening`.

**Extended 2026-09-12 (same date, later pass, docs-agent):** section 7 below
("The 'hazard detected' saga, continued") and its lead-in amendment at the
end of section 6 were added after the original version of this write-up
already existed and covered sections 1–6 and the (renumbered) desync
section. The rest of the document is otherwise unchanged from that original
pass; the "Named gaps," "Test evidence," and this intro were updated to stay
consistent with the addition.

**Status: not a phase close. Deliberately not filed as Phase 8 or Phase 9.**
Shaked's own framing, verbatim: "we made so much changes and mistakes and
didn't write it or update in anywhere... i don't know if it counts as phase
8 or 9 its pretty much neither." That's accurate, and I'm not going to force
it into a phase number it doesn't fit. It reads as Phase 8 hardening (the
video-delay and alert-sound work is squarely UI/serving) braided together
with what turned out to be six separate `cv/` bug fixes in one saga (not
two, as this write-up originally described before being extended) — all
Phase 4/5 debt (hazard-map occlusion handling) surfacing because Phase 8 is
the first time this system has been driven by someone standing in front of
the camera on a real wireless network for extended stretches. `PHASE_PLAN.md`'s
own Phase 8 status is updated in this same session (see its own entry) to
`[~]`, not `[x]` — real progress, real open items, not a close, and one of
those open items is now marked CRITICAL rather than routine.

## How this write-up was produced

Read/Grep/Glob/Write/Edit only, no Bash this session either. Every code claim
below was checked by reading the actual current file, not by trusting the
task briefing that prompted this write-up (which I was explicitly told not
to take on faith). Specifically, I read in full or in relevant part:
`cv/risk_engine.py` (the module docstring, `AlertDispatcher`,
`HazardMap.apply_scan_candidates`, `fingerprint_changed`, and the surrounding
tuning constants), `cv/test_risk_engine.py` (all tests touching occlusion,
person, fingerprint, and `AlertDispatcher.current_clip`), `ui/app.py` (the
video-quality flags, the PIN gate, the `alert_audio`/`AlertSoundPolicy`
wiring), `ui/alert_audio.py` in full, `ui/test_ui.py`'s alert-sound test
block, `ui/requirements.txt`, `backend/API.md`'s `alert` object
documentation, `SETUP.md` in full, `PHASE_PLAN.md`'s Phase 8 entry, and
`docs/decision-log.md`'s entire 2026-09-06 through 2026-09-12 run (the
"App Store Flet app," "first wireless test," "reverted the video throttle,"
"iPad delay root cause," "alert sound rebuilt," and "real bug found and
fixed" entries) to confirm what was and wasn't already logged before I added
anything.

Where a claim rests on something I did not personally re-run (a live iPad
test, a terminal log Shaked or the orchestrator watched happen) I say so
explicitly below rather than presenting it with the same weight as something
I verified by reading the code directly.

**For the section-7 extension (the "hazard detected" saga) added in this
same pass:** I additionally read, directly and in full or in the relevant
part, `cv/risk_engine.py`'s `AlertDispatcher` class end to end (the
collection-window mechanism — `_join_collection`/`_flush_collection`/
`_record_silently`), `person_occludes()`, `HazardMap.propose_named()` and
`HazardMap.apply_scan_candidates()` (specifically the person-gated retry at
line 1069 and the now-unconditional wide match inside `propose_named`), and
`main()`'s `occlusion_person_boxes`/`PersonTracker` wiring — to confirm the
mechanisms described in the six new decision-log entries genuinely exist in
the code as described, not just in the log's prose. I also read
`docs/decision-log.md`'s full run of six 2026-09-12 entries plus its
capstone entry, the new `docs/unsolved-gaps.md` in full, and `PHASE_PLAN.md`'s
Phase 8 section in full, specifically to check these documents against each
other and against the code for contradictions before extending this
write-up — findings from that cross-check are folded into section 7 and the
"Named gaps" list below, not hidden in a separate report.

## 1. The iPad video was 5–7 seconds behind real time

**What it looked like:** watching the live camera picture on an iPad over
WiFi, the picture was real but stale — motion from several seconds ago,
catching up in bursts rather than smoothly.

**The wrong turn first, corrected quickly and worth recording honestly.**
The first hypothesis was that this was a bandwidth problem — a weak WiFi
hotspot-class connection struggling to carry enough picture data per second.
Shaked corrected this directly: the delay was present and roughly constant
on his normal home WiFi router too, not proportional to how loaded the
network was, and the Mac browser on the very same network showed almost no
delay at all (loopback aside — see below for why that distinction turns out
to matter). That ruled out "the network is just slow" as the explanation.

**The real root cause, found by reading the actual library source rather
than guessing.** Flet (the Python UI framework this whole app is built on)
sends every update to a connected browser/tablet as a message over a
websocket. Reading `flet_web/fastapi/flet_app.py` (the file inside Flet's own
installed package, not this project's code) directly: every outbound message
goes into one plain Python `asyncio.Queue()` with **no size limit**, and
nothing ever removes an item from that queue except sending it — first in,
first out, always, no matter how old an item gets while it's waiting its
turn.

**In dummy's terms:** imagine a single-file line at a ticket window, and the
rule is "the line never skips anyone, no matter how long they've been
waiting." If new people join that line faster than the window can serve
them, the line just gets longer and longer — and the person currently being
served might have joined the line five minutes ago, even though ten newer
people have shown up since. The camera picture works the same way: every new
video frame our own code hands to Flet joins this line. Over a real WiFi
link (unlike a same-computer "loopback" connection, which has no real travel
time and basically never falls behind), if frames arrive faster than the
WiFi link can actually get them out the door, they queue up — and because
nothing is ever thrown away, every single one of those stale frames still
gets delivered and shown, in order, before the newest one gets its turn.
That queue depth *is* the 5–7 second delay.

**Why the first fix attempt (shrink the picture) barely helped, and why
that was actually useful information, not just a failed attempt.** Before
the real cause was known, the team tried making each picture smaller
(lower resolution, lower JPEG quality, lower frame rate — `WEB_VIDEO_MAX_
WIDTH`/`_QUALITY`/`_MAX_FPS` all turned down). This should help if the
problem is "too many bytes per second." It barely moved the needle. That's
actually a meaningful clue, read correctly in hindsight: if smaller pictures
barely help, the bottleneck probably isn't the *size* of what's being sent —
it's *how many separate turns in the line* are being taken, each with its
own overhead regardless of how big or small the item is. That's exactly what
the FIFO-queue explanation predicts, and once the team suspected the queue,
that earlier "barely helped" result stopped being a discouraging dead end
and became confirming evidence.

**The actual fix, and how it was proven, not just theorized.** Send fewer
pictures per second, not smaller ones — `--web-max-fps 2` (two pictures a
second instead of Flet's uncapped default). Decisive test, done live with
Shaked watching a real device: the synthetic test camera used for
development (`docs/design/phase-8/fake_camera_server.py`) already burns the
real wall-clock time into every frame's pixels, so pointing the iPad at it
lets you directly compare "what time does the picture say" against a real
clock. At `--web-max-fps 2`, the two matched — "identical," in Shaked's own
words comparing the iPad screen to his phone's clock side by side. Repeated
against the real camera with a hand-wave test: "wow, works pretty good!
almost no delay im satisfied with the result."

**What this fix does and doesn't cost — worth stating precisely, since it's
a safety product.** `--web-max-fps` only throttles the video picture. It
does **not** touch `RISK_STATUS_POLL_HZ` (the separate loop, running at
4 times/second, that drives the risk banner, zone color, and hazard counts),
and it does **not** touch the actual alert sound, which is triggered and
played locally by `cv/risk_engine.py`'s own detection loop on the Mac,
completely independent of whether anyone is even looking at a screen. So the
iPad's *picture* looks choppier at 2 frames/second, but a hazard alert on
that same iPad's own screen, and the Mac's own speaker, both still react at
full speed. Confirmed by reading the code (`RISK_STATUS_POLL_HZ` and the
video-fetching loop are two structurally separate `asyncio` tasks in
`ui/app.py`), not just asserted.

**Verification status:** live-verified on real hardware, twice (synthetic
clock test, then real camera + hand-wave), with Shaked watching. This is
solid, first-person, sensory evidence in the same tradition this project has
used before (e.g. "Shaked confirmed hearing it" at Phase 5/6 close) — a
specific claim against a specific test, not a vague "it worked."

## 2. A regression along the way, and a full revert to fix it

While chasing the delay above, an early fix tied the picture-quality
downscale to `page.web` — the reasoning being "any browser-based session
(as opposed to the native desktop window) probably has a weaker link than
loopback, so give it the lighter picture automatically." This shipped, and
broke something that had nothing to do with the iPad: `page.web` is `True`
for the **ordinary Mac browser session** (`http://127.0.0.1:8551`) exactly
as much as it is for the iPad — Flet can't tell "a browser on this same
computer" from "a browser on a phone three rooms away" from that one signal
alone. So the same downscale meant for a struggling wireless link also
silently hit the Mac browser, which had never had a bandwidth problem in the
first place. Shaked: "its not good, the delay is way too big and unefficient,
and the live feed on the web app is now too small."

**In dummy's terms:** the fix asked "is this a web browser?" when the
question that actually mattered was "is this connection slow?" — those
aren't the same question, and answering the wrong one degraded a session
that was never broken.

**The fix, and I confirmed it's genuinely what's in the code now, not just
what the decision log claims:** `page.web` was removed from this decision
entirely. Reading `ui/app.py` directly: `WEB_VIDEO_MAX_WIDTH = 0`,
`WEB_VIDEO_QUALITY = 70`, `WEB_VIDEO_MAX_FPS = 0.0` — all off/full-quality
by default, for every session, no exceptions. The `--web-max-width`/
`--web-quality`/`--web-max-fps` CLI flags remain as the **only** way to
reduce quality, opt-in per launch. There is no `page.web`-gated conditional
left anywhere touching video quality in the file — the only remaining
`page.web` reference is in a diagnostic print line (`f"web={bool(page.web)}"`)
that reports the value, never branches on it for quality. This matches
exactly what the task briefing described should be true, and I verified it
by reading the file rather than taking that description as given.

## 3. The alert-sound feature: built, fully reverted, then rebuilt properly

This is the piece of the session most worth being honest about in full,
because the dead end is part of the real history, not something to smooth
over.

**First attempt.** A screen (iPad, Mac browser, or the desktop window)
playing the same alert sound the Mac's own speaker plays was built and wired
into the live app directly, on an "is this possible" exploratory basis,
without first checking scope with Shaked. It worked — genuinely: 11 unit
tests plus a live end-to-end run showed exactly one sound per distinct alert
window, no repeats, no failures. **The problem was never a defect in what
was built — it was how it was built:** wired into the running app on
exploration rather than on an agreed plan. Shaked's instruction was explicit
and total: revert everything back to the prior state. Since no commit
existed at that point to revert to, this was done by hand, file by file, and
confirmed clean by a repo-wide grep plus an exact test-count match back to
the pre-feature baseline.

**Worth naming plainly, because it's an easy thing to misread from the
decision log alone:** the 2026-09-11/12 revert entries can read like "the
feature had a bug." It didn't. This was a process correction (build with
agreement first), not a quality correction.

**The rebuild, done properly this time.** Before writing any code, three
real product questions were asked and answered explicitly (via
`AskUserQuestion`, i.e. actually put to Shaked as decisions, not assumed):

1. **A real ~1.5-second gap right after unlocking a screen, where the sound
   isn't armed yet.** This is a genuine Flet library timing quirk, not a bug
   in this project's own code: calling `.play()` on a freshly-constructed
   `flet_audio.Audio` control immediately after creating it can time out,
   because the client hasn't finished registering the sound service yet.
   **Decision: accept it as a disclosed limitation, don't fight the library.**
   `ALERT_AUDIO_SETTLE_SECONDS = 1.5` in `ui/app.py` — the app waits this
   long after unlock before treating the sound as armed; a hazard that fires
   in that exact 1.5-second window is silently skipped *on that one screen
   only* (logged, not silently dropped from the terminal's perspective), with
   the Mac's own speaker completely unaffected either way.
2. **Should a cooldown apply even to two genuinely *different* alerts
   arriving close together, on top of "don't repeat the same alert"?**
   **Decision: yes** — a fast sequence of distinct hazards firing sounds
   back-to-back would itself feel like spam even though each one is
   individually a real, new alert.
3. **Alert sound doesn't play in Chrome/Edge on the Mac at all**, per Flet's
   own documentation (an autoplay-style restriction those browsers impose).
   **Decision: accept as-is** — the Mac's own speaker already covers that
   exact device regardless of what any browser tab does, so this costs
   nothing safety-relevant.

Also flagged, deliberately not built: iOS often mutes in-app sound
automatically when the iPad's physical silent switch is on. Not a code
problem to solve — a testing precondition to remember (switch must be off
to test this feature at all).

**What actually got built, verified directly against the current code:**

- `cv/risk_engine.py`'s `AlertDispatcher` gained `current_clip(now)`
  (`cv/risk_engine.py:1670-1683`), built by factoring a shared
  `_banner_active(now)` check out of the existing `banner(now)` method
  (`:1653-1668`) so the two can never disagree about whether an alert is
  currently showing. `build_risk_status()` gained an `alert_clip` parameter
  (`:2109`) that lands in `/risk_status`'s JSON as `alert.clip`
  (`:2217`) — confirmed matching `backend/API.md`'s documented example
  (`"clip": "immediate_danger.wav"`, `backend/API.md:131`) character for
  character.
- **In dummy's terms, what this field is for:** every screen viewing
  GuardianEye polls `/risk_status` on its own, independently. Before this
  change, that JSON told a screen *that* an alert was active and what text
  to show, but not *which* of the three pre-recorded sound files
  (`hazard_detected.wav` / `baby_getting_close.wav` / `immediate_danger.wav`)
  belongs to it. `alert.clip` answers that, from the exact same place
  (`current_clip()`) that already decides which sound the Mac's own speaker
  plays — so a remote screen and the Mac speaker can never disagree about
  which sound corresponds to which alert.
- `ui/alert_audio.py`'s `AlertSoundPolicy` (read in full) is deliberately
  **pure logic with no Flet import at all** — a dict in, a filename-or-`None`
  out, fully testable without a running app or real audio hardware. It does
  two separate things, confirmed by reading the code, not just the module
  docstring's description of it:
  - **Identity dedup**, keyed on `(text, clip)` together, not on "did
    `active` just become `True`." This distinction matters concretely: a
    second, genuinely different alert can arrive and refresh the on-screen
    banner while the first one's banner is technically still showing —
    `active` never dips back to `False` in between — so a check that only
    looks at the rising edge of `active` would silently miss the second, real
    alert.
  - **A 2.5-second cooldown** (`ALERT_SOUND_COOLDOWN_SECONDS = 2.5`),
    deliberately matching `cv/risk_engine.py`'s own
    `GLOBAL_ALERT_MIN_INTERVAL_SECONDS` rather than inventing a fresh number
    — reusing "this project's existing answer to how fast is too fast" for
    the Mac's own speaker, on the reasoning that it should be the right
    starting point for a screen's speaker too. A suppressed sound is **not**
    remembered as played — once the cooldown clears, whatever is active at
    that moment gets a real turn, so this delays rather than permanently
    drops a legitimate alert.
- The three `.wav` files are duplicated into `ui/assets/audio/` from
  `cv/audio/` rather than served over a new backend route — confirmed by
  `ui/requirements.txt`'s own comment and by `ui/app.py`'s `alert_audio =
  fa.Audio(src="audio/hazard_detected.wav", ...)` construction, a local
  asset path, not an HTTP URL back to the backend. This keeps the feature
  entirely client-side: no new server endpoint, no new privacy surface.

**A real, small inconsistency I found while verifying this, worth fixing
before commit even though I'm not allowed to fix code this session:**
`ui/requirements.txt`'s own top comment block (lines 45-49) still says
*"Deliberately NOT included: ... flet-audio: voice alerts are pre-recorded
clips played by `cv/risk_engine.py`. The UI does not play audio"* — directly
contradicted twenty-some lines further down in the same file, where
`flet-audio==0.86.5` is pinned with a full paragraph explaining exactly why
it *is* now included. This is a stale comment left over from before this
feature was built, not a functional bug (the actual pin is correct and
present), but it's the kind of thing that will confuse the next person who
reads only the top of that file. Flagging rather than fixing, per this
session's documentation-only scope.

**Test evidence, verified by reading the actual test file, not by trusting
the count claimed in conversation:** `ui/test_ui.py` contains **8**
`AlertSoundPolicy`-specific tests, not the "6 (up from 3)" the decision log
originally stated — I found and corrected that arithmetic in
`docs/decision-log.md` as part of this audit (see that file's 2026-09-12
entries). The 8 break down as 5 identity-dedup tests (sustained-alert
dedup, second-different-alert-after-cooldown, replay-after-going-quiet,
closed-signal-plays-nothing, missing-alert-object-doesn't-crash) plus 3 new
cooldown tests (fast-flurry-suppressed, suppressed-alert-gets-its-turn-once-
cooldown-clears, cooldown-clock-survives-a-brief-inactive-blip). The
project-wide aggregate the decision log states — 178 tests passing at that
point (110 cv, 26 ui, 26 server, 16 persistence) — is internally consistent
with the file-level count I verified for `ui` (26 `def test_` functions,
grep-confirmed).

**End-to-end verification, as reported (I did not watch this myself):**
against the synthetic camera harness, through the real unmodified `ui/app.py`,
7 consecutive `[alert_audio] played '...'` log lines landing roughly 12
seconds apart (matching one full zone-escalation cycle) with zero failures.
**Not yet confirmed on Shaked's actual iPad and Mac browser** — this is
explicitly named as an open item in the decision log itself, and I'm
repeating that here rather than letting it read as settled.

## 4. "Double sound" from two open windows — checked, and it's expected
behaviour, not a bug

Shaked reported hearing an alert sound twice — once from an open desktop
window, once from Chrome, at the same time. I checked this directly against
the code rather than accepting the "expected behaviour" explanation at face
value: `ui/app.py`'s `async def main(page: ft.Page)` is Flet's per-session
entry point — it runs once for every new connecting client (desktop window,
each browser tab, the iPad). `alert_sound_policy = AlertSoundPolicy()`
(`ui/app.py:177`) is constructed *inside* `main()`, not as a module-level
object shared across connections. There is no mechanism anywhere in
`ui/app.py` or `backend/server.py` that would let one session know another
session already played a sound. **In dummy's terms:** each open screen is
its own independent listener with its own private memory of "have I already
played this one" — nothing coordinates between them. Two screens open at
once, both hearing about the same real alert from `/risk_status`, will both
correctly decide to play it. That's not a bug to fix; it's the direct,
correct consequence of each screen being an independent client of the same
backend, which is this project's whole architecture (CLAUDE.md decision 1).
**Gap found while checking this:** neither `docs/decision-log.md` nor
`SETUP.md` said this anywhere before this audit — `SETUP.md`'s alert-sound
section explains the settle window and the cooldown but says nothing about
multiple open screens. Added to the decision log this session (see its
2026-09-12 entry); **not** added to `SETUP.md`, since editing that file
wasn't part of what this documentation pass was asked to produce — flagged
here as a concrete, low-effort follow-up.

## 5 & 6. Two real bugs found in `cv/risk_engine.py`: a person's own body
being mistaken for "the hazard changed or disappeared"

Both bugs share one root cause category: the periodic room scan (CLAUDE.md
decision 3 — every few seconds, list what's occupying reachable surfaces,
compare against what's already known) is deliberately built to ignore
anything a person is standing on or in front of, **so it never proposes a
person's own body as a hazard.** That's correct and necessary. What wasn't
handled correctly is what that suppression does to hazards the system
*already knows about* when a person happens to be standing near them — which
is exactly what a toddler approaching a hazard looks like.

### Bug A — an existing, unreviewed hazard being deleted and "rediscovered"
as brand new

**In plain terms:** picture an object sitting on a low table, not yet
reviewed by a parent. A child walks up to it (the thing the whole proximity
alarm exists to detect). The room-scan camera looks at that spot, sees the
child's body covering it, and — because the scan is built to ignore
anything a person overlaps — reports nothing there. Before this fix, "we
looked and saw nothing" and "the object is genuinely gone" were
indistinguishable. Two scans of that (about 10 seconds) deleted the hazard
entirely. Once the child stepped back and the spot became visible again, the
system saw it as showing up for the very first time — and CLAUDE.md decision
4 says a brand-new object alerts on arrival *as well as* on approach. Net
effect, exactly matching what Shaked reported: "no new hazards... it alarmed
baby getting close AND ALSO new hazard detected for no reason" — a real,
correct proximity alert, PLUS a fake "new hazard" alert for an object that
never actually moved.

**The fix**, confirmed by reading `HazardMap.apply_scan_candidates()`
directly (`cv/risk_engine.py:1004-1037`): before counting an unmatched
entry as one scan closer to being deleted, check whether any currently
tracked person's box overlaps *that entry's own bounding box* by more than
`PERSON_OVERLAP_THRESHOLD` (0.3 — the exact same constant and the exact same
comparison function, `overlap_frac_of_a`, that the upstream person-
suppression already uses, imported from `measure_segmentation.py`, so the
two mechanisms can never disagree about what counts as "a person is
covering this"). If a person is covering it, this scan cycle counts as *no
evidence at all* — the absence counter isn't touched, not even capped. This
protection applies to every entry regardless of whether a parent has already
judged it (confirmed or dismissed), not just unreviewed ones — reading the
code, this check runs before, not instead of, the existing
`JUDGED_ABSENT_SCANS_REQUIRED` protection from an earlier fix in this same
session's family of bugs.

**Test evidence, read directly:** `cv/test_risk_engine.py` has three
dedicated tests —
`test_a_person_standing_on_an_entrys_spot_does_not_count_as_absence`,
`test_person_occlusion_protects_judged_entries_too_not_just_pending_ones`,
and `test_a_person_elsewhere_in_frame_does_not_protect_an_unrelated_entry`
(the last one specifically checking the fix is *location-specific* — a
person standing anywhere in frame must not give every unrelated hazard a
free pass, only the one they're actually near). All three read as real
regression tests exercising the actual bug's shape, not restatements of the
fix.

**Confirmed logged:** yes — `docs/decision-log.md`'s 2026-09-12 "Real bug
found and fixed" entry already existed before this audit and matches the
code precisely.

### Bug B — the same family, one code path deeper: a dismissed hazard
re-raising itself because a person's body changed the "fingerprint" pixels

**In plain terms, and why this is genuinely a separate bug from Bug A, not
a duplicate of it.** When a parent dismisses a hazard ("not a hazard"), the
system remembers what that exact spot looked like at that moment — a small
saved crop of pixels, its "fingerprint." Every later scan re-crops that same
fixed region and compares it, pixel by pixel, against the saved fingerprint.
If it looks meaningfully different for two scans in a row, the system
assumes something real changed there and puts it back in front of the
parent for review (Shaked's deliberate "better safe than sorry" rule from
earlier in the project). The bug: that pixel comparison has no way to know
*why* the pixels changed. A child reaching toward, or standing in front of,
an already-dismissed object genuinely does change those pixels (skin instead
of whatever was there before) — and the comparison, doing exactly its job,
correctly reports "this looks different now." Two scans of a child's arm
being in that spot, and a dismissed hazard re-raises itself for no real
reason, firing a fresh "new hazard detected" alongside the real "baby
getting close" alert.

**Why Bug A's fix doesn't already cover this.** Bug A protects the
*removal* path — a candidate that never showed up at all this scan, because
a person covered the whole segment. Bug B's code path is different: the
candidate box for the dismissed spot *does* still show up and match (the
object's own outline survived the upstream person-suppression), so the code
proceeds to the fingerprint-comparison branch — a completely separate
region of pixels (the fixed fingerprint crop) than the segmentation
candidate box being matched. A person can overlap the fingerprint region
without ever showing up as "covering the segment" to the upstream check,
because they aren't the same box. Confirmed directly in code
(`cv/risk_engine.py:951-1002`, inside the `if match.state ==
HAZARD_STATE_DISMISSED:` branch) and in the code's own comment, added at
the same time as the fix: *"This is a DIFFERENT code path from the
arrived/removed occlusion fix above... and was not covered by it."*

**The fix:** the same shared mechanism reused again — before running
`fingerprint_changed()`, check whether any tracked person overlaps the
entry's `fingerprint_bbox` above `PERSON_OVERLAP_THRESHOLD`. If so, this
scan counts as no evidence either way — the `changed_scans` counter is left
exactly where it was, neither bumped (which would eventually mistrigger a
re-raise) nor reset to zero (which would itself be a false "confirmed
unchanged" signal).

**Test evidence, read directly:** one dedicated test,
`test_a_person_over_the_fingerprint_spot_does_not_false_reraise_a_dismissed_entry`
(`cv/test_risk_engine.py:670-714`). Its own docstring calls it out
explicitly as "a SECOND bug in the same family" and "a DIFFERENT code
path... not covered by" the first fix — the team's own record of having
correctly recognized these as two distinct fixes, not one. The test also
confirms the fix doesn't weaken the real re-raise mechanism: with the person
removed from frame, the same genuinely-changed spot still correctly
re-raises after two consecutive changed scans.

**Not logged before this audit — now fixed.** This is the specific gap the
task that produced this write-up asked me to check for by name, and it was
real: `docs/decision-log.md` had an entry for Bug A but none for Bug B, even
though the code comments for both fixes explicitly say they were "discovered
live 2026-09-12" in the same investigation. I added the missing entry this
session (see `docs/decision-log.md`, 2026-09-12, "A second bug in the same
family"). Grep-counting `cv/test_risk_engine.py` directly: 114 `def test_`
functions currently exist, one more than the 113 the last-logged entry
stated — consistent with exactly Bug B's one new test having landed after
that entry was written and never being reflected anywhere. **I did not run
the suite this session (no Bash access)**, so 114 is a function-definition
count, not an independently confirmed pass count; stated with that caveat
in the decision log too.

### Neither Bug A nor Bug B has been watched happen on the real camera yet — true when this was first written, and superseded almost immediately

Both fixes are backed by a faithful, targeted reproduction (fixture values
in the tests are literally "a child standing in front of a low-table spot,"
not an abstract edge case) and by regression tests that pin the exact
failure shape described live by Shaked. That is real, meaningful evidence.
It is **not** the same as watching it happen on the actual camera feed —
placing an unreviewed object, walking up to it, and confirming on-screen
that nothing spurious fires; and separately, dismissing an object, then
reaching toward it, and confirming it stays dismissed.

**Amendment, added while extending this write-up (2026-09-12), not a
rewrite of the paragraph above — it was accurate the moment it was
written.** Shaked did exactly this camera test, immediately, several more
times the same session. Bugs A and B themselves held up fine under a real
child walking into frame — but the *reported symptom* they were fixed to
stop ("hazard detected" sounding alongside a legitimate proximity alert)
kept happening anyway, for four more distinct, real reasons that had
nothing to do with either fix being wrong. See the new section below for
the full account. This is exactly why this write-up says "not the same
claim as verified live" a few lines up — the gap between "the specific
mechanism I fixed works" and "the symptom the user reported is gone" turned
out to be a real, six-round gap, not a rhetorical one.

## 7. The "hazard detected" saga, continued: five more fix attempts, five
more times the symptom came back, and a final decision to stop and defer

**This is the single most important section of this write-up, and the
reason docs-agent was called in specifically to make sure it is written
down accurately.** Sections 5 and 6 above (Bug A and Bug B) were the
**first** of six real, distinct, verified fixes made in this one session
chasing one reported symptom: walking into frame produces
`hazard_detected.wav` playing alongside a legitimate `baby_getting_close.wav`
/ `immediate_danger.wav`, for no real new hazard. Each of the six fixes
below was correct for the evidence available at the time it was made. Each
one was then disproven, or shown to be incomplete, by the *next* live test
Shaked ran. **Shaked has since marked this CRITICAL and UNSOLVED, and is
taking it on directly rather than continuing to iterate blind in this
session.** This section, and the new `docs/unsolved-gaps.md` tracker file,
exist specifically so that fact cannot be lost or quietly re-litigated as
if it were a fresh problem next time someone picks it up.

**Reading order matters here, and it wasn't quite what a simplified retelling
suggests.** Verified directly against `docs/decision-log.md`'s own entry
titles (which explicitly number "Third live test," "Fourth pass," "Fifth
pass," "Sixth pass") and against the current code in `cv/risk_engine.py`,
the six real fix-and-fail cycles run in this order:

**Attempt 1 — Bugs A and B (sections 5 and 6 above).** Found and fixed
together, in the same live investigation, before any of what follows. Both
reused the file's existing, deliberately lenient `PERSON_OVERLAP_THRESHOLD`
(0.3) — the same number the room scan already uses to decide whether to even
*suggest* a candidate object to a parent. That number is tuned for "wrong
costs one tap," not for "wrong costs an unwanted sound," which turns out to
matter a lot below.

**Attempt 2 — two different alerts audibly overlapping, then a full
redesign the same day.** Shaked reported hearing two sounds essentially at
once ("baby getting close" + "immediate danger" together). Root cause,
confirmed by reading `AlertArbiter.offer()`: a RED (critical) signal is
*deliberately* allowed to bypass the normal pacing window at any time
(so real danger is never delayed) — but it had no way to notice a
lower-priority signal had been approved and spoken a split second earlier,
so both got voiced back to back. First fix: a short 0.4s "settle window"
(`ALERT_SPEAK_SETTLE_SECONDS`) that lets a RED signal cancel a still-pending
lower one. **Live-retested the same day and found genuinely incomplete**:
that fix only helps when RED is one of the two competing signals — "baby
getting close" and "hazard detected" are BOTH non-RED, so neither could
cancel the other under that mechanism, and both still got voiced a few
seconds apart. **Redesigned, same day:** collection now happens *before*
either signal ever reaches the pacing arbiter at all — every non-RED signal
joins a short shared window first, and only the single highest-priority
signal in that window is ever submitted onward, so two competing non-RED
alerts can no longer both get a turn. In the same investigation, a
dedicated backstop (`new_object_alert_suppressed_by_person()`) was added at
the one single place every "new object" alert of any origin has to pass
through, checking a live person's overlap against the exact hazard entry
being announced — described in the decision log at the time as "the real
root cause... found and fixed."

**In dummy's terms, attempt 2:** imagine two people trying to shout
different warnings at you within the same second. The first fix taught the
*more urgent* shouter to shove the other one aside — fine if one of them is
actually the urgent one, useless if neither is. The redesign instead makes
both shouters whisper to a referee first, and only the referee's pick
actually gets shouted at you.

**Attempt 3 (decision log's own "Third live test") — the same number was
doing two incompatible jobs.** Shaked reported the exact same symptom yet
again. Rather than patch again, every person-occlusion check in the file
was re-read end to end, and a real structural gap turned up: all three
occlusion guards built so far (Bugs A and B, plus attempt 2's backstop) were
still reusing that same lenient 0.3 `PERSON_OVERLAP_THRESHOLD` — a bar
correctly tuned for "should we even suggest this to a parent" but far too
loose for "should we interrupt with a sound." A child's own outstretched
arm or foot regularly cleared 30% overlap with a tight YOLO person box just
barely, which is a "close enough, flag it" result for a review-queue
suggestion and a "not really occluded, sound the alarm" result for an alert
decision — the same number, asked to do two jobs with two very different
acceptable error rates. **Fix: a new, separate, stricter, padded check,**
`person_occludes()` — 0.05 overlap threshold (much less overlap needed to
count) plus a 30%-of-box padding margin around every person box (so a limb
just outside YOLO's tight edge still counts) — and it now replaces all three
of the earlier inline checks, verified directly in the code
(`cv/risk_engine.py:896-913`). Confirmed by a test built to isolate exactly
this scenario: a stray limb that shows **zero** overlap against the raw,
unpadded person box, correctly caught once padding is added.

**Attempt 4 (decision log's "Fourth live test") — the real problem was
never the threshold; it was that a person can simply be missing from a
frame.** Shaked tested again, same symptom, and asked a sharper question
this time: is there *any* scenario where every occlusion check, however
carefully tuned, still has nothing to compare against? Yes — YOLO does not
detect a person on literally every single frame; a child mid-stride or at
an odd angle can simply miss the confidence threshold for one frame, and
every occlusion check up to this point read only *that exact frame's* raw
detections. When that one frame comes up empty, no threshold and no padding
in the world can flag "a person is here," because the list handed to the
check has nothing in it. **Fix, confirmed directly in `main()`
(`cv/risk_engine.py:2865-2866`):** every occlusion check now reads
`PersonTracker.entries` — a rolling ~1-second memory of recently seen
people — instead of the current frame's raw list. A person detected up to a
second ago still counts, even if the exact frame a scan or dismissal check
happened to run on missed them. The proximity/risk-scoring path
(`score_frame`) is untouched by this — it was already reading through
`PersonTracker` from an earlier phase, which is part of why "baby getting
close" kept firing correctly throughout this whole saga while the
occlusion checks kept failing: the two systems had quietly been built on
two different strengths of person data the entire time.

**In dummy's terms, attempts 3 and 4 together:** attempt 3 fixed "the ruler
we're using to measure 'is a person here' was the wrong ruler for this job."
Attempt 4 fixed a different problem entirely: "sometimes there's nothing to
measure with any ruler, because the camera simply blinked for one frame."
Both are real, and neither one would have caught the other's mistake.

**Attempt 5 (decision log's "Fifth pass") — the first time Shaked sent
actual terminal output instead of a description, and it named a completely
different bug.** The real log showed a refrigerator — a fixed, geometry-
matched appliance entry that should never re-announce itself mid-session —
duplicating as a fresh "new hazard." Reading `HazardMap.propose_named()`
directly showed the real cause: it has **zero** tolerance for a detected
box drifting even slightly between frames, and an ordinary partial
occlusion (a person walking in front of the fridge) can shift or shrink
YOLO's box for it just enough to fail the match, creating a second, "new"
entry for the same physical fridge. This is a matching-tolerance bug, not a
person-occlusion-threshold bug — none of attempts 1–4 touch matching at
all. **Fix:** a second, wider matching-tolerance pair
(`OCCLUDED_MATCH_IOU_THRESHOLD` / `OCCLUDED_MATCH_CENTER_DIST_FRAC`), used
as a retry, *only* when `person_occludes()` confirms a person is genuinely
right there — deliberately narrower than "just loosen matching everywhere,"
to avoid accidentally merging two real, separate objects that happen to sit
near each other with no person involved.

**Attempt 6 (decision log's "Sixth pass") — a second terminal log
disproved attempt 5's own core assumption within the same session.** Shaked
tested again immediately and sent another real log: the *same* refrigerator
duplicated **twice**, within the first couple of seconds of the program
starting — before a single video frame had even been pushed to the stream.
Nobody can have been occluding the fridge in the first one or two seconds
of a cold start. This disproves attempt 5's premise outright: it is
ordinary YOLO detection jitter between frames, full stop, with no person
involved at all. **Fix, confirmed directly in the code
(`cv/risk_engine.py:957-995`):** `propose_named()` now uses the wider
matching tolerance *unconditionally*, with no person check and no two-step
retry — the `person_boxes` parameter and the person-gated retry from attempt
5 are gone from this function entirely; verified by reading it directly,
neither call site in `main()` passes person boxes to it any more.
Reasoning for why unconditional widening is safe *specifically here* and
not a general "just loosen everything" move: this label set (oven,
microwave, refrigerator, wall socket) is small, curated, and each label
names something large, static, and effectively one-of-a-kind in a real home
— a duplicate alert for a fridge that's been on record since second one is
judged a worse failure than the (still bounded, not unlimited) risk of two
separate same-labeled appliances merging. The room-scan's own matching
(arbitrary, potentially numerous small objects — a different risk profile)
deliberately keeps attempt 5's narrower, person-gated retry, unchanged —
verified directly at `cv/risk_engine.py:1069`, which still calls
`person_occludes()` before retrying with the wider thresholds.

**In dummy's terms, attempts 5 and 6:** attempt 5 assumed "if a box moved
between two look-and-compare passes, a person probably explains it" — a
reasonable, testable guess. Attempt 6's evidence (the fridge duplicating
before anyone could have walked anywhere) showed that guess was simply
wrong for this specific object type: ordinary camera/detector noise alone
is enough to make a static fridge's box wobble slightly frame to frame, so
tolerating that wobble has to be unconditional for a small set of things
that are never going to actually move.

**Where this was left, in Shaked's own words as recorded in the decision
log: stopped, marked CRITICAL, and handed to himself directly** rather than
continuing to iterate blind in this session. After six real, verified
fixes, the exact original symptom — "I walked into frame and got two
alerts" — has not been confirmed gone by a live test run *after* the sixth
fix landed. Every one of the six fixes addressed exactly the mechanism the
most recent evidence pointed to, and no more; there may be a seventh
mechanism nobody has surfaced yet. **The standing advice for whoever picks
this up next, taken directly from the decision log's own retrospective on
why this took six rounds: get real terminal output from the exact failing
moment before trying anything else** — reasoning from the symptom alone
produced attempts 1–4; only real logs (attempts 5 and 6) found the bugs
those four attempts, however careful, had no way to see. Full details:
`docs/decision-log.md`'s six 2026-09-12 entries (search "hazard detected"
or "person_occludes"), and the new `docs/unsolved-gaps.md`, which is now
the single place to check "is this still broken" without re-reading the
whole decision log.

**Also done this session, unrelated to fixing the bug itself, but worth
recording precisely: the runtime database and saved test clips were wiped
clean at Shaked's request**, so the next test starts against a genuinely
empty system rather than a session's worth of test data. Confirmed against
the actual `.gitignore`: `backend/guardianeye.db` (791 event rows, 35 clip
rows) and every file under `cv/clips/pending/` (12 `.mp4` files) were
deleted — neither path was ever tracked by git, so nothing committed was at
risk. **`cv/captures/`** — the separate folder of screen-recording captures
kept for research/fine-tuning — **was explicitly left untouched**; it is a
different thing entirely and was never in scope for this cleanup.

**One more open question, raised by Shaked but not yet confirmed either
way: do the "first scan" (pending, drawn in blue/cyan) hazard-map boxes
really disappear across a session restart, including an actual change of
physical location?** CLAUDE.md decision 3 is explicit that a session must
start with a fresh, empty map every time, since a camera's room/angle/
lighting can't be assumed identical to the last session. Investigated so
far by reading the code only, not yet tested live with a real location
change: `HazardMap()` is constructed fresh inside `main()` on every run,
nothing anywhere reads the SQLite database back into it at startup, and
`--seed-hazard` (the only startup-time hazard injector that exists) is off
by default and, even if it were on, seeds entries as CONFIRMED (red), not
PENDING_FIRST_SCAN (blue), so it could not be the source either way. The
working conclusion — genuinely not yet verified — is that nothing in the
code should persist, and that the same blue boxes reappearing after a
restart is more likely explained by testing happening in the same physical
room each time (the same real objects get freshly re-detected in the same
place) than by an actual persistence bug. This needs a real test in a
different location before either conclusion can be trusted. Tracked as
item #2 in `docs/unsolved-gaps.md`, filed as "next to check," not claimed
as fixed or as confirmed broken.

## 8. Open, unfixed: the iPad and Mac browser show "totally
unsynchronized" alerts

Shaked's report: watching the iPad and a Mac browser side by side, "they
shout completely different alerts." **This is not fixed, and it should not
be read as fixed by anything else in this session.**

**What was checked, and what was ruled out.** `--web-max-fps` — the flag
that fixed the video delay — only throttles `MjpegClient`, the component
that fetches and displays camera pictures. `RISK_STATUS_POLL_HZ = 4.0`, the
separate loop that drives the banner text, zone color, and (as of this
session) which sound to play, runs independently and is untouched by that
flag. Confirmed directly by reading `ui/app.py`: these are two structurally
separate `asyncio` tasks, not two settings on the same one.

**The working hypothesis — plausible, not measured.** The same underlying
Flet transport property that caused the video delay (an unbounded outbound
queue that never drops a stale message, confirmed against Flet's own
library source earlier this session — see section 1 above) could equally be
backing up the risk-status-driven updates a real WiFi client sends and
receives, independent of the video path entirely. If true, that would mean
one screen is reacting to a meaningfully older snapshot of `/risk_status`
than another screen, at any given instant — which would look exactly like
"they shout completely different alerts" if the room's actual state changed
in between.

**What has NOT been done, and should be named precisely rather than
implied:** nobody has instrumented two simultaneous clients to actually
measure which one (or both) is stale, by how much, or confirmed this is the
mechanism at all — a much less interesting explanation (ordinary jitter
between two independently-scheduled 4-times-a-second poll loops, one on each
device) would produce a broadly similar-looking symptom. The fix that
worked for video (send fewer, not smaller, updates) has no confirmed
equivalent here yet, and nothing should be assumed to carry over without
separately measuring it. I added this as an explicit, dated open item to
`docs/decision-log.md` (2026-09-12) so it isn't lost between sessions, and
`PHASE_PLAN.md`'s Phase 8 status (updated this session, see that file) names
it as a reason Phase 8 is not being closed.

## Test evidence, aggregated and caveated honestly

- `cv/test_risk_engine.py`: 130 `def test_` functions as of the sixth pass
  of the "hazard detected" saga (section 7) — grep-counted directly against
  the current file this session, matching the decision log's own final
  running count for this session.
- `ui/test_ui.py`: 26 `def test_` functions, grep-counted directly,
  matching every count claimed in the decision log across this session.
- `backend/`: not touched this session (only `backend/API.md`, a docs file,
  changed) — the last-known 26 `test_server.py` / 16 `test_persistence.py`
  counts stand, unverified again by me since nothing in that code moved.
- **None of these counts were re-run by me this session — no Bash access.**
  This is a function-definition count, a floor on "at least this many tests
  exist," not a pass/fail confirmation. Worth an actual
  `cd cv && python test_risk_engine.py` (and the `ui`/`backend` equivalents)
  before trusting any of these numbers fully — the same caveat this
  project's write-ups have attached before when Bash access wasn't
  available (see Phase 5's write-up).
- **A separate, minor accuracy note found while auditing the decision log's
  own running test-count trail across the six "hazard detected" fixes**:
  summing the increments each entry states by hand (117 after the settle-
  window fix, +5 across the collection-window/backstop fix, +4 for
  `person_occludes`) lands on 126 *before* the PersonTracker-memory fix's own
  one new test is added — yet that entry's own text states the total as
  "126," the same number, not 127. This reads as a genuine off-by-one in the
  decision log's arithmetic, not a claim I could resolve without actually
  running the suite (which I could not do this session). Flagged in
  `docs/decision-log.md` itself at the relevant line, in the same
  self-correcting style the project already uses elsewhere, rather than
  silently changed.

## Named gaps, in priority order

1. **CRITICAL — the "hazard detected" bug (section 7) is unsolved after six
   real fix attempts and is now explicitly deferred to Shaked.** This is the
   top item, full stop — not because it's the newest finding, but because
   Shaked himself marked it this way and stopped iterating on it in this
   session. See `docs/unsolved-gaps.md` for the live-tracked status going
   forward.
2. **The iPad/Mac alert desync (section 8) is open, real, and unmeasured.**
   For a child-safety product, two screens disagreeing about what's
   currently happening is a genuinely worse failure mode than one screen
   being uniformly slow.
3. **Whether PENDING_FIRST_SCAN ("blue") hazard boxes persist across a real
   location change (section 7's closing item) is unverified** — code
   reading says they shouldn't, but nobody has tested it with an actual
   change of physical location yet. Filed as `docs/unsolved-gaps.md` item 2.
4. **`ui/requirements.txt` contains a stale, self-contradicting comment**
   (says flet-audio is "deliberately NOT included" directly above where it
   *is* pinned with a full justification). Not a functional bug — the pin
   itself is correct — but confusing for the next reader. Not fixed this
   session (documentation-only scope; the file is under `ui/`).
5. **The decision log itself had a few accuracy gaps this audit found and
   corrected in place**: a missing entry for Bug B (added earlier this
   session), an arithmetic error in the alert-sound-rebuild entry's own test
   count ("6 total, up from 3" corrected to "8 total, up from 5"), the
   off-by-one test-count note above, and a duplicated restatement of the
   `GLOBAL_ALERT_MIN_INTERVAL_SECONDS` 5.0→3.5 change appearing in two
   consecutive entries as if it happened twice.
6. **"Double sound" from multiple open screens (section 4) is explained and
   logged in the decision log, but not yet added to `SETUP.md`**, where a
   future tester is more likely to actually look for it. Small, low-effort
   follow-up.

None of these six are fixed by this write-up itself — this was a
documentation-only pass, per the task that produced it. What's above is the
accurate current state, not a plan executed.

## For the write-up's teaching purpose: check your own understanding

1. Explain, without re-reading section 1 above, why making the video
   pictures *smaller* barely helped the iPad delay, and why that specific,
   unhelpful result was actually a useful clue rather than just a failed
   attempt. What does the "ticket-line" analogy get right, and where would
   you extend it to explain why `--web-max-fps` (fewer pictures) worked when
   shrinking each picture didn't?
2. `alert.clip` was added to `/risk_status`'s JSON rather than each screen
   deciding for itself which sound file matches a given zone/alert text.
   Explain the concrete failure mode this design choice avoids, and connect
   it to CLAUDE.md decision 1's general rule about where a decision is
   allowed to get made once versus decided independently by every consumer.
3. Bug A and Bug B are both described as "the same family" but genuinely
   different bugs, not a duplicate finding. In your own words, explain what
   two DIFFERENT boxes each fix is comparing a person's position against,
   and why a person could overlap one of those boxes without overlapping the
   other.
4. The alert-sound feature was built once, fully reverted, then rebuilt to
   an almost identical design. Explain, without saying "it had a bug,"
   what was actually wrong the first time, and why that distinction (process
   vs. defect) matters for how the team should read the revert in hindsight.
5. Section 8's open item is described as "not measured," not "not fixed."
   Explain the difference between those two claims in your own words, and
   name one concrete experiment (not necessarily the one this write-up
   proposes) that would move it from "plausible hypothesis" to "confirmed
   or ruled out."
6. Section 7 names six fix attempts. Without re-reading it, list which
   attempt fixed a person-detection *data* problem (a person missing from a
   single frame) versus which attempt fixed a person-detection *threshold*
   problem (the right person was detected, but the pass/fail bar for
   "occluded" was wrong for the job) — and explain in your own words why
   fixing the threshold first, before realizing the data problem existed,
   still felt like "the same bug, still broken" to whoever was testing it.
7. Attempt 5 and attempt 6 both concern the same refrigerator, and attempt 6
   directly disproves an assumption attempt 5 was built on. State that
   assumption in one sentence, state the exact piece of evidence that broke
   it, and explain why "add a person-gate to a fix" and "make a fix
   unconditional" are opposite responses to superficially similar-looking
   evidence — and why the room-scan's own matching logic deliberately did
   NOT get the same "make it unconditional" treatment attempt 6 gave to
   named-appliance matching.
8. This saga is explicitly marked "deferred to Shaked," not "fixed." If an
   advisor asked "so is the hazard-alert system reliable or not," what is
   the honest one-paragraph answer, and what is the one piece of evidence
   (not a vibe) that would change that answer from "unsolved" to "solved"?
