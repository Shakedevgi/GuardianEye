# Session write-up — iPad video delay, the alert-sound feature (built, reverted, rebuilt), and two real occlusion bugs in the hazard map

**Dates covered:** 2026-09-11 to 2026-09-12, on top of commit `d58c2af`
("Phase 8: Flet UI (Console direction)...") on branch
`phase-8-ui-and-serving-hardening`.

**Status: not a phase close. Deliberately not filed as Phase 8 or Phase 9.**
Shaked's own framing, verbatim: "we made so much changes and mistakes and
didn't write it or update in anywhere... i don't know if it counts as phase
8 or 9 its pretty much neither." That's accurate, and I'm not going to force
it into a phase number it doesn't fit. It reads as Phase 8 hardening (the
video-delay and alert-sound work is squarely UI/serving) braided together
with two `cv/` bug fixes that are really Phase 4/5 debt (hazard-map
occlusion handling) surfacing because Phase 8 is the first time this system
has been driven by someone standing in front of the camera on a real
wireless network for extended stretches. `PHASE_PLAN.md`'s own Phase 8
status is updated in this same session (see its own entry) to `[~]`, not
`[x]` — real progress, real open items, not a close.

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

### Neither Bug A nor Bug B has been watched happen on the real camera yet

Both fixes are backed by a faithful, targeted reproduction (fixture values
in the tests are literally "a child standing in front of a low-table spot,"
not an abstract edge case) and by regression tests that pin the exact
failure shape described live by Shaked. That is real, meaningful evidence.
It is **not** the same as watching it happen on the actual camera feed —
placing an unreviewed object, walking up to it, and confirming on-screen
that nothing spurious fires; and separately, dismissing an object, then
reaching toward it, and confirming it stays dismissed. Both decision-log
entries for these fixes say this plainly, and I'm repeating it here rather
than letting the strength of the unit-test evidence read as more than it is.
This is exactly the kind of interaction a real toddler will cause
constantly and repeatedly, not an edge case — worth prioritizing as the
very next camera session's first test, ahead of anything else.

## 7. Open, unfixed: the iPad and Mac browser show "totally
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

- `cv/test_risk_engine.py`: 114 `def test_` functions, grep-counted
  directly this session. The most recent decision-log entry before this
  audit claimed 113 (before Bug B's one test was logged) — now reconciled.
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

## Named gaps, in priority order

1. **The iPad/Mac alert desync (section 7) is open, real, and unmeasured.**
   For a child-safety product, two screens disagreeing about what's
   currently happening is a genuinely worse failure mode than one screen
   being uniformly slow — this is the top item for the next session.
2. **Neither hazard-map occlusion fix (sections 5 and 6) has been watched
   happen on the real camera.** Both are backed by faithful, targeted
   reproductions and real regression tests, which is real evidence — but not
   the same claim as "verified live," and both involve exactly the
   interaction a real toddler causes constantly.
3. **`ui/requirements.txt` contains a stale, self-contradicting comment**
   (says flet-audio is "deliberately NOT included" directly above where it
   *is* pinned with a full justification). Not a functional bug — the pin
   itself is correct — but confusing for the next reader. Not fixed this
   session (documentation-only scope; the file is under `ui/`).
4. **The decision log itself had two accuracy gaps, both now corrected in
   place**: a missing entry for Bug B (added this session), and an
   arithmetic error in the alert-sound-rebuild entry's own test count ("6
   total, up from 3" corrected to "8 total, up from 5," verified by reading
   `ui/test_ui.py` directly rather than trusting the original claim).
5. **"Double sound" from multiple open screens (section 4) is explained and
   logged in the decision log, but not yet added to `SETUP.md`**, where a
   future tester is more likely to actually look for it. Small, low-effort
   follow-up.

None of these five are fixed by this write-up itself — this was a
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
5. Section 7's open item is described as "not measured," not "not fixed."
   Explain the difference between those two claims in your own words, and
   name one concrete experiment (not necessarily the one this write-up
   proposes) that would move it from "plausible hypothesis" to "confirmed
   or ruled out."
