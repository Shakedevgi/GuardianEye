# GuardianEye — Unsolved Gaps

A single, easy-to-find list of problems that are known, real, and **not
closed** — as opposed to `docs/decision-log.md`, which is a chronological
record of everything that happened (including the dead ends and partial
fixes along the way to each item below). Read this file first if the
question is "what's still broken." Read the decision log for "what did we
already try, and why didn't it work."

Each entry links to its full history in the decision log rather than
repeating it. Update this file the moment an item is genuinely closed —
move it to a "Resolved" section with the date and the decision-log entry
that closed it, don't just delete it.

---

## 1. CRITICAL — "hazard detected" firing alongside a real proximity alert
(unsolved after 6 fix attempts, 2026-09-12)

**Symptom, as reported live, repeatedly:** walking into frame produces
`hazard_detected.wav` alongside a legitimate `baby_getting_close.wav` /
`immediate_danger.wav`, for no real new hazard.

**Status: marked critical and deferred.** Shaked is picking this up
directly rather than continuing here. Six attempts were made in one
session (2026-09-12), each fixing a real, verified bug that the live
evidence at the time pointed to — and each time, further live testing
showed the underlying symptom persisted, pointing to yet another distinct
mechanism.

**Corrected 2026-09-12 by docs-agent's audit of this file against the
actual decision-log entries and the current code**: the list below was
originally out of order and, in one place, attributed a fix to the wrong
attempt (`person_occludes()` did not exist yet at attempt 1 — it was
introduced at attempt 3, replacing the plainer check attempt 1 actually
used). Re-ordered and re-attributed to match the decision log's own entry
titles (which explicitly number "Third live test," "Fourth pass," "Fifth
pass," "Sixth pass") and confirmed against the code that exists right now
(`cv/risk_engine.py`):

1. Shaked's first report of the exact symptom led to two fixes made
   together, in one investigation: a person's own body overlapping a
   hazard's tracked spot was being counted as "hazard removed," then
   "hazard re-arrived" (for an already-known, unreviewed entry); and,
   separately, a person's body overlapping a *dismissed* entry's saved
   "fingerprint" crop was making that entry re-raise itself as if it had
   changed. Both fixes reused the file's existing, deliberately lenient
   `PERSON_OVERLAP_THRESHOLD` (0.3) — the same bar the room scan uses to
   decide whether to even suggest a candidate to a parent.
2. Live-tested again: two DIFFERENT non-RED alerts ("baby getting close"
   and "hazard detected") were both still getting voiced, a few seconds
   apart. Added a priority/cooldown settle window so a RED alert could
   cancel a still-pending lower one — found the same day, by testing again,
   that this only covers RED preempting a lower alert, not two non-RED
   alerts competing with each other. Redesigned as a collection window in
   front of the alert-pacing arbiter, so only the single highest-priority
   signal in a short window is ever submitted at all. In the same
   investigation, added a dedicated backstop
   (`new_object_alert_suppressed_by_person`) at the one place every
   "new object" alert passes through regardless of origin.
3. Live-tested again, same symptom: re-read every person-occlusion check
   end to end and found the same lenient 0.3 threshold from attempt 1 was
   being reused for a much stricter job (deciding whether to interrupt with
   a sound, not whether to suggest a review item) — a child's own limb
   regularly cleared that bar as "not really occluded." Fixed by replacing
   all the inline checks from attempts 1 and 2 with one shared, much
   stricter, padded helper, `person_occludes()` (0.05 overlap, 30% padding)
   — this is where that function is actually introduced, not attempt 1.
4. Live-tested again, same symptom: found that `person_occludes()` itself
   was correct but was being fed only the CURRENT frame's raw person
   detections, which YOLO can simply miss for one frame (a child mid-stride,
   an odd angle) — with nothing detected that frame, no threshold or
   padding can help. Fixed by reading `PersonTracker`'s ~1-second memory
   (`.entries`) instead of the single frame's raw list, everywhere an
   occlusion check runs.
5. Real terminal output (not a description) showed a refrigerator entry
   already on record re-announcing itself as newly discovered — a
   DIFFERENT bug: the match between this frame's detected box and the
   stored one is failing, not anything about person overlap or missed
   detections. Fixed with a person-gated wider-tolerance retry on the
   matching logic, used only when `person_occludes()` confirms a person is
   right there.
6. A SECOND terminal log showed the same refrigerator duplicating twice
   within the first couple of seconds of a session — before anyone could
   plausibly have been occluding it. This disproved the assumption behind
   fix #5 (that a person was involved at all) — it's ordinary detection
   jitter. Fixed by making named-appliance matching (fridge/oven/
   microwave/socket) unconditionally wider, not gated on a person; the
   room scan's own, separate matching logic (arbitrary, potentially
   numerous small objects — a different risk profile) deliberately keeps
   the person-gated retry from fix #5, unchanged.

**Full blow-by-blow, each attempt's reasoning and evidence:** the six
`docs/decision-log.md` entries dated 2026-09-12, starting with "Real bug
found and fixed: an existing, unreviewed hazard firing a fresh 'new object'
alert..." and its immediate follow-up "A second bug in the same family...",
then running through "Real-camera test surfaced two more bugs...", "Live
re-test found the settle-window fix incomplete...", "Third live test...",
"Fourth live test...", "Fifth pass..." and "Sixth pass..." (search that file
for "hazard detected" or "person_occludes"; the "Two smaller session items"
entry in between is documentation bookkeeping, not one of the six fixes).

**What this means going forward:** after six real, verified fixes, the
core reported symptom — the *specific* moment of "I walked into frame and
got two alerts" — has not been confirmed gone by a live test since the
sixth fix landed. Every fix so far addressed the exact mechanism the most
recent evidence pointed to and no more; there may be a seventh mechanism
neither Shaked nor this session has surfaced yet. **If you pick this back
up: get real terminal output from the exact failing moment before trying
anything else** — that is what actually found bugs 3 through 6; reasoning
from the symptom alone did not.

---

## 2. To investigate next — do "first scan" (blue/cyan) boxes really not
persist across sessions, including a location change?

**Raised by Shaked, 2026-09-12, as a question, not yet a confirmed bug.**
CLAUDE.md decision 3 is explicit: "No persistence between sessions/boots —
every session starts with a fresh, empty map (camera angle/room/lighting
cannot be assumed identical to last time)." If the same PENDING_FIRST_SCAN
(cyan/"blue") boxes are reappearing after a restart in a genuinely
different location, that would be a real violation of this decision.

**Investigated so far (code-reading only, not yet tested with an actual
location change):**
- `HazardMap()` is constructed fresh inside `main()` every run — nothing
  loads prior entries into it at startup.
- `EventWriter`/`backend/persistence.py`'s SQLite DB is write-only from
  `cv/risk_engine.py`'s point of view — nothing reads it back to
  reconstruct hazard-map state.
- `--seed-hazard` (`cv/risk_engine.py`) is the only mechanism that injects
  hazards at startup, and it: (a) is off by default (`start.sh` passes no
  seed flags), and (b) seeds entries as `HAZARD_STATE_CONFIRMED` (red), not
  pending/blue, so it couldn't be the source even if it were on.
- No other file-based persistence (no pickle, no JSON state file) exists
  in `cv/risk_engine.py`.

**Working conclusion, not yet verified live:** by code, nothing should
persist. The most likely explanation for "the same blue boxes came back"
is that testing so far has been in the same room, so the same real objects
get freshly re-detected and land in the same place — not actual
persistence. This needs an actual test in a genuinely different location
to confirm either way before assuming it's a real bug.

---

*Items 3–10 below were added 2026-11-24 from a gap review with Shaked.*
*Shaked's own list was: double alarms (item 1), Mac/iPad alert desync*
*(listed under "Also open" below), too many false alarms, re-alarms on*
*already-reviewed objects, the design/interface, and the voice clip. The*
*review also found the other items in this file, `PHASE_PLAN.md` and the*
*recent commits. Items 3–6 are Shaked's; items 7–10 came from the review.*

## 3. Too many false alarms overall

**Reported by Shaked, 2026-11-24.** Separate from item 1's specific
double-alert symptom: the system raises alerts too often in general.
Probably overlaps with items 1 and 4, but the cause hasn't been
investigated. Not measured: no count of false alarms per session exists.
A first step would be to log every alert during a normal session and label
each one real or false, so it's clear which mechanism produces most of them.

## 4. Re-alarms on objects the parent already reviewed

**Reported by Shaked, 2026-11-24.** Objects already marked "hazard" or "not
a hazard" alert again as if they were new. Related to the dismissal
fingerprint check (`fingerprint_changed()`, `cv/risk_engine.py`) and the
hazard-matching jitter fixed in item 1's attempts 5–6. The 2026-09-12 fixes
targeted parts of this, but Shaked still sees it. Not confirmed fixed live.

## 5. Design and interface quality

**Raised by Shaked, 2026-11-24.** The current Flet UI ("Console" direction,
`ui/app.py`) works but isn't good enough for a parent. Phase 8 work, lower
priority than the alert bugs above.

## 6. Replace the voice alert clips

**Raised by Shaked, 2026-11-24; suggestion from the lecturer.** The current
pre-recorded clips aren't good. The lecturer suggested a recorded voice note
of the mother saying "NO", aimed at stopping the child from getting closer.
**Design note:** today's clips ("Hazard detected", "Baby getting close")
address the *parent*; a "NO" addresses the *child*. That likely means two
outputs: the "NO" played loudly in the child's room, and the normal alert
on the parent's screen. Still pre-recorded audio, so it stays within
CLAUDE.md decision 4.

## 7. iPad video only smooth at 2 fps (workaround, not a fix)

`--web-max-fps 2` on the `flet run --ios` launch fixed the iPad video delay
(2026-09-11) by throttling, not by fixing Flet's unbounded outbound
websocket queue. Higher rates (3–4 fps) were never tried. Picture smoothness
only, not alert latency. See `docs/decision-log.md`, search
"--web-max-fps 2".

## 8. cv/ fixes never verified on the real camera

Several fixes were reproduced and verified only with isolated scripts and
unit tests, not live. Example: commit 57b0fc7's two occlusion fixes ("neither
has been verified against the real camera yet"). Items 1 and 4 can't be
called closed until a live run confirms them.

## 9. UI panels not re-verified

The event log, settings panel, and clip keep/discard controls named in
Phase 8's goal line haven't been checked by docs-agent against the code (see
`PHASE_PLAN.md`, Phase 8). They aren't known to be broken.

## 10. No long hands-off run yet (Phase 9)

Phase 9's done-when bar is "full run-through works unattended for an
extended session without manual intervention." Not started (`[ ]` in
`PHASE_PLAN.md`). Several items above, such as false alarm rates, are only
really measurable in such a run.

---

## Resolved

*(move items here, with the date and the decision-log entry that closed
them, once they're genuinely done — none yet from this file.)*

---

## Also open, lower priority (tracked in `docs/decision-log.md` /
`PHASE_PLAN.md`, not repeated in full here)

- **iPad/Mac browser alert-banner desync** — the two screens can show
  different alerts at the same moment. Diagnosed (suspected same unbounded-
  queue transport property as the already-fixed video delay, now
  suspected on the `/risk_status` poll path) but not measured or fixed.
  See `docs/decision-log.md`, 2026-09-12, "iPad/Mac browser 'shouting
  completely different alerts'."
- **PersonTracker ID churn** — a real, visible symptom (the same physical
  person gets a new tracked ID every couple of frames in some conditions),
  a pre-existing documented simplification, not fixed here and not
  confirmed to be the cause of any specific bug on this list.
