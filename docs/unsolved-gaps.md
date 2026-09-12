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
