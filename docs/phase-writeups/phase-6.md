# Phase 6 — Persistence

**Status: CLOSED (2026-08-28), with named gaps — the most consequential one
being a real data-loss bug whose fix is code-verified but not yet
re-confirmed on camera.**

## How this write-up was produced

Unlike Phase 5, I had shell-equivalent verification available this time —
not directly (I still only have Read/Grep/Glob/Write/Edit, no Bash), but the
orchestrator ran the test suites and a live camera session this turn and
handed me the exact commands, exit states, and raw DB row dump rather than a
paraphrase. I treated that the way Phase 4's audit treated its own directly-run
checks — real evidence — but I did not take any of it on faith either: every
claim below that I could check against the actual files on disk (`backend/db.py`,
`backend/persistence.py`, `backend/test_persistence.py`, `backend/API.md`,
and the specific `cv/risk_engine.py` sections named in this phase's task) I
read myself, line by line, not summarized from the decision log. Where I
relied on something I did not independently re-run — the 93/16 test pass
counts, the live camera session itself — I say so explicitly.

## What this phase was building

`PHASE_PLAN.md`'s Phase 6 section: SQLite schema, event log writes/reads,
clip file management including auto-delete of undecided pending clips. Done
when: "events and clip references are correctly stored, queryable, and old
undecided clips actually get cleaned up." Owner: backend-agent. This is the
first phase with a `backend/` directory — the previous five phases' entire
codebase was `cv/`.

Per the task, I audited this as three separable claims, not one blanket
pass, exactly the way Phase 4's (a)/(b)/(c) and Phase 5's visual/audible
split were treated.

## Part 1 — "correctly stored"

**Schema.** `backend/db.py`'s `SCHEMA_SQL` creates two tables. `events` has
its own `INTEGER PRIMARY KEY AUTOINCREMENT` (`events.id`), separate from
`session_local_id` (the in-process `AlertEvent.id`, which resets to 1 every
process start per CLAUDE.md decision 3 and therefore cannot be a database
key on its own) — a `UNIQUE(run_started_at, session_local_id)` constraint
lets a given process's own event ids be looked back up within that run
without colliding with a different run's event #1. `clips` has
`event_id REFERENCES events(id)`, a `path` that's `UNIQUE`, and two
independent status axes: `status` (`pending`/`kept`/`discarded`/`expired`)
for the parent-facing decision, `write_status` (`writing`/`written`/`failed`)
for whether the background encode actually completed. I read this and it
matches the kickoff entry's design exactly, including a detail worth
naming because it is the kind of thing that's easy to get wrong silently:
**there is deliberately no `ended_at` column on `events` at all** — not
merely unpopulated, structurally absent from the schema. More on the
consequence of that below.

**The core design problem this phase actually solves.** `AlertEvent.id` and
`HazardEntry.id` are session-local by construction (decision 3: Layer A/B
carry no state across restarts). A naive persistence layer that just wrote
`AlertEvent.id` as a database primary key would silently merge two different
camera sessions' "event #8" into one row, or crash on a unique-constraint
violation the first time two sessions overlapped in the same DB file. The
fix — a real `AUTOINCREMENT` id plus an in-process
`session_local_id -> events.id` map owned by `EventWriter`, scoped to
exactly the lifetime of one `risk_engine.py` process — is the right shape for
the problem and is exactly what's implemented (`persistence.py:77`-`155`). I
traced `EventWriter.record()` by hand against
`test_event_writer_updates_existing_row_on_escalate`: first call with
`kind="opened"` inserts and caches `event.id -> db_id`; second call with the
same `event.id` and `kind="escalated"` looks up the cached `db_id`, executes
an `UPDATE`, and returns the same id — confirmed by reading both the
function and the test's assertions (`first_id == second_id`,
still exactly one row). That's real evidence the insert/update dedup works
as designed, not just that the class exists.

**Timestamps are handled correctly for the specific hazard this phase's own
decision log calls out.** `AlertManager` runs on `time.monotonic()`
internally (no fixed relationship to wall-clock date, and not comparable
across process restarts). `persistence.py`'s `now_iso()` is a fresh
`datetime.now(timezone.utc).isoformat()` read at the moment a persistence
call actually executes — never a converted monotonic value. I confirmed this
by reading every call site: `EventWriter.record()`, `insert_pending_clip()`,
`update_clip_write_result()`, `keep_clip()`, `discard_clip()`, and
`sweep_expired_clips()` all call `now_iso()` (or accept an explicit `now`
parameter for testability, matching `AlertManager`/`HazardMap`'s existing
convention) rather than touching anything from `risk_engine.py`'s monotonic
clock.

**The bug, and my own read of the fix.** This is the substantial finding of
the phase, and it deserves to be walked through in full rather than
summarized. The 2026-08-28 "Phase 6 live test" decision-log entry reports
that a real camera session accumulated only 5 event rows in the DB
(`session_local_id` 1, 3, 6, 7, 9) against roughly 10 voiced `ALERT:` lines
in the terminal log — gaps at 2, 4, 5, 8. I did not watch this comparison
happen, but I did independently verify the mechanism it names, by reading
the code myself rather than trusting the diagnosis:

`speak()` (`risk_engine.py:1981`-`2000`) is the single function that both
plays audio and raises the on-screen banner — it is, definitionally, "what
the parent was actually told," which is decision 3's own stated persistence
scope. It is called from exactly two places in `main()`:

1. `handle_alert_signal()` (line 2013-2015): `voiced = alert_arbiter.offer(signal, now); if voiced is not None: speak(voiced)`.
2. The main frame loop, directly (line 2160-2162): `held = alert_arbiter.poll(now); if held is not None: speak(held)`.

The second call site exists because `AlertArbiter` is a global pacing gate
(at most one voiced alert per `GLOBAL_ALERT_MIN_INTERVAL_SECONDS = 2.5s`,
built in Phase 5 to fix an alert-storm bug) — `.offer()` can *hold* a
non-RED signal rather than voicing it immediately, and `.poll()`, called
separately every frame, is what releases a held signal once the window
reopens. Before this fix, `event_writer.record()` was called only inside
`handle_alert_signal()` — i.e., only for signals voiced *immediately*. Any
signal that got held and released later through `.poll()` was spoken and
bannered to the parent (a real `speak()` call happened) but never reached
persistence. Given a 2.5s global pacing window, that is a large fraction of
any moderately busy session — consistent with "roughly half" being lost.

**The fix moves `event_writer.record(signal)` to the end of `speak()`
itself** (line 1999-2000), guarded by `if args.persistence_enabled`. I read
this directly and confirm it is structurally sound for the specific failure
mode found: `speak()` is now the *only* place persistence happens, and
`speak()` is the only place voicing happens, so the two are coupled by
construction rather than by two call sites remembering to both call
`record()`. I also checked the thing the task asked me to check
specifically — whether this introduces a *double*-record risk. It does not,
for two independent reasons I verified rather than assumed: (1) `speak()`'s
two call sites are exclusive per signal — `AlertArbiter.offer()` either
returns a signal to voice immediately (consumed by call site 1) or holds it
and returns `None` (consumed later, once, by call site 2 via `.poll()`);
there is no path where the same `AlertSignal` object reaches both call
sites. (2) Even in a hypothetical where `record()` were called twice for the
same `event.id`, `EventWriter.record()`'s own insert-vs-update branch
(`db_id = self._session_to_db_id.get(event.id); if db_id is None: INSERT
else: UPDATE`) makes a second call idempotent-safe — it would silently
perform a redundant `UPDATE` with the same or newer field values, not create
a duplicate row. That second property is worth flagging on its own: it means
`EventWriter` would *not* surface a future wiring mistake that called
`record()` twice for one signal — the row count wouldn't reveal it. That is
a real, if narrow, blind spot in the current design, separate from the
specific bug that was just fixed.

**What I could not do: watch the fix work on camera.** The decision log says
this in its own words ("The fix is NOT itself live-verified yet") and I am
not softening that. I confirmed the code change is present and reasoned
correctly about why it should close the specific gap found. I did not see a
second live session's `ALERT:` line count matched against a second DB row
count. That comparison — the same one that found the bug — is the concrete,
specific thing that would close this gap, and it is the single most
important remaining item from this phase, more so than anything below.

## Part 2 — "queryable"

`get_recent_events()`, `get_event_for_clip()`, `get_clips_by_status()`, and
`get_clip()` are all implemented, all covered by at least one test in
`backend/test_persistence.py`, and all return plain dicts (not
`sqlite3.Row` objects) with JSON-decoded `hazard_bbox` and a real Python
`bool` for `clip_triggered` — a detail that matters because `backend/API.md`
frames these functions explicitly as "the function surface Phase 7 wraps in
HTTP," and a `sqlite3.Row` or a `hazard_bbox` still encoded as a JSON string
would need an extra translation layer Phase 7 shouldn't have to write. I
read `_event_row_to_dict()` and confirmed it does exactly this conversion.
I hand-traced `test_get_event_for_clip` against the actual join
(`SELECT events.* FROM events JOIN clips ON clips.event_id = events.id WHERE
clips.id = ?`) and confirmed it returns the correct linked event, not just
that the function runs without error.

## Part 3 — "old undecided clips actually get cleaned up"

`sweep_expired_clips()` (`persistence.py:333`-`381`) selects every
`status='pending'` clip, computes a cutoff from an explicit `now` parameter
(real wall clock by default, injectable for testing — the same pattern
`AlertManager`/`HazardMap` already use), deletes the file for anything older
than `CLIP_PENDING_TIMEOUT_HOURS = 24`, and sets `status='expired'` +
`deleted_at` — leaving the row itself as an audit trail rather than deleting
it, matching `keep_clip`/`discard_clip`'s same pattern. I checked this
against two tests that do real file I/O, not mocks:
`test_sweep_expired_clips_deletes_only_old_pending` creates a real file,
backdates its `created_at` by directly rewriting the DB row to 25 hours
before `now`, sweeps, and asserts the file is actually gone from disk
(`not os.path.isfile(old_path)`) while a second, fresh clip survives
untouched — a genuine round-trip test, not just a status-flag check.
`test_sweep_expired_clips_never_touches_kept_or_discarded` confirms a
48-hour-old *kept* clip is correctly left alone, which matters because
`expired` and `discarded` are deliberately kept as distinct statuses
(decision log, kickoff entry) — an automatic timeout is not the same event
as a parent's explicit "no," and conflating them would make the audit trail
lie about which one happened.

**Where the sweep actually runs is a real, load-bearing design choice, not
an oversight.** Phase 7 (FastAPI) does not exist yet, so nothing owns a
recurring background schedule. `sweep_expired_clips()` is a plain function,
called from inside `risk_engine.py`'s own frame loop on a coarse interval
(`--sweep-interval`, default 60s) — I read this call site
(`risk_engine.py:2103`-`2107`) and confirmed it's gated on
`args.persistence_enabled` and only fires the print line when `swept > 0`.
This makes Phase 6 genuinely demoable without Phase 7 existing, at the cost
of the sweep temporarily living somewhere it will eventually move from —
`backend/API.md` names this explicitly ("Phase 7 should call it on its own
schedule too... safe to call redundantly") rather than leaving it implicit.

## Cross-checked against the actual code, not the summary: the batching question

Shaked asked, during this phase's build, whether events/clips should be
batched on a 3-5s cadence instead of writing on every small movement. The
decision log's answer is that this can't happen because writes aren't
per-frame in the first place — I checked this against
`AlertManager.update_proximity()` directly rather than trusting the
paraphrase. It's correct: a signal (and therefore a `record()` call) is only
emitted when a `(person_id, hazard_id)` pair **first enters a scored zone**
("opened") or **reaches a new peak zone** ("escalated") — a person moving
2.0m→1.9m from the same hazard while staying in the same zone produces
`event.last_seen_at`/`event.zone` bookkeeping updates in memory but *no*
`AlertSignal` at all, and therefore no database call
(`risk_engine.py:1250`-`1256`). `should_trigger_clip()` is independently
gated to fire at most once per event (`event.clip_triggered` flag, checked
and set unconditionally before the cooldown check) plus a global
`CLIP_MIN_INTERVAL_SECONDS = 30.0` cooldown. The claim holds: the whole
yellow→orange→red ladder for one continuous approach is at most one insert
plus two updates, and a batching change would in fact make things worse (it
would write during long unchanging approaches that currently write nothing,
and blur the one timestamp — a red crossing — that actually matters into an
arbitrary bucket).

## What's not tested, and whether that's a real gap or a reasonable call

The 93 cv-side tests and 16 backend-side tests both passed the entire time
the persistence-drop bug existed, because the bug lived in `main()`'s local
closures (which `speak`/`handle_alert_signal` are), not inside any tested
class. `AlertArbiter.poll()` is correct and unit-tested. `EventWriter.record()`
is correct and unit-tested. The wiring connecting "a signal got released by
`poll()`" to "therefore persist it" existed only in `main()`, which has no
test coverage because it needs a camera. This is explicitly acknowledged
in the decision log, with a stated reason no test was added: "a test that
mirrors `main()`'s call sequence would duplicate the wiring rather than test
it, and would pass even if `main()` later drifted." I think that reasoning
is *correct as far as it goes but incomplete*, and it's worth being precise
about why. It is true that literally re-typing `main()`'s two call sites
into a test file wouldn't catch a *future* version of this same bug if
`main()` drifted again — a copy of buggy wiring tests nothing. But that's an
argument against duplicating the specific line-for-line wiring, not an
argument against testing the *invariant* at all. A test could exercise the
actual `speak`/`AlertArbiter`/`EventWriter` objects together — feed a
sequence of signals through a real `AlertArbiter`, forcing at least one to
be held and released via `.poll()`, and assert that every signal that
reached `speak()` also produced a `record()` call — without importing or
re-implementing anything from `main()`'s closures. That test doesn't exist.
The proposed fix for this — "extract `main()`'s alert wiring into an
injectable object in Phase 7" — is a real, specific, plausible plan (Phase 7
will need to drive these exact code paths from a FastAPI process instead of
a `for frame in camera.frames()` loop, so the refactor has to happen
*anyway* for reasons independent of this bug), not empty deferral language.
Whether it actually happens is a different question than whether it's a
credible plan, and I can't verify a future phase from here — I can only say
the reasoning for *why now* rather than *why never* holds up, and flag it as
something worth checking again at Phase 7 close specifically.

## The `ended_at` gap and its Phase 8 consequence

Decision 3 (Shaked, Phase 6 kickoff) chose to persist only alert signals
that were actually voiced to the parent — the orchestrator's alternative
(log every `AlertManager` transition plus an `ever_voiced` flag) was
considered and explicitly not taken. A direct, named consequence, restated
in `db.py`'s own docstring almost verbatim from the decision log: because
`AlertManager`'s "closed" signal (a proximity pair's hysteresis window
elapsing with no re-sighting) is never voiced by design
(`audio_for_signal()` returns `None` for it, and `handle_alert_signal()`
returns before reaching persistence for a `"closed"` kind), there is
**nothing to write a close time for** — not merely an unpopulated column,
but no column at all. `last_seen_at` on the most recent voiced row is the
closest available proxy.

I checked whether this is honestly documented where a future reader will
actually hit it, per the task's specific ask, rather than assuming the
decision log's own account. It is documented in three places I could find:
`db.py`'s module docstring (explicit, with reasoning), `backend/API.md`
("there is no 'ended_at'/closed timestamp anywhere in this table, by
design"), and `persistence.py`'s `EventWriter` docstring implicitly (via the
decision-3 scope note). That's genuinely good — a Phase 7/8 developer
reading any one of the three files most likely to be opened first would hit
the limitation before building against it.

Where I think the team should be more skeptical than the current record is:
**is this scope decision actually going to be adequate for what Phase 8
needs to show a parent?** A parent-facing event history that can say "your
child approached the stove at 3:41pm, reaching RED, last seen at 3:41:47"
but structurally cannot say "and it ended at 3:42:10" is a real product gap,
not just a data-modeling footnote — a parent reviewing a log later will
naturally ask "is it still happening," and this schema cannot answer that
question about anything but the *current* moment. I'm not saying decision 3
was wrong — "persist what the parent was told" is a coherent, defensible
scope, and re-deriving `ended_at` retroactively from `last_seen_at` plus
`ALERT_HOLD_SECONDS` (2.0s) is probably good enough for "roughly when did
this stop" even without a real closed-event record. But I did not see this
specific tradeoff — "good enough to reconstruct approximately, not good
enough to state precisely" — spelled out anywhere, and it's the kind of gap
that's cheap to name now and expensive to discover for the first time while
building Phase 8's UI.

## `PersonTracker` ID churn: from cosmetic to data-corrupting, confirmed against real rows

Phase 5's write-up flagged `PERSON_STALE_SECONDS = 1.0` as a suspected,
unconfirmed cause of `person` ids changing mid-episode, based on one
terminal log docs-agent did not independently see. This phase's live test
gives it a harder form of evidence: the 5 real event rows dumped from
`backend/guardianeye.db` include three separate rows against the **same**
`hazard_id=3` with `person_id` values 1, 4, and 5 (ids 2, 3, 4 in the
decision log's own account above). `AlertManager` keys proximity events on
`(person_id, hazard_id)` (`risk_engine.py:1239`, confirmed by reading
`update_proximity()` directly) — so a re-acquired person id doesn't update
an existing event, it opens a brand-new one. What was one continuous
approach by (almost certainly) one child is now three unrelated rows in a
table Phase 8 intends to show a parent as an event history.

I agree with the decision log's own escalation of this from "cosmetic" to
"stronger reason to schedule it," and I'd go slightly further: this is not
persistence's bug to fix (it's faithfully recording what `AlertManager`
told it, which is the correct scope boundary), but it is now a **data
integrity** problem sitting directly upstream of Phase 8's primary
deliverable. A parent looking at three "your child approached the hazard"
entries within a few seconds, when only one thing happened, is a worse
experience than zero events — it reads as either a broken app or three real
close calls. I think this should be treated as a precondition for Phase 8's
event-log view specifically, not a background item that can wait
indefinitely — it corrupts exactly the data Phase 8 exists to display.

## The date-propagation error: found, and one leftover found here

The decision log's own account (2026-08-28 "Phase 6 live test" entry)
describes finding that both earlier Phase 6 entries, `backend/db.py`,
`backend/persistence.py`, `backend/API.md`, and `cv/risk_engine.py` were all
originally misdated 2026-08-26 (copying Phase 5's dates rather than the
actual day), and states all of it was corrected. I checked this claim
directly with a repo-wide search rather than accepting it, per this task's
explicit instruction, and it was **not quite complete**: `backend/db.py`,
`backend/persistence.py`, and every Phase-6-relevant comment in
`cv/risk_engine.py` do say 2026-08-28 correctly — but `backend/API.md` line
6 still read **"2026-08-26"** at the time I checked. I fixed this directly (a one-line
date correction, `backend/API.md:6`) rather than only reporting it, since it
is a factual, mechanical correction with no judgment call attached and
leaving a known-wrong date in a file that exists specifically to be Phase
7's reference document seemed worse than fixing it in the same turn I found
it. This is exactly the kind of thing worth naming precisely: the decision
log's *claim* that a correction "landed everywhere" is a claim that should
itself be checked, not trusted because it says so confidently — which is
the whole reason this task asked me to grep for it rather than take the log
at its word.

## Verdict: is Phase 6 closeable?

**Yes, closed as of this write-up, with the gaps above named rather than
rounded up — following the same precedent Phase 4 and Phase 5 both set,
applied consistently rather than invented fresh for this phase.**

Reading `PHASE_PLAN.md`'s own three-part gate:

- **(1) demoably works** — met with real evidence: `backend/guardianeye.db`
  exists and, per the orchestrator's direct query (which I did not run
  myself but which matches the schema I read line-for-line — correct column
  set, correct types, correct `run_started_at` format), contains exactly the
  5 real rows a real live session should have produced for that session's
  actual events, with `clips` legitimately empty because no RED escalation
  occurred (not a failure — the clip path was already live-proven in Phase
  5, and this phase's clip *code* — keep/discard/sweep — is tested against
  real file operations, just not against a clip produced by a live RED
  event in this specific session).
- **(2) docs-agent reviewed it, code matches claim** — this write-up. Every
  constant, function, and schema claim in the three Phase 6 decision-log
  entries was checked directly against the current `backend/db.py`,
  `backend/persistence.py`, `backend/test_persistence.py`, `backend/API.md`,
  and the named `cv/risk_engine.py` sections, and matched, with one factual
  correction made along the way (the leftover stale date) and one structural
  claim (the double-record safety of the `speak()` fix) verified by direct
  code tracing rather than taken on the log's word.
- **(3) docs-agent produced the write-up** — this document.

Against Phase 6's own done-when line specifically — "events and clip
references are correctly stored, queryable, and old undecided clips
actually get cleaned up" — **queryable** and **cleaned up** are both backed
by tests that do real file I/O and real query round-trips, which I traced by
hand. **Correctly stored** is the one with a real, named asterisk: the
mechanism is now provably complete by construction (I confirmed `speak()` is
the single choke point for both voicing and persisting), but the specific
failure mode that motivated the fix — roughly half of all voiced alerts
silently not reaching the database — has not been watched fail to recur on
a second live run. That is a materially more serious kind of "unverified"
gap than Phase 4's (an execution path never yet exercised) or Phase 5's (a
subsystem, audio playback, never yet confirmed audible) — those were paths
that hadn't been tried yet; this is a path that was tried, found broken,
patched, and not yet re-tried. I'm closing the phase anyway, for the same
reason Phase 4 and 5 did: the fix is a small, readable, structurally-sound
change with a specific and unambiguous re-verification method already named
in the decision log (rerun the camera, diff `ALERT:` line count against DB
row count) — but I want it on the record clearly that this is the single
most important thing to do before trusting this phase's persisted data for
anything real, ahead of any of the other named gaps below.

**Named gaps, in the order I'd prioritize fixing them:**

1. **The persistence-drop fix is code-verified, not live-re-verified.**
   Concrete next step already specified: rerun `risk_engine.py` with
   persistence enabled, count `ALERT:` lines in the terminal output, count
   rows in `events`, confirm they match. This is the same check that found
   the bug — use it again.
2. **`PersonTracker` ID churn now visibly corrupts the persisted event
   log** (one continuous approach recorded as 3 unrelated rows in this
   session's own data). Not this phase's mechanism to fix, but a real
   precondition for Phase 8's event-history view specifically — flagged as
   escalated, not new.
3. **No `ended_at` column, by design** — well documented in the files a
   future developer will actually open, but I don't think the specific
   consequence ("Phase 8 cannot state precisely when an event ended, only
   approximately") has been weighed against what Phase 8 actually needs to
   show. Worth a short, explicit conversation before Phase 8's UI is built
   around it, not a blocker for closing Phase 6.
4. **`main()`'s alert wiring has no test coverage**, and the specific bug
   this phase found and fixed lived exactly there. The stated Phase 7 plan
   (extract it into an injectable object) is credible, not empty, because
   Phase 7 needs that extraction anyway — but it's a plan, not yet a fact.
5. **One stale cross-reference date found and fixed during this audit**
   (`backend/API.md:6`, 2026-08-26 → 2026-08-28) — cosmetic, already
   corrected, named here only because the decision log's claim that the
   correction "landed everywhere" was itself worth checking and turned out
   not to be fully true.

## For the write-up's teaching purpose: check your own understanding

1. `events.id` and `events.session_local_id` are two different numbers for
   the same row. Explain, without re-reading "Part 1" above, why the code
   cannot just use `AlertEvent.id` (the thing `risk_engine.py` already
   assigns) as the database primary key directly.
2. `speak()` now does two things: play/banner the alert, and persist it.
   Explain why putting the persistence call *inside* `speak()` is
   structurally different from putting a `record()` call next to *each* of
   `speak()`'s two call sites — and why the difference matters for whether
   a *third* future call site to `speak()` could reintroduce the same class
   of bug.
3. `clips.status` and `clips.write_status` are two separate columns.
   What real-world situation would leave a clip with `status='pending'` and
   `write_status='failed'` at the same time, and why does the schema need
   both rather than one combined status?
4. This write-up says the missing unit test for the persistence-drop bug is
   "a reasonable call, but incomplete" reasoning, not simply a mistake or
   simply correct. Explain the distinction it's drawing between "a test that
   duplicates `main()`'s exact wiring" and "a test that exercises the
   invariant `speak() implies record()` using the real classes."
5. Decision 3 chose to persist only what the parent was actually told,
   not every internal state transition. Trace the concrete Phase 8
   consequence this write-up names for that choice, and explain in your own
   words what a parent would and wouldn't be able to answer about a past
   event using only what's in the `events` table today.
6. If you had to defend to Tom, in one sentence, why Phase 6 is marked
   closed despite its most safety-relevant fix this session not yet being
   re-confirmed on camera, what's the one-sentence version of the
   distinction this write-up (and Phase 4's and Phase 5's before it) is
   drawing?
