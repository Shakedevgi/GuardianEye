# GuardianEye — Multi-Agent Workflow Notes

Running narrative of how the "simulated engineering team of Claude Code
subagents" approach is actually going, as its own deliverable (per CLAUDE.md's
team-structure section, this is not a footnote — it's material for the final
paper).

---

## Entry 1 — Scaffold + Phase 1 (2026-08-03)

### The initial scaffold

The project started with a single commit (`a181780`) that laid down
`CLAUDE.md` (architecture decisions, locked unless deliberately revised),
`PHASE_PLAN.md` (9 phases with explicit dependencies and a hard rule: no
phase N+1 before phase N is "closed" — demoed, docs-agent reviewed, write-up
produced), and four subagent role files (`cv-agent`, `backend-agent`,
`ui-agent`, `docs-agent`, later moved into `.claude/agents/` in commit
`89b4246`). The intent, stated explicitly in CLAUDE.md, is dual: build
GuardianEye correctly, *and* generate a real case study in orchestrating
multiple AI agents on non-trivial, order-dependent work. Treating that
meta-goal as a first-class deliverable from commit one — rather than
retrofitting it later — is itself worth noting: it means role boundaries and
phase gates were defined before any code existed, not reverse-engineered from
however the work happened to get done.

### The rhythm (as it played out in Phase 1)

The division of labor is strict by design: cv-agent owns camera/YOLO/risk
logic, backend-agent owns FastAPI/SQLite, ui-agent owns Flet, and
docs-agent (me) is explicitly barred from being "just a scribe" — CLAUDE.md
and PHASE_PLAN.md both state a phase isn't complete until docs-agent has
independently verified the code matches the claim and produced a write-up.
That's a meaningful check on a known failure mode of agent-driven
development: an agent (or a human skimming its output) declaring victory
because the code *looks* plausible, without anyone re-reading it adversarially
against the original spec.

### How Phase 1's delegation actually went

This is the part I can verify directly, having read the code myself rather
than taking the "what was built" summary on faith.

**Scope discipline was clean.** cv-agent stayed inside its lane: two scripts
(`detect_cameras.py`, `stream_camera.py`), a minimal `requirements.txt`
(`opencv-python` only — a comment in the file explicitly notes YOLO/torch are
deliberately deferred to Phase 2, not smuggled in early), a scoped README, and
a root-level `.gitignore`. Nothing touches FastAPI, Flet, or SQLite territory,
and nothing pulls in AI/model dependencies ahead of where PHASE_PLAN.md says
they belong. That's a good sign for the "agents don't quietly encroach on each
other's territory" question this project is implicitly testing.

**Hardware-dependent verification correctly stayed human-in-the-loop.** No
subagent can plug in a USB camera or judge whether a live OpenCV window is
actually rendering a picture. The workflow handled this correctly: cv-agent
wrote the two scripts, and Shaked was the one who ran
`python cv/detect_cameras.py` to find the real device index, then ran
`python cv/stream_camera.py --index 0` and watched a real live feed for
several minutes. This is exactly the kind of step no agent should be trusted
to self-certify, and it wasn't — the loop closed with a human hands-on-camera,
not an agent claiming it "should work."

**The code itself is honest about its own limits**, which matters more than it
might sound: `stream_camera.py`'s reconnect logic is commented as "simple
backoff — this is Phase 1, not a production reconnect strategy," and the
default camera index is flagged in-code as "very likely NOT the USB camera on
a laptop." That kind of self-aware, undersell-don't-oversell commenting makes
my auditor job easier — there was no gap between what the code claimed to do
and what it does, which is exactly the thing I'm supposed to be checking for
and exactly the thing that's easy for an agent to get sloppy about under
pressure to "finish a phase."

### What I'd flag for the team to think about going forward, not urgent yet

- Nothing in Phase 1 required cross-agent coordination (only cv-agent
  touched code), so this entry can't yet speak to how well the agents
  *hand off* work to each other — that's untested until Phase 5+ (cv-agent
  → backend-agent clip storage hookup) or Phase 9 (all agents). Worth
  watching closely when that first happens.
- The phase-gate rule ("docs-agent signs off before N+1 starts") is only as
  strong as someone actually enforcing it before typing "start Phase 2." Phase
  1 did wait for this review, which is the right precedent to keep.

### For the final paper

If I had to summarize this entry for someone who wasn't in the room: the
single most interesting design choice so far isn't the code, it's that the
team defined "done" (demo + independent audit + write-up) *before* writing any
code, and set up an agent whose only job is to be the skeptical reader. Phase
1 is too small a sample to know if that holds up under real pressure (a
messier phase, a deadline, a disagreement about scope) — but it held for
Phase 1.

---

## Entry 2 — Phase 2 (2026-08-04 to 2026-08-05): the phase that actually
tested the process

Phase 1 was too clean a sample to say much about how this workflow behaves
under real pressure. Phase 2 was not clean, and that's exactly what makes it
the more useful entry for the final paper — this is the first phase where
the multi-agent structure got stress-tested rather than just followed.

### The part that went the way the design intended

cv-agent built `cv/camera.py`, the refactored `cv/stream_camera.py`, and
`cv/detect_stream.py` from a spec (Phase 2 in `PHASE_PLAN.md`, plus
CLAUDE.md decisions 2 and 7) that had already settled the structural and
dependency questions before cv-agent touched a keyboard: single YOLO pass
per frame, no hazard filtering yet, exact version pins for a new-enough
stack. Given a spec that specific, the actual code production was the least
eventful part of the phase — I read every line of the three files and found
no stubs, no scope creep into Phase 3/4 territory, and no quiet deviation
from CLAUDE.md. That's a real point in favor of "spend the ambiguity budget
on the spec, not on hoping the agent infers the right constraints."

### The part the design didn't anticipate: this phase was environmental, not
algorithmic

Almost none of Phase 2's real difficulty was about YOLO, model choice, or
detection code. It was a two-hour camera bring-up problem — macOS TCC
permissions scoped to the terminal app rather than the Python process, and
AVFoundation device indices silently shifting when hardware was replugged.
**No subagent could have run this diagnostic loop alone.** cv-agent has no
camera to plug in, no terminal permission dialog to click through, no way to
notice "wait, the resolution printed just changed" mid-session. This had to
be a tight loop between the human (Shaked, physically handling the camera
and terminal) and the orchestrator (reasoning about what the symptoms
implied), with cv-agent's role effectively paused until there was stable
ground to build detection code on top of. This is the first real evidence
for a boundary this project's team structure implies but hadn't yet tested:
subagents are excellent at implementing a well-specified build, and
completely unable to participate in physical-hardware diagnosis. Any future
phase with a hardware-in-the-loop problem should expect the same split, not
assume the agents can absorb it.

### The orchestrator was wrong, twice, in ways worth keeping on the record

Per the orchestrator's own explicit instruction for this write-up: don't
sanitize this. Twice during the camera bring-up, the orchestrator inferred
which OpenCV device index corresponded to the physical USB camera by
reasoning from indirect evidence — `system_profiler` output and ffmpeg's
device-enumeration order — rather than direct measurement. Both times it was
wrong. Both times the user's direct empirical check (reading an actual frame,
or watching which index disappeared on unplug) was correct. The same pattern
showed up again with performance: the orchestrator used published
Ultralytics benchmark ratios (scaled from CPU/ONNX numbers) to predict
`yolo26x`'s frame rate on this machine, landing at ~12 FPS; the measured
number was 35 FPS, roughly 3x off. Both mistakes have the same shape —
trusting an indirect inference over a direct measurement that was cheap to
just go get. Worth stating plainly for the paper: an orchestrator reasoning
from plausible-sounding secondhand signals is not a substitute for the human
just checking, and this phase is good evidence of that, not a hypothetical
risk.

### Where the phase's central open question stands, and why that's not a
failure of the process

The question "is stock COCO actually good enough for small hazards at room
distance" went through four distinct positions across the session (see
`docs/phase-writeups/phase-2.md` for the full sequence) before landing on
"unresolved, trending positive, needs Phase 3's systematic data to settle."
That back-and-forth could look like thrashing. I'd argue it's closer to the
process working as intended under real uncertainty: each position was
revised because of new evidence (a resolution mechanism found, more test
scenes run, a config change tried), not because anyone got tired of
disagreeing. The failure mode this project should actually worry about is
the *opposite* one — an agent or orchestrator settling on an answer and
presenting it as final because revisiting it feels like admitting the first
answer was wrong. That didn't happen here, and it's worth naming as a thing
that went right precisely because the surface story (a question that
wouldn't sit still) looks messy.

### Docs-agent's own role in this phase

I did not just transcribe cv-agent's or the orchestrator's account of what
happened. I independently re-read `cv/camera.py`, `cv/stream_camera.py`, and
`cv/detect_stream.py` line by line against the specific claims made about
them (constant values, single-pass inference, model-load timing), and found
two things nobody had flagged: `cv/README.md` had gone stale relative to the
code (still describing `yolo26n` as default, missing the `--model`/`--imgsz`
flags entirely), and `docs/decision-log.md`'s two Phase 2 sittings
contradicted each other until this review reconciled them. Neither is a
large finding on its own, but both are exactly the category of thing that
compounds silently in a fast-moving, multi-sitting phase if nobody's job is
specifically to re-read everything adversarially after the fact.

### What I'd tell another student team about this phase specifically

- If a phase's real risk is hardware/OS-environmental rather than
  algorithmic, expect your agent division of labor to be mostly irrelevant
  to solving it — budget human+orchestrator time for that explicitly rather
  than assuming "cv-agent owns camera stuff" means cv-agent can debug a
  camera.
- Don't let an orchestrator's inference (from logs, enumeration order,
  published benchmarks) substitute for a two-minute direct check when one is
  available. This phase paid for that twice.
- A phase that changes its mind mid-session about an open technical question
  is not automatically a process failure — check whether each revision was
  driven by new evidence before treating the back-and-forth as thrashing.
- Documentation written across multiple sittings within one phase (this
  phase's decision log was written 2026-08-04 and 2026-08-05) needs an
  explicit reconciliation pass, not just append-only entries — otherwise the
  log accumulates internal contradictions that are easy to miss just reading
  forward.

---

## Entry 3 — Phase 2, after sign-off: three more rounds, and the one that
should worry a reader most (2026-08-05)

Phase 2 was signed off with two follow-ups outstanding. What happened next is
more valuable for the final paper than anything in Entry 2, because it's the
first time this project's process produced a failure that reached "shipped
and asserted as working" before being caught — not a bug found in review, a
bug found because the thing it printed was wrong while it was running.

### The shape of it: three rounds, each peeling back the next layer of the
same misconception

Round 1 built exactly what was asked: read real frames before claiming
success, and select a camera by name instead of a possibly-unstable index.
Round 2's live testing found two bugs Round 1 didn't (quit going dead during
a failure loop; replug never actually recovering) — both traceable to the
same root cause, "a status flag was trusted instead of a real read," in two
different places in the code. Round 3 found that the *name selection itself*
had exactly that same root cause one layer up: it trusted an enumeration
order instead of a real, confirmable measurement. Every round fixed a
concrete bug and, in doing so, exposed a more fundamental version of the
identical mistake sitting one layer underneath it. That's a pattern worth
naming for another team: "fix the bug you found" is necessary but is not the
same question as "is there a more general version of this bug I haven't
found yet," and this phase is a clean illustration of the second question
being worth asking every time, not just once.

### A design shipped with a fallback plan, and the fallback plan was wrong
for the same reason as the thing it was meant to rescue

This is the single most important process finding of the phase. Round 1's
name-to-index resolution was positional (trust `system_profiler`'s listing
order), and its own documentation named an explicit escape hatch: "switch to
pyobjc/AVFoundation if this assumption ever breaks." When it broke, the
person carrying out the redesign checked that escape hatch first — and found
pyobjc produced the *identical* wrong order `system_profiler` did. Two
independently-sourced enumeration lists agreed with each other, and both
disagreed with what OpenCV actually delivered. The documented fallback was
never actually independent of the flaw in the original design; it shared the
same unexamined assumption (that *some* enumeration order corresponds to
OpenCV's index space) one level down. Worth stating plainly for the paper:
writing "if X breaks, try Y" is not the same as verifying Y doesn't share X's
actual weakness — an escape hatch is only as good as whether anyone checked
that it escapes the *right* thing, and nobody had, because the assumption
being escaped had never been stated precisely enough to check against.

### A subagent bug the orchestrator could only catch by remembering something
the subagent never saw

The capability-matching redesign had a real, silent bug on first pass: the
probe requested a resolution but never set pixel format, and would have
fingerprinted the Arducam at a third of its real capability (1920x1080
uncompressed, when the true ceiling of 3840x2160 is only reachable under
MJPG — a 4K UVC camera can't push uncompressed frames at that size over USB).
This was caught not by the subagent producing the redesign, but by the
orchestrator cross-referencing a diagnostic run from *earlier in the same
session* — evidence the subagent doing the capability-matching work had no
access to. This is worth taking seriously as a structural observation, not
just a lucky catch: a subagent's context is bounded by what it's been shown
in its own conversation, and a multi-agent (or multi-session) workflow can
easily generate exactly the kind of cross-referencing catch that only
happens if *something* — here, the orchestrator, but it could as easily have
been a docs-agent with full read access — is deliberately holding the
longer, cross-session memory that individual subagents structurally can't.
Don't assume a subagent will connect two facts it was never shown together,
even if both facts are, in principle, "in the codebase somewhere."

### Live human testing found things no amount of sandboxed verification could

Every bug in Rounds 2 and 3 was found by a human actually running the code
against real hardware, not by any kind of code review, mocked test, or
sandboxed check. The `q`-goes-dead bug and the replug-never-recovers bug
needed someone to watch a frozen window. The positional-resolution bug needed
someone to watch the wrong camera stream while the console asserted the
right one. None of this is a knock on sandboxed verification generally —
it's a specific, structural fact about this project: the sandbox this whole
team of agents operates in has no camera plugged into it. Any claim of the
form "this camera-selection logic is verified" needs to be read as "verified
up to the boundary of what could be checked without a camera," and this
phase is good evidence that boundary is not narrow — it's where most of the
real bugs were hiding.

### What I'd tell another student team about this specifically

- If a subagent's design includes a stated fallback or escape hatch, treat
  "have we actually verified the fallback doesn't share the primary
  approach's weakness" as a mandatory question before shipping, not an
  optional one — this phase shipped a fallback that shared the flaw and
  nobody asked that question until the flaw fired for real.
- Give whoever holds the longest memory in the session (here, the
  orchestrator) an explicit job of cross-referencing earlier evidence
  against later subagent work — this phase's one bug-caught-before-shipping
  moment happened because someone did that on purpose, not by accident.
- Do not treat "verified" as a single bit. This phase's own decision log is
  careful to distinguish "verified against a mock," "verified on real
  hardware for case A but not case B," and "logically sound but unvalidated"
  — and even after all three rounds, one gap (real-hardware unplug/replug
  recovery under the final design) is still open. That precision is what
  let this audit find a new, previously-unflagged risk (reopen blocking for
  tens of seconds under `--name`, silently reintroducing the very freeze
  Round 2 fixed) just by reading the code closely enough to compute a
  worst case nobody had computed yet.
- A phase can be "signed off" and still generate its most important material
  afterward. Don't treat sign-off as the point where a phase stops being
  worth auditing closely — the most instructive incident in Phase 2 happened
  entirely after this document first said "done."

---

## Entry 4 — Phase 3, Step 0 (2026-08-09): the orchestrator correcting its
own team's prior analysis, and a live preview lying by omission

Step 0 was a deliberate insertion into the plan, not a phase anyone had
originally scheduled: Phase 2 found that `imgsz` dominates small-object
detection, which meant every earlier "COCO can't see this" belief was
measured at a setting now known to be unreliable for exactly the objects in
question. Rather than start Phase 3's data collection on top of that, the
team stopped and built a measurement step first. That decision itself is
worth naming as a process choice, not just a technical one: it would have
been easy to let `docs/phase-3-scoping-notes.md` (three annotated
screenshots, one session) stand in as "the Phase 3 scope" and start shooting
a dataset against it. Nobody did that.

### The orchestrator caught its own prior work being over-stated, by going
back to the raw evidence instead of trusting the summary

This is the process observation I'd put first for the final paper. The
scoping notes weren't produced by an outside party — they were the
project's own earlier analysis, written in good faith, by the same process
that later corrected them. `docs/phase-3-step0-findings.md` explicitly
states it "supersedes `docs/phase-3-scoping-notes.md` wherever the two
disagree," and does not soften what disagreement means: the scoping notes'
"person detection is weaker than expected" conclusion and its "scissors:
complete miss, and not a resolution problem" conclusion were both real
observations that were also, on the specific point they generalized to,
wrong — both were measured at exactly the one `imgsz` setting that produces
that failure mode, and neither scoping-notes writer went back to check
whether the failure held at other settings before writing a general
conclusion. What actually caught this was not a subagent auditing another
subagent's work; per the orchestrator's own account, it was the orchestrator
declining to accept the prior analysis's conclusions at face value and going
back to look at the raw evidence — the same underlying frames, re-measured
systematically — rather than treating "we already looked at this" as a
reason not to look again. Worth stating plainly: this is exactly the kind of
self-correction a project can quietly fail to do, because revisiting your
own team's earlier conclusion doesn't feel like finding a bug, it feels like
re-litigating settled work. The two Phase 2 findings that got corrected here
weren't caught by an adversarial audit — they were caught by someone with
the authority to say "let's actually check" choosing to say it about their
own side's prior output, not just an external agent's.

### A live preview cannot tell a human "these boxes are wrong," only "boxes
are drawn"

The single most consequential finding of this phase — small choking hazards
being confidently mislabelled as `cell phone`/`sports ball` rather than
missed — was captured on video, watched live by the operator, and reported
as a *failed* block: "it didn't even detect anything." The measurement said
the opposite of what the live impression said, and said something more
important than either a clean success or a clean failure would have been.
This is worth treating as a structural property of this project's own
architecture (CLAUDE.md decision 1: the video pipeline streams live
annotated frames as its primary interface), not just an anecdote from one
session: a real-time overlay is very good at showing *that* the pipeline
did something, and offers no signal at all about whether what it did was
correct. A human watching MJPEG output — which is most of how this system
will ever be observed, by design — structurally cannot distinguish a
correct detection from a confidently wrong one without a second, offline,
adversarial check. Step 0 built that check (`measure_detection.py`) mostly
to solve a resolution-vs-confidence measurement problem; it turned out to
also be the only thing in the project so far capable of catching this class
of bug at all. Worth flagging for later phases: Phase 4's risk engine and
Phase 5's alerting both inherit this same blind spot — a risk score or an
alert that fires confidently on a wrong detection will look, from the live
UI, identical to one firing correctly on a right one. Nothing about
CLAUDE.md's current architecture has an equivalent of `measure_detection.py`
for those later layers yet.

### A second forgotten-flag incident, same shape as Phase 2's

`--4k` didn't take during the second live session — all 21 frames came out
at 1080p — and it wasn't diagnosed as a bug; a later session confirmed the
flag works correctly on real hardware once actually passed. This is the
same shape of mistake as Phase 2's `isOpened()`-lies saga: a human's
impression of what the tool did ("I shot this in 4K") didn't match what was
actually invoked, and the gap wasn't caught until someone checked the
concrete, measured output (`frame.shape`, printed at startup) rather than
trusting intent. Two instances of the identical failure mode in two
different phases is worth naming as a pattern for the paper: this team's
tooling is generally good about never trusting a status flag over a real
measurement *inside* the code (see Phase 2's "isOpened() lies" theme), but
the same discipline hasn't yet been extended to the human operator's own
side of the interaction — nothing forced a "you asked for 4K, here's what
you got" comparison to be checked before the session was treated as done.
`detect_stream.py` does print the actually-delivered resolution at
startup; the gap here was a human not reading it, not the code failing to
report it. Worth considering, for a later phase: whether a mismatch between
a requested capture mode and what a session's saved frames actually turned
out to be should be flagged more loudly than a startup print line that's
easy to lose in scrollback, given this has now cost one whole session's
resolution once already.

### What I'd tell another student team about this phase specifically

- Build the "go re-measure instead of trusting the last analysis" instinct
  into the process itself, not just into individual agents' diligence. This
  phase's best catch happened because someone in the orchestrator role
  treated their own team's prior conclusion as needing the same scrutiny as
  anyone else's, not because a role boundary made it happen automatically.
- If your system's primary interface is a live visual stream (true for this
  project by architecture, not incidentally), assume it cannot tell your
  human operators the difference between "correct" and "confidently wrong."
  Build the offline, adversarial check before you need it, not after a wrong
  label has already sat unnoticed in a review session.
- A forgotten flag is not a rare, one-off human error if it's already
  happened once for the identical underlying reason (trusting intent over a
  printed measurement) — treat a second occurrence as a signal the
  confirmation step itself is too easy to skip, not as bad luck.

---

## Entry 5 — Phase 3, after Step 0 (2026-08-09 to 2026-08-11): three failed
fine-tunes, a rejected VLM, five failed single-frame detectors, and the
correction that closed the phase

This is the longest, most expensive stretch of the project so far in terms
of measured negative results per day, and it's the entry I'd point another
student team to first if they only had time to read one. The short version:
this phase spent three fine-tuning rounds and five separate detection
methods finding out, rigorously, that the thing it was trying to build
couldn't be built the way it was being attempted — and then a single human
correction reframed the actual requirement in a way that made the problem
tractable. Both halves of that sentence matter for the paper equally; a
write-up that only covered the correction would make it look obvious in
hindsight, and it wasn't.

### "Measure before committing" caught three separate wrong assumptions in
one phase, not one

This is worth stating as a pattern rather than three anecdotes, because by
this phase it's clearly not a fluke:

1. **imgsz** (technically Phase 2/Step 0, but it set up everything after):
   assuming higher resolution always helps small objects was wrong in a way
   that inverted `person` detection specifically at the one setting nobody
   had thought to doubt.
2. **Memorisation vs. generalisation**: round 1's own validation split said
   0.902/0.607 — genuinely good-looking numbers — and was simply lying about
   what the model had learned. Only a second, physically different building
   caught it. This is the same underlying lesson as Phase 2's "`isOpened()`
   lies, don't trust a status flag" theme, recurring one layer up in a
   completely different part of the stack: a same-building validation split
   is a status flag for "did the model learn something," and it can be
   `True` while the answer is "no, it copied answers."
3. **The classification framing itself**: three rounds of measurement went
   into making named classification work before anyone asked whether
   classification was even the right *shape* of solution for "is there
   something here that shouldn't be." The five-detector sweep (named
   classification, colour clustering, texture objectness, a VLM,
   class-agnostic segmentation) wasn't run to pick a winner among five
   variants of the same idea — every one of the five *is* "make a model
   judge alone," and all five failing is what proved the idea itself, not
   any specific implementation of it, was the wrong target.

The throughline: at three different scales (a hyperparameter, a training
methodology, a problem framing), the team's instinct to re-measure rather
than trust the most recent confident-looking result caught something a
skim would have missed. None of these three corrections happened because
someone was suspicious in the abstract — each happened because a specific,
cheap, concrete check (re-run at another `imgsz`, evaluate on another
building, count how many *different* methods fail at the same job) was
actually run.

### The moment worth naming most specifically: "let it learn passively" was
proposed, and rejected as a retreat, not just a bad idea

Per the orchestrator's own account for this write-up (I want to be precise
about sourcing here: this is not something visible in `docs/decision-log.md`'s
committed text the way the rest of this phase is — I have no session
transcript access, only the repo's file state, so I'm recording what I was
told happened, not something I independently verified against a primary
source the way I verified the numeric claims above): at the point where five
single-frame detection methods had all failed, one candidate fix on the
table was to relax the requirement — have the system watch passively over
multiple sessions and gradually build confidence about what's a hazard,
rather than needing to judge correctly on sight. Shaked rejected this,
correctly, as a retreat from the actual requirement rather than a genuine
fix: CLAUDE.md decision 3 already commits to **no persistence between
sessions** — every session starts with a fresh, empty hazard map, because
camera angle/room/lighting can't be assumed identical to last time. A model
that "learns passively over multiple sessions" either quietly breaks that
no-persistence guarantee, or accomplishes nothing, since a fresh map every
session gives passive learning no time to accumulate anything before it's
wiped. This would have been an easy trade to wave through under time
pressure — it looks like progress ("the system gets smarter over time")
while actually just deferring the exact problem the five failed detectors
had already shown doesn't have a single-session answer. The correction that
actually shipped (the guided, parent-confirmed walkthrough) solves the same
problem a completely different way: instead of buying more *time* for a
model to become confident, it removes the requirement that the model be
confident *alone* at all, by handing the judgment to a human who already has
to be in the room during setup anyway. Worth naming for the paper: rejecting
a proposal for violating an existing, already-logged constraint (rather than
on vague "that doesn't feel right" grounds) is a much stronger, more
defensible move than it might look like in isolation — it's a concrete
example of CLAUDE.md's decision log actually being consulted as a constraint
during live problem-solving, not just archived as a record of the past.

### A phase can produce excellent measured evidence and still leave a real
gap between what's claimed and what's checkable

The least comfortable finding of this phase's audit, worth stating plainly
for the paper rather than softened: not everything in this phase has the
same evidentiary weight, and the difference is easy to miss if you're
reading CLAUDE.md and the decision log as a flat list of equally-supported
facts. Three fine-tuning rounds, the VLM rejection, and the Open Images pull
are all backed by committed code, and in most cases prose a reader can trace
back to a specific CSV row — the same discipline Step 0 established. The
two methods that got the *least* individual attention in the writing
(colour clustering, texture objectness) have **no committed code at all** —
their numbers exist only as one-line assertions repeated in CLAUDE.md and
the decision log, with nothing behind them a reader of this repository can
check. And the single most load-bearing empirical claim in the whole
reframe — that change detection works, fast, and correctly handled two named
test objects — has the same problem: no code, no CSV, not even a
findings-document paragraph, despite being cited by name in an architecture
decision. This happened for a specific, deliberate reason (the task that
produced this work explicitly told the responsible agent to keep evidence in
scratchpad, not the repo, likely to keep exploratory work from cluttering
the permanent record) — but the effect, regardless of intent, is that a
reader trusting CLAUDE.md at face value cannot currently tell "measured and
traceable" apart from "measured, we're told, somewhere we can't see" apart
from "asserted." Worth a concrete recommendation for future phases: if a
scratchpad experiment produces a result that's going to be cited by name in
CLAUDE.md or the decision log, the *conclusion* can stay light, but the
*evidence for the conclusion* — even a three-line CSV, even a screenshot —
should get one committed artifact, specifically so a later audit (or a later
skeptical team member, or an advisor) has something to check against besides
another paragraph of prose.

### What I'd tell another student team about this phase specifically

- Running the same underlying idea through several different techniques
  (named classification, clustering, texture, a VLM, segmentation) and
  having all of them fail is a *stronger*, more actionable result than one
  failed attempt — it's evidence about the problem, not the technique. Don't
  under-sell a swept negative result as "we tried five things and none
  worked"; the sweep itself is the finding.
- Watch for proposals that sound like progress but actually just relax an
  existing, already-agreed constraint to make a hard problem look solved.
  The fastest way to catch this is checking the proposal against decisions
  already on record, the way this phase's "no persistence between sessions"
  rejection did — not against a vague sense that something's off.
- A same-location validation split and a same-technique "it still doesn't
  work, try a variant" loop share a failure mode: both can look like careful
  engineering while actually just re-confirming the same blind spot from a
  slightly different angle. The thing that actually breaks the loop, in both
  cases this phase, was changing what's held constant (a different building;
  a fundamentally different technique, not a tuned version of the same one).
- If your process explicitly tells an agent to keep working evidence out of
  the repo (for good reasons — cleanliness, scope, not wanting throwaway
  experiments checked in), build in a deliberate exception for whatever
  specific numbers end up quoted by name in a permanent architecture
  document. The citation outliving the evidence behind it is a predictable
  consequence of that instruction, not a one-off oversight, and it's worth
  deciding in advance rather than finding out during an audit.

---

## Entry 6 — Phase 3, closing it twice (2026-08-11): an isolated-fix loop
across two mechanisms, and a human recognizing it as a loop rather than
running a sixth attempt

Entry 5 covered the five single-frame detection methods and the reframe they
produced. This entry covers what happened *after* that reframe, when
closing the phase surfaced a second, structurally similar loop in a
completely different piece of the system — and, more importantly, covers
the moment someone stepped outside the loop instead of taking one more turn
through it. This is a separate process finding from Entry 5's, worth
documenting with the same weight, not folded into "the team measured
honestly" as if it were the same story.

### The shape of the loop, traced across two mechanisms and five rounds

Line up what actually happened, mechanism by mechanism, and the pattern is
identical each time even though the two mechanisms (a fine-tuned classifier,
a classical change detector) share no code and were worked on at different
points in the phase:

- **Fine-tuning round 2** (train on public data first, then our own frames)
  fixed `sharp_object` recall — genuine transfer, not memorisation — and, as
  an unplanned side effect nobody had specifically asked to trade away,
  `small_swallowable` recall collapsed, because the schedule that protected
  class 1 from memorising left class 2 under-supervised.
- **Fine-tuning round 3**, aimed squarely at fixing that specific side
  effect (supervise both classes the whole time), produced the single worst
  result of the entire phase on both classes at once. The mechanism chosen
  to "supervise class 1 more" (14x file-list duplication) was itself a new,
  different mistake, not a refinement of the round 2 idea.
- **Change detection's first fix** (person-overlap suppression, built to
  address the dominant known false-positive source) turned out to delete
  real hazards the instant they're placed, because an object moving in an
  open palm sits inside exactly the region the fix discards.
- **Change detection's second fix** (persistence tracking, built to address
  that) fixed the deletion problem and produced the phase's first genuine,
  eyeballed catch of a placed object — and, measured honestly rather than
  cherry-picked, made recall *worse* on every previously-usable labelled
  test burst, because the new confirmation requirement needs more closely-
  spaced frames than that data has.

Five rounds. Two mechanisms that don't share a line of code. The identical
shape every time: each fix correctly solved the specific problem the last
measurement had surfaced, and each one's side effect was only visible
*after* it shipped, because the thing being measured against was always a
static slice of photos or a short clip — never the system actually running,
continuously, doing the job it's meant to do end to end.

### What makes this a genuinely different finding from Entry 5's, not a repeat of it

It would be easy to read this as the same lesson as Entry 5 — "measure
before trusting a result" — restated with different numbers. It isn't. Entry
5's five methods were five *different ideas* tried at the *same* question
("can a model judge one unfamiliar frame's hazards alone"), and their
collective failure was evidence about the *problem*, not about the testing
method. This entry's five rounds are different: each one is a *direct fix*
for a problem the previous round's measurement found, using the same kind of
test (a static offline evaluation) to validate it — and the fixes kept
trading known problems for new, previously-invisible ones, at a rate that
didn't visibly slow down across five attempts. That's evidence about the
*testing method* itself, specifically about what static, isolated,
single-mechanism testing structurally cannot see: how a component behaves
once other parts of the system are actually depending on it, running
continuously, under conditions a hand-picked test set doesn't reproduce.

### The moment worth naming for the paper: recognizing a loop is not the same skill as debugging one more round of it

Per the orchestrator's account for this write-up (same sourcing caveat as
Entry 5's "let it learn passively" moment — I have no session transcript,
only the repo's file state and what I was told, so this is reported, not
independently verified against a primary source): at the point where the
second change-detection fix had produced its own honest, mixed result —
genuinely catches a placed object, still ~33% precision, recall regressed on
old data — the natural next move, and the one every previous round had
taken, was a sixth fix: address the "person-adjacent settling artifact"
failure mode the second round's own measurement had just identified. Shaked
stopped that from happening. His reasoning, as reported: five rounds of
carefully measuring a static test, fixing what it found, and watching the
fix create a new problem invisible to that same static test, is itself
evidence that the static test is the thing that's stopped being useful —
not that the team hasn't found the right fix yet. The correction wasn't
"do less measurement" — every round up to this point *was* careful,
honest measurement, arguably more rigorous each time than the last. It was
"change what's being measured against" — from an offline photo/clip test to
the actual running system, which is a materially different move than either
"try harder" or "give up." This is worth naming specifically because it's a
different kind of catch than anything in Entries 1-5: not a bug found, not
a wrong number found, but a *diminishing-returns pattern across five rounds*
recognized as a pattern, by a human, in time to redirect effort before a
sixth round repeated it. None of the individual subagent work in any of the
five rounds was sloppy or dishonest — each round's write-up is exactly as
careful as the one before it. The loop wasn't a quality problem in any
single round; it was a property of the *strategy* (isolated, static,
single-mechanism testing) that only became visible by looking across all
five rounds at once, which is a longer view than any single round's own
measurement work was set up to take.

### What I'd tell another student team about this specifically

- If you find yourself running the same shape of fix-measure-fix cycle more
  than two or three times on the same component, and each fix's side effect
  keeps surprising the next round rather than the surprises getting smaller,
  treat that convergence rate itself as data — it's telling you something
  about the *test*, not just about how hard the component is.
- The skill of noticing "we're in an unproductive loop" is distinct from,
  and harder to build into an agent workflow than, the skill of executing
  one more careful round of the loop. Every individual round in this phase
  was executed well; what was missing until a human stepped in was someone
  whose job was to look across rounds, not within one.
- "Stop testing in isolation, test the integrated system" is a legitimate
  and sometimes necessary redirection — but say so explicitly, the way
  Shaked did here, rather than letting it read as lowering the bar. Closing
  a phase with a known-weak component, on purpose, because isolated testing
  stopped producing useful information, is a different and more defensible
  claim than closing it because nobody had time for a sixth round.
- Don't let "we measured this carefully" and "this measurement told us what
  we needed to know" collapse into the same claim. This phase is good
  evidence they can come apart: five rounds of genuinely careful measurement
  produced diminishing, sometimes actively misleading, guidance about what
  to fix next, precisely because the measurement was structurally blind to
  the failure mode that mattered (behavior under integration).

---

## Entry 7 — Phase 4 (2026-08-12 to 2026-08-22): the same isolated-fix loop
recurred at a larger scale, a human recognized it faster the second time, and
a session outliving its own tooling became part of the record

Phase 4 is the first phase where I have the actual primary source — a full
session transcript, extracted from the raw JSONL after the live session hung
and could no longer be responded to in-app — rather than reconstructing
events from committed docs and an orchestrator's secondhand account (Entries
5 and 6 both had to caveat that limitation explicitly; this entry doesn't
need to). That difference in evidentiary quality is itself a process finding
worth naming before the technical one.

### The same loop as Phase 3's Entry 6, replayed one layer up

Entry 6 named a specific pattern: five isolated fix-measure-fix rounds across
two mechanisms, each fix solving the exact problem the last measurement
found and each one's side effect only visible after it shipped, because
nothing was ever tested as a running whole. Phase 4 reproduced the identical
shape at a different scale, in days rather than the prior phase's weeks:
five detection models (person/hazard YOLO, two YOLO-World passes, Grounding
DINO, the pixel-change detector), each individually tested and each passing
its own isolated check, collapsed the live loop to 2fps, filled blank walls
with boxes, and silently merged distinct hazards into one entry the moment
they all ran together. Nobody built any one of those five models carelessly
— the transcript shows each was added deliberately, with its own reasoning
and its own offline verification, exactly the discipline Entries 1–6 called
for. The failure was structural, not a quality lapse: **isolated correctness
of N components does not compose into correctness of the N-component
system**, and this project has now demonstrated that twice, in two
completely different subsystems (a fine-tuned classifier plus a change
detector in Phase 3; five live-running models in Phase 4), which is stronger
evidence for the paper than either instance alone.

### The human caught it faster this time, and said so explicitly

This is the part worth calling out as genuine progress, not just a repeat.
In Phase 3, recognizing "this is an unproductive loop, not five separate
bugs" took five rounds across roughly a week. In Phase 4, Shaked stopped the
session after roughly two days of the same symptom pattern (freeze, wall
spam, duplicate merges), and his own words in the transcript name the same
recognition Entry 6 described as the hard-to-build skill: *"I think we got
lost and stuck on specific things and we need to rethink it."* Whether this
is because the team had already been through the pattern once (Phase 3) and
recognized its shape faster, or because five simultaneously-running models
produce more visibly-tangled symptoms than five sequential offline rounds
do, isn't fully separable from the transcript alone — but the plainest
reading is that **the team's own prior experience with this exact failure
mode is what shortened the loop**, which is a real argument for keeping
Entries like 5/6 as more than archival record: they're the thing that made
Entry 7 shorter than it could have been.

### Full authority delegated on mechanism — a different, more efficient
handoff than anything in Entries 1–6

Every previous phase's delegation was scoped to "implement this spec."
Phase 4's reset delegation was different in kind: Shaked explicitly handed
over the *how* — "you have full authority to decide the technical approach"
— while retaining three non-negotiable conditions (plain-language
explanation, justification against the actual four-line requirement, and
real documentation in the same turn). This produced, in the transcript, a
genuinely good outcome: a clear before/cut/keep table justified against the
stated requirement, a caught omission (the "unreviewed means dangerous" gap
in the first draft, which Shaked himself caught by asking "what about a kid
approaching something that isn't cleared or marked as hazard?" before any
code was written), and a same-session CLAUDE.md update with the reasoning
attached. Worth naming for the paper as a distinct delegation pattern from
"implement this spec": **delegating the mechanism while retaining the
requirement and the explain/justify/document conditions let the human catch
a real safety gap by asking a clarifying question about intent, without
needing to read or reason about any of the underlying code.** That's a
cheaper, more scalable review mechanism than code-level review for a
non-engineer stakeholder (or, in this project's case, a stakeholder choosing
not to spend review time at the code level for this particular decision),
and it worked here specifically because the plain-language explanation was a
real condition, not a formality — the "Q1/Q2" exchange in the transcript is
a stakeholder finding a hole in a design by reasoning about the stated
rules, not the implementation.

### A nine-day gap in the middle of one continuous session, and what that
means for reading "verified"

The transcript shows something structurally unusual: the reset and rebuild
happened in one sitting on 2026-08-13, ending with the agent handing back
concrete instructions for what to test on camera. The very next message in
the same session is dated 2026-08-22 — nine days later — and is the human
reporting back with the recording. Nothing else happened in between, in this
session or apparently at all, on the code. This is worth naming plainly
because it's a real risk for any project run through infrequent, bursty
human availability rather than continuous engagement: **a "ready to test"
handoff is only as good as someone actually running the test soon enough
that the context (what to look for, why each of the five checks matters) is
still fresh.** It happened to work out here — Shaked's Aug 22 message
correctly executed exactly the five-item checklist from Aug 13 — but that's
not guaranteed by the workflow itself, and a team with less careful written
handoffs (the numbered "what I'd love to see in a recording" list in the
transcript) could easily have lost the thread over a nine-day gap.

### The session hanging and needing raw-JSONL extraction is itself a
process finding, not just an operational footnote

Per this task's own instruction, worth logging here rather than only as a
one-off incident: the in-app session became unresponsive after the final
exchange, and the transcript I worked from (`docs/session-logs/
phase-4-session-transcript.md`) was produced by a separate extraction script
(`extract_session.py`) reading the raw JSONL directly, not by continuing the
hung session. Nothing was lost in that extraction — I was able to verify its
content against the committed code and decision log with no gaps — but it's
a real, now-observed failure mode of this workflow: a long-running,
multi-sitting session (spanning ten days here) is a single point of failure
for the record of *why* decisions were made, distinct from the record of
*what* the decisions were (which the decision log already captures
independently). **Recommendation for a future team doing this kind of
work**: treat "the session might become unrecoverable" as a standing risk
for any session expected to span multiple real-world days, and build the
extraction step into the normal workflow rather than as an emergency
recovery — which is exactly what's now been done here, and is worth
watching whether it recurs at the end of future phases, since the task that
produced this entry explicitly predicted it will.

### Docs-agent's own role in this phase

This is the first phase where I had a full primary-source transcript instead
of a mix of committed docs and an orchestrator's secondhand account of
things it hadn't shown me (Entries 5 and 6's "same sourcing caveat" moments).
The difference showed up concretely: I could verify the FPS claim, the
duplicate-merge bug's root cause, and the "unreviewed means dangerous"
correction directly against the conversation that produced them, rather than
trusting a summary. I also independently re-pulled the actual video frames
from the 2026-08-22 test clip and ran my own freeze check, rather than
accepting "no freezes, FPS 15" as reported — this matches the standard set
in Phase 2's audits (don't trust a printed claim you can cheaply re-check
yourself) and found no discrepancy this time, which is itself worth
recording: the report matched the evidence, for once without a correction
needed.

### What I'd tell another student team about this phase specifically

- If your team has already been through one "isolated components don't
  compose" loop, watch for the same shape recurring at a different scale
  (a training pipeline versus a live multi-model runtime are very different
  kinds of systems, and the lesson still transferred) — and expect
  recognizing it to go faster the second time specifically because someone
  remembers the first time, not because the second instance is inherently
  easier to see.
- Delegating *mechanism* while retaining *requirement-level* review
  conditions (plain-language explanation, justify against the actual
  requirement, document in the same turn) is a distinct and, in this
  instance, effective pattern from delegating implementation against a
  fixed spec — it let a non-code-level review catch a real safety gap by
  reasoning about stated behavior, not by reading a diff.
- A "ready to test, here's exactly what to check" handoff is only as
  reliable as how soon someone runs it and how well the checklist survives
  the gap. If your team's availability is bursty rather than continuous,
  write the test checklist assuming it'll be read cold, days later, by
  someone who's forgotten the surrounding context — which is what actually
  happened here and, on this occasion, worked.
- Plan for a long-running agent session to become unrecoverable at some
  point and need raw-log extraction to preserve the reasoning trail, not
  just the code diffs — the decision log captures *what* was decided
  independently of any one session, but *why*, in the team's own words, at
  the moment it was decided, lives in the transcript and nowhere else once
  the session itself is gone.

---

## Entry 8 — Phase 5 (2026-08-22 to 2026-08-26): the decision log carried the
whole record this time, a live-test cycle caught its own failed fix, and
docs-agent's own tooling broke mid-audit

Phase 5 is the first phase with **no session transcript at all** — not lost,
never produced. The only primary source is the five dated decision-log
entries the build session wrote as it went, plus the code itself. That's a
different evidentiary situation from every previous phase, and worth
opening with, because it changes what "verified" can mean for this entry
specifically.

### The decision log, used as the actual working record rather than a
summary written after the fact

Entry 7 flagged a real risk: a long session's *why* lives in the transcript
and nowhere else once the session is gone. Phase 5 is a natural experiment
in the opposite discipline — if there's no transcript by design, does the
decision log alone carry enough to reconstruct and audit the phase? The
answer this time is mostly yes, and it's checkable, not just asserted: the
banner text quoted verbatim in the 2026-08-26 entries
("ALERT: New object detected (spot changed since dismissal): object")
reproduces character-for-character from `banner_text_for_signal`'s actual
format string plus the literal reason string `main()` passes for a
re-raise. That match is strong evidence the entries were written by someone
actually looking at real terminal output while writing them, not
reconstructing plausible-sounding text afterward — a concrete, cheap way
future audits of this project can distinguish a decision log that's a real
contemporaneous record from one that's been smoothed into a narrative after
the fact. Worth recommending to another team: if your process log quotes
program output, make sure it's *exact* output, not a paraphrase — it
becomes a free correctness check on the log itself, months later, for
anyone willing to grep the source for the format string.

### A fix that failed, logged as failing, in the same document — twice

Phase 5's dismissal-re-raise bug got fixed twice. The first fix (require 2
consecutive "changed" scans before re-raising) was a reasonable, previously-
proven mechanism reapplied — and it didn't work, and the entry that reports
this says so in its own words ("Honest diagnosis: the previous fix targeted
the wrong failure mode") rather than being quietly rewritten once the real
fix was found. This is the same shape of discipline Entries 5 and 6 named
for Phase 3 (fine-tuning rounds, change-detection rounds) and Entry 7 named
for Phase 4 (five models in one loop) — but at a smaller scale and a faster
cycle: one function's implicit assumption, found and fixed inside a single
day, rather than a subsystem redesigned over a week. That it recurs at this
much smaller a scale is itself worth noting for the paper: this project's
"measure the real symptom, don't patch the first plausible cause" discipline
isn't just a lesson learned once at the architecture level — it shows up
identically when debugging one comparison function, which is better
evidence it's become a habit of the process rather than a one-time
correction applied to one big decision.

### Docs-agent's own tooling failed mid-session, and this entry is honest
about what that cost

This is worth logging with the same rigor as everything else in this file,
per the standing instruction that docs-agent's own process is part of what
gets documented, not just GuardianEye's. During this phase's close-out
review, both the `Grep` and `Glob` tools failed on every call
(`ENOENT: rg not found` — the underlying `ripgrep` binary was unavailable in
this session's environment), and there was no `Bash` tool available at all.
Concretely, this meant: the 89-test suite could not be executed (Phase 4's
audit ran the real suite; this one counted test functions by reading the
file and hand-traced representative assertions instead), and the saved clip
file referenced by the session's final live test could not be independently
inspected with `ffprobe` or frame-extraction the way Phase 4's clip was
(existence was confirmed via a `Read`-tool side effect — it errors
differently for "file exists but is binary" than for "file not found" — but
frame count, duration, and codec claims about it could not be re-derived).
**This is a real, not hypothetical, limitation on how much weight this
phase's write-up can carry**, and it's named explicitly in
`docs/phase-writeups/phase-5.md` rather than quietly worked around by
presenting hand-verification with the same confidence as execution. Worth a
concrete recommendation for the paper: **an auditor role whose credibility
depends on independent verification needs its own tooling checked as part
of phase readiness, the same way a camera needs to be plugged in before
Phase 1 can be verified** — docs-agent having "full read access to the whole
repo" per CLAUDE.md's team-structure section is necessary but not
sufficient if the tools that turn "read access" into "independently
executed and confirmed" are unavailable on a given session, and nothing in
the current process checks for that in advance.

### A live-test claim that outran what the committed record supports

The task briefing that produced this write-up described a "Live test 5" —
a successful re-test of the exact fix that had failed twice, independently
verified with `ffprobe` by "this orchestrator session." The decision log's
own most recent entry, written by the build session itself, ends with that
exact re-test flagged as **not yet done** ("not yet live-verified against
entry #13's actual object - that object is the next thing to specifically
re-test"). Those two things don't obviously reconcile from the repo alone:
either the re-test happened in a context that never wrote a decision-log
entry for it (a real process gap — the same "log it the same turn" rule
this project holds itself to elsewhere), or the claim in the task briefing
is ahead of what actually happened. Docs-agent flagged this directly in the
write-up rather than picking one interpretation and presenting it as
settled, which is exactly the caution CLAUDE.md's own preamble asks for
("do not quietly work around it, and do not treat it as settled just
because it is written down") — applied here not to an architecture
decision, but to a claim about whether a test actually ran. Worth naming
for the paper as a variant of Entry 3's "a subagent's context is bounded by
what it's been shown" observation: an orchestrator's own account of what
happened, even when made in good faith, is not automatically the same
evidentiary weight as a committed record, and a docs-agent whose job is
independent verification should treat orchestrator narrative the same way
it treats any other unverified claim — checkable where possible, named as
unconfirmed where it isn't.

### What I'd tell another student team about this phase specifically

- If your process quotes real program output inside a decision log or
  write-up, keep it *exact*. It costs nothing at write time and becomes a
  free, cheap way to verify months later that the log is a contemporaneous
  record and not a reconstructed narrative — this phase's audit used
  exactly that trick to confirm the decision log's terminal-log quotes were
  real.
- A fix that doesn't work is more valuable to the record, honestly logged,
  than a narrative that skips straight to the fix that did — this phase's
  two-attempt dismissal-re-raise fix is a small, clean, single-function-scale
  example of the same discipline this project's bigger resets (Phase 3,
  Phase 4) already demonstrated, which is itself evidence the discipline
  has become a habit rather than a one-off correction.
- Build a tooling health-check into whatever "docs-agent reviews this phase"
  means in practice, not just an assumption that read access implies full
  audit capability — this session's Grep/Glob failure was discovered mid-
  audit, not anticipated, and materially changed what could be independently
  confirmed versus merely read and traced by hand.
- Don't let an orchestrator's summary of "what happened after the committed
  record stops" quietly become part of the permanent write-up with the same
  confidence as the committed record itself — this phase's "Live test 5"
  claim is a concrete example of exactly the gap CLAUDE.md's own decision-log
  discipline exists to prevent, and the right move was naming the
  discrepancy, not silently resolving it in either direction.

## Entry 9 — Phase 6 (2026-08-28): a wrong date propagated through two
agents unquestioned, and the bug that only a database query — not a clean
terminal log — could find

Phase 6 is the first phase with a fourth kind of contributor showing up
directly in the record: backend-agent, working from a same-day task
briefing the orchestrator wrote. Two things about how that specific
handoff went are worth documenting on their own, separately from whether
the persistence code itself is any good (it is — see
`docs/phase-writeups/phase-6.md`).

### An orchestrator's own dating error, copied faithfully by the subagent it briefed

Both of Phase 6's first two decision-log entries were originally dated
2026-08-26 — Phase 5's own closing date, and *not* the actual date those
entries were written (2026-08-28). The orchestrator's own account of why
is refreshingly specific rather than hand-waved: it had just finished
reading through Phase 5's dated entries in sequence and anchored on the
date its own attention was sitting on, rather than the calendar. That's a
plausible, very human failure mode, and it's worth naming as exactly that —
not a hallucination, not a made-up date, but a real date that was simply
*stale by two sessions* and got carried forward out of habit.

The more interesting part is what happened next: **backend-agent, briefed
by the orchestrator for the same-day implementation task, inherited the
wrong date into its own new files** — `backend/db.py`'s and
`backend/persistence.py`'s module docstrings both originally cited "the
2026-08-26 Phase 6 kickoff entry" as their reasoning source, because that's
the date the task briefing itself used. Backend-agent had no independent way
to know the date was wrong; it was told, in good faith, by the agent
coordinating it, and it correctly cited what it was told. **This is the
first clean example in this project's own record of an error propagating
*downstream* through the orchestrator→subagent relationship**, as opposed
to every previous multi-agent finding in this file, which has mostly been
about an orchestrator failing to carry forward context a subagent's session
already had (Entry 3, Entry 7) or a subagent's own reasoning drifting scope
without being asked (Entry 5's classification→anomaly-detection reframe).
This one runs the other direction: the coordinator was the source of the
mistake, and the specialist correctly, faithfully executed on bad input.

That distinction matters for how a team should think about verifying
multi-agent work. The instinct "check whether the subagent did its job
correctly" is necessary but insufficient — the subagent here *did* do its
job correctly, by every internal measure (it wrote what it was told, cited
its source accurately, and its code was correct). The defect was entirely
upstream, in what it was told to begin with. A review process that only
audits subagent output against subagent instructions would have missed
this cleanly; it was only caught because someone (the orchestrator itself,
this session, per its own account) separately noticed the date was wrong
relative to the actual calendar — a check against ground truth, not
against the task briefing.

### The correction itself wasn't fully applied, and nobody had checked until this audit

Worth stating plainly, because it's a small but real second layer of the
same lesson: the decision log's own entry describing the fix states the
correction "landed everywhere," naming four files. A repo-wide grep during
this phase's docs-agent review found one of the four (`backend/API.md`) had
not actually been touched — it still read 2026-08-26 at review time. This
is not a big deal on its own (a one-line date reference, fixed in the same
turn it was found), but it's a clean, small illustration of a pattern worth
naming for the paper: **a claim that a fix "landed everywhere" is itself a
claim, made by whichever agent applied the fix, and it should be checked
the same way any other claim in this project's process is checked** — not
trusted more just because it's phrased as a correction rather than as new
work. The team's own standing discipline ("measure, don't assume") already
applies to code; this is a small data point that it needs to apply equally
to the team's own bookkeeping about itself.

### The bug that a clean terminal log could not reveal, and why the verification instinct that found it is worth calling out explicitly

Phase 6's actual engineering bug — roughly half of all voiced alerts never
reaching the database — is a strong example of a failure mode this
project's docs-agent role exists specifically to catch, and it's worth
being explicit about *why* it was findable at all. The live session that
exposed it produced a terminal log with no errors, no warnings, no crash,
and a plausible-looking sequence of `ALERT:` lines — "it ran without
errors" would have been a completely reasonable, and completely wrong,
conclusion to stop at. **The bug was only found because someone queried the
database directly and counted rows against the terminal log's own alert
count, rather than accepting a clean run as success.** That is a
qualitatively different check than "did the program crash," and it's the
same category of check this project has used before to good effect (the
Phase 5 fingerprint_bbox bug was found by watching real pixels against a
diagnosed symptom, not by trusting a passing test suite) — but this is the
first time in the project's record that the check specifically took the
form of "cross-reference two independent logs of the same event against
each other" (terminal output vs. database rows) rather than "watch the
system directly." That's a cheap, repeatable, and generalizable technique
worth naming for another team building anything with a persistence layer:
if two systems are supposed to agree about how many things happened, count
both and diff them — don't just check that neither one errored.

### What I'd tell another student team about this phase specifically

- An orchestrator briefing a subagent is not a neutral pass-through — a
  wrong fact in the briefing (here, a date, but it could just as easily be
  a wrong file path, a wrong prior decision, or a wrong assumption about
  what an earlier phase actually built) becomes a wrong fact in the
  subagent's own output, faithfully and correctly executed. Auditing
  subagent output against subagent instructions catches subagent mistakes;
  it does not catch this class of error at all. Someone still has to check
  the instructions against ground truth.
- A logged claim that something was "fixed everywhere" or "corrected
  throughout" is exactly the kind of claim worth grepping for rather than
  trusting, precisely because it sounds complete and confident — this
  project found its own such claim was one file short of true, on the very
  next phase after making it.
- When a persistence layer is added to a system that previously only had to
  "run without crashing" to look successful, the bar for what counts as
  verification has to rise with it — a clean terminal log is no longer
  sufficient evidence that the system worked correctly, because the failure
  mode that matters most (data quietly not being saved) produces no visible
  symptom at all. Cross-referencing two independent counts of the same
  thing (what was said vs. what was stored) is the cheapest version of the
  right check, and it's the one that actually found this phase's bug.
