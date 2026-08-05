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
