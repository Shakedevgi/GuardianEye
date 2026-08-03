# GuardianEye — Claude Code Workflow Guide

This is the "how do I actually press go" guide. Follow it in order the first
time; after that it's just: open Claude Code in this folder and talk to it.

## One-time setup

1. **Install Claude Code** if you haven't already (`npm install -g
   @anthropic-ai/claude-code` — but check `claude.com/docs` for the current
   install method before running this, since install steps do change).

2. **Get this folder onto your machine** with this structure already in place:
   ```
   guardianeye-project/
   ├── CLAUDE.md
   ├── PHASE_PLAN.md
   ├── WORKFLOW_GUIDE.md
   └── .claude/
       └── agents/
           ├── cv-agent.md
           ├── backend-agent.md
           ├── ui-agent.md
           └── docs-agent.md
   ```
   (All of these files were generated for you already — just drop them in.)

3. **Initialize git** in this folder if you haven't (`git init`, first commit)
   — the decision log and phase write-ups are much more useful with real
   history behind them.

4. **Open a terminal in this folder and run `claude`** to start your first
   session. This first session is the one that needs to "see" the new
   `.claude/agents/` folder for the first time — that's normal, it's a one-time
   thing.

## The rhythm, phase by phase

For each phase in `PHASE_PLAN.md`, in order:

1. **Kick it off** — tell the orchestrator (the main session) which phase
   you're starting and let it delegate. You don't need to name the subagent
   yourself every time (the `description` fields are written so Claude
   auto-routes), but you can force it if you want to be explicit. Example
   opening prompt for Phase 1:

   > "We're starting Phase 1 from PHASE_PLAN.md — foundation setup: dev
   > environment, git, and getting the USB camera streaming raw frames via
   > OpenCV. Use the cv-agent for this. Ask me anything you need about the
   > camera model or my machine before starting."

2. **Work the phase** — let the orchestrator and cv-agent/backend-agent/
   ui-agent go back and forth with you. Answer their clarifying questions.
   Test what they build as you go rather than only at the end.

3. **Close the phase** — once it demoably works, explicitly invoke docs-agent:

   > "Phase 1 looks done to me — use docs-agent to audit it against
   > PHASE_PLAN.md and CLAUDE.md, update the decision log for anything we
   > decided along the way, and write up the phase writeup. Then walk me and
   > Yahli through what we actually built."

4. **Don't start Phase N+1 until docs-agent has actually signed off** — if it
   flags gaps, go fix those first. This is the whole point of the gate.

5. **Mark the phase `[x]` in `PHASE_PLAN.md`** once closed.

## Ongoing habits (not tied to any one phase)

- Any time a real architectural call gets made mid-phase (not just at the end),
  ask docs-agent to log it right away:
  > "Log this decision: we're using X instead of Y because Z."
- If you or Yahli genuinely don't understand something an agent built, ask
  docs-agent to re-explain it differently — that's exactly its job, don't
  just nod along.
- Periodically (every few phases is reasonable) ask docs-agent to update
  `docs/agent-workflow-notes.md` with how the multi-agent setup itself is
  going — don't leave that entirely to the very end of the project.

## When something feels wrong

If an agent seems to be working outside its lane (e.g. cv-agent starts writing
Flet UI code), just say so — "that should be ui-agent's job" — and redirect.
The role boundaries in each agent's file are guidelines for Claude's routing
and for keeping your own mental model clean, not a hard technical wall; you're
allowed to correct it in the moment.

If you want to add a new agent later (e.g. splitting the risk-engine logic out
of cv-agent once it gets big), just create a new `.claude/agents/<name>.md`
file the same way these were made, and restart the session once to pick it up.
