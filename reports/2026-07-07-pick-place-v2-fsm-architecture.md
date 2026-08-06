# v2 FANUC pick expert as a learnable FSM / option structure (architectural direction, 2026-07-07)

**Status: design direction (recorded, not yet implemented).** The deterministic v2 controller stays the initial
expert / baseline policy; this note frames it as an explicit FSM/option structure so it can *later* be optimized
by RL — without RL ever overriding the hard validators. Ties to `project-hymeko-planner-roadmap` (Phase 5
RL-bounded search) and `project-fsm-structured-rl`. **No RL training now; no BC/DAgger until v2 passes the
clearance gate.**

## Core idea

Do not frame the v2 expert as a hand-written trajectory forever. Build it as an explicit FSM of phases, each with a
declared contract, so the deterministic controller is the *baseline policy* and RL later learns **over** the FSM
(which phase, which bounded parameters, small residuals, search priority) — never raw behaviour from scratch.

## Phases (each a first-class option with a contract)

`HOME_RETRACT_OR_PRESHAPE · TRANSIT_ABOVE_TABLE · ABOVE_OBJECT_ALIGN · VERTICAL_DESCENT · GRASP · LIFT ·
PLACE_TRANSIT · PLACE_DESCEND_RELEASE · RECOVERY/RETRY (later)`

Per phase, expose:
- **preconditions** — when the phase is legal to enter;
- **target waypoint / controller primitive** — the commanded goal for the phase;
- **exit condition** — commanded/seed/waypoint progress (sag-independent) that advances to the next phase;
- **failure condition** — what trips RECOVERY/RETRY;
- **hard safety checks** — IK validity, collision/contact legality, positive clearance, phase preconditions;
- **logged metrics** — commanded & physical clearance, contacts, sag, per-phase timing, success flags;
- **tunable parameters** — hover height, waypoint offsets, dwell time, IK seed choice, approach speed, grasp
  timing, retry threshold.

## What RL learns later (over the FSM, not raw control)

1. **Phase-selection / option policy** — which phase or recovery branch to execute next.
2. **Parameter policy** — bounded tuning of the per-phase params above.
3. **Residual correction** — small bounded corrections around the deterministic waypoints, only inside
   phase-specific safety envelopes.
4. **Search guidance** — prioritise waypoint branches, allocate sampling budget, choose the fallback planner branch.

## Hard rule (the invariant)

RL may **choose among legal options and tune bounded parameters, but it may not override the hard validators.**
Validators remain the sole authority: IK validity · collision/contact legality · positive clearance · phase
preconditions · object/task success metrics. This is the same soft-bound-vs-hard-proof rule as the planner
roadmap's Phase 5 — RL accelerates and focuses; correctness is the validators'.

## Long-term model

FSM / HyMeKo hypergraph **declares** the legal skill structure → classical validators **enforce** physical
legality → RL **learns** which legal transition/parameter/recovery choices work best.

## For now (near-term, deterministic)

- Keep the deterministic v2 controller as the baseline expert.
- Refactor it into the explicit FSM above: a phase enum + per-phase (precondition, primitive, exit, failure,
  safety, metrics, params) records; the current `_expert_action_v2` branches map onto these phases already.
- **Expose state/phase/parameters cleanly** and **log everything needed for later RL** (commanded & physical
  clearance, contacts, sag, per-phase timing) — the CIP/DirectLiNGAM diagnostics layer
  (`project-cip-lingam-rl-diagnostics`) consumes exactly these logged rollout variables.
- **Do not train RL yet. Do not run BC/DAgger until the v2 expert passes the clearance gate**
  (`hymeko_rl/eval/pick_clearance.py`, the frozen safety authority).

## Sequencing note

This FSM refactor is the **next task**, kept separate from the in-flight clearance fixes (one change at a time, each
smoke-evaluated in isolation). The current blocker is the physical sag (gains) — the FSM refactor does not fix it;
it structures the controller so that, once clearance passes, RL-over-FSM (roadmap Phase 5) becomes the natural
next lever.
