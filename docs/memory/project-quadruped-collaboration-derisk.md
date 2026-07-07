---
name: project-quadruped-collaboration-derisk
description: "2026-06-30: Hajdu's \"frame quadruped/humanoid as collaboration\" idea DE-RISKED. 4-leg CTDE trains but LOSES to flat HSiKAN on goal-REACH (non-cyclic): median -84 vs -33, s1 degrades (non-stationarity). Goal-reach is the WRONG test; GAIT (cyclic=holonomy) is the real one. A negative control consistent with structure-load-bearing-only-where-coordination-matters."
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

Hajdu's provocative 2026-06-30 idea: frame quadruped/humanoid control as a **collaborative** scenario (each limb = an agent, like the 2-arm coin-toss). De-risked on the existing `QuadrupedGoalEnv` (8 DOF = 4 legs × [hip,knee]) by building a **4-leg CTDE** policy (shared HSiKAN backbone + 4 per-leg heads, split `(2,2,2,2)`, reuses `CollaborativeGalambos`/`CTDEActorCritic` — only the partition changes from arm→leg). Prototype: `scratchpad/quad_ctde_prototype.py` (`leg_partition`, `build_quad_ctde`). The CTDE `action_mean` returns the full 8-d action → drop-in for single-agent `train_ppo`.

**Measured (30 iters × 2 seeds, real budget):** 4-leg CTDE final median return **−83.67** (s0 −124→−66 learns; s1 −92→−102 **degrades**), flat HSiKAN **−32.81** (both seeds stable −35/−31). CTDE has **fewer params** (18.8k vs 28k). Plot: `reports/figures/quad_ctde_curves.png`; json `reports/overnight/quad_ctde.json`.

**Verdict (measured / inferred / hypothesis, per CLAUDE.md):**
- **Measured:** the CTDE *trains* (de-risk "does multi-agent train at all" = YES, s0 clearly learns) but **loses to flat** on goal-reach and is **unstable across seeds** (s1 degrades — the non-stationarity I flagged).
- **Inferred:** goal-*reach* is **non-cyclic** — there is no coordination/gait structure for the leg-decomposition to exploit, so the loss is expected. This is a clean **negative control**, consistent with [[project-topology-control-matching-law]] and [[project-gauge-holonomy-signed-hsikan]] (structure pays only where coordination is load-bearing).
- **Hypothesis (the real test, NOT yet run):** a **gait** reward (forward velocity, periodic) makes coordination = a **gait cycle = holonomy** → the place the collaborative framing + the **rotor** (the holonomy reader, [[project-fast-rl-sanity-suite]]) should *finally* win where flat ties/loses. Needs a gait `.hymeko` reward (edit the reward file, never in-memory — [[feedback-reward-definition-in-hymeko]]) on the quadruped, then CTDE+rotor vs flat.

So the idea is **viable but unproven**: the gate "does it train" passed; the payoff gate is a **cyclic/gait task**, which goal-reach is not. Next = build the gait reward and re-test. Don't claim collaboration helps quadruped control from the goal-reach numbers — it doesn't there, and that's the point.
