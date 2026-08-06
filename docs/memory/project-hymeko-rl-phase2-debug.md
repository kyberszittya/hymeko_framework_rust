---
name: project-hymeko-rl-phase2-debug
description: hymeko_rl Phase 2 PPO is PAUSED mid-debug; PPO degrades the BC policy; top lead is the truncation-bootstrap bug; tree code is unverified
metadata: 
  node_type: memory
  type: project
  originSessionId: 5fd53482-77c4-4262-be8f-95b4bcaaa64f
---

`hymeko_rl/` (the Kato-collab robot RL line — see [[project-kato-collaboration-grasping]])
Phase 2 PPO is **paused mid-debug** (2026-06-18). Phase 0 (the kinematic-hypergraph bridge,
HSiKAN policy reads the compiled hypergraph) and Phase 1 (BC reaching) **work and are
reported**. Phase 2 PPO does **not yet beat BC** — it degrades a good policy.

**Diagnostic state (verified):** critic cold-start under the dense-negative reward corrupts
the shared backbone → critic warm-up (`ppo.py` `_warmup_critic`) stops HSiKAN degrading but
PPO still doesn't *improve* past BC. Action std was too high; lowered. BC's ~0.27 reach is
the covariate-shift floor (not a bug).

**✅ RESOLVED 2026-06-20: the truncation-bootstrap bug is ALREADY FIXED in `ppo.py`.** Verified by
reading `_collect` (ppo.py:108-118): it separates `terminated`/`truncated` and folds `gamma*V(next)`
into the reward on truncation; the checkpoint report's UPDATE 2 confirms PPO then improves
monotonically on reach. Do NOT re-chase this — it is the FIRST thing future sessions wrongly suspect.
(Older note, now obsolete, said this was the "top untested lead"; it was fixed same checkpoint.)

The real live blocker moved to **Galambos planar grasp (0 goals)** — see
[[project-galambos-reward-shaping]] for the measured diagnosis (the arms approach to ~0.06 m but
NEVER contact; reward is ~100% a near-flat disk→zone `pull` term; no dense fingertip→coin gradient).
Remaining reach-line TODO if revisited: revert any delta-action change, separate actor/critic if
needed, repair tests, multi-seed matched ablation, algebraic-entropy-feedback test
([[project-kato-collaboration-grasping]]). Emitter/PyO3 bugs B-003/004/005 in `docs/BUGS.md`.
