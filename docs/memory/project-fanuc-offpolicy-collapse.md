---
name: project-fanuc-offpolicy-collapse
description: "FANUC off-policy refine collapses the BC clone to 0 even WITH the warm-start bridge — root cause is gross under-budgeting; fix = TD3+BC anchor + ≥1e5 steps (user chose \"Both\")"
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

Measured 2026-06-24 (`reports/2026-06-24-policy-gifs-fanuc.log`): BC→DDPG/TD3 on FANUC pick-place collapsed
refine_place to **0.0 in all four cells** (ddpg/td3 × hsikan/mlp) from a working BC (noisy 0.12–0.75 at n=8 eval),
**even with** `warm_start=True`, `critic_warmup=2000`, `noise_scale=0.05`. The earlier "fix verified on Galambos
(0.083→0.083)" did NOT transfer to FANUC.

**Not a wiring bug** (verified): `behaviour_clone(ac,...)` and `train_offpolicy(ac,...)` act on the SAME module;
the clone IS carried into the off-policy actor. **Root cause: gross under-budgeting** — `refine=12000` steps on a
620-step episode ≈ ~19 episodes, far below off-policy norms (1e5–1e6). The critic never becomes meaningful and
~10k actor updates against a garbage Q walk the actor off the BC manifold.

**Fix (user chose "Both"):** (a) **TD3+BC anchor** — add `λ·MSE(actor, demo)` to the actor loss (`bc_coef` field in
`ddpg.py`, non-core), the standard Fujimoto&Gu offline/warm-start fix so refine can't destroy the clone; (b) a real
budget (≥1e5 steps). The anchor is reused by the Kato CTDE refine [[project-kato-dual-discriminator-plan]].

**Why:** prevents re-declaring the warm-start bridge as "the fix" — it is necessary but insufficient on sparse
reward without the anchor + budget. **How to apply:** the working lever on these tasks stays **BC / BC→PPO**; treat
off-policy refine as needing the anchor AND 1e5+ steps before expecting improvement. Each 12k cell ≈ 30min CPU, so
1e5 ≈ ~4h/cell — smoke one cell before fanning out. Supersedes the optimistic note in
[[project-hymeko-rl-phase2-debug]]; see also [[project-rl-algorithm-roadmap]].
