# Guarded CQL actor smoke — FAIL, new mechanism (Q-scale runaway, not ranking inversion)

**Date:** 2026-07-07 · Git SHA `4320202` (dirty). Non-core. One seed, CPU. Full safety stack PASS throughout
(schema + provenance). **No SAC / residual / multi-seed; v2b reward unchanged; monitor external.** This is the one
guarded CQL actor smoke unlocked by the STRONG_PASS critic; it FAILED acceptance.

## Result — FAIL, but the failure mode changed

The STRONG_PASS gate worked and the critic ranking *held* through actor training — yet the actor still collapsed,
by a **different** mechanism than the baseline smoke.

| stage | ft_dom | monitor_pass | monitor_score | exploit | reward |
|---|---:|---:|---:|---:|---:|
| baseline (this DAgger checkpoint) | **0.75** | 0.417 | 0.278 | 0.0 | −115.2 |
| after CQL actor smoke | **0.0** | 0.0 | −0.249 | 0.167 | −234.1 |

**Acceptance: FAIL** (ft_dom / monitor_pass / monitor_score / no-body-driven-rise / q-not-inverted all fail; guards
pass). **Abort tripped: `delivery_collapse`** (ft_dom dropped 0.75 → 0.0). Violation: `coin_pushed_away_during_approach`.

## What went right (the gate + the critic ranking)

- **Critic pre-training reproduced STRONG_PASS**: margins exploit **12.71**, one-finger 7.78, ood_gap 16.28 — the
  harness gated on this before touching the actor (would have aborted a non-STRONG critic).
- **Provenance + tensor-contract PASS** the whole run; null baseline defined from the exact checkpoint (ft_dom 0.75).
- **The dagger>exploit ranking SURVIVED actor training.** After the smoke, critic Q = {dagger **−174.8** > exploit
  −193.7, one_finger −180.3}: `Q_flip_exploit_over_dagger = False`. CQL fixed the category-B inversion *and held
  it* — unlike the baseline smoke where the critic ranked the exploit above DAgger. The `inverted=True` flag is only
  a minor one-finger-vs-exploit swap, not the dagger-vs-exploit axis.

## What went wrong — Q-scale runaway under continued conservative pressure

The critic **Q deflated without bound**: `Q = −36.5 → −55.7 → −79.3 → −108 → −143 → −182` over the 6000 steps, and
the actor loss (`−Q + bc`) blew up in lockstep (`act = 36.6 → 181`). Keeping the CQL conservative penalty
(`logsumexp_a Q − Q_data`, α=1.0) active during actor training pushes Q down on out-of-support actions; as the
actor explores, more of the action space becomes "OOD" and is deflated, so Q recedes faster than the actor can
climb it. The actor — even at `actor_lr=5e-5`, `policy_delay=2`, with a BC anchor (bc loss only 0.008 → 0.059) —
**drifts off the DAgger manifold chasing a receding Q landscape**, and delivery collapses. The drift is roughly
**uniform across phases** (APPROACH 0.447 / CONTACT 0.378 / DELIVERY 0.542; not concentrated in CONTACT/PUSH), i.e.
a whole-policy off-manifold drift, not a local contact error.

**So: passing the STRONG critic-ranking benchmark is necessary but STILL not sufficient.** CQL removed the ranking
inversion (the baseline pathology) and kept it removed — but introduced/exposed a *second* pathology, **Q-magnitude
runaway + off-manifold actor drift**, that a static-ranking benchmark cannot see. The BC anchor at this weight
cannot hold the actor against an unboundedly-deflating value.

## The 14 required fields

reward −234.1 · ft_dom 0.0 · monitor_pass 0.0 · monitor_score −0.249 · violation `coin_pushed_away_during_approach` ·
reward-vs-monitor **misaligned** (reward prefers exploit) · critic-vs-monitor **inverted** (one-finger vs exploit;
dagger still top) · tensor-contract **PASS** · policy-provenance **PASS** · actor ckpt `edf4fe81…` · anchor ckpt
`edf4fe81…` · selected DAgger stage d3 · reward file `galambos_task_deliver_v2b.hymeko` · env file `PlanarGraspEnv
v2 graded`. Critic tier at init: **STRONG_PASS**.

## Verdict + next branch

**RL stays frozen.** Per the directive, a failed CQL actor smoke means the next branch is **residual or
phase-gated RL, not more plain actor-critic** — and the mechanism points the same way:

- **Residual RL** (freeze the DAgger actor, learn a small *bounded* residual Δ with `‖Δ‖ ≤ ε`): structurally caps
  off-manifold drift, so a deflating critic cannot walk the policy away from DAgger. This directly addresses the
  observed failure (uniform off-manifold drift).
- **Phase-gated correction**: only correct in the phase(s) the monitor flags, freezing the rest.
- Orthogonal tuning that would likely help but is *not* the structural fix: a **Lagrangian / annealed CQL α** (the
  constant α=1.0 kept active during actor training is the proximate cause of the Q runaway) — worth trying inside
  the residual formulation, not as another plain actor-critic run.

Do **not** run: SAC, plain TD3/CTDE-TD3+BC again, multi-seed. A tiny 3-seed confirmation is moot — the one-seed
smoke failed on a structural (not stochastic) mechanism.

## Files / artifacts

- Harness `scratchpad/v2_cql_actor_smoke.py`; checkpoints `experiments/v2_cql_actor_smoke/{cql_critic0_s1.pt,
  cql_actor_smoke_s1.pt}`; result `experiments/v2_cql_actor_smoke/results.json`; log `scratchpad/cql_smoke.log`.
- Code added this task: `train_offpolicy(critic_regularizer=…)` hook (ddpg.py, non-core, eager path) +
  `cql_regularizer` factory (critic_repair.py) + `classify_critic` 3-tier gate (critic_benchmark.py). All tested
  (15 unit tests, ruff clean).

**Status:** critic ranking benchmark + 3-tier margin gate in place; CQL is STRONG_PASS and actor-safe *by ranking*;
the one guarded CQL actor smoke **failed via Q-scale runaway / off-manifold drift**. Imitation baseline
(MLP+DAgger, ft_dom 0.452 deployable / 0.75 this checkpoint) stands. Next: residual / phase-gated, gated on the
same full stack + STRONG_PASS critic.
