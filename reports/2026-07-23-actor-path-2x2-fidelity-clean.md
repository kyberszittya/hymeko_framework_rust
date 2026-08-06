---
title: Actor-path fidelity-clean 2x2 — the critic learns proximity but its action-ranking is not physically faithful
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: MARKOV_CRITIC_REPAIR_CONFIRMED + CRITIC_ACTION_RANKING_NOT_PHYSICALLY_FAITHFUL + TRUST_REGION_PROTECTIVE_BUT_NOT_SUFFICIENT + PHYSICAL_POLICY_CONVERSION_NOT_ACHIEVED
tags: [coin, markov, actor-update, critic-exploitation, physical-rollout, bc-anchor, trust-region]
---

# ACTOR_PATH_FIDELITY_CLEAN_2x2_V1 — the physical rollout overturns "BC+trust block a good gradient"

The critique demanded a physical proof: ΔQ is critic-space; only a physical rollout or K6 ladder shows real policy
improvement. This experiment supplies it — one trained Markov critic, four candidate actors from a 2×2
{Q-only, Q+BC} × {no-trust, trust}, each multi-step (100 steps), then a **physical rollout of every candidate on a
graded-ladder panel**. The result changes the conclusion.

## Setup
- Same trained Markov critic (Arm A, 4000 updates). Graded-ladder panel = the 31-state transport/braking/settling dev
  bank — **genuinely graded**: pi_0 K1 0.226, K3 0.226, K5 0.226, **K6 0.161** (K1 > K6). (The late_dev panel used for
  training is flat, k1=k3=k5=k6, and cannot detect conversion.)

## Result — Q improves but the policy physically regresses
| candidate | ΔQ (critic) | counter-use | K1 | K3 | K5 | **K6** | exit | dwell |
|---|---|---|---|---|---|---|---|---|
| **pi_0** | — | — | 0.226 | 0.226 | 0.226 | **0.161** | low | — |
| Q-only / no-trust | **+29** | 0.024 | 0.258 | 0.097 | 0.097 | **0.097** | **0.355** | 0.74 |
| Q+BC / no-trust | +30 | 0.044 | 0.194 | 0.032 | 0.032 | 0.032 | 0.355 | 0.35 |
| Q-only / trust | +0 | 0.0005 | 0.194 | 0.129 | 0.129 | 0.097 | 0.032 | 0.84 |
| Q+BC / trust | +0 | 0.0005 | 0.194 | 0.129 | 0.129 | 0.097 | 0.032 | 0.84 |

- **Unconstrained Q-only:** +29 in critic-space and *does* use the counter — but **physically K6 drops 0.161 → 0.097 and
  target-exit jumps 0.032 → 0.355 (~11×, not "triples")**. It gets *more* coins to enter (K1 0.258 > 0.226) but
  destabilizes them (holds collapse, exit ~11×). This is **critic exploitation**, not policy improvement.
- Adding BC makes it worse still (K6 0.032). The trust-constrained candidates ≈ the trained actor (0 new accepts), K6
  0.097 — already below pi_0. Note this is the decisive evidence the trust region is **not sufficient**: the actor was
  trained *with* the trust region (only small steps accepted), yet those accepted steps cumulatively still moved K6
  0.161 → 0.097. The trust region blocks the big 100-step exploit but the accumulated accepted small steps drift
  physically wrong.

## Corrected verdict
- **`MARKOV_CRITIC_REPAIR_CONFIRMED`** — the critic represents distance-to-terminal (strict-conditioned Q rises toward
  K6; the actor *can* be pushed to use the counter). The representation repair is real and stands.
- **`CRITIC_ACTION_RANKING_NOT_PHYSICALLY_FAITHFUL`** — following the critic's action-gradient raises Q (+29) but
  **physically regresses** K6 (0.161→0.097) and raises exit (→0.355). The critic's *value ordering near the terminal* is
  correct, but its *which-action-is-better* ranking is not physically faithful off pi_0 (classic TD3 over-optimism when
  the actor moves off-distribution; here the move drives entry at the cost of stability).
- **`BC_GRADIENT_CONFLICT_CONFIRMED`** — Q-grad vs BC-grad cosine −0.395, and Q+BC/no-trust is physically worse than
  Q-only/no-trust (0.032 vs 0.097). The strict-blind BC anchor does oppose the counter — but it is not the load-bearing
  blocker of a *good* move, because the unconstrained move is not good.
- **`TRUST_REGION_PROTECTIVE_BUT_NOT_SUFFICIENT`** — the trust region blocked the big 100-step exploit (0/100 accepts;
  protective against the regression the unconstrained gradient caused) — but it is **not sufficient**: the actor was
  trained *with* the trust region, and the small steps it *did* accept still cumulatively degraded physical K6
  (0.161 → 0.097). So the remedy is not to remove the trust region but to add a **stricter physical/certificate-based
  actor-authorization gate** alongside it.
- **`PHYSICAL_POLICY_CONVERSION_NOT_ACHIEVED`** — no cell physically beat pi_0's K6 on the graded panel.

This **overturns the previous `BC_ANCHOR_AND_TRUST_REGION_BOTH_BLOCK`** framing (the actor-update-path-audit, critic-space
only): the physical test shows the blocked gradient was not physically beneficial. The critique's strictest form — keeping
`PHYSICAL_POLICY_CONVERSION_PENDING` until a physical test — was exactly right, and the pending test now resolves to
NOT_ACHIEVED because the critic's action-ranking is the wall, not the trust region.

## Honest limitation
The graded panel (transport/braking/settling) is partly **out-of-distribution** for this critic (trained on
target_entry/braking/settling; target_entry ≠ transport), and the in-distribution panel is flat (cannot test conversion).
So "critic action-ranking not physically faithful" is demonstrated on a graded-but-partly-OOD panel; the large
off-distribution actor move (Q+29 via ~6.7 action-norm drift) would exploit critic errors even in-distribution, and the
physical trend (more entry / more exit / lower K6) is a coherent mechanism, not OOD noise — but a fully in-distribution
graded panel is the clean follow-up.

## What this means for the arc (complete, agreed causal chain)
hidden strict-counter → non-Markov critic → strict-counter made visible → terminal-proximity **value-ordering** restored →
**action-ranking still not physically faithful** → actor learns a Q-exploit (more entry, worse stability) → the trust
region protects against the large regression **but is not sufficient** (accepted small steps still drift physically wrong).
So the binding wall is **critic action-fidelity**, not the Markov state and not (primarily) the actor-update contract. The
problem has narrowed to a normal RL question: *how to learn a local action-ranking that does not conflate arriving-at-target
with staying-stably-in-target?*

## Next lever — LOCAL_ACTION_RANKING_FIDELITY_V1 (not CQL/SAC/TD3 yet)
A short, fully in-distribution diagnostic on {target_entry, braking, settling_dwell} states only (removes the OOD confound):
perturb pi_0 by tiny action offsets — ±actuator basis and the critic-gradient direction, at norms {0.005, 0.01, 0.02, 0.04}
— and physically roll out each. Measure critic ΔQ, physical return, dtz, speed, dwell, exit, K1/K3/K5/K6, and the
**critic-ranking ↔ physical-ranking correlation**. Decision tree: locally-faithful-but-bad-at-large-drift → small-step TD3 +
physical gate; entry-faithful-but-settling-unfaithful → phase-conditioned / separate settling critic; unfaithful-even-locally
→ more on-distribution critic data / paired-difference ranking / pessimistic critic; locally-faithful → matched TD3 & SAC are
finally warranted. Keep the trust region as a safety wall throughout.

## Files & commits
- entry: `experiments/…/rl_entry/coin_actor_path_2x2.py`; result `…/actor_path_2x2_v1.json`.
- refines `reports/2026-07-23-actor-update-path-audit-v1.md` (frozen; not modified) — the physical test supersedes its
  BOTH_BLOCK framing.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Arm A critic 4000 updates; graded panel = transport_dwell
dev (31 states). No new campaign; no reward/task change. Single-thread. No CORE.YAML items.
