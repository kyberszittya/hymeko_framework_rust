# Galambos coin-grasp: HSiKAN vs params-matched MLP — the real-topology structure test

**Date:** 2026-06-22 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Summary
The experiment cart-pole could not give: does the signed-hypergraph (HSiKAN) backbone earn its keep on a task
with **real topology** (the Galambos two-arm coin-grasp — 6 link vertices, a genuine kinematic hypergraph, 4
actions)? Single seed, 300 iters PPO from `galambos_strategy.hymeko` (reverse curriculum), HSiKAN vs a
parameter-matched MLP.

| net | params | init | final | best | last-10 mean |
|---|---|---|---|---|---|
| HSiKAN | 28 745 | −65.9 | −6.4 | 10.0 | −26.4 |
| MLP-96 | 28 521 | −68.7 | −9.6 | **28.9** | −22.7 |

## The result — no structure advantage (a second negative, now on real topology)
Both learn the coin task well (≈ −67 → single digits / teens). **HSiKAN does not beat the params-matched
MLP**: they are comparable within the (large) run-to-run noise, and the MLP reached higher peaks (best 28.9 vs
10.0) with a marginally better last-10. So on **both** tasks tested fairly — the 2-vertex cart-pole and the
6-vertex coin-grasp — reading the signed kinematic hypergraph buys nothing measurable over matched capacity.
This is consistent with the 2026-06-21 cart-pole control and the 2026-06-18 rotor-vs-MLP-embed ablation: a
matched-capacity baseline closes the apparent structural gap each time.

## Caveats (do not over-read — it is preliminary, not a firm verdict)
- **Single seed, high variance.** The curves oscillate wildly (HSiKAN final −6.4, last-10 −26.4, best 10.0
  span 36 points; only ~6% of HSiKAN's last-50 iters are positive). One seed cannot settle a
  variance/robustness question — **multi-seed is required** before this is firm. The honest status is
  "comparable, no advantage observed," not "structure proven useless."
- **The structure may be too thin.** Two simple 2-link arms is a small, near-tree graph; richer topology
  (cycles, more joints, the Kato tensorised hand) might carry information a 6-vertex chain does not.
- **Fixed-incidence HSiKAN.** This uses the kinematic adjacency as a fixed buffer; the learned-incidence
  `signedkan` variant and the structural-entropy-feedback seat (SAC `α·H`) are the *un-tested* ways to
  "exploit the structure further" (the user's intuition) — not yet run.
- **Single-env PPO**, not the strongest learner; SAC (off-policy, the strongest baseline) on this task may
  differ and is cheaper.

## Implication
The *current* way of using the structure (message-passing over a fixed kinematic adjacency, HSiKAN as a
drop-in backbone) is not winning on control. That is not a dead end — it sharpens the real question: the value,
if any, is in a *different exploitation* of the structure (learned incidence, structural-entropy feedback,
richer-topology / harder tasks), not in HSiKAN-as-a-better-MLP. The standing rule holds: always run the
params-matched control; never credit structure on a single seed.

## Artifacts
- Policies: `checkpoints/galambos/ppo_hsikan.pt`, `checkpoints/galambos/ppo_mlp96.pt`. HSiKAN stored as
  `data/nn/galambos_hsikan_policy.hymeko` (provenance: ppo/hsikan/galambos).
- Curves: `checkpoints/galambos/ppo_hsikan.json` (full returns); MLP summary in this report.
- Bug found + fixed mid-run: the first matched-MLP attempt passed `obs_dim=8` (per-vertex feat) instead of
  `48` (flat = n_vertices×feat) — an mlp-baseline shape error; corrected.

## Addendum — quadruped (14-vtx, richest topology, 2026-06-22)
Extended the structure test to a quadruped (HyMeKo scaling fixture `quadruped_d3_t0`, emitted + floating base
+ floor + forward-velocity reward; 14 link vertices, 13 actuated joints — genuinely branching topology). A
**40-iter PPO smoke** (single seed):

| net | params | init | final | best |
|---|---|---|---|---|
| HSiKAN | **27 035** | 4.5 | 6.7 | 11.2 |
| MLP-112 | 33 403 | 5.3 | 7.3 | 11.1 |

Both learn (the quadruped starts walking). **HSiKAN ties the MLP's peak with ~18% fewer params** — but the MLP
was *over*-parameterized (33k vs 27k), and this is a **40-iter smoke** (locomotion needs hundreds; absolute
reward ~11 = barely moving), single seed. So: competitive, a *hint* of param-efficiency on rich topology —
not a win. The minimal QuadEnv was a one-off experiment script (floating base via `<freejoint>`, floor seated
below the lowest geom at qpos=0); formalize into `hymeko_rl/env/` only if a proper run shows signal.

**Three-task picture:** cart-pole (2-vtx) tie · coin-grasp (6-vtx) tie · quadruped (14-vtx) param-competitive.
HSiKAN is **never worse, never decisively better** — parity with matched capacity, a whiff of param-efficiency
that surfaced only as topology got rich. The decisive test is now a **longer, properly param-matched,
multi-seed quadruped run** — if HSiKAN's efficiency edge widens with harder coordination, that's the first
real signal.

## Follow-ups
1. **Multi-seed** (≥5) HSiKAN vs matched-MLP on coin-grasp — the only thing that makes this a firm result.
2. **SAC on coin-grasp** (off-policy, sample-efficient) — both backbones, with the matched control.
3. The "exploit further" line: **signedkan** (learned incidence) and **structural-entropy feedback** on a
   richer-topology task — where the structure could actually matter.
