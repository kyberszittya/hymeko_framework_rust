# RL sequential overnight — BC-anchor cross-port, coin confirm, quadruped-as-collaboration de-risk

**Date:** 2026-06-30 (work ran 00:14–03:55 JST, unattended/throttled)
**Context:** user said "after [the BC-anchor A/B] finishes go sequentially" through the scenario-roadmap list
(`docs/plans/2026-06-30-rl-scenario-roadmap/`). Three steps banked; the fourth held for a steer (below).

## Summary

| step | result | verdict |
|---|---|---|
| 1. BC-anchor verdict (pick-place) | HSiKAN s0 0.5/0.125→**0.875/0.5**, s1 0.375/0→**0.625/0.25**; MLP 0/0=0/0; mixture 0.75/0.375→**0/0** | anchor **helps HSiKAN**, neutral MLP, **hurts mixture** — *not universal* |
| 2. coin-toss BC-anchor confirm | single 0.333/0.125 (med 0.23), collab 0.208/0.167 (med 0.19) | **neutral** — same ~0.2 band as baseline |
| 3. quadruped-as-collaboration de-risk | 4-leg CTDE med **−84** (s1 degrades) vs flat HSiKAN **−33**, fewer params | **trains but loses on goal-reach** (non-cyclic) |
| 4. closed-loop topology matching | **held** | needs user steer (see fork) |

## The throughline (honest)

The collaborative / structural framing pays **where the task structure makes coordination load-bearing, and
(for the anchor) where there is a good policy to protect** — *not universally*:

- **BC anchor is scenario-specific.** It prevents PPO from collapsing a *good* BC policy. Pick-place had a good
  BC policy (0.5–1.0) → the anchor is a clear win for HSiKAN. Coin-toss's BC policy was already modest (~0.25) on
  a harder 2-arm task → little collapse to prevent → neutral. And it *hurt* mixture (gate × anchor instability).
- **The collaborative reframe is not a free win.** On goal-*reach* (quadruped) the 4-leg CTDE trains (s0
  −124→−66) but loses to flat HSiKAN and is unstable across seeds (s1 −92→−102, the multi-agent non-stationarity).
  On coin-toss, collab ≈ single on delivery. Neither task has the structure the decomposition should exploit.

This is consistent with the matching-law / structure-is-load-bearing thesis: structure pays where coordination
matters. The de-risk result is therefore a useful **negative control**, not a refutation.

## What this measured vs. what it implies (per CLAUDE.md)

- **Measured:** the three verdicts above (multi-seed where applicable).
- **Inferred:** goal-reach and the modest-BC coin task lack the coordination structure the collaborative/anchor
  levers exploit — so the neutral/negative outcomes are expected, not bugs.
- **Hypothesis (the real test, NOT run):** a **gait** reward (forward velocity, periodic) makes coordination =
  a **gait cycle = holonomy**, the one place the collaborative framing + the **rotor** (the holonomy reader,
  acc 1.0 vs additive 0.5) should win where flat ties/loses.

## The fork (user steer)

- **(A) Gait reward.** Author a gait `.hymeko` reward on the quadruped (reward = the lever; `.hymeko` is the
  source of truth) → re-test 4-leg CTDE (+rotor) where coordination is load-bearing. *The direct continuation of
  the quadruped/humanoid-as-collaboration idea; de-blocks the humanoid milestone.*
- **(B) Closed-loop topology matching** (step 4 as listed) — take the supervised matching law into RL. Independent
  thread.

## Artifacts

- `reports/figures/bc_anchor_ab.png` — BC-anchor A/B (pick-place, 6 arms).
- `reports/figures/quad_ctde_curves.png` — 4-leg CTDE vs flat HSiKAN learning curves.
- `reports/overnight/quad_ctde.json`, `reports/overnight/coin_anchor_s{0,1}.log` — raw.
- `reports/gifs/coin_toss_success.gif` — a successful collaborative delivery (coin into zone, seed 20002).
- Memory: `project-quadruped-collaboration-derisk`.
- Prototype: `scratchpad/quad_ctde_prototype.py` (`leg_partition`, `build_quad_ctde`) — promote to a module only
  on fork (A).

## Tests / hygiene

- BC-anchor cross-port (`exp_collaborative.py` `--bc-coef`): ruff + mypy clean, smoke passed.
- `rl_prescreen.py` + test: ruff + mypy clean (latency screen; states it does NOT screen PPO-stability).
- The quad prototype lives in scratchpad (de-risk only) — **not** added to the tree pending fork (A); no
  CORE.YAML touched anywhere this session.

## Open issues

- 4-leg CTDE seed instability (s1 degrades). If fork (A): add the BC anchor / per-agent value baselines to the
  CTDE refine to tame non-stationarity before reading any rotor result.
- Promote the quad prototype to a real module (`hymeko_rl/collaborative.py` generalized partition) only if (A).
