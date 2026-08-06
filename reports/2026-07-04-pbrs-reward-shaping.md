# Potential-based reward shaping for the galambos coin-toss

**Date:** 2026-07-04 · **Branch:** `hymeko-neuro-migration` · **Scope:** replace the farmable galambos reward with a provably farm-proof potential-based one, certified by the oracle and put up against baseline in an RL A/B.

## The problem (and that it is textbook)

The galambos reward farmed: in the 4-hour RL run, delivery *fell* (0.14 → 0.04) while `both_contact` *rose* (0.021 → 0.060). The oracle (`reward_oracle.certify`, ms-fast) confirmed it analytically — the reward-optimal trajectory oscillates in/out of the zone to milk the per-step `in_zone·10 + both·5` **annuity** without ever completing delivery.

A **novelty search** (2026-07-04) placed this precisely: it is the canonical **specification-gaming** example — DeepMind's *Coast Runners* boat, shaped to hit green blocks, circles forever hitting the same blocks instead of finishing the race. Our annuity is the same failure. So this is **not a new phenomenon**, and the fix below is **not a new method** — it is the correct application of an established one.

## The principled fix: PBRS (Ng, Harada & Russell 1999)

A shaping term of the form
```
F(s, s') = γ·Φ(s') − Φ(s)
```
provably **leaves the optimal policy invariant**: summed over a trajectory it telescopes to `γ^T·Φ(s_T) − Φ(s_0)` — path-independent. Therefore it **cannot be farmed**: dipping into the zone and back out nets ≈ 0 (only the `γ<1` residual), so there is no per-step annuity to milk. It supplies the *same dense guidance* the farmable per-step potentials gave (progress toward the goal), with the farming incentive removed by construction. This is exactly what HPRS (hierarchical PBRS from task specs) uses for robotics.

## What we built

- **`reward.py`**: `_pbrs_shaping(env, Φ_now, attr) = γΦ(s')−Φ(s)`, with `Φ(s)` tracked per-episode on the env (first step returns 0, initialising `Φ(s0)`; `γ=0.99` matches the training discount for exact invariance). Two potentials:
  - `zone_progress`  — `Φ = −disk_to_zone`  (progress toward the zone), replacing the farmable `in_zone` annuity;
  - `grasp_progress` — `Φ = −max(left_tip, right_tip)` (both fingertips close at once), replacing the farmable `both_contact` annuity.
- **`planar_grasp_env.reset()`**: clears the tracked potentials so each episode re-initialises `Φ(s0)`.
- **`data/robotics/galambos_task_pbrs.hymeko`** (reward-in-hymeko): the sparse aligned **task** (`terminal_deliver 30` one-shot on completion + `oob 2`) plus the two **PBRS** terms (`zone_progress 20`, `grasp_progress 20`). Weights scale the potentials; any positive scaling is still a valid potential, so the optimum is unchanged — only the gradient magnitude.
- **`meta_reward.hymeko`**: vocabulary entries for the two potential terms.

## Oracle certification + farm-proof check

| Check | Result |
|-------|--------|
| baseline reward optimum | **farms** (`certify.delivers = False`) |
| sparse task (`terminal_deliver + oob`) | **delivers** (`True`) — the optimum PBRS preserves |
| `galambos_task_pbrs` | **delivers** (`True`) — shaping nets to 0 in the oracle; theorem preserves the task optimum |
| telescoping | first step `0`; a progress step pays `γΦ(s')−Φ(s) > 0`; **in→out nets ≈ 0** (unfarmable) |

## Tests

`test_reward.py`: `test_pbrs_shaping_is_progress_and_telescopes_unfarmable` (exact `γΦ(s')−Φ(s)`, progress pays, oscillation nets ~0, 0 on non-planar); `test_pbrs_hymeko_is_task_plus_potential_shaping`. `test_reward_oracle.py`: the de-annuitization/farming certifications. **29 passed.** Env-integration smoke: PBRS reward runs a full rollout, rewards finite, per-episode tracking correct. No CORE.YAML.

## The A/B (in flight)

`exp_galambos_coord_ab.py --treatment-hymeko galambos_task_pbrs.hymeko --treatment-name pbrs` — baseline (farms) vs PBRS, collab off-policy (sa_hsikan, TD3+BC), 1000 demos (cached) / 500k steps / 3 seeds, delivery scored on the **baseline** env (the true task, not the shaped reward). Task ID `bmeincoxx`. This is the *internal* validation: does the ms-certified PBRS fix transfer to real RL and lift delivery off the farming ceiling?

## Honest positioning

- The farming problem and the PBRS solution are both **established** — this report applies a known principle correctly, replacing an ad-hoc de-annuitization with the provably-invariant version.
- The oracle remains a useful **internal certifier** (catch farming before RL), not a headline contribution.
- Any *paper* angle is the broader integrated frame — a declarative reward substrate that is simultaneously PBRS-shapeable, planner-certifiable, and manifold/graph-searchable — which needs its own novelty search before it is claimed.
