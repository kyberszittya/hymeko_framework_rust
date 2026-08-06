# Fast RL sanity suite — k-bandit, grid world, hexa world, collaborative

**Date:** 2026-06-29 · **Modules:** `hymeko_rl/sanity_rl.py`, `hymeko_rl/sanity_worlds.py` · **Tests:** 13 (all green).

## Why

Every RL env in the repo is **MuJoCo** (`arm_reach`, `pick_place`, `planar_grasp`, `quadruped`, …): physics, 200–620 steps, **minutes-to-hours** per verdict. A backbone regression, a wiring bug, or an architecture comparison should not cost an hour. This suite is **physics-free, in-process, and fast**, exercising the *real* `build_policy` backbones (`mlp` / `hsikan` / `sa_hsikan` / `mixture`, with the `cr` vs `cr_cheby` cell) across the three RL regimes — bandit, sequential, cooperative. It is the RL analog of the supervised `structural_probe`.

## The four testbeds

| testbed | module | regime | tests | speed |
|---|---|---|---|---|
| **k-bandit** | `sanity_rl.ContextualBandit` | 1-step, signed ring | accuracy + **B=1 deploy latency** | ~seconds |
| **grid world** | `sanity_worlds.LatticeNav("grid")` | multi-step, 4-neighbour | credit assignment | ~tens of s |
| **hexa world** | `sanity_worlds.LatticeNav("hex")` | multi-step, 6-neighbour | credit assignment; **grid-cell / path-integration substrate** | ~tens of s |
| **collaborative** | `sanity_worlds.CollabBandit` | 2-agent (CTDE) | coordination | ~seconds |

**k-bandit** — context on a signed ring `(N, feat)`; the optimal action is a pooled function of the context. `flat` target = pooled context (no graph); `structural` = pooled signed 1-hop `mean(A·x)`. Trained by REINFORCE.

**grid / hexa world** — a point agent reads per-vertex features `[landmark − agent, is_goal]` over the lattice hypergraph and navigates to a goal landmark over a short horizon (multi-step REINFORCE, vectorised over a batch of agents). `hex` is the 6-neighbour lattice — the **grid-cell / path-integration** substrate (ties to the backlog's grid-cell probe, P2).

**collaborative** — two agents see the context; reward `−‖(a_A + a_B) − target‖`: only the *sum* matches, so the agents must coordinate (shared reward = centralised-training). The fast analog of the cooperative Galambos delivery.

## Results

**k-bandit — accuracy + deploy latency (B=1, flat target, one fast run):**

| variant | reward | deploy (B=1) | params |
|---|---|---|---|
| mlp | −0.157 | 0.12 ms | 3 847 |
| **sa_hsikan** (B^L collapse) | −0.158 | **0.91 ms** | **775** |
| **hsikan-cheby** (deploy-Chebyshev) | **−0.125** | **3.63 ms** | 8 071 |
| hsikan-cr (vanilla) | −0.133 | 4.67 ms | 8 071 |
| mixture | −0.146 | 5.17 ms | 11 833 |

**grid / hexa / collab — backbones learn (median final reward, 0.0 = optimal):**

| world | mlp | hsikan / sa_hsikan |
|---|---|---|
| grid nav | −0.67 | −0.61 (hsikan) |
| hexa nav | −0.65 | −0.62 (sa_hsikan) |
| collaborative | −0.11 (coordinated) | −0.12 (coordinated) |

## What the suite already exposes

1. **The recent enhancements, in seconds.** SA-HSiKAN's B^L collapse: **~5× faster deploy + 10× fewer params** than vanilla HSiKAN. CR-Chebyshev's deploy path: **~22% faster at B=1**, better accuracy, same params (via `set_deploy_mode`, train-CR / deploy-Chebyshev).
2. **Why HSiKAN ties MLP on linear tasks** — a *linear* graph target (A·x) is trivially representable by an MLP, so flat + structural-linear both tie (~−0.25). Reproduces the standing tie finding in seconds. The genuine *accuracy* discriminator must be a **nonlinear** graph property (cycle parity / Z₂ holonomy) — the holonomy-discriminator toy (P1).
3. **The launch-bound, re-demonstrated** — multi-step nav makes HSiKAN ~3× slower than MLP over the rollout (the per-forward cost compounds across the horizon), which is exactly where SA-HSiKAN / batched-rollout matter.

## Usage
```
python -m hymeko_rl.sanity_rl     --target flat|structural
python -m hymeko_rl.sanity_worlds --world grid|hex|collab
```

## Next
- **`--target holonomy`** (k-bandit) + the holonomy-discriminator toy — the nonlinear task where HSiKAN *should* beat MLP (the real accuracy discriminator; P1).
- **Grid-cell population structure** on the hexa world — does a StructuralActor path-integrate into toroidal/grid cells (the comp-neuro "crown jewel", P2)?
- A `--gif` of an agent navigating the grid/hex world (§9 animated output) for the deck.
