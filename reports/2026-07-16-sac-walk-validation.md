---
title: SAC-from-scratch walking validation + kato15 campaign (the "collapse" resolved)
date: 2026-07-16
status: local validation COMPLETE (decisive) — kato15 campaign PREPARED (unreachable from here)
core_yaml_touched: none
---

# SAC-from-scratch walking — the "so-called collapse" resolved

## Question

The recurring TD3+BC "collapse" (coin-toss, Aibo standing, the CIP grid): is it a fundamental wall, or an
artifact? And can these bodies be walked with **RL**? The CIP grid + DAgger validation pointed at two
confounds — **TD3+BC value drift** and **weak scripted demonstrators**. This isolates them.

## Local validation (cheetah, 200k steps, decisive)

`exp_sac_walk_validation.py` — A/B on the demonstrator, pure SAC vs gait-warm-started SAC. Result
(`experiments/2026_07_16_sac_walk_validation/result.json`, figure `figures/2026-07-16-aibo/sac_walk_validation.png`):

| arm | forward dx | CIP propel-edge (leg⇒fwd) |
|---|---|---|
| scripted CpgGait | +0.04 | 0.00 |
| **SAC from scratch** | **+0.23** (6×) | **+0.84** |
| SAC + weak-gait warm-start | +0.01 (traps) | 0.00 |

**Two clean findings:**
1. **Pure SAC from scratch WALKS** — dx +0.23 (6× the scripted shuffle) with a strong causal propel-edge
   (+0.84: leg motion genuinely drives forward speed). Already +0.62 at 20k. The "collapse" is **not** a wall
   — it was TD3+BC value drift + a weak anchor. RL-from-scratch un-collapses and locomotes.
2. **The weak-gait BC warm-start TRAPS** — dx +0.01, *worse than the shuffle*, propel-edge 0. The BC anchor
   dragged SAC into the demonstrator's dead basin — a direct reproduction of the arc's **BC-warm-start-traps**.
   This also explains the CIP-grid degeneracy: Aibo/humanoid were 0 not because the actors were equal, but
   because their weak demonstrators never produced forward motion for the metric to see.

So the honest resolution: **pure SAC from scratch** is the lever; do **not** warm-start from a weak gait.

## kato15 campaign (prepared — GPU-bound, not a CPU job)

`exp_sac_walk_campaign.py` + `scripts/kato15/run_sac_walk.sh` scale it: **pure SAC from scratch × {flat,
structural} × 3 bodies (Aibo goal-reach, biped humanoid, planar cheetah) × 5 seeds**, 500k steps, measuring
forward dx + the CIP propel-edge — asking the original thread's question (does the **structural/relational
actor** beat flat) at scale, on policies that actually walk. The **structural SAC actor constructs** (verified,
no fallback). Pure SAC only (warm-start traps). JSONL-resumable, live-logged, `systemd-run MemoryMax=16G` (§4),
`HYMEKO_DEVICE=cuda`, the handoff's `.venv_stand` (torch 2.11+cu128).

**Why kato15:** 500k × 2 actors × 3 bodies × 5 seeds = 30 cells × ~500k steps. The Mac CPU did *one* cheetah
cell at 200k in 22 min; the full grid is a GPU job (kato15 ~1135 steps/s, ~40× local). **kato15 is unreachable
from this Mac** (SSH timeout — VPN/network), so it's packaged for the user to sync + launch:
`bash scripts/kato15/run_sac_walk.sh full` (sync command in the script header).

## Files

New: `hymeko_rl/experiments/{exp_sac_walk_validation, exp_sac_walk_campaign}.py`, `scripts/kato15/run_sac_walk.sh`,
figure `sac_walk_validation.png`. Earlier this session: `exp_aibo_cip_walk`, `exp_cip_verification_campaign`,
`env/reward.py` (+`vertical_bounce`), the Aibo restore. 56 tests pass, ruff clean; no CORE.YAML touched.

## Provenance

Branch `integration/fanuc-pick-place-canonical`. cheetah SAC 200k, seed 0, CPU, 22 min/arm. Env: Python 3.11
`.venv`, mujoco 3.10.0, macOS Apple-Silicon. Artifacts under `experiments/2026_07_16_sac_walk_*/`.
