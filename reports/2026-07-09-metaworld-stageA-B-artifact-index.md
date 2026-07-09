# MetaWorld reward-ablation Stage A→B — artifact index

**Date:** 2026-07-09 · closure commit `a92a8c6` · branch `hymeko-neuro-migration`
Technical map of every report, figure, JSON, GIF, mechanism file, command, and commit for the Stage A→B arc. All
paths are repo-relative and verified to exist (see the *Path verification* section).

---

## Reports

| Report | Content |
|---|---|
| [reports/2026-07-09-metaworld-reward-sot-lingam-sh-integration.md](2026-07-09-metaworld-reward-sot-lingam-sh-integration.md) | HyMeKo reward as SoT for MetaWorld; fidelity R² |
| [reports/2026-07-09-metaworld-reward-ablation-stageA.md](2026-07-09-metaworld-reward-ablation-stageA.md) | Stage A `mw_grasp_off` (NOT_SUPPORTED, negative) |
| [reports/2026-07-09-metaworld-reward-ablation-positive-control.md](2026-07-09-metaworld-reward-ablation-positive-control.md) | `mw_in_place_off` positive control + `mw_dist_off` secondary |
| [reports/2026-07-09-metaworld-reward-ablation-multiseed.md](2026-07-09-metaworld-reward-ablation-multiseed.md) | 5×N=80 median/IQR robustness (controls confirmed) |
| [reports/2026-07-09-metaworld-stageb-training-smoke-setup.md](2026-07-09-metaworld-stageb-training-smoke-setup.md) | Stage B gated harness + smoke run (plumbing) |
| [reports/2026-07-09-metaworld-stageb-training-result.md](2026-07-09-metaworld-stageb-training-result.md) | Stage B **single-seed** result (seed 0: 62.5% → 0%) |
| [reports/2026-07-09-metaworld-stageb-multiseed-result.md](2026-07-09-metaworld-stageb-multiseed-result.md) | Stage B 5-seed REINFORCE: success NOT robust; disagreement robust |
| [reports/2026-07-09-metaworld-stageb-ppo-multiseed.md](2026-07-09-metaworld-stageb-ppo-multiseed.md) | **Stage B 5-seed PPO: success refuted (both ~100%); disagreement survives** |
| [reports/2026-07-09-metaworld-stageA-B-kato-brief.md](2026-07-09-metaworld-stageA-B-kato-brief.md) | Executive brief (updated post multi-seed) |
| [reports/2026-07-09-metaworld-stageA-B-claim-discipline.md](2026-07-09-metaworld-stageA-B-claim-discipline.md) | 3-tier claim discipline (updated post multi-seed) |

## Stage A figures + JSON

| Artifact | Path |
|---|---|
| Stage A `mw_grasp` summary | `reports/figures/2026_07_09_19_35_cip_reward_ablation_stageA/reward_ablation_summary.json` |
| Positive-control panel PNG | `reports/figures/2026_07_09_poscontrol_cip_reward_ablation/positive_control_panel.png` |
| Positive-control JSON | `reports/figures/2026_07_09_poscontrol_cip_reward_ablation/reward_ablation_comparison.json` |
| **Multiseed aggregate JSON** | `reports/figures/2026_07_09_multiseed_cip_reward_ablation/reward_ablation_multiseed.json` |
| **Multiseed panel PNG** | `reports/figures/2026_07_09_multiseed_cip_reward_ablation/multiseed_panel.png` |

## Stage B figures + JSON + GIFs + mechanisms

Base dir: `reports/figures/2026_07_09_metaworld_stageb_real/`

| Artifact | Path |
|---|---|
| **Train JSON** (metrics + returns + comparison) | `reports/figures/2026_07_09_metaworld_stageb_real/stage_b_train.json` |
| **Comparison PNG** | `reports/figures/2026_07_09_metaworld_stageb_real/stage_b_comparison.png` |
| **Side-by-side GIF** (original ∥ off) | `reports/figures/2026_07_09_metaworld_stageb_real/stage_b_compare.gif` |
| Per-profile GIF — original | `reports/figures/2026_07_09_metaworld_stageb_real/original/rollout.gif` |
| Per-profile GIF — mw_in_place_off | `reports/figures/2026_07_09_metaworld_stageb_real/mw_in_place_off/rollout.gif` |
| Mechanism `.hymeko` — original (cross-view ✅) | `reports/figures/2026_07_09_metaworld_stageb_real/original/reward_mechanism_original_trained.hymeko` |
| Mechanism `.hymeko` — mw_in_place_off (cross-view ✅) | `reports/figures/2026_07_09_metaworld_stageb_real/mw_in_place_off/reward_mechanism_mw_in_place_off_trained.hymeko` |
| Checkpoint — original | `reports/figures/2026_07_09_metaworld_stageb_real/original/policy.pt` |
| Checkpoint — mw_in_place_off | `reports/figures/2026_07_09_metaworld_stageb_real/mw_in_place_off/policy.pt` |

### Multi-seed (5 seeds) — the robustness pass

| Artifact | Path |
|---|---|
| **Multi-seed JSON** (per-seed + aggregate + verdict) | `reports/figures/2026_07_09_metaworld_stageb_multiseed/stage_b_multiseed.json` |
| **Multi-seed panel PNG** | `reports/figures/2026_07_09_metaworld_stageb_multiseed/stage_b_multiseed_panel.png` |
| Seed-0 GIFs + mechanisms (representative) | `reports/figures/2026_07_09_metaworld_stageb_multiseed/seed_0/` |

### PPO multi-seed (5 seeds) — the decisive optimizer pass

| Artifact | Path |
|---|---|
| **PPO multi-seed JSON** | `reports/figures/2026_07_09_metaworld_stageb_ppo_multiseed/stage_b_multiseed.json` |
| **REINFORCE-vs-PPO synthesis PNG** | `reports/figures/2026_07_09_metaworld_stageb_ppo_multiseed/stage_b_optimizer_synthesis.png` |
| Seed-0 GIFs (both profiles succeed) | `reports/figures/2026_07_09_metaworld_stageb_ppo_multiseed/seed_0/` |
| PPO trainer | `hymeko_rl/experiments/stage_b_ppo.py` |

## Code

| Module | Role |
|---|---|
| `hymeko_rl/eval/cip/reward_ablation_metaworld.py` | Stage A: `ablate_reward_spec`, `run_reward_ablation_stage_a` / `_comparison` / `_multiseed`, `_condition` |
| `hymeko_rl/experiments/exp_metaworld_reward_stageb.py` | Stage B harness: config, gate, cert, env-override, BC, REINFORCE, `launch`, `post_eval` |
| `hymeko_rl/experiments/stage_b_eval.py` | Stage B post-eval: policy-rollout metrics, `evaluate_and_render`, GIF, `compare_profiles` |
| `hymeko_rl/tests/test_reward_ablation_metaworld.py` | Stage A tests (10) |
| `hymeko_rl/tests/test_metaworld_stageb.py` | Stage B tests (11) |
| `data/robotics/metaworld_reward.hymeko` | the reward SoT (terms `mw_in_place, mw_grasp, mw_near, mw_dist`) |

## Commands

Reproduce (read-only Stage A):
```
python -c "from hymeko_rl.eval.cip.reward_ablation_metaworld import run_reward_ablation_multiseed; \
from pathlib import Path; run_reward_ablation_multiseed('pick-place', batches=5, n=80, seed0=0, \
  out_dir=Path('reports/figures/2026_07_09_multiseed_cip_reward_ablation'))"
```
Stage B (training — run only with authorization):
```
python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --launch-training \
  --profiles original mw_in_place_off --total-env-steps 6000 \
  --out reports/figures/2026_07_09_metaworld_stageb_real
```
Post-eval a trained checkpoint (no training):
```
python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --post-eval \
  --profile original --checkpoint reports/figures/2026_07_09_metaworld_stageb_real/original/policy.pt
```
Dry-run validation (no training):
```
python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --dry-run --profiles original mw_in_place_off mw_grasp_off
```

## Commits (Stage A→B arc)

| SHA | Milestone |
|---|---|
| `8cbaaba` | HyMeKo reward SoT ↔ LiNGAM-SH mechanism pipeline |
| `a85e451` | LiNGAM-SH step 4A: per-tail mechanism weights |
| `23fecc5` | Stage A `mw_grasp` ablation (NOT_SUPPORTED, honest negative) |
| `022a0fc` | Positive control `mw_in_place` (SUPPORTED) + `mw_dist` secondary |
| `99bff8f` | Multi-seed 5×N=80 (controls confirmed) |
| `ec33cfb` | Stage B gated harness (no training) |
| `4122179` | Stage B smoke run (plumbing verified) |
| **`a92a8c6`** | **Stage B result: 62.5% → 0% policy collapse (closure)** |

## Path verification

All paths in this index were checked for existence on 2026-07-09 at commit `a92a8c6` (see the consolidation
report's *checks run*). Reports use repo-relative markdown links; figure/JSON/GIF/`.hymeko` paths are listed as
plain repo-relative paths.
