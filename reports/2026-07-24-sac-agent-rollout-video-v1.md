---
title: SAC_AGENT_ROLLOUT_VIDEO_V1 — reproducible video evidence of the Stage-5b SAC carry-option agent
date: 2026-07-24
branch: feat/sac-rollout-video-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: SAC_AGENT_ROLLOUT_VIDEO_V1_DELIVERED
---

# SAC_AGENT_ROLLOUT_VIDEO_V1 (2026-07-24)

Reproducible video evidence of the frozen-baseline Stage-5b SAC carry-option agent — **not** a cherry-picked highlight.
Rendered from the **real evaluation rollout** (the deployed controller `θ_center → fixed b=8 search → committed
push/brake/release option → frozen settling pi_0`), physics/timestep unchanged, deterministically replayable, with
per-video provenance. Linked to the frozen baseline `executable-hymeko-option-rl-v1` @ `772a11a4`.

## Artifacts (`reports/2026-07-24-sac-rollout-video/`)
| # | video | MP4 + GIF | what it shows |
|---|---|---|---|
| 1 | `01_matched_seed_update0_vs_sac` | side-by-side | **matched seed** (same start state + search seed): update-0 proposal (left) **K6 0** vs Stage-5b SAC (right) **K6 1** — the coin the RL actually delivers |
| 2 | `02_sac_success_k6` | single | a clean SAC success: full PUSH → BRAKE → RELEASE → HANDOFF → SETTLE with **K6 held** |
| 3 | `03_sac_honest_failure` | single | an honest failure/near-miss (no handoff, full horizon) — the remaining failure mode, no cherry-picking |

Each carries a restrained HUD: controller (Stage-5b SAC / update-0), object (coin), search budget (b=8), phase, K6 dwell
counter (n/6), handoff, and the final verdict card. Each has a sibling provenance JSON.

## Honest scan (no cherry-picking)
Over 30 held-out final-panel states (seeds 12200–13600) at fixed search seed 8000+i: **SAC K6 6/30, update-0 K6 4/30** —
consistent with the frozen Stage-5b result (`CONSISTENT_POSITIVE_SAC_LEAN`). The three videos are *selected* from this scan
(first SAC-beats-update-0 state; first SAC success; first SAC failure), and the scan totals are recorded in `manifest.json`,
so the failure video is not hidden.

## Provenance (per video JSON) — deterministically replayable
Each video pins: controller, checkpoint filename + SHA-256(16), env seed + prefix + family, **start-state hash**
(`snapshot_hash`), search seed, search budget (8), **θ_center** (the Bellman action) and **θ_selected** (the search-selected
committed option — provenance only), and the K6 / handoff / containment-exit certificate. Example (matched-seed video):
- update-0 — `carry_proposal_refined.pt` sha `095a9cde6232afad`, seed 12201, start `1d3b316ae23d`, search 8002 → **K6 0**
- SAC — `carry_rlb_sac_seed3_selected.pt` sha `868775c4c8367d41`, seed 12201, start `1d3b316ae23d`, search 8002 → **K6 1**

Same env seed + same start-state hash + same search seed on both panels: the only difference is the controller.

## Faithfulness — rendered from the real rollout
The frame capture is a **non-behavioral** `frame_hook` added to `structured_carry_rollout` (the eval-path rollout) and
`execute_one_option`: it observes the live MuJoCo scene each physics step; it does not touch the physics or timestep. Gated:
`test_coin_carry_option.py::test_structured_carry_rollout_frame_hook_non_behavioral` and
`test_coin_carry_option_rl.py::test_frame_hook_is_non_behavioral` assert the rollout outcome (return + certificate) is
bit-identical with the hook on vs off. The scan-K6 and the rendered-K6 come from the **same** `structured_carry_rollout`, so
the on-screen verdict equals the reported metric. Reuses the framework renderers (`mujoco.Renderer`, `topdown_camera`,
`_draw_overlay`, `compare_gif`) — no new rendering code.

## Reproduce
```
git switch feat/sac-rollout-video-v1        # from executable-hymeko-option-rl-v1
# provision the gitignored frozen pi_0 (1902454c) + build the CLI (cargo build -p hymeko_cli)
PYTHONPATH=. .venv/bin/python experiments/2026_07_22_coin_v3_learning/rl_entry/coin_sac_rollout_video.py
```
Deterministic: fixed panel seeds, fixed search seeds; the same checkpoints (SHAs above) reproduce the same rollouts.

## Next
The same video template extends to the object variations (`feat/object-to-target-variants-v1`): coin, rectangle, triangle,
ring — each with the matched-seed update-0-vs-RL side-by-side, so the added value of RL is visible per object.
