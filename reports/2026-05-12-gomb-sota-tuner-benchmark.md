# Gömb SOTA chase: tuner protocol, VRAM clamps, benchmark smoke (2026-05-12)

## Summary

Changes target **higher AUROC under the same `80_10_10` protocol** while reducing wasted OOM trials and aligning the external benchmark script with an **honest val-first selection** policy.

### Code / tooling

- **`run_gomb_tune`**
  - **`--pick-best-by {test_auroc,val_auroc}`** (default `test_auroc`): ranking metric for `best_score` / `best_meta`; summary adds `tuner_pick_best_by`, `best_val_auroc` when present, and still reports `best_test_auroc` for the selected row.
  - **Bitcoin wide joint** (`for_joint_mix_tuning`): stricter VRAM caps — `topk` ≤ **56**, `d_embed` ≤ **48**, `M_outer` ≤ **10**, `d_middle` / `d_core` capped relative to `d_embed`, **`n_tiers` ≤ 3**, walk caps unchanged (32k).
  - Wide LR menu adds **1e-4**.
- **`signedkan_wip/src/gomb_jsonl_summarize.py`**: prints a Markdown table from `tuner_phase_summary` rows.
- **`signedkan_wip/experiments/run_gomb_external_auc_tuning.sh`**
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default.
  - `PICK_BEST_BY` default **`val_auroc`**; passed to all `run_gomb_tune` invocations.
  - `NEPOCHS` default **42 → 72** (Bitcoin); Slashdot defaults unchanged.
  - End-of-script calls to **`gomb_jsonl_summarize`** (best-effort).

### Tests

- `test_tuner_objective_val_vs_test`
- `test_gomb_jsonl_summarize.py` (subprocess module smoke)
- Bitcoin joint clamp test updated for new caps.

`pytest … test_hymeko_gomb.py test_gomb_jsonl_summarize.py test_konect_datasets.py` → **43 passed** (after fixes).

## CUDA benchmark (single dataset, short)

Command:

`PYTHONPATH=. python -m signedkan_wip.src.run_gomb_tune --datasets bitcoin_alpha --joint-mix --trials 6 --search-seed 99 --data-seed 0 --edge-split 80_10_10 --n-epochs 56 --device cuda --pick-best-by val_auroc --out reports/gomb_tune_sota_chase_alpha_joint_2026_05_12.jsonl`

- **Best val AUROC** (ranking metric): **≈ 0.9075** (trial 0).
- **Test AUROC of that same row**: **≈ 0.9081**.
- **vs in-repo tuned SGCN** (0.927 on Alpha, draft table): **~0.019 AUROC gap remains**.
- **vs published SGCN ~0.91**: Gömb test **≈ 0.908** is **in-band** on this seed / short search.
- One trial still **CUDA OOM** (trial 5) before `n_tiers` cap was added; re-run after cap recommended.

## CORE.YAML

None.

## Follow-up (not done here)

- Multi-seed `data_seed` grid + frozen `trial_params` (see `docs/plans/2026-05-12-gomb-auc-break-benchmark/plan.pdf`).
- Long second stage: **80–120 epochs** on val-picked configs only.
- OTC joint repeat with same protocol (target tuned SGCN **0.957** is still far for Gömb on prior reports).

## Artifacts

- `reports/gomb_tune_sota_chase_alpha_joint_2026_05_12.jsonl`

## Wide multi-seed sweep (many parameter draws)

Driver: **`signedkan_wip/experiments/run_gomb_wide_param_sweep.sh`**

- Re-runs ``run_gomb_tune`` once per ``SEARCH_SEEDS`` value (default **six** seeds) so the same discrete menus yield **independent** hyperparameter samples.
- Defaults: ``TRIALS=32``, ``NEPOCHS=72``, joint-only (set ``VANILLA=1`` for a second loop), append to ``reports/gomb_wide_sweep.jsonl``.
- Wide search space: ``weight_decay`` menu includes **2e-5**.
- External AUC script: default ``TRIALS`` **10 → 24**, Slashdot ``TRIALS_SLASH`` **12 → 16**.

Smoke: ``DATASETS=bitcoin_alpha TRIALS=2 SEARCH_SEEDS="81 83"`` → ``reports/gomb_wide_smoke.jsonl`` (two phase summary rows).
