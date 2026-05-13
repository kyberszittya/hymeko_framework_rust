# Gömb tuner: `n_params` in logs/summary + SNAP joint AUC search tweaks

## Summary

- **`run_gomb_tune`**: Each trial log line appends **`n_params=…`** when the smoke JSON row includes `n_params`. Per-dataset phase summary JSONL adds **`best_n_params`**; the final `DONE` line prints **`n_params=`** for the best row.
- **Search space (compact, Slashdot/Epinions)**: `d_embed` menu gains **26**; joint-mix SNAP **compact `topk` cap** raised from **24 → 28** (walk caps and `joint_slot_cap` unchanged).
- **`run_gomb_external_auc_tuning.sh`**: Defaults **`TRIALS_SLASH=12`**, **`NEPOCHS_SLASH=48`** (was 4 / 24) for deeper Slashdot tuning.
- **Tests**: `_n_params_from_row`, compact slashdot `d_embed` includes 26, joint topk clamp ≤ 28, narrow compact bound updated to 26.

## Files touched

- `signedkan_wip/src/run_gomb_tune.py`
- `signedkan_wip/experiments/run_gomb_external_auc_tuning.sh`
- `signedkan_wip/tests/test_hymeko_gomb.py`
- `docs/plans/2026-05-12-gomb-tune-n-params-auc/plan.{tex,pdf,mmd,tikz}`
- `reports/2026-05-12-gomb-tune-n-params-auc.md` (this file)

## CORE.YAML

None.

## Test results

`PYTHONPATH=. pytest -p no:randomly signedkan_wip/tests/test_hymeko_gomb.py -q` → **35 passed** (~11 s).

`ruff` not installed on host (exit 127); not run.

## CUDA tuning run (Slashdot joint compact)

Command:

`python -m signedkan_wip.src.run_gomb_tune --datasets slashdot --joint-mix --trials 8 --search-seed 7 --data-seed 0 --edge-split 80_10_10 --n-epochs 48 --device cuda --architecture compact --out reports/gomb_tune_slashdot_joint_auc_2026_05_12.jsonl`

- Wall **~761 s** for 8 trials; best **test_auroc ≈ 0.8962**, **`best_n_params` = 1 887 440** (trial 0: `lr=3e-3`, `d_embed=22`, `topk=28`, `pos_weight_auto=false`).
- **Note:** Earlier ~0.75 test AUROC used **`n_epochs=10`**; this run uses **48 epochs**, which dominates the gain versus the small `topk` / `d_embed` menu changes.

## Dependencies

None.

## Open issues

- Re-baseline prior “0.74” comparisons using the same **`n_epochs`** when claiming architecture-only deltas.
- If VRAM errors appear on smallest GPUs at `topk=28`, lower cap locally or pass a smaller `--joint-slot-cap` from the tuner (not wired automatically).

## Provenance

- Artifact: `reports/gomb_tune_slashdot_joint_auc_2026_05_12.jsonl`
- Seeds: `search_seed=7`, smoke `seed=0`, `data_seed=0`

## Anti-patterns (CLAUDE 6.5)

No new Cartesian CLI surface; `best_n_params` is additive metadata.
