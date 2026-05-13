# Slashdot joint-mix Gömb: per-slot cap + default joint on external script

## Summary

Joint-mix Slashdot previously OOM’d on ~8 GB VRAM because four joint slots each held large Rust-built pools. `run_gomb_smoke` now subsamples joint pools on SNAP datasets (`slashdot`, `epinions`) by default to **12 000 rows per slot** (override with `--joint-slot-cap`; `0` disables). `run_gomb_tune` passes `joint_slot_cap` for those datasets in `for_joint_mix_tuning`. External AUC script defaults **`RUN_SLASHDOT_JOINT` to 1** (set `0` to skip). Tests cover `_subsample_joint_pools`, tuner clamps including `joint_slot_cap`, and `_build_cmd` for `--joint-slot-cap`.

## Files touched

- `signedkan_wip/src/run_gomb_smoke.py` — `_subsample_joint_pools`, `--joint-slot-cap`, default cap on SNAP when `joint_slot_cap` unset
- `signedkan_wip/src/run_gomb_tune.py` — `_build_cmd` appends `--joint-slot-cap`; `for_joint_mix_tuning` sets `joint_slot_cap` for slashdot/epinions
- `signedkan_wip/tests/test_hymeko_gomb.py` — `test_subsample_joint_pools_reduces_rows`; joint_slot_cap assertions on slashdot compact/wide; build_cmd already asserts `--joint-slot-cap`
- `signedkan_wip/experiments/run_gomb_external_auc_tuning.sh` — default `RUN_SLASHDOT_JOINT=1`, comment update
- `reports/2026-05-12-gomb-slashdot-joint-slotcap.md` — this report

## CORE.YAML items touched

None.

## Test results

- Command: `PYTHONPATH=. pytest -p no:randomly signedkan_wip/tests/test_hymeko_gomb.py -q`
- Result: **33 passed** in ~14 s (CPU).

## Performance / smoke (CUDA)

- Command: `PYTHONPATH=. python -m signedkan_wip.src.run_gomb_tune --datasets slashdot --joint-mix --trials 1 --search-seed 41 --data-seed 0 --edge-split 80_10_10 --n-epochs 10 --device cuda --timeout-s 7200 --architecture compact --out reports/gomb_tune_slashdot_joint_slotcap.jsonl`
- Result: **exit 0**; subprocess wall ~91 s; row includes `"joint_slot_cap": 12000`, `"returncode": 0`, `"model": "joint_mix_gomb[c3,c4,w2,w3]"`.
- Peak RSS not measured in this run (short smoke; no `systemd-run` wrapper).

## New dependencies

None.

## Open issues / follow-up

- Epinions joint-mix not re-smoked here; same cap path should apply.
- For reproducibility papers, document `joint_slot_cap` alongside walk/topk when comparing to full-pool joint runs.

## Experiment provenance (smoke)

- Artifact: `reports/gomb_tune_slashdot_joint_slotcap.jsonl` (2 lines: trial row + phase summary).
- Seeds: `tuner_search_seed=41`, trial `seed=0`, `data_seed=0`.
- Working tree: not clean (repo has many unrelated changes); this report lists only the files above for this task.

## Anti-patterns (CLAUDE.md 6.5)

No new Cartesian PyO3 surface; cap is a single config field and CLI flag, not N wrapper names.
