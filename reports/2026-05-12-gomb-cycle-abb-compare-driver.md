# Gömb cycle ABB compare driver

## Summary

Added `run_gomb_cycle_abb_compare` to run paired `run_gomb_smoke` subprocesses per ABB mode, print a Markdown table, optional JSONL append; added `signedkan_wip/docs/gomb_cycle_abb_optimization.md` (narrative + CLI); cross-linked usage in `run_gomb_smoke` module docstring.

## Files touched

- `signedkan_wip/src/benchmarks/run_gomb_cycle_abb_compare.py` (new)
- `signedkan_wip/tests/test_run_gomb_cycle_abb_compare.py` (new)
- `signedkan_wip/docs/gomb_cycle_abb_optimization.md` (new)
- `signedkan_wip/src/run_gomb_smoke.py` (docstring usage block)

## CORE.YAML

None.

## Tests

`pytest -p no:randomly signedkan_wip/tests/test_run_gomb_cycle_abb_compare.py` — 2 passed (~6.5 s wall, sbm_n200, 1 epoch, none vs start_local).

## Dependencies

None added.

## Follow-up

- Optional: breach-after-each-trial in `run_hsikan_optuna_chase`; fix `runtime_config` import for dot/quaternion Optuna trials.
