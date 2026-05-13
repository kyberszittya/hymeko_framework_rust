# Gömb + P-graph MSG/SSG driver (A + B)

## Summary

Implemented **(A)** `hymeko_driver --backend gomb` (single smoke + grid sweep with `gomb.*` targets) and **(B)** `run_gomb_msg_sweep.py` plus Rust `hymeko_pgraph_dump` / `analyze_source` for JSON MSG/SSG/ABB export.

## Files touched (high level)

- `hymeko_pgraph`: `src/dump.rs`, `src/bin/hymeko_pgraph_dump.rs`, `Cargo.toml` (serde/serde_json, bin), `tests/gomb_dump_msg.rs`
- `data/hsikan/gomb_training.hymeko`, `sweep_msg_gomb.hymeko`, `sweep_grid_gomb.hymeko`
- `signedkan_wip/src/hymeko_driver.py`, `gomb_pgraph_mapping.py`, `run_gomb_msg_sweep.py`, `tests/test_gomb_pgraph_driver.py`
- `signedkan_wip/docs/gomb_cycle_abb_optimization.md` (see-also)

## CORE.YAML

No protected items edited.

## Tests

- `cargo test -p hymeko_pgraph gomb_toy`
- `pytest signedkan_wip/tests/test_gomb_pgraph_driver.py` (5 passed)
- `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings`

## Usage

```bash
cargo build -p hymeko_pgraph --bin hymeko_pgraph_dump
python -m signedkan_wip.src.hymeko_driver --backend gomb --device cpu
python -m signedkan_wip.src.run_gomb_msg_sweep \
  --pgraph data/hsikan/sweep_msg_gomb.hymeko --algorithm ssg --device cpu --max-runs 2
```

## Follow-up

Extend `GOMB_UNIT_TO_KNOBS` for each new `@` unit name in custom P-graphs; for `|O_MSG|>30` use `--algorithm abb` instead of full SSG.
