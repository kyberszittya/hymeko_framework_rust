# Report: workspace `clippy -D warnings` green (2026-05-11)

## Summary

Drove `cargo clippy --all-targets -- -D warnings` to exit 0 across the Rust workspace: mechanical clippy fixes in `hymeko_compute`, `hymeko_cli`, `hymeko_clifford`, small daemon / bench / parser / query harness adjustments, and integration-test crate attributes where appropriate.

## CORE.YAML

No `CORE.YAML` entries modified.

## Files touched (high level)

- `hymeko_compute`: `buffers.rs`, `kernels/{force_directed,signed_spmv,vector_add}.rs`, `examples/spmv_from_json.rs`
- `hymeko_cli`: `main.rs`, `repl.rs`
- `hymeko_clifford`: `algebra/blade.rs`, `algebra/multivector.rs`
- `hymeko_daemon`: `worker.rs`, `iox_ingress.rs`, `service.rs`, `common.rs`
- `hymeko_query/tests`: `mod.rs`, `test_helpers.rs`
- `hymeko_hre/tests`: `test_expansion.rs`, `test_fixture_expansion.rs`
- `hymeko_bench/src/bin`: `binary_vs_hypergraph.rs`, `bench_scaling_tensor.rs`, `artifact_generation.rs`, `bench_control_cycle.rs`, `bench_scaling.rs`
- `parser`: `benches/parser_bench.rs`, `tests/mod.rs`

## Tests run

- `cargo test -p hymeko_clifford -p hymeko_compute -p hymeko_hre --tests` — all executed tests passed (GPU tests ignored as designed).
- `cargo clippy --all-targets -- -D warnings` — pass (re-run after `rustfmt` on edited files).

## Performance

Not applicable (lint-only / trivial arithmetic rewrites).

## Dependencies

None added or removed.

## Follow-ups

- `cargo fmt --check` still fails on pre-existing diffs in `parser/tests/simd_lexer_tests/simd_lexer_tests.rs` and `parser/tests/using_alias.rs` (not part of this change set).
- Workspace profile warnings from `hymeko_wasm` / `hymeko_monitor` `Cargo.toml` remain (Cargo emits them at build start).

## Open issues

None for the clippy gate itself.
