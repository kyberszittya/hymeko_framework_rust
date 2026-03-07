# Project Changelog — 2026-03-06

## Tensor Grid Telemetry & COO Export Benchmarks
- Added `py/coo_tensor/coo_tensor_grid_eval.py`, a configurable harness that sweeps node/edge/density grids across both star and clique expansions.
- Each run records per-iteration parse, extraction, and tensor materialization timings, tracking NNZ counts so tensor sparsity can be profiled over time.
- The harness now emits two timestamped CSVs (`coo_tensor_grid_<stamp>.csv` and `coo_tensor_grid_raw_<stamp>.csv`) that separate summary statistics from per-trial telemetry for downstream analysis or visualization notebooks.

## Parser Layout Guidance
- Documented in the root `README.md` why the LALRPOP-driven `parser/` crate remains nested inside the workspace instead of being hoisted to the repository root.
- Clarified that keeping the parser co-located with its data fixtures and build script avoids `cargo` feature bleed-through, simplifies CI caching, and mirrors how the Python bindings expect paths to be resolved.

## PathID / PathKey & IR Tree Hygiene
- Captured the recent `PathKey` improvements (slice borrowing + serde derives) plus the double-ended `DeclNode` sibling tracking work so downstream contributors know how path lookups avoid needless allocations.
- Called out the new `ResolveError::UnexpectedTopLevelArc` guardrails and the tensor CSR helper additions to highlight how resolver safety and math kernels evolved together.

## Measurement & Troubleshooting Notes
- Summarized the known `maturin develop` hang troubleshooting steps (verbose logging, cargo timings, environment probes) so future incidents have an immediate playbook.
- Linked the parser benchmark grid (`py/parsing/benchmarks/grid_expansion_bench.py`) with the new COO tensor harness to give a single reference for expansion-related performance profiling.

