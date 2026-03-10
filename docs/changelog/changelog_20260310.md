# Project Changelog — 2026-03-10

## Daemon Data-Plane Task Closure
- Completed and re-traced `docs/plans/daemon/checklist_task2.md` so Phase 2 now reflects the delivered direct-memory path end-to-end.
- Marked Task 2.1 as implemented with the daemon loop publishing `ExpansionHeader + COO` payloads via shared-memory loaned slices.
- Kept Task 2.3 wording focused on the concrete bridge contract (raw pointer writes in core + `pyarrow.foreign_buffer` offsets in Python).

## Daemon Control-Plane Checklist Trace
- Updated `docs/plans/daemon/checklist_task3.md` to reflect completed groundwork already present in code.
- Marked `moka` cache dependency + `HymekoDaemon` cache initialization as done under Task 3.1; cache hit/skip routing remains open.
- Marked Task 3.2 runtime pieces as active end-to-end: `#[tokio::main]`, Zenoh session setup/subscriber registration in `hymeko_daemon/src/service.rs::run`, and the heartbeat `tokio::select!` reactor loop.
- Traced the logging migration in `hymeko_daemon/src/main.rs`: daemon status output now uses `tracing`/`tracing-subscriber` with geometric/ascii markers instead of pictograms.
- Marked Rayon dependency setup as done under Task 3.3, and documented `hymeko_daemon/src/worker.rs::compute_expansion` as scaffold-only (oneshot + Rayon flow is outlined but not activated yet).

## Daemon Bootstrap Modularization
- Updated `hymeko_daemon/src/main.rs` to a thin bootstrap that declares `config`, `service`, and `worker` modules and defers operational flow into those units.
- Kept runtime logging initialization in `main` via `tracing_subscriber::fmt()` with `EnvFilter::try_from_default_env()` fallback to `info`.
- Routed startup through `Args::parse()` and `DaemonConfig::from(args)` before constructing `HymekoDaemon`.
- Handed execution to `Arc::new(HymekoDaemon::new(config)).run().await`, keeping `main` focused on orchestration only.

## Zero-Copy Bridge Runtime Notes
- `hymeko_core/src/tensor/shared_state.rs` remains the canonical layout boundary for `ExpansionHeader` and transport payload framing.
- `hymeko_daemon/src/main.rs` gates publication with `service.dynamic_config().number_of_subscribers() > 0` and sends contiguous `ExpansionHeader + COO` frames via `publish_star_expansion`.
- `hymeko_py/src/interface_python/api.rs` (`PySharedExpansion::buffers`) exports `(k, i, j, val)` as foreign buffers while tying ownership to the Python object for lifetime safety.

## Random COO Builder Benchmark Telemetry
- Recorded a fresh suite export at `hymeko_core/target/benchmarks/coo_builder_random_benchmark.csv` from `hymeko_core/tests/benchmarks/bench_coo_builder_random.rs::bench_random_hypergraph_coo_builder_suite`.
- Captured 28 benchmark rows (11 case configurations) over `(nodes, edges)` scales from `(64, 32)` up to `(2048, 1024)` and densities `0.01`, `0.05`, and `0.20` (where configured).
- Observed `total_ms` spanning `4.4408` to `1432.9884`, with throughput envelope `ns_per_entry` spanning `10128.738` to `249896.226`.
- Confirmed run schema stability for downstream analysis (`nodes,edges,density,run_idx,nnz,parse_ms,compile_ms,view_ms,coo_ms,total_ms,ns_per_entry`).

## Architecture Documentation Consolidation
- Expanded `architecture/README.md` with a stronger layer-oriented view and links to per-domain diagram docs.
- Added/extended sub-READMEs under `architecture/` so Mermaid and SysML assets are documented in place.
- Continued README polish (`README.md`) to surface the architecture index and logo-first navigation from the project landing page.

## Follow-ups
- If CI starts compiling `iceoryx2` from source in additional targets, add an explicit toolchain note (including `clang`) in CI docs and pin it in the next changelog entry.
- If benchmark artifacts should be retained long-term, mirror key CSV snapshots from `target/` into a tracked data folder (for example under `hymeko_core/data/benchmarks/`) and reference that durable path in future changelogs.
