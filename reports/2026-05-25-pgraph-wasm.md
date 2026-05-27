# Browser (WASM) P-graph application — 2026-05-25

## Summary

Exposed the P-graph engine to the browser via `wasm-bindgen`, reusing the exact
engine the native CLI uses. A meta-model `.hymeko` source string (with
`@"meta_pgraph.hymeko"` include + `<isa>` typing) is resolved **without a
filesystem** and solved (MSG/SSG/ABB) / rendered (DOT), returning JSON / strings
to JavaScript.

Per the user's decision, this **extends the existing `hymeko_wasm` crate**
(approved intra-workspace `hymeko_pgraph` path dep; `wasm-bindgen` already
present — no new external crate).

Plan: `docs/plans/2026-05-25-pgraph-wasm/` (tex/pdf/tikz/mmd).

## Design

```
browser JS ─(instance, meta strings)→ wasm.rs cfg(wasm32) #[wasm_bindgen]
          → hymeko_wasm::pgraph (native core) → hymeko_pgraph::compile_sources
          → ModuleStore<MemProvider> (in-memory resolve, no FS) → LoweredPGraph
          → MSG/SSG/ABB/DOT (reused) → JSON / DOT string → JS
```

**Filesystem blocker → fix.** `compile_to_lowered` uses `StdFsProvider`
(unavailable in the browser). Added `hymeko_pgraph::compile_sources(root, files)`
using `hymeko_core`'s public in-memory `MemProvider`; it shares one generic
`lower_compiled<P: SourceProvider, R: HymekoParser>` helper with
`compile_to_lowered` — no duplication. The native core fns are pure over source
strings (so they unit-test natively); the `cfg(wasm32)` shims are one-line
wrappers.

**rusqlite / wasm incompatibility → `pgip` feature.** `hymeko_pgraph` pulled
`rusqlite` (bundled C SQLite, for `.pgip`), which cannot target
`wasm32-unknown-unknown`. Made it the optional, **default-on** `pgip` feature
(native behaviour unchanged); `hymeko_wasm` depends with `default-features =
false`, dropping `rusqlite`/C from the wasm graph.

## Files touched

| File | Change |
| --- | --- |
| `hymeko_pgraph/src/meta_resolve.rs` | + `compile_sources` + generic `lower_compiled` helper |
| `hymeko_pgraph/src/lib.rs` | export `compile_sources`; `#[cfg(feature="pgip")]` gate `pgip_io` + its re-exports |
| `hymeko_pgraph/src/cli.rs` | `#[cfg(feature="pgip")]` gate the `.pgip` branch (clear error when off) |
| `hymeko_pgraph/Cargo.toml` | `rusqlite` optional; `[features] default=["pgip"]`; both bins `required-features=["pgip"]` |
| `hymeko_pgraph/tests/{meta_resolve,pgip_io}.rs` | `compile_sources` test; gate pgip test file |
| `hymeko_wasm/Cargo.toml` | + `hymeko_pgraph` dep (`default-features = false`) |
| `hymeko_wasm/src/pgraph.rs` | **new** native core: `solve_json` / `transform_text` / `dot` (meta→literal fallback) |
| `hymeko_wasm/src/wasm.rs` | + `cfg(wasm32)` `pgraph_solve` / `pgraph_dot` / `pgraph_transform` shims |
| `hymeko_wasm/src/lib.rs` | `pub mod pgraph` |
| `hymeko_wasm/tests/test_pgraph.rs` | **new** native tests |

## CORE.YAML items touched

Empty list. `hymeko_core`/`parser` used read-only. The added dependency
(`hymeko_pgraph` → `hymeko_wasm`, intra-workspace) was **explicitly approved by
the user** (the WASM-placement decision); no new external crate, no `hymeko_core`
manifest change.

## Test / verification results (real exit codes)

| Check | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` (pgip on) | **pass** — 15 test binaries green (`PG_TEST=0`), incl. new `compile_sources` test |
| `cargo build -p hymeko_pgraph --no-default-features --lib` | **pass** (`NODEF_LIB=0`) — lib compiles with no `rusqlite` (wasm-readiness proof) |
| `cargo test -p hymeko_wasm` (pgip-off dep) | **pass** (`WASM_TEST=0`) — 21 tests incl. 4 new P-graph-app tests |
| `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` | **pass** (`=0`) |
| `cargo clippy -p hymeko_pgraph --no-default-features --lib -- -D warnings` | **pass** (`=0`) |
| `cargo clippy -p hymeko_wasm --all-targets -- -D warnings` | **pass** (`=0`) |

New native tests assert: `solve_json` reports MSG `{u1,u4,u5}` + ABB and prunes
`u2`; `transform_text` shows the M/O partition; `dot` is a valid `digraph`; a
literal-tag instance solves with empty `meta` (fallback).

## ⚠ Unverified: the `wasm32` compile (environmental, not code)

`rustup target add wasm32-unknown-unknown` **failed twice** — the `rust-std-…-
wasm32-unknown-unknown` download timed out (`source: TimedOut`) against
`static.rust-lang.org`. The target stdlib is therefore unavailable in this
environment, so the `#[wasm_bindgen]` shims (`cfg(target_arch="wasm32")`) could
**not** be compiled here. Per the operating contract I am flagging this rather
than asserting it works.

Mitigation: the shims are one-line wrappers over the **fully tested** native
`hymeko_wasm::pgraph` fns, and the `rusqlite`/wasm blocker is removed (verified by
the no-default-features lib build + clippy). To finish on a networked machine:

```
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown -p hymeko_wasm     # compile check
wasm-pack build hymeko_wasm --target web                        # browser bundle
```

## §6.5 anti-patterns

None. `compile_sources` and `compile_to_lowered` share one generic helper
(no FS/in-memory duplication); the WASM surface reuses the engine + the CLI
renderers; the `pgip` gate removes a dep cleanly rather than `#[cfg]`-scattering.

## Open issues / follow-up

1. **`wasm32` compile + browser bundle** — pending a networked machine (commands
   above). Optionally add a tiny `index.html` demo driving `pgraph_solve`.
2. The pre-existing crate-wide rustfmt drift note stands (earlier reports); my
   touched files are rustfmt-clean (`--edition 2024`).

## Experiment provenance

N/A (no training runs / no persistent-state mutation). Git SHA at task start:
`db99de0`. Working tree dirty with this task's additions plus the prior
hypergraph-queries / meta-resolve / CLI changes and the orthogonal
`Core.yaml`/`*ools.yaml` edits. `wasm32` target install attempted (failed,
network) — no source impact.
