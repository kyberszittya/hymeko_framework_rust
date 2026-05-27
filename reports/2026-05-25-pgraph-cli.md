# `pgraph` CLI over the underlying structures — 2026-05-25

## Summary

Added a first-class `pgraph` command-line tool that drives the P-graph engine
through the real data structures (meta-model resolution → `LoweredPGraph` →
MSG/SSG/ABB), with four subcommands matching the P-graph workflow:

- **read** — materials (with raw/product roles) and operating units (cost +
  signed I/O).
- **transform** — the bipartite P-graph: M/O partition + directed signed
  incidence.
- **solve** — MSG, SSG (guarded ≤ 30 units), and ABB; `--json` reuses the
  existing analysis emitter.
- **generate** — Graphviz `DOT` (dependency-free) or a P-graph Studio `.pgip`.

Input (`.hymeko` meta-model, `.hymeko` literal-tag, or `.pgip`) is auto-routed by
one loader, `cli::load_pgraph`. The binary is a thin dispatcher; all logic is in
tested library functions. No new dependency (manual arg parsing — `clap` would be
CORE-gated); `hymeko_core`/`parser` used read-only; the engine is reused, not
duplicated (§6.1).

Plan: `docs/plans/2026-05-25-pgraph-cli/` (tex/pdf/tikz/mmd).

## Verified output (real binary)

```
$ pgraph solve hymeko_pgraph/data/prgraph_ex_3_1.hymeko
P-graph: prgraph_ex_3_1  (strict no-excess)
  MSG  maximal structure: { u1, u4, u5 }   [2 of 5 units pruned]
  SSG  feasible solution structures: 1
  ABB  optimum: { u1, u4, u5 }   cost 3.00   [explored 7]
```

`read`, `transform`, and `generate` (DOT) verified on the same file. (Strict
no-excess forces `u5`: `u4`'s byproduct `C` must be consumed, so `{u1,u4}` is
infeasible — optimum is 3.0, exercising the engine's real semantics.)

## Files touched

| File | Δ | Change |
| --- | --- | --- |
| `hymeko_pgraph/src/cli.rs` | **new** | `load_pgraph` (pgip/meta/literal routing), `render_entities`/`render_pgraph`/`render_solution`, `to_dot`, `CliError` |
| `hymeko_pgraph/src/bin/pgraph.rs` | **new** | subcommand dispatch + manual flag parsing |
| `hymeko_pgraph/src/lib.rs` | +2 lines | `pub mod cli` + re-exports |
| `hymeko_pgraph/Cargo.toml` | +3 lines | `[[bin]] name = "pgraph"` (no dependency change) |
| `hymeko_pgraph/tests/cli.rs` | **new** | 7 tests |

## CORE.YAML items touched

Empty list. `hymeko_pgraph` is not core; `hymeko_core`/`parser` used via public
API only; no dependency added (adding `clap` etc. would be CORE-gated, so arg
parsing is manual, matching `hymeko_pgraph_dump`).

## Loader routing

`load_pgraph`: `.pgip` → `read_pgip`; else `.hymeko` → `compile_to_lowered`
(meta-model). A `MetaResolveError::MissingArchetype` means a literal-tag file →
fall back to `lower(parse_description(..))`. Any other meta error is a malformed
meta-model P-graph and is surfaced, not masked. One tool handles all three input
shapes.

## Test results

| Layer | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` | **112 passed, 0 failed, 1 ignored** (TEST_EXIT=0) |
| New `tests/cli.rs` | 7 pass: meta routing, literal fallback, missing-file error, the four renderers (substring + structural asserts), DOT validity |
| Existing suites | unchanged pass |

## Static analysis

- `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` — **passes**
  (CLIPPY_EXIT=0, verified by real exit code — not a piped `tail`).
- New files rustfmt-clean (`--edition 2024`, matching the crate edition). One
  unused-import warning (`AbbSolution`) was fixed during build.
- No `#[allow]`, no `unwrap`/`expect` in non-test code.

## Performance

Toy scale; SSG guarded at `|MSG| ≤ 30` (existing cap). The binary is a thin
shell; logic lives in tested library fns. Suite wall unchanged; ≪ 16 GB. No
criterion bench (CLI front-end, no new algorithm).

## §6.5 anti-patterns

None. The CLI **reuses** `maximal_structure` / `ssg_enumerate` / `solve` /
`analyze_lowered_with_full_options` / `write_pgip` — no algorithm duplication.
Subcommands share one `load_pgraph` and the `render_*` formatters.

## Open issues / follow-up

1. **WASM application** — being assessed separately (the loader is filesystem-
   bound via `StdFsProvider`; a browser build needs an in-memory
   `SourceProvider` + `wasm-bindgen`, the latter a CORE-gated dependency).
2. **`hymeko_pgraph_dump`** now overlaps with `pgraph solve --json`; could be
   deprecated in favour of the new CLI in a later cleanup.
3. Pre-existing crate-wide rustfmt drift + the prior clippy-debt note stand (see
   earlier reports); my new files are clean.

## Experiment provenance

N/A (no training runs / no persistent-state mutation). Git SHA at task start:
`db99de0`. Working tree dirty with this task's additions plus the prior
hypergraph-queries / meta-resolve changes and the orthogonal `Core.yaml`/
`*ools.yaml` edits.
