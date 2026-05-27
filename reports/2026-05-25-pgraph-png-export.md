# PNG / SVG export for the `pgraph` CLI — 2026-05-25

## Summary

`pgraph generate <file> --format png|svg --out PATH` now renders the P-graph to
an image by piping the existing `to_dot` output through the system Graphviz
`dot` binary. User-approved approach: **shell-out, no new Cargo dependency**.
Graphviz is an optional *runtime* tool — if `dot` is absent the command fails
with an actionable install hint, never a panic.

Plan: `docs/plans/2026-05-25-pgraph-png-export/` (tex/pdf/tikz/mmd).

## Files touched

| File | Change |
| --- | --- |
| `hymeko_pgraph/src/cli.rs` | + `render_graphviz(dot, format, out)`; + `CliError::Render` |
| `hymeko_pgraph/src/lib.rs` | export `render_graphviz` |
| `hymeko_pgraph/src/bin/pgraph.rs` | `generate` gains `png`/`svg` arms (require `--out`); usage/error text |
| `hymeko_pgraph/tests/cli.rs` | + test that adapts to `dot` availability |
| `docs/guides/pgraph.md` | document `--format png\|svg` + Graphviz requirement |

## CORE.YAML items touched

Empty list. No new dependency (subprocess to a runtime tool, not a manifest dep);
no core crate edited.

## Implementation notes

`render_graphviz` spawns `dot -T<format> -o <out>`, feeds the DOT on piped
stdin, and checks exit status. `ErrorKind::NotFound` → a "install graphviz / use
`--format dot`" message; non-zero exit → surfaces `dot`'s stderr. No
`unwrap`/`expect` in non-test code: the piped stdin handle is taken via
`ok_or_else`, with a comment noting it is `Some` by construction.

## Test results

| Check | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph --test cli` | **8 passed, 0 failed** (`CLI_TEST=0`) |
| `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` | **pass** (`=0`) |
| Smoke: `pgraph generate … --format png --out …` | exits 1 with the clear "Graphviz `dot` not found" hint (no panic) |

The new test is deterministic regardless of environment: with `dot` installed it
asserts the output begins with the PNG magic bytes (`89 50 4E 47`); without it,
it asserts `CliError::Render` names graphviz/dot. Not flaky — it verifies the
correct branch for the host.

## ⚠ Unverified here: actual PNG/SVG rendering (environmental)

Graphviz `dot` is **not installed** in this environment (checked on both the
bash and Windows PATHs). So real PNG/SVG output could not be smoke-tested; the
verified parts are (a) the DOT input (`to_dot`, already tested), (b) the
subprocess wiring, and (c) the missing-`dot` error path. The `dot -Tpng -o`
invocation is the standard Graphviz idiom and is expected to work where Graphviz
is present. Flagged for honesty (same stance as the wasm32-target item).

## §6.5 anti-patterns

None. Reuses `to_dot`; adds one render entry point; no dependency added.

## Open issues / follow-up

1. Verify real PNG output on a machine with Graphviz installed.
2. Browser/WASM PNG remains out of scope (done JS-side via viz.js / @hpcc-js-wasm
   on the DOT that `pgraph_dot` already returns).

## Experiment provenance

N/A (no training runs / no persistent-state mutation). Git SHA at task start:
`db99de0`. Working tree carries this plus the prior hypergraph-queries /
meta-resolve / CLI / WASM changes and the orthogonal `Core.yaml`/`*ools.yaml`
edits.
