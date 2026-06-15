# Report — SMC #6: in-process vs. subprocess generation ablation

**Date:** 2026-06-14
**Slug:** `smc-inproc-vs-subprocess-ablation`
**Plan:** `docs/plans/2026-06-14-smc-inproc-vs-subprocess-ablation/` (`plan.tex/.pdf/.tikz`+`plan-figure.pdf`/`.mmd`)
**Author:** Csaba Hajdu
**Branch:** `feature/ac-hsikan`

## Summary

Implements SMC paper additions item #6: a benchmark that decomposes HyMeKo's
codegen-cost advantage into a **representational** term and an **architectural**
term by timing three ways to turn one compiled IR into a robot description —
C1 (signed-hypergraph, in-process, real codegen), C2 (binary-graph mock,
in-process), C3 (subprocess toolchain, analytic). A new non-core
`hymeko_formats::binary_graph` module supplies the clique-expansion + mock
emitter (reusing the arithmetic from `binary_vs_hypergraph.rs`); a new
`hymeko_bench --bin bench_ablation` produces `ablation.csv`.

## Headline finding (honest — a measured assumption was overturned)

**The advantage is almost entirely architectural, not representational.**

| config | median | IQR | worst | n | kind |
|---|---|---|---|---|---|
| C1 hypergraph, in-process | 0.838 ms | 0.053 | 1.11 | 200 | measured |
| C2 binary mock, in-process | 0.009 ms | 0.000 | 0.025 | 200 | measured |
| C3 subprocess (xacro/gz) | 464.0 ms | — | — | — | analytic |

- **Architectural gain (C2−C3):** in-process emission (sub-millisecond) vs the
  subprocess toolchain (~464 ms) is a **~500× gap** — this is the robust,
  paper-worthy result and dominates everything else by ~3 orders of magnitude.
- **Representational term (C1−C2): +0.83 ms, i.e. the *wrong sign* for a
  hypergraph "win".** The real hypergraph→URDF template render (C1, 288 B valid
  output) is *slower* than the trivial binary mock (C2), because the mock does
  not produce a real artifact — it is a deliberate **lower bound** on
  binary-graph emission, not a real generator. So C1−C2 is not a fair
  representational comparison and must **not** be reported as one. This was the
  plan's stated `sec:risk` "mock-emitter fairness" hazard, now confirmed
  empirically (§11: measurement contradicts an assumption → reported, not
  massaged).

**For the paper:** cite the architectural finding (≈500× from avoiding subprocess
spawn). The representational comparison needs a *real* binary-graph URDF emitter
(emitting an equivalent artifact) before C1−C2 means anything — that is follow-up,
flagged below. The arity-sweep in `binary_vs_hypergraph.rs` remains the place the
representational story is told (edge-count growth), not this timing mock.

## Files touched

| Path | Action | Lines |
|---|---|---|
| `hymeko_formats/src/binary_graph.rs` | new (`BinaryGraph`/`BinaryEdge` + mock + 6 tests) | 151 |
| `hymeko_formats/src/lib.rs` | modify (+1 `pub mod`, +1 `pub use`) | +2 |
| `hymeko_bench/src/bin/bench_ablation.rs` | new (C1/C2/C3 harness + CSV) | 213 |
| `hymeko_bench/Cargo.toml` | modify (+1 `[[bin]]`) | +4 |
| `docs/plans/2026-06-14-smc-inproc-vs-subprocess-ablation/*` | new (4 formats) | — |
| `reports/2026-06-14-smc-inproc-vs-subprocess-ablation.md` | new (this file) | — |
| `hymeko_bench/results/ablation.csv` | new (artifact) | — |

## CORE.YAML items touched

**None.** `grep CORE.YAML` returns no entry for `hymeko_formats` or
`hymeko_bench`. **No dependency added** — `csv`/`anyhow`/`clap` were already in
`hymeko_bench`. `criterion` was deliberately **not** added: it is not a workspace
dep and adding it is a §1 core change; the bench follows the crate's existing
measurement-bin idiom (`binary_vs_hypergraph`, plain `Instant` loop) while
meeting §3's stability bar (≥5 iters after warm-up; median, IQR, worst). If true
criterion rigor is wanted, that is a §1 escalation — not assumed here.

## Test results

- **Unit** (`cargo test -p hymeko_formats`): **10 passed** (6 new `binary_graph`
  + 4 pre-existing). New coverage: triangle k=3→3 edges, k=4→6 edges, k<2→0
  edges (boundary), `mock_emit` length monotone, plus **two failure cases**
  (member index ≥ n_verts → `debug_assert` panic; members/signs length mismatch
  → panic) — exercises the §8 preconditions.
- **Integration** (the bin): `bench_ablation` on `examples/paper/hymeko_robot.hymeko`
  builds a 23-vertex / 40-edge contextual binary graph and writes a well-formed
  `ablation.csv` with C1/C2 `measured`, C3 `analytic`; the bin `assert!`s each
  measured median > 0 and n ≥ 5.
- **Static gate:** `cargo clippy -p hymeko_formats -p hymeko_bench --bin
  bench_ablation` — **0 warnings**; `rustfmt --check --edition 2024` clean on
  both new files. No `unwrap`/`expect` in non-test code except the documented
  sort comparator on finite timings; errors propagate via `anyhow` at the bin
  boundary.

## Performance results

C1 0.838 ms / C2 0.009 ms (200 iters each, after warm-up) / C3 464 ms (analytic),
on the host below. Peak RSS negligible (one IR compile + string emission, tens of
MB). Caveat: C2 (~9 µs) is near the timer-resolution floor (IQR≈0) — fine for the
order-of-magnitude conclusion, not for a tight C1−C2 difference (which is not
claimed anyway).

## Risk anticipation outcomes

- **Mock-emitter fairness** — the flagged risk *materialised*: C2 is too cheap to
  be a fair C1 counterpart. Handled by reporting C1−C2 as not-a-fair-comparison
  and recommending a real binary emitter. Conservative framing preserved (the
  mock cannot over-state HyMeKo's advantage; if anything it under-states C2's
  real cost).
- **C3 analytic** — tagged `kind=analytic` in the CSV and prose; when xacro/gz
  are present the bin measures C3 with no code change.

## §6.5 anti-patterns

None introduced. Clique-expansion arithmetic is shared (one definition, reused),
the bench is one mode-free bin (#13), no new function-name axis. Waiver: criterion
not used (documented §1/§10 reasoning above).

## Open issues / follow-ups

1. **Real binary-graph emitter** for a fair representational C1−C2 (emit an
   equivalent URDF from the flattened graph). Until then the paper should lead
   with the architectural ≈500× result only.
2. **Measure C3 for real** on a machine with `xacro`/`gz`/`mujoco` (absent here)
   to replace the 464 ms analytic constant.
3. **Paper wiring** (smc_02): the architectural number + ablation.csv into the
   codegen-cost subsection; deferred to the manuscript edit pass.

## Provenance

Git SHA: working tree dirty (this change + prior uncommitted edits; see
`git status`). Host: Windows 11, cargo 1.93.1 / rustc 1.93.1, release profile.
Input: `examples/paper/hymeko_robot.hymeko`. Artifact:
`hymeko_bench/results/ablation.csv`. Iters: 200 (after 1 warm-up) per measured
config. C3 constant: 464 ms (documented xacro spawn+parse estimate).
