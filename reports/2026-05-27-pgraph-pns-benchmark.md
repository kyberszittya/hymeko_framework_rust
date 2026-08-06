# Report — Real PNS benchmark: decision-mapping SSG + costed ABB parity

**Date:** 2026-05-27
**Plan:** `docs/plans/2026-05-27-pgraph-pns-benchmark/` (tex/pdf/tikz/mmd)
**Crate:** `hymeko_pgraph` (non-core)

## Summary

Gave the P-graph chapter two literature-anchored, reproducible empirical
results against Friedler / Orosz / Pimentel Losada,
*P-graphs for Process Systems Engineering* (Springer):

1. **Structural parity — Example 3.3** (`data/pgraph/Chapter4/example4_3`,
   35 declared units). Implemented the canonical recursive **decision-mapping
   SSG** (book Def. 5.1 / Fig. 5.13) in `ssg_dm.rs`. On the 29-unit relaxed
   maximal structure it reproduces the book's published **3465
   solution-structures exactly**. The pre-existing brute `ssg` (generate-and-test
   over $2^{29}$ subsets) cannot reach this; the decision-mapping recursion does
   it in ~32 ms.

2. **Costed parity — Example 14.1** (`data/pgraph/book/example14_1`, 12 units,
   authored from book Table 14.1). ABB returns the book's optimum
   $\{u_1,u_4,u_8,u_{11}\}$ at weight **16.0** (matching the book's feasible
   solution extension $\delta$, $w(\delta)=16 < w(\varepsilon)=28$).

The producer query $\Delta(x)=\{u : x\in\mathrm{outputs}(u)\}$ used by SSG is
expressed as a query over the signed incidence
(`LoweredPGraph::producers` = `schema.predecessors`), reinforcing the chapter's
thesis (process information *is* a query, not a parallel store).

## Files touched

**New:**
| File | Lines |
|:--|--:|
| `hymeko_pgraph/src/ssg_dm.rs` (decision-mapping SSG + 4 unit tests) | 342 |
| `hymeko_pgraph/tests/ssg_decision_mapping.rs` (5 integration tests) | 175 |
| `data/pgraph/book/example14_1.hymeko` (fixture, Table 14.1) | 58 |

**Modified (additive):**
| File | Δ |
|:--|--:|
| `hymeko_pgraph/src/lib.rs` (module + re-exports) | +16 / −8 |
| `hymeko_pgraph/src/lowering.rs` (`producers`, `consumers` accessors) | +30 / −14 |
| `hymeko_pgraph/benches/pgraph_bench.rs` (`bench_ssg_dm`) | +43 / −2 |

No existing function's observable behaviour was changed; `msg`, `abb`, and the
brute `ssg` are untouched. The `consumers` accessor is the documented dual of
`producers` and is exercised indirectly; `producers` is covered by the SSG tests
and the unit-level `three_producers_yields_seven` case.

## CORE.YAML items touched

**None.** `hymeko_pgraph` is not a CORE.YAML crate. No pinned dependency changed:
`criterion` was **already** a `[dev-dependencies]` entry of the crate (0.8.2), so
the plan's flagged dependency sign-off turned out moot — no manifest dependency
edit was required. `Cargo.toml` was not modified.

## Test results

`cargo test -p hymeko_pgraph` — **all green, no regressions.**

| Layer | Tests | Result |
|:--|--:|:--|
| `ssg_dm` unit (in-module) | 4 | pass (7-producers, single-chain, unproducible, cap) |
| `ssg_decision_mapping` integration | 5 | pass |
| lib unit total | 29 | pass |
| other integration suites (pre-existing) | 88 | pass (1 pre-existing ignored) |

Headline integration assertions:
- `example3_3_reproduces_3465_solution_structures` — relaxed MSG = 29 units;
  `ssg_dm` returns exactly **3465** distinct structures.
- `example14_1_abb_matches_book_optimum` — `{u1,u4,u8,u11}`, cost **16.0**.
- `hda_decision_mapping_structures` — 3 structures (hand-derived); note this
  differs from the brute strict SSG, which force-includes a disposal sink that
  reaches no product (violates axiom S4). See below.
- `decision_mapping_structures_are_brute_feasible` — soundness: every emitted
  structure passes the brute relaxed `is_feasible`.
- `brute_ssg_still_refuses_above_30_units` — contract preservation: the brute
  `n>30` guard is unchanged.

Determinism: enumeration is deterministic (material pick order = ascending
`DeclId`); no RNG. Duplicate-freeness asserted (exactly-once generation).

## Performance results (criterion, release; 15 samples post-warmup)

Host: Windows 11 (10.0.26200), 16 logical cores.

| Benchmark | median | budget (plan) | verdict |
|:--|--:|--:|:--|
| `ssg_dm/example3_3` (3465 structures) | **32.4 ms** [32.19, 32.66] | <500 ms | ✓ 15× under |
| `ssg_dm/example14_1` (ABB optimum) | **39.8 µs** [37.5, 42.5] | <50 ms | ✓ |

Peak RSS not separately probed; the working set is 3465 `BTreeSet<DeclId>`
(< a few MB), far under the 16 GB cap (§4). No regression risk: all benchmarked
paths are new or unchanged.

## Semantic finding (documented, not coerced)

The decision-mapping SSG enforces axiom **S4** (every included unit reaches a
product — units enter only when a needed material justifies them). The brute
`ssg::is_feasible` checks forward-producibility + optional strict no-excess but
**not** S4, so the two are deliberately **not** set-equal: on HDA the brute
strict path yields `{Mixer,Reactor,Disposal}`, but `Disposal` reaches no product,
so it is not a true solution-structure. The decision-mapping count is the
canonical one (and the one the book's 3465 figure uses). The cross-check is
therefore *soundness* (dm ⊆ brute-feasible), not equality — a refinement of the
plan's cross-check, made after the implementation surfaced the S4 distinction.

## Dependencies

None added or removed. (`criterion 0.8.2` was already a dev-dependency.)

## Open issues / follow-ups

- **ABB silent-suboptimal incumbent (pre-existing, out of scope):** when
  `AbbOptions::max_explored` trips, ABB returns whatever incumbent it holds —
  possibly suboptimal — with no signal. Example 3.3 / 14.1 explore 13–59 nodes,
  nowhere near the 1M cap, so the benchmark is unaffected. Worth a follow-up
  (return a `capped` flag like `SsgDmResult`).
- **Crate-wide fmt drift (pre-existing):** `hymeko_pgraph`'s sibling modules
  (`axioms.rs`, `dump.rs`, `pgip_io.rs`, several test files) are not
  `rustfmt`-clean in the committed tree. A `cargo fmt -p` sweep was reverted to
  keep this change minimal; the drift predates this work and is left for a
  dedicated formatting pass. My changed files are individually fmt-clean.
- **Chapter wiring:** the chapter draft (`docs/plans/2026-05-27-pgraph-chapter`)
  can now cite both results; `ssg_dm` should appear in the evidence map.

## Provenance

- **Git SHA:** `9abfc3435f55f7443cb07bde4583a17126ac3fc1` (branch
  `feature/pgraph_engine`). Working tree **dirty** — uncommitted for this change:
  `hymeko_pgraph/src/{ssg_dm.rs,lib.rs,lowering.rs}`,
  `hymeko_pgraph/tests/ssg_decision_mapping.rs`,
  `hymeko_pgraph/benches/pgraph_bench.rs`, `data/pgraph/book/example14_1.hymeko`.
  Also dirty and **unrelated**: `tools.yaml` (pre-existing case-collision
  manifest defect — `Tools.yaml`/`tools.yaml`; see the case-collision note).
- **Toolchain:** rustc/clippy 1.93.0; criterion 0.8.2; gnuplot absent (criterion
  plotters backend).
- **Static analysis:** `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings`
  passes (fixed one `derivable_impls` on `SsgDmOptions`). No new
  `#[allow(...)]`, no `unwrap`/`expect` in non-test code.
- **Determinism:** no RNG; fixtures committed; Example 3.3 source is the
  committed `pgip_to_hymeko.py` conversion of `example4_3.pgip`.
- **§6.5 anti-patterns:** none introduced. `ssg_dm` is a single focused
  enumerator (no Cartesian wrappers, no string-typed config, no global state);
  it reuses the existing query layer and `SolutionStructure` type rather than
  duplicating them.
