# Report — MSG/ABB canonical-semantics fix (Pimentel report)

**Date:** 2026-05-27
**Plan:** `docs/plans/2026-05-27-msg-abb-canonical-fix/` (tex/pdf/tikz/mmd)
**Crate:** `hymeko_pgraph` (non-core)
**Trigger:** Jean Pimentel (book co-author) reported MSG/ABB "broken".

## Summary

Verified against the book (Friedler, Orosz, Pimentel Losada, *P-graphs for
Process Systems Engineering*, Springer) that `maximal_structure` diverged from
the canonical Friedler MSG, returning **too-small / empty** maximal structures,
and that ABB inherited the error. Replaced MSG with the canonical **reduction +
composition** algorithm (book Ch. 4) and made the canonical (no-excess-free)
behaviour the default for MSG / SSG / ABB and both CLI binaries.

**Two root causes (both fixed):**
1. **Non-canonical `strict_no_excess` default.** The backward pass imposed a
   "every output consumed" rule that is not among axioms S1–S5. (`true` was the
   default and was mislabeled "canonical Friedler 1992" in the 2026-05-19 report.)
2. **Raw-reachability forward pass.** The forward feasibility used
   `close_producible` (reachability *from raws*), which dropped structurally
   valid **cycles** — e.g. book Example 4.1's `{u3,u6,u10}`, where `u3` produces
   the product but the trio can't bootstrap from raws.

Canonical MSG now: *reduction* (drop raw-producers + cascade-drop consumers of
no-producer materials; availability = raws ∪ outputs-of-survivors, so cycles
survive) + *composition* (backward reachability from products). The
`strict_no_excess` knob is retained as an explicit, default-off, non-canonical
"no-waste" filter.

## Before → after vs the book (verified)

| Example | book maximal | before (strict default) | after (canonical) |
|:--|:--|:--|:--|
| 3.2 ≡ 6.1 | **7** `{O1..O7}`, 19 sol-structs | 3 ✗ | **7 ✓**, dm-SSG = **19 ✓** |
| 4.1 | **7** `{u2,u3,u4,u5,u6,u8,u10}` | 3 ✗ | **7 ✓** |
| 3.3 (example4_3) | **29**, 3465 sol-structs | 0 ✗ | **29 ✓**, dm-SSG = **3465 ✓** |
| 5.1 | 6 | 6 ✓ | 6 ✓ |
| 14.1 | 12 (all product-reaching) | 0 ✗ | **12 ✓** |

| ABB optimum | book/correct | before | after |
|:--|:--|:--|:--|
| 4.1 | `{u2,u4,u8}` = **13** | `{u2,u5,u8}` = 15 ✗ | **13 ✓** |
| 6.1 | `{O2,O5,O7}` = **9** | `{O1,O3,O6}` = 18 ✗ | **9 ✓** |
| 14.1 | `{u1,u4,u8,u11}` = **16** | `None` ✗ | **16 ✓** |

How the bug stayed hidden: the 2026-05-19 validation checked only the ABB
*optimum* (often the bootstrappable core), never the *maximal structure* against
the book — and on Example 4.1 it blessed the suboptimal 15.

## Files touched (all non-core)

**Source:**
- `src/msg.rs` — rewrote `maximal_structure_with_options` to reduction +
  composition; `MaximalStructureOptions::default()` now canonical;
  `strict_no_excess` repurposed as a default-off post-filter (~+90 LOC net).
- `src/abb.rs`, `src/ssg.rs` — `AbbOptions`/`SsgOptions` default
  `strict_no_excess = false`.
- `src/bin/hymeko_pgraph_dump.rs`, `src/bin/pgraph.rs` — canonical is the
  default; new `--strict-no-excess` opt-in; `--relaxed-msg`/`--relaxed` kept as
  deprecated no-ops.
- `src/builder.rs` — doctest updated (HDA optimum 350).

**Tests:** updated to canonical book values —
`tests/{pgraph_e2e,relaxed_msg,pgip_io,multi_objective,axiom_witness,builder}.rs`;
**new** `tests/book_validation.rs` (5-test regression spine asserting the table
above). `multi_objective` switched from `strict_no_excess: true` to canonical
(the methanol weighted-route optima — capex→blue, CO2-heavy→green, H2O-heavy→SMR
— all hold; scalar optimum 2940).

## CORE.YAML items touched

**None.** `hymeko_pgraph` is not a CORE crate; the incidence queries in
`schema.rs` are used read-only. No pinned-dependency change.

## Test results

`cargo test -p hymeko_pgraph --no-fail-fast` — **128 passed, 0 failed** (1
pre-existing ignored). 21 tests across 7 binaries were updated from the buggy
values to the book-verified canonical ones (none loosened to pass — each new
value is the book/hand-verified truth). `book_validation.rs` (new) is the spine;
`ssg_decision_mapping.rs` (3465, Ex 14.1 = 16) still green.

`cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` — clean. Changed
files are individually `rustfmt`-clean (the crate has pre-existing fmt drift in
untouched sibling modules, left alone for a dedicated pass).

## Performance (criterion, release)

| Benchmark | median | vs baseline |
|:--|--:|:--|
| `msg/chain/512` | 10.8 ms | no regression (polynomial; reduction+composition O(\|O\|·\|M\|)/step) |
| `abb/chain/64` | 1.72 ms | — |
| `ssg_dm/example3_3` (3465) | 34.1 ms | +5.2% (under 10% gate; dm-SSG code unchanged — machine variance) |

## Semantic notes (documented design choices)

- **Structural vs operational.** The canonical maximal structure is *structural*
  (axioms S1–S5; admits non-bootstrappable cycles). ABB's `is_feasible` keeps the
  *operational* `close_producible` (bootstrap) check, so ABB still returns
  operable optima (Ex 4.1 → `{u2,u4,u8}`, not a cycle). `assert_minimality`
  (axiom_witness) was switched to the decision-mapping SSG for the
  "MSG = union of solution-structures" identity (book Def. 3.3), which is
  structural; the brute SSG over-counts/under-counts on the operational axis
  (e.g. Ex 3.2 brute = 25 vs structural 19).
- **HDA disposal sink.** Canonical MSG excludes `Disposal` (reaches no product,
  S4); the old strict-vs-canonical S4 divergence on disposal sinks is therefore
  gone. Default HDA optimum is now `{Mixer,Reactor}` = 350.
- **Strict no-waste is now lightly supported & cascade-prone.** With sinks
  excluded by composition, strict mode can collapse (e.g. methanol → 1 unit). It
  is retained as a clearly-labeled non-canonical opt-in; `byproduct_filter_phase11`
  still exercises it on cases where it is well-defined.

## Open issues / follow-ups

- **ABB `max_explored` silent-suboptimal incumbent** (pre-existing, unchanged):
  returns a possibly-suboptimal incumbent silently when the cap trips. The book
  examples explore ≤ tens of nodes, far from the 1M cap.
- **Strict no-waste consistency:** a self-consistent no-waste variant would need
  an MSG that retains byproduct-consumer sinks. Out of scope; flagged for Jean.
- **Crate-wide fmt drift** in untouched sibling modules (pre-existing).

## Provenance

- **Git SHA:** `9abfc3435f55f7443cb07bde4583a17126ac3fc1` (branch
  `feature/pgraph_engine`). Working tree dirty/uncommitted; this change layers on
  top of the earlier (also uncommitted) decision-mapping-SSG work. `tools.yaml`
  dirty = pre-existing case-collision defect (not this change).
- **Verification source:** `d:\Toshokan\P-graphs for process systems engineering (2).pdf`
  (Examples 3.2/3.3/4.1/6.1/14.1; MSG Ch.4; axioms Ch.3).
- **Toolchain:** rustc/clippy 1.93.0; criterion 0.8.2.
- **Determinism:** no RNG; book fixtures committed; assertions are exact unit
  sets / integer counts / costs.
