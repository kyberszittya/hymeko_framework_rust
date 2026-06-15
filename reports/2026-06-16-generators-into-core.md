# Hypergraph generators into `hymeko_core` (canonical home)

**Date:** 2026-06-16 · **Plan:** `docs/plans/2026-06-16-generators-into-core/`
(plan.tex/pdf/tikz/mmd) · **Persona:** Aiko

## CORE.YAML approval

This was a **core edit** (`hymeko_core` is `lockdown: full`). Authorised in chat
(2026-06-16) with the token, quoted verbatim per CORE.YAML §approval and to be
repeated in the commit footer:

```
APPROVED-CORE-EDIT: generators-in-core
```

The token applies only to this slug.

## Summary

The user wanted the hypergraph generators available from `hymeko_core`, the
foundational crate. The only Rust generators lived in `hymeko_hive::generators`
and produced HIVE-store types. Naively copying them into core would duplicate the
algorithm across two crates (forbidden by §6.1 / §6.5 #1). So the algorithm
**moved** to core as a dependency-free, index-level generator, and `hymeko_hive`
was refactored into a thin adapter that delegates. Net result: generators are in
core (the ask), and the algorithm exists exactly once (no duplication).

- **New `hymeko::generators`** (`hymeko_core/src/generators.rs`): a dependency-free
  module producing an index-level `HypergraphDesign { n_vertices, edges:
  Vec<Vec<usize>> }`. Families: Steiner triple systems (canonical Fano `n=7` /
  closed-form Bose `n≡3 mod 6` / MRV backtracking `n≡1 mod 6`), sunflower /
  Δ-system, complete `r`-uniform `K_n^(r)`. Helpers `binom`, `combinations`.
  Typed `HypergraphGenerator` enum with `.design()`. `GeneratorError`.
- **`hymeko_hive::generators`** rewritten as an adapter: `steiner_triple_system`,
  `sunflower`, `complete_uniform` keep their signatures but call
  `hymeko::generators::*` for the index design and map indices → `HiveNode` /
  `HiveRelation` (with HIVE node-role typing). The duplicated algorithm (Bose,
  backtracking, `binom`, `combinations`) and the hive-local
  `HypergraphGenerator` were removed; `GeneratorError` / `HypergraphGenerator`
  are re-exported from core.

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_core/src/generators.rs` | **new**, ~480 LOC | canonical algorithm + design type + enum + 10 tests |
| `hymeko_core/src/lib.rs` | +1 line | `pub mod generators;` |
| `hymeko_hive/src/generators.rs` | 748 → ~330 LOC | rewritten as delegating adapter; 18 tests retained |
| `hymeko_hive/Cargo.toml` | +1 dep | `hymeko_core = { path = "../hymeko_core" }` |
| `docs/plans/2026-06-16-generators-into-core/` | new | 4-format plan |
| `reports/2026-06-16-generators-into-core.md` | new | this report |

## CORE.YAML items touched

- **`hymeko_core` crate (`lockdown: full`)** — new module + one `lib.rs` line.
  **No** edit to `hymeko_core/Cargo.toml`; the generators are pure, no new
  dependency for core. No existing core logic touched, so no published-benchmark
  or RTL-parity path is affected.
- **`hymeko_hive/Cargo.toml`** gains an internal path dependency on
  `hymeko_core`. A dependency add is normally itself a core-protocol event (§1);
  it is part of this approved edit (the wiring that lets the existing hive
  consumer use the core generators without duplication) and is flagged here. It
  is an internal workspace path dep — no external crate, no version pin — and
  introduces no cycle (`hymeko_core` does not depend on `hymeko_hive`).

All covered by `APPROVED-CORE-EDIT: generators-in-core`. No new **external**
dependency anywhere.

## Test results

| Crate / layer | Result |
|---|---|
| `cargo test -p hymeko_core --lib generators` | **10 passed** — binom; combinations distinct/ascending/counted; Steiner valid for every offered n {7,9,13,15,19,21,25} (pair coverage = 1); complete-uniform count/arity/distinctness; sunflower core-intersection contract; enum dispatch; failure cases; S(2,3,25) perf guard |
| `cargo test -p hymeko_hive` | **18 passed** — Fano/Bose shape+coverage, association queries on generated Fano + sunflower (proves the adapter is faithful across all orders incl. the MRV 13/19/25), complete-uniform via HIVE types, error propagation |

The pure-combinatorics property tests moved to core (they are algorithm tests);
hive keeps the HIVE-output tests, which now also prove the delegation is faithful.

## Static analysis

- `cargo clippy -p hymeko_core --all-targets -- -D warnings`: **clean** (the new
  module respects the crate's existing approved allow-list; introduced no new
  suppressions).
- `cargo clippy -p hymeko_hive --all-targets -- -D warnings`: **clean**.
- `cargo fmt -- --check` on both crates: **clean**.

## Performance

Pure combinatorics, `O(output)` except the MRV `n≡1` search (backtrack-free at the
offered orders, step-capped at 2,000,000). `S(2,3,25)` median build ≪ 100 ms
(in-crate perf guard). No runtime dependency added; no published benchmark path
changed. RSS ≪ 16 GB.

## §6.5 anti-patterns

The point of this change was to **remove** a §6.1/§6.5 #1 duplication risk rather
than create one: the algorithm is now single-sourced in core, with hive as a thin
adapter (no second copy). Enum-with-data over string dispatch (#7). No globals
(#11). No `unwrap`/`expect` in non-test code. No new artifact created without a
discovery pass (#12 — the existing `generators.rs` files were found and
refactored, not duplicated).

## Open issues / follow-up

- `hymeko_hive` now compiles `hymeko_core` (heavier build). Correct layering
  (HIVE is above the core IR); accepted as the cost of single-sourcing.
- Other consumers (WASM editor, a future `hymeko_cli generate`) can now call
  `hymeko::generators` directly — the enum is the natural typed entry.
- The earlier `reports/2026-06-15-hive-generators-parity.md` documents the
  now-superseded self-contained hive implementation; this report is its
  follow-on (the algorithm relocated to core).
