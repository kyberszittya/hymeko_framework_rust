# HIVE hypergraph generators — parity with the web editor

**Date:** 2026-06-15 · **Plan:** `docs/plans/2026-06-15-hive-generators-parity/`
(plan.tex/pdf/tikz/mmd) · **Persona:** Aiko

## Summary

The web editor (`docs/editor/views/generators.js`) had three parametric
hypergraph generators — Steiner triple systems `S(2,3,n)`, sunflowers
(Δ-systems), and complete `r`-uniform `K_n^(r)`. The Rust framework
(`hymeko_hive/src/generators.rs`) had only two, and the Steiner one was partial.
This change brings the Rust generators to parity with the editor and adds a typed
dispatch entry, so the generators are usable programmatically (CLI/tests/demos),
not only in the browser. Per the user's follow-up, tests cover every generator
family and every offered parameter.

Concretely:

1. **Steiner `n ≡ 1 (mod 6)`** — a deterministic most-constrained-pair (MRV)
   backtracking search with a step cap, mirroring the editor's `stsBacktrack`.
   This is what unlocks `n = 13, 19, 25` (previously unbuildable in Rust). `n = 7`
   keeps the canonical hard-coded Fano block set; `n ≡ 3 (mod 6)` keeps the
   closed-form Bose construction.
2. **Complete `r`-uniform `K_n^(r)`** — new `complete_uniform(n, r)` emitting all
   `C(n,r)` size-`r` relations, with private `combinations` (lexicographic) and
   exact `binom` helpers.
3. **Unified typed entry** — `HypergraphGenerator` enum-with-data
   (`SteinerTriple`, `Sunflower`, `CompleteUniform`) with one `generate()` method,
   mirroring the JS registry. Enum-with-data over string dispatch
   (CLAUDE.md §6.5 #7): an out-of-range request fails at construction, not in an
   inner arm.

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_hive/src/generators.rs` | 305 → 748 LOC (~+443) | backtracking search, complete-uniform, `binom`/`combinations`, `HypergraphGenerator` enum, contracts, 8 new tests |
| `docs/plans/2026-06-15-hive-generators-parity/` | new | plan.tex/pdf/tikz/mmd |
| `reports/2026-06-15-hive-generators-parity.md` | new | this report |

`hymeko_hive/` is an **untracked** crate (not yet committed), so there is no git
diff baseline; line counts are absolute. No other files changed.

## CORE.YAML items touched

**None.** `hymeko_hive` is not listed in CORE.YAML (locked crates: `hymeko_core`,
`hymeko_query`, `hymeko_client`, `hymeko_daemon`, `parser`). The foundational
`hymeko_core` crate (`lockdown: full`) was **not** edited — it holds the GGK
4-tuple types; generators do not belong there. No dependency added or changed.

## Interface changes (additive only)

- `pub fn complete_uniform(n, r) -> Result<GeneratedHypergraph, GeneratorError>`
- `pub enum HypergraphGenerator { SteinerTriple{n}, Sunflower{petals,core,petal}, CompleteUniform{n,r} }`
  with `pub fn generate(&self) -> Result<GeneratedHypergraph, GeneratorError>`
- `GeneratorError`: new variants `SteinerSearchExhausted(usize)`,
  `InvalidComplete{n,r}`. `steiner_triple_system` now accepts a strictly **wider**
  domain (all valid Steiner orders); no signature changed; no caller breaks.

## Design by contract

- `steiner_triple_system`: pre `n ≥ 7 ∧ n ≡ 1|3 (mod 6)`; post `n(n-1)/6` arity-3
  relations, every pair covered once (`debug_assert!` on `pair_coverage`).
- `complete_uniform`: pre `2 ≤ r ≤ n`; post `C(n,r)` distinct arity-`r` relations
  (`debug_assert_eq!` on count).

## Test results (`cargo test -p hymeko_hive`)

| Layer | Result |
|-------|--------|
| Unit — `binom` known values; `combinations` distinct/ascending/counted + r=0 and r>n edge cases | pass |
| Property — `steiner_systems_valid_for_every_offered_order` (7,9,13,15,19,21,25): count = `n(n-1)/6`, arity 3, every pair covered exactly once. **13/19/25 are the regression cases** that the pre-parity code could not build | pass |
| Failure — Steiner rejects {4,5,6,8,10,11,12,14}; complete rejects `r>n` and `r<2` | pass |
| Complete — `C(n,r)` edges, each arity `r`, all distinct, for (4,3)(5,3)(6,2)(6,4) | pass |
| Integration — `HypergraphGenerator::generate()` == the free functions; error surfaces through the enum | pass |
| Perf guard — `steiner_25_builds_under_budget` (median-of-5, <100 ms) | pass |

**22 passed / 0 failed** for the whole crate (12 generator + 10 store/query),
0.35 s. Doc-tests: 0.

## Performance

- `S(2,3,25)` (worst MRV case in the offered list, 100 triples): median build well
  under the 100 ms guard — the MRV search is effectively backtrack-free at these
  sizes. The step cap (2,000,000) bounds any pathological larger `n` to a clean
  `Err`, never a hang and never a silently truncated result.
- Memory: a few KB per generated object; RSS ≪ 16 GB cap. No allocation in inner
  loops beyond the `completions` candidate vector.
- This is a **regression guard** (`Instant`, diagnostic per CLAUDE.md §10), not a
  reportable criterion benchmark — no perf claim is made beyond "builds fast
  enough to gate the offered parameter list," matching the JS test's intent.

## Static analysis

- `cargo clippy -p hymeko_hive --all-targets -- -D warnings`: **clean**.
- `cargo fmt -p hymeko_hive -- --check`: **clean** (applied).
- Complexity: each new function is small; the backtracking is split into a
  `SteinerSearch` struct with focused methods (`completions`, `pick_pair`,
  `solve`) rather than one large free function — under the §6.2 ceilings.

## §6.5 anti-patterns

None introduced. The three families are exposed both as direct builder functions
(existing style) and behind one `HypergraphGenerator` enum (no string-typed
dispatch, #7; no per-cell function explosion, #1). No new globals, no `unwrap`/
`expect` in non-test code, no broad error swallowing. Discovery pass found the
existing `generators.rs` and extended it rather than creating a new module (#12).

## Open issues / follow-up

- More design families (Pasch configuration, projective planes, `K₅³`) are a
  separate backlog item — the editor lists them as P4 too. The enum makes adding
  them a one-variant change.
- No WASM/CLI surface wired yet; this is the library layer. If a `hymeko_cli`
  `generate` subcommand is wanted, the enum is the natural entry.
