# P-graph engine minimality + HSIKAN/Gömb gain assessment — Phase 5 (2026-05-19)

## Summary

Two questions in scope:

1. **Does the engine produce minimal sets?** Verified on every shipped
   textbook fixture + the HSIKAN and Gömb architecture-search P-graphs
   used by the real `run_gomb_msg_sweep` driver.
2. **Did HSIKAN / Gömb gain anything from the canonical-axiom fix?**
   Quantified as: the existing search outputs are **already**
   canonical-feasible (the audit certifies them); the engine's
   behaviour is unchanged (so the same architectures get picked); a
   *future* gain is available by opting into the extension bundle.

## Files touched

| File | Status | Notes |
| --- | --- | --- |
| `hymeko_pgraph/tests/axiom_witness.rs` | extended | +9 tests: 5 minimality witnesses + 4 HSIKAN/Gömb canonical-feasibility certificates + 1 diagnostic. Total now 22 tests in this file. |

## What "minimal" means here

Two precise notions the engine must satisfy:

1. **MSG-minimality.** The set returned by `maximal_structure` is
   exactly the set of operating units that appear in *some*
   combinatorially feasible solution structure:

   $$\mathrm{msg.units} \;=\; \bigcup_{s \in \mathrm{ssg\_enumerate}} s.\mathrm{units}.$$

   Equivalently: every unit dropped by MSG cannot appear in any
   feasible solution; every unit kept does appear in at least one.
2. **ABB-optimality.** `abb_solve` returns the solution structure of
   minimum total cost:

   $$\mathrm{abb.cost} \;=\; \min \{ \mathrm{cost}(s)\;\big|\;s \in \mathrm{ssg\_enumerate} \}.$$

Both properties are pinned by `assert_minimality()` in the witness
test file, which (a) computes MSG, (b) enumerates the full SSG, (c)
checks MSG = union of SSG units, (d) brute-force computes the
minimum-cost SSG solution, and (e) asserts ABB returns that cost.

## Minimality verification results

| Fixture | MSG-minimal | ABB-optimal | Notes |
| --- | --- | --- | --- |
| `Chapter6/ex1` (textbook canonical) | ✓ | ✓ (cost 18) | Reproduces the published cost optimum. |
| `Chapter4/ex1` (textbook messy) | ✓ | ✓ (cost 15) | MSG correctly prunes 8 of 11 unfeasible units. |
| `HDA` (with disposal sink) | ✓ | ✓ (cost 400) | Even with the canonical-S4-violating `@Disposal` unit selected, the selection is minimal and cost-optimal under the engine's strict-no-excess predicate. |
| `HSIKAN/sweep_msg` (NAS) | ✓ | ✓ (cost 60) | Engine picks the architectural-cost minimum. |
| `Gomb/sweep_msg` (NAS) | ✓ | ✓ (cost 30) | Engine picks the architectural-cost minimum. |

All 5 minimality witness tests pass:

```
test engine_is_minimal_on_chapter6 ... ok
test engine_is_minimal_on_chapter4_1 ... ok
test engine_is_minimal_on_hda_strict ... ok
test engine_is_minimal_on_hsikan_architecture_sweep ... ok
test engine_is_minimal_on_gomb_architecture_sweep ... ok
```

## Concrete engine outputs on HSIKAN and Gömb sweeps

```
[HSIKAN/sweep_msg]
  MSG units = [cycle_topk_m4, cycle_topk_m16, cycle_topk_m64,
               model_h8, model_h16, model_h32,
               train_short, train_long]   (8 units)
  ABB picks  = [cycle_topk_m4, model_h8, train_short]   (cost 60)

[Gomb/sweep_msg]
  MSG units = [gomb_fast, gomb_slow, gomb_fit]   (3 units)
  ABB picks  = [gomb_fast, gomb_fit]   (cost 30)
```

### HSIKAN: how to read the selection

The HSIKAN architecture sweep has three orthogonal axes:

- **cycle setup** $\in$ {`cycle_topk_m4` (10), `cycle_topk_m16` (40),
  `cycle_topk_m64` (160)}
- **hidden width** $\in$ {`model_h8` (20), `model_h16` (60),
  `model_h32` (200)}
- **training length** $\in$ {`train_short` (30), `train_long` (120)}

ABB picks one from each axis at the cheapest end:
`cycle_topk_m4 + model_h8 + train_short = 10+20+30 = 60`. This is
the cost-minimum architecture **under the canonical S1..S5
feasibility constraints** — i.e. you need *some* cycle setup,
*some* hidden width, and *some* training step, but the cheapest
choice in each slot is structurally feasible. The engine correctly
avoids picking redundant alternatives (e.g.\ two cycle sizes).

### Gömb: how to read the selection

The Gömb sweep has only three units; ABB picks the two cheaper ones
(`gomb_fast` for the cycle-pool builder, `gomb_fit` for the
fitting step), dropping the more expensive `gomb_slow` alternative.

## Canonical-feasibility certificate for HSIKAN/Gömb

The most important finding: **both architecture-search P-graphs
satisfy canonical Friedler S1..S5 — both the full sweep schema and
the ABB-selected sub-schema.** Concretely:

| P-graph | Canonical full schema | Extension full schema | Canonical engine output | Extension engine output |
| --- | --- | --- | --- | --- |
| `HSIKAN/sweep_msg` | **PASS** | **PASS** | **PASS** | **PASS** |
| `Gomb/sweep_msg` | **PASS** | **PASS** | **PASS** | **PASS** |

Compare against the textbook fixtures:

| Fixture | Canonical full | Extension full | Canonical engine | Extension engine |
| --- | --- | --- | --- | --- |
| `HDA` | FAIL [S4] | FAIL [E-NoExcess, E-WellFormed] | FAIL [S4] | FAIL [E-NoExcess, E-WellFormed] |
| `Chapter4/ex1` | FAIL [S2, S4, S5] | FAIL [E-NoExcess, E-ConsumedHasProducer] | PASS | PASS |
| `Chapter4/ex3` | FAIL [S4] | FAIL [E-NoExcess] | (no ABB) | (no ABB) |
| `Chapter6/ex1` | PASS | FAIL [E-NoExcess] | PASS | PASS |

The HSIKAN and Gömb architecture P-graphs are **even cleaner** than
the textbook Chapter 6 fixture — they pass *both* bundles on both
the full schema and the engine output. Chapter 6, by contrast, has
a by-product (material `B`) that the extension bundle flags on the
full schema; the engine correctly drops it.

## Question 1: minimality — verified

The engine produces minimal sets in both the MSG-minimal and
ABB-optimal senses on every measured P-graph: 3 textbook fixtures +
both production architecture-search P-graphs. The chapter-6 cost
optimum of 18 (the published reference) is reproduced. The ABB
selections on HSIKAN/Gömb are the cost-minimum canonical-feasible
selections.

## Question 2: HSIKAN / Gömb gain — honest assessment

### Direct gain: none in the search outcome

The canonical-axiom fix in Phase 1 changed only `axioms.rs` — a
schema-level validator. The MSG/SSG/ABB engine modules were
unchanged (Phase 2 audit confirmed corollary-of-canonical). The
HSIKAN/Gömb sweeps invoke the engine, not the validator. Therefore
**the architectures selected by ABB are byte-identical before and
after the audit**. There is no behaviour change to measure on
HSIKAN/Gömb training.

### Indirect gain: formal certification

The HSIKAN and Gömb architecture-search P-graphs were *already*
satisfying canonical S1..S5 — but until the audit no one had checked
this against verbatim Friedler 1992. The witness suite now provides
the certificate. Concretely:

1. The HSIKAN search space (8 units) and the cost-minimum selection
   (3 units) are canonical Friedler-feasible.
2. The Gömb search space (3 units) and the cost-minimum selection
   (2 units) are canonical Friedler-feasible.
3. The extension bundle (E-NoExcess, E-WellFormed,
   E-ConsumedHasProducer) is also clean on both, so even the
   stricter "no waste" reading accepts both.

This matters for the paper / Pimentel-meeting context: when the
multi-objective P-graph plan ships, the architecture-search results
can be cited as "canonical Friedler 1992-feasible architectures"
with a unit-test backing the claim.

### Future gain: extension bundle as a NAS lever

Where the audit *could* matter for HSIKAN/Gömb in the future:

1. **Stricter waste-free filter.** Add a "memory_overhead" or
   "compute_overhead" raw-material that every architecture choice
   must "consume". If an architecture leaves overhead unaccounted,
   the extension bundle's E-NoExcess will fire and the architecture
   is filtered out of the search. This injects engineering
   constraints (e.g., "no half-used cycle quality") into the NAS.
2. **Source-unit prohibition.** Add E-WellFormed enforcement to
   reject any architecture choice that "consumes from nothing"
   (a useful NAS sanity check; not currently a risk on the HSIKAN/
   Gömb encodings).
3. **A/B test.** Run the existing sweeps with `strict_no_excess =
   true` (current default) vs. `false` and report whether the
   chosen architecture differs. On the current HSIKAN/Gömb encodings
   there is no by-product so this should be a null comparison; if
   anyone adds a by-product material it becomes informative.

These are *opt-in* paths — Phase 1+2's design preserves the engine's
existing behaviour and exposes the canonical / extension bundles as
new tools, not as new defaults.

## §6.5 anti-pattern audit

No new anti-patterns. The minimality helpers (`brute_force_minimum_cost`,
`assert_minimality`) are single-responsibility, parametrised by the
fixture name, and reused across five tests — no Cartesian copy/paste.

## Open issues and follow-up items

1. **Add canonical/extension validation hooks to
   `run_gomb_msg_sweep`.** Currently the driver runs MSG/SSG/ABB
   and reports the selection. Adding a one-line axiom validation
   pass on the selected sub-schema (and printing PASS / FAIL [tags])
   would surface the certificate at sweep time.
2. **Cite this witness in the multi-objective P-graph plan.** The
   `docs/plans/2026-05-19-pgraph-multi-objective/` plan can now
   reference the witness suite as evidence that the multi-objective
   ABB will preserve canonical feasibility on the existing search
   spaces.
3. **Empirical NAS-divergence test.** A useful next step: hand-craft
   a search-space variant of `sweep_msg.hymeko` with an explicit
   by-product (e.g., an architecture that produces an "unused
   gradient" material) and confirm the canonical vs. extension
   selections differ. This is the empirical test of the
   "we gained anything?" question.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (working tree still carries Phases
  1–5 + cortical Slice 1 + book regenerations, all uncommitted per
  the "no commits without explicit ask" policy).
- **Tests:** `cargo test -p hymeko_pgraph` → 22 axiom-witness +
  61 other pgraph + 1 ignored doctest = 84 tests pass.
- **Diagnostic:** `cargo test -p hymeko_pgraph --test axiom_witness
  print_hsikan_and_gomb_abb_selection -- --nocapture` for the
  per-fixture ABB selection.

## Acceptance check

- [x] Engine MSG-minimality verified on Chapter 6, Chapter 4/1, HDA,
      HSIKAN sweep, Gömb sweep.
- [x] Engine ABB-optimality verified on the same five fixtures
      against brute-force SSG minimum.
- [x] HSIKAN architecture-search P-graph + engine output pass
      canonical S1..S5 + extension bundle (4-way PASS).
- [x] Gömb architecture-search P-graph + engine output pass
      canonical S1..S5 + extension bundle (4-way PASS).
- [x] Concrete ABB selections on HSIKAN/Gömb captured for citation.
- [x] Honest assessment: no HSIKAN/Gömb behaviour change because
      the engine's predicate didn't change; the gain is formal
      certification + new opt-in NAS levers.
- [x] Report on disk.
