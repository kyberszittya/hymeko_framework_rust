# P-graph axiom witness — Phase 4 (2026-05-19)

## Summary

Phase 4 builds the **concrete witness** the user asked for: run both
axiom bundles (canonical Friedler 1992 S1..S5 and the orthogonal
`ExtensionAxiomBundle`) on every shipped P-graph fixture, then on the
MSG/ABB **engine output** of each, and assert what's actually true.
The exercise surfaced a substantive semantic finding the earlier
phases had not articulated: **the engine's strict-no-excess
feasibility predicate is not the same as canonical Friedler S1..S5**.
They diverge cleanly on by-product disposal sinks, and the witness
suite documents this with worked examples on the textbook fixtures.

## Files touched

| File | Status | Notes |
| --- | --- | --- |
| `hymeko_pgraph/tests/axiom_witness.rs` | **new integration test** (12 tests) | Per-fixture validation + engine-output validation + synthetic divergence + diagnostic dump |

## Diagnostic dump

Running `cargo test -p hymeko_pgraph --test axiom_witness -- --nocapture`
produces this table (the `S{i}` tags name the canonical axiom that
fired; `E-*` tags name the extension axiom):

```
[HDA]
  full schema    canonical = FAIL [S4]
  full schema    extension = FAIL [E-NoExcess, E-WellFormed]
  MSG units = 4
  ABB units = 3 (cost 400)
  engine output  canonical = FAIL [S4]
  engine output  extension = FAIL [E-NoExcess, E-WellFormed]

[Chapter4/ex1]
  full schema    canonical = FAIL [S2, S4, S5]
  full schema    extension = FAIL [E-NoExcess, E-ConsumedHasProducer]
  MSG units = 3
  ABB units = 3 (cost 15)
  engine output  canonical = PASS
  engine output  extension = PASS

[Chapter4/ex3]
  full schema    canonical = FAIL [S4]
  full schema    extension = FAIL [E-NoExcess]
  MSG units = 0
  ABB = NONE (infeasible)

[Chapter6/ex1]
  full schema    canonical = PASS
  full schema    extension = FAIL [E-NoExcess]
  MSG units = 3
  ABB units = 3 (cost 18)
  engine output  canonical = PASS
  engine output  extension = PASS
```

## The four substantive findings

### 1. Chapter 6 is the only fixture that is canonical-feasible as shipped

Chapter 6 example 1 passes canonical S1..S5 on its full schema. It
*does* fire `E-NoExcess` on material `B` (a by-product produced by
`@O2` that no unit consumes downstream), so the extension bundle
flags it. MSG correctly prunes the producer of `B` from the maximal
structure; the ABB-selected sub-schema passes **both** bundles. This
is the textbook "clean" case.

Test: [`chapter6_full_schema_passes_canonical`](hymeko_pgraph/tests/axiom_witness.rs#L101-L118)
+ [`chapter6_engine_output_satisfies_canonical_and_extension`](hymeko_pgraph/tests/axiom_witness.rs#L120-L135).

### 2. HDA's `@Disposal` sink permanently violates canonical S4

The HDA reference P-graph encodes the by-product-disposal pattern:
`@Reactor` produces `+Methane` as a by-product, and `@Disposal`
consumes Methane producing nothing. Under **canonical S4** ("∀ O-node,
∃ path to a product"), `@Disposal` has no outputs → no path possible
→ canonical S4 fires. Under **strict-no-excess feasibility** (Friedler
1992 §3 orthogonal refinement), `@Disposal` is *required*: without
it Methane has no consumer, so the selection isn't strict-feasible.

The engine's default predicate (`strict_no_excess = true`) selects
`@Disposal` and therefore the **ABB output also fails canonical S4**.
This is not a bug — it is the canonical/engineering divergence the
audit's whole point was to expose. When the user opts into
`strict_no_excess = false`, the engine drops `@Disposal` (Methane is
vented) and the ABB output then satisfies canonical S1..S5.

Tests:
- [`hda_full_schema_violates_canonical_s4_on_disposal_sink`](hymeko_pgraph/tests/axiom_witness.rs#L140-L153)
- [`hda_engine_output_under_strict_no_excess_still_violates_canonical_s4`](hymeko_pgraph/tests/axiom_witness.rs#L155-L177)
- [`hda_engine_output_under_relaxed_no_excess_drops_disposal_and_passes_canonical`](hymeko_pgraph/tests/axiom_witness.rs#L179-L201)

### 3. Chapter 4 example 1 is a "messy" textbook fixture; MSG prunes it to canonical

Chapter 4-1 ships with 18 materials and 11 units, deliberately
including: 5 non-raw materials with no producer (`K, L, N, Q, V`), 1
raw that is also produced inside (`H`), an isolated material (`L`),
and dead-branch units that don't reach the product (`u11`). All four
canonical-violation kinds fire: S2 forward + S2 reverse + S4 + S5.
This is a stress-test for MSG: the textbook expects MSG to prune the
unfeasible parts and find the clean sub-structure.

After MSG: 3 units survive. After ABB: cost-15 optimum selected. The
selected sub-schema **passes canonical S1..S5**. This is the
strongest empirical evidence that the engine correctly extracts a
canonical-feasible solution from a messy input.

Tests:
- [`chapter4_1_full_schema_is_intentionally_messy`](hymeko_pgraph/tests/axiom_witness.rs#L205-L228)
- [`chapter4_1_engine_output_satisfies_canonical_after_msg_prune`](hymeko_pgraph/tests/axiom_witness.rs#L230-L238)

### 4. Chapter 4 example 3 has no strict-feasible structure

MSG in strict mode prunes every unit (existing
`relaxed_msg::chapter4_3_strict_collapses_relaxed_does_not` test
already documented this). ABB returns `None`. Engineering reading:
the fixture has no waste-free solution. Relaxed-mode MSG keeps units
and ABB returns a (more permissive) solution that satisfies canonical
S1..S5.

Tests:
- [`chapter4_3_msg_prunes_every_unit_strict_mode`](hymeko_pgraph/tests/axiom_witness.rs#L240-L255)
- [`chapter4_3_engine_output_under_relaxed_satisfies_canonical`](hymeko_pgraph/tests/axiom_witness.rs#L257-L278)

### 5. Synthetic divergence pinned at the API level

Two minimal hand-built schemas exercise the divergence with no
fixture dependence:

- A schema with a by-product B (canonical passes; extension fires
  E-NoExcess).
- A schema with a source unit U (canonical passes; extension fires
  E-WellFormed).

Tests:
- [`synthetic_byproduct_canonical_passes_extension_fails`](hymeko_pgraph/tests/axiom_witness.rs#L282-L307)
- [`synthetic_source_unit_canonical_passes_extension_fails`](hymeko_pgraph/tests/axiom_witness.rs#L309-L329)

## Engine vs canonical — clean statement of the relationship

The engine's feasibility predicate is **not** equivalent to canonical
Friedler S1..S5. The precise relationship is:

```
Engine-feasible (strict_no_excess = true)
    ≡  (products reachable from raws via the selection)
        ∧  (every input of every selected unit is raw or produced)
        ∧  (every produced non-product non-raw is consumed)
```

vs.

```
Canonical Friedler S1..S5
    ≡  S1 (products are M-nodes)
        ∧  S2 (M has no ancestor ⟺ M is raw)
        ∧  S3 (O-node in catalogue)
        ∧  S4 (every O-node has a path to a product)
        ∧  S5 (every M-node has ≥ 1 incident edge)
```

**Where they agree:** on schemas with no by-products (Chapter 6,
the "clean" case) and on the *core feasible substructure* extracted
by MSG when the input is messy (Chapter 4-1).

**Where they diverge:** disposal-sink units (HDA `@Disposal`). The
engine accepts them under strict-no-excess because they consume
by-products; canonical S4 rejects them because they have no
forward path to a product. Both readings are internally consistent.
They model two different definitions of what "feasible operating
unit" means:

- **Engineering reading (engine):** an operating unit is anything
  in the catalogue that consumes / produces materials. Disposal
  units are fine.
- **Canonical Friedler reading:** an operating unit must
  *contribute to product synthesis*. Disposal units don't, so they
  are outside the synthesis structure.

## Test results

| Sub-suite | Tests | Status |
| --- | --- | --- |
| `axiom_witness` integration | 12 | **12 pass** |
| `hymeko_pgraph` total | 76 | 76 pass + 1 ignored doctest |

Pre-Phase-4 totals were 23 lib unit + 40 integration + 1 doctest =
64; Phase 4 adds 12 integration tests → **76 total**.

## CORE.YAML items touched

None.

## §6.5 anti-pattern audit

No new anti-patterns. The witness test file:
- Uses one helper per concern (`parse_and_lower`, `validate_*`,
  `project_schema`) — no Cartesian variants.
- Per-fixture tests are uniform in shape (parse → validate → project
  → reassert). Three lines apart from the assertion is acceptable —
  trying to abstract this further would obscure the headline finding
  per test.
- No globals, no string-typed config, no `unwrap()` outside test
  expectations.

## Implications for downstream work

1. **For NAS / multi-objective P-graph (the morning's
   `2026-05-19-pgraph-multi-objective` plan):** declare *which*
   feasibility notion the multi-objective ABB enforces. If the
   target is canonical S1..S5, disposal-sink units must not appear
   in the catalogue or must be modelled as having a "waste-product"
   M-node on the product list. If the target is
   engineering-feasibility, the current engine behaviour is
   correct.
2. **For PGIP ingest:** the `pgip_io::read_pgip` path should
   probably emit a warning when the input schema would fail
   canonical S1..S5 on raws-produced-inside or non-raw-without-
   producer. This is essentially an opt-in lint, not a hard error.
3. **For the user's "HSiKAN / GömbSoma might regress" hypothesis:**
   if the architecture-enumeration P-graph contains disposal-sink
   units (or analogues — e.g., a layer that consumes a signal but
   produces no further output), the canonical vs. extension choice
   will affect the search space materially. Easiest path: A/B test
   the same NAS run with `strict_no_excess = true` vs. `false`
   and compare optima.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (working tree carries the four-phase
  edits + GömbSoma cortical Slice 1 + earlier book regenerations,
  all uncommitted per the user's "no commits without explicit ask"
  policy).
- **Tests:** `cargo test -p hymeko_pgraph` → 76 pass / 1 ignored
  doctest.
- **Diagnostic dump:**
  `cargo test -p hymeko_pgraph --test axiom_witness -- --nocapture
   diagnostic_dump_canonical_vs_extension_vs_engine`.

## Acceptance check

- [x] No `CORE.YAML` items touched.
- [x] Witness test file lands as `tests/axiom_witness.rs`.
- [x] All 12 witness tests pass.
- [x] Full `hymeko_pgraph` sweep still passes (76 / 76).
- [x] Engine-vs-canonical divergence is articulated with concrete
      examples on every shipped fixture.
- [x] Both bundles' results are dumped per fixture for transparent
      inspection.
- [x] Report on disk.
