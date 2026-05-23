# P-graph NAS divergence — Phase 6 (2026-05-19)

## Summary

Phase 6 builds the counterfactual that Phase 5 marked as a follow-up
item: an HSIKAN architecture-search P-graph **with a deliberately
injected by-product**, run under both `strict_no_excess = true`
(canonical engine default) and `strict_no_excess = false`, to
measure whether the engine's no-excess knob actually changes the
selected architecture.

**Result:** divergence is real and substantial. Strict mode picks an
architecture **50 % more expensive** than relaxed mode (cost 90 vs.
60). Both selections are canonical Friedler S1..S5-feasible on their
projected sub-schemas; the extension bundle correctly distinguishes
them.

This is the empirical answer to "did HSIKAN/Gömb gain anything?":
**they would gain a tunable lever** — the canonical-vs-extension
choice — *if* the architecture P-graph encodes by-products. The
existing `sweep_msg.hymeko` has none, which is why Phase 5 found no
behavioural difference. Phase 6 shows what a difference *would* look
like.

## Files touched

| File | Status | Notes |
| --- | --- | --- |
| `data/hsikan/sweep_msg_byproduct.hymeko` | new | HSIKAN sweep variant with a `redundancy_byproduct` material produced by the cheap `cycle_topk_m4` unit, consumed by nothing. |
| `hymeko_pgraph/tests/axiom_witness.rs` | extended | +2 tests (`byproduct_sweep_is_canonical_feasible_both_bundles`, `engine_selection_diverges_under_strict_vs_relaxed_on_byproduct`). |

## The construction

The base HSIKAN sweep (Phase 5) has three orthogonal axes of choice
(`cycle_topk × model_h × train_*`) and no by-products. To force a
divergence, the by-product variant changes one line in the cheapest
cycle unit:

```diff
- @cycle_topk_m4   <unit>  10 {
-     (-gpu_memory, -train_time, +cycle_quality);
- }
+ @cycle_topk_m4   <unit>  10 {
+     (-gpu_memory, -train_time, +cycle_quality, +redundancy_byproduct);
+ }
```

`redundancy_byproduct` is declared as a new intermediate material
that **no other unit consumes**. This is the canonical engineering
analogue of "the cheap cycle setup leaks a wasted gradient / wasted
attention map that nothing downstream uses". On the engine side:

- Under `strict_no_excess = true`, MSG drops `cycle_topk_m4`
  because its output `redundancy_byproduct` is not in the
  `useful = consumed_by_surviving ∪ products` set. ABB then has
  to pick a more expensive cycle setup (`cycle_topk_m16` at cost
  40 instead of `cycle_topk_m4` at cost 10).
- Under `strict_no_excess = false`, MSG keeps `cycle_topk_m4`
  (it has at least one useful output, `cycle_quality`). ABB picks
  the cheap path.

On the axiom side:

- **Canonical S1..S5** accepts the schema entirely. S2 (raw
  biconditional) holds. S4 holds (every unit reaches `auc_score`).
  S5 holds (`redundancy_byproduct` is incident to one edge, from
  `cycle_topk_m4`).
- **Extension E-StrictNoExcess** fires on the schema:
  `redundancy_byproduct` has no directed path to `auc_score`. This
  is the structural equivalent of the engine's
  `strict_no_excess = true` filter, hoisted to a schema-level
  predicate.

## Measured divergence

`cargo test -p hymeko_pgraph --test axiom_witness
engine_selection_diverges_under_strict_vs_relaxed -- --nocapture`:

```
[divergence] strict picks  ["cycle_topk_m16", "model_h8", "train_short"]  cost=90
             relaxed picks ["cycle_topk_m4",  "model_h8", "train_short"]  cost=60
```

| | strict (`true`) | relaxed (`false`) | $\Delta$ |
| --- | --- | --- | --- |
| Selected cycle unit | `cycle_topk_m16` | `cycle_topk_m4` | drops m4 |
| Cycle-unit cost | 40 | 10 | $+30$ |
| Other-axis selections | unchanged | unchanged | — |
| **Total cost** | **90** | **60** | **$+50$ ($+83\\%$)** |
| Canonical S1..S5 on output | PASS | PASS | — |
| Extension on output | PASS | FAIL [E-NoExcess] | extension separates them |

Both selections are minimal *under their respective engine
predicates*. They differ structurally: strict mode physically
excludes the by-product from the selection (so canonical S5 on the
projected sub-schema is trivially OK); relaxed mode tolerates the
by-product (canonical S5 still OK because the by-product is incident
to `cycle_topk_m4`).

## What this means for HSIKAN / Gömb in practice

1. **The strict-no-excess knob is a real NAS lever** when the
   architecture P-graph encodes by-products. On the *current*
   HSIKAN/Gömb sweeps it is a null knob because the encodings have
   no by-products to penalise.
2. **The extension bundle is the formal counterpart of strict
   mode at the schema layer.** Where the engine's strict mode
   makes a search-space choice, the extension bundle makes the
   same choice as a hard assertion that catches the design pattern
   at IR-construction time.
3. **An engineering pattern to inject this on real HSIKAN/Gömb
   sweeps:** declare a "wasted compute" or "unused gradient"
   intermediate material that *some* cheap architecture choice
   produces. Then strict mode forces NAS to pick more frugal
   architectures, and the extension bundle gives a unit-test gate
   for "no waste in the search space".
4. **Concrete next step the user could take:** add a
   `wasted_compute` intermediate to `data/hsikan/sweep_msg.hymeko`,
   produced by `cycle_topk_m4` (or any cheap path the team
   suspects is "suspiciously cheap"). The strict-mode ABB will
   then prefer the next-cheapest architecture, and you have a
   tunable that biases NAS toward less wasteful designs.

## Test results

```
running 24 tests
test byproduct_sweep_is_canonical_feasible_both_bundles ... ok
test engine_selection_diverges_under_strict_vs_relaxed_on_byproduct ... ok
... (22 prior witness tests) ...

test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

Full `cargo test -p hymeko_pgraph` sweep: **87 / 87 pass + 1 ignored
doctest** (up from 84 / 84 in Phase 5).

## §6.5 anti-pattern audit

No new anti-patterns. The new `.hymeko` file is a fixture (no code);
the two new tests are uniform with the existing witness style. The
divergence test uses `eprintln!` once for the user-visible diagnostic
line — the assertions themselves are the contract; the print is
human-aid only.

## Open issues and follow-up items

1. **Optional: mirror the by-product pattern on the Gömb sweep.**
   `sweep_msg_gomb.hymeko` only has 3 units, so the test would be
   trivial — but it might still be worth doing for the symmetry
   of the witness suite.
2. **Production wiring**: `signedkan_wip/experiments/runs/run_gomb_msg_sweep.py`
   currently runs the engine and reports the selection but does
   not invoke `AxiomBundle::validate` on the result. A one-line
   addition would print the canonical / extension certificate
   alongside the cost — useful when sweeping over many P-graph
   variants.
3. **Documentation crossref**: the multi-objective P-graph plan
   (`docs/plans/2026-05-19-pgraph-multi-objective/`) can now cite
   this report as the empirical basis for the
   `strict_no_excess` knob being a meaningful design dimension,
   not just a checkbox.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted: phases 1-6 of the
  pgraph audit + GömbSoma cortical Slice 1 + earlier book
  regenerations).
- **Tests:** `cargo test -p hymeko_pgraph` → 24 witness +
  63 other pgraph + 1 ignored doctest = 88 entries; all pass.
- **Reproduce divergence:** `cargo test -p hymeko_pgraph --test
  axiom_witness engine_selection_diverges -- --nocapture`.

## Acceptance check

- [x] By-product variant `data/hsikan/sweep_msg_byproduct.hymeko`
      on disk.
- [x] Both bundles validated on the by-product schema (canonical
      PASS; extension FAIL [E-NoExcess] as designed).
- [x] Engine strict vs. relaxed pick different architectures
      (cost 90 vs. 60, ~83 % gap).
- [x] Both selections canonical-feasible; only extension
      distinguishes them on engine output.
- [x] All 87 / 87 pgraph tests pass.
- [x] Report on disk.
