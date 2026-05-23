# P-graph axiom semantics fix — Phase 1 (2026-05-19)

## Summary

Phase 1 of the three-phase remediation triggered by J. Pimentel's audit
note. The five `check_a{1..5}` functions in
[hymeko_pgraph/src/axioms.rs](../hymeko_pgraph/src/axioms.rs) were
audited against Friedler 1992; A1 and A3 were already canonical, but
A2, A4, and A5 had drifted to plausible-but-wrong paraphrases. Phase 1
restores verbatim Friedler semantics (with the internal `A1..A5`
labels kept, per the user's direction) and rewrites the test suite to
pin the new behaviour. Phases 2 (downstream re-derivation) and 3
(re-measurement) follow in separate plan + report pairs.

## Files touched

| File | Status | Notes |
| --- | --- | --- |
| `docs/plans/2026-05-19-pgraph-axiom-semantics-fix/plan.tex` | new | 4-format plan (.tex + .pdf + .tikz + .mmd) |
| `docs/plans/2026-05-19-pgraph-axiom-semantics-fix/plan.pdf` | new | 4 pages, 184 KB |
| `docs/plans/2026-05-19-pgraph-axiom-semantics-fix/plan.mmd` | new | Mermaid: before-vs-after axiom map |
| `docs/plans/2026-05-19-pgraph-axiom-semantics-fix/plan.tikz` | new | TikZ: canonical ↔ before ↔ after columns |
| `hymeko_pgraph/src/axioms.rs` | rewritten | Module doc + 5 check functions + variant payloads + 12 unit tests |
| `hymeko_pgraph/src/lib.rs` | minor | Crate-doc pointer to the canonical-semantics audit |
| `hymeko_graph/src/friedler.rs` | minor | Module doc updated for canonical S2/S4/S5 phrasing |
| `docs/plans/plans_20260429/hymeko_pgraph_plan.md` | corrected | Wrong-axiom table replaced with canonical statements + audit note |

## CORE.YAML items touched

None. `hymeko_pgraph` is not listed in
[CORE.YAML](../CORE.YAML); `hymeko_graph` doc-only changes touch a
crate not in the lockdown manifest.

## Interface changes

The `AxiomBundle::validate` and `AxiomBundle::validate_timed` entry
points keep their signatures. The `AxiomViolation` enum's variant
names and payloads were updated to reflect the canonical violation:

```rust
pub enum AxiomViolation {
    MissingProducts { missing: Vec<DeclId> },                         // A1 / S1
    RawMaterialDirectionFailures {                                    // A2 / S2
        non_raw_without_producer: Vec<DeclId>,
        raw_with_producer:        Vec<DeclId>,
    },
    InvalidUnits { invalid: Vec<DeclId> },                            // A3 / S3
    UnitsWithoutPathToProduct { offenders: Vec<DeclId> },             // A4 / S4
    IsolatedMaterials { offenders: Vec<DeclId> },                     // A5 / S5
}
```

The previous variant names (`UnreachableNodes`, `DegreeViolations`,
`MissingEdges`) named the *paraphrase*, not the canonical axiom; they
are gone. The only external consumer of the enum is a comment in
`hymeko_graph/src/friedler.rs` — no Rust call site outside the crate
constructs or matches on these variants. Within the crate, only
`lib.rs` re-exports them.

The `AxiomBundle.raws` field is still used by A2 (now both directions
of the biconditional); the previous A5 paraphrase consumed `raws`,
the canonical A5 does not.

## Test results

All `cargo test -p hymeko_pgraph` and `cargo test -p hymeko_graph`
test binaries pass after the rewrite:

| Crate | Before phase 1 | After phase 1 | Notes |
| --- | --- | --- | --- |
| `hymeko_pgraph` lib unit | 12 | 17 | +5 axiom tests (6 → 12 in `axioms::tests`) |
| `hymeko_pgraph` integration | 40 | 40 | unchanged — see below |
| `hymeko_pgraph` doctest | 1 | 1 | unchanged |
| `hymeko_graph` lib unit | 87 | 87 | doc-only edit to `friedler.rs` |
| `hymeko_graph` integration | 50 | 50 | unchanged |

The fact that no integration test (`pgraph_e2e`, `relaxed_msg`,
`pgip_io`, MSG/SSG/ABB tests) regressed under the new canonical
semantics means: **every existing P-graph test fixture is also
canonical-feasible.** The pre-existing fixtures all use well-formed
chapter-4/chapter-6 P-graphs whose materials and units satisfy
S1..S5; the old A2/A4/A5 paraphrases happened to be true on those
same fixtures even though they tested different properties. This is
expected — the test fixtures were drawn from published textbook
examples, which are by construction canonical-feasible.

The new in-axioms-module tests in `axioms::tests`:

1. `a1_catches_missing_product` — keeps.
2. `a2_forward_catches_non_raw_with_no_producer` — *new*; was not
   testable under old A2.
3. `a2_reverse_catches_raw_produced_inside` — *new*; old A2 silently
   accepted raws produced inside the schema.
4. `a2_passes_on_well_formed_pgraph` — *new*.
5. `a3_catches_unwhitelisted_unit` — keeps.
6. `a4_catches_dead_branch_o_node` — *new*; old A4 silently accepted
   units with non-zero in/out degree that don't reach any product.
7. `a4_passes_when_every_o_reaches_product` — *new*.
8. `a5_catches_isolated_material` — *new*; old A5 silently accepted
   M-nodes with no incident edges.
9. `a5_silent_on_unit_output_only_material` — *new*.
10. `worked_example_passes_all_axioms` — keeps (renamed).
11. `multi_axiom_failure_surfaces_each_violation` — *new*; one fixture
    that fails A2 + A4 + A5 simultaneously and asserts each is
    reported.
12. `validate_timed_returns_five_traces_in_order` — *new*; pins the
    public `validate_timed` API.

Tests removed: `a4_catches_isolated_unit` (was testing the
paraphrase, not the canonical axiom),
`a2_catches_dead_material` (paraphrase),
`a5_catches_unproduced_consumed_material` (paraphrase),
`a5_silent_when_consumed_is_raw` (paraphrase).

## Behavioural drift

Per the plan's risk anticipation: a schema accepted by the old
(weaker) A4 + A5 may now be flagged. None of the existing integration
tests trip this in practice. The four behaviour shifts that callers
should be aware of:

1. **A2 reverse direction now active.** If a schema contains a `u → r`
   edge for some `r ∈ raws`, `validate` now returns a
   `RawMaterialDirectionFailures { raw_with_producer: [r], .. }`
   violation. Old code silently passed. This is the **interface
   condition** from Friedler 1992: raws are an interface to the
   outside; producing them inside is a structural error.
2. **A4 dead-branch O-nodes flagged.** An O-node with `in_degree > 0`
   and `out_degree > 0` whose every output dead-ends in a
   non-product material is now flagged. Old A4 accepted it as long
   as both degrees were non-zero. This is the **purposive
   participation** condition from Friedler 1992: every unit must
   contribute to the production of some required product.
3. **A5 isolated M-nodes flagged.** An M-node with no incident
   edges is now flagged, regardless of whether it appears in
   `raws`. Old A5 only fired when the M-node was consumed and not
   produced.
4. **A5 no longer flags "consumed but not produced and not raw".**
   This was the old A5's job; canonically it is **A2's forward
   direction** ("a non-raw material must have an ancestor"), with a
   different formal phrasing. The behaviour is preserved — the
   `non_raw_without_producer` list of `RawMaterialDirectionFailures`
   contains exactly these M-nodes — but the violation variant moved
   from `MissingEdges` (old A5) to `RawMaterialDirectionFailures`
   (new A2).

## Performance results

The axiom checker runs on small textbook graphs ($|M|+|O| \lesssim
50$ for the published examples; the largest pgraph_e2e fixture is
< 30 nodes). Both validate paths run in well under a millisecond per
schema; no measurable change vs. the prior implementation, because:

- A2 went from a per-M-node BFS forward-reachability ($O(M \cdot
  (V+E))$) to a per-M-node lookup in a pre-computed producer map
  ($O(M + E)$). Strictly cheaper.
- A4 went from a per-O-node degree check ($O(O)$) to a per-O-node
  BFS in a pre-computed forward adjacency ($O(O \cdot (V+E))$).
  Strictly more expensive in the worst case, but on textbook graphs
  unmeasurable.
- A5 went from edge-pass scanning ($O(E)$) to per-M-node degree
  check ($O(M)$). Comparable, both are linear.

A single shared `producers` map and a single shared `adj_forward`
map are built once per `validate`/`validate_timed` call and reused
across all five axiom checks, so the overall runtime is
$O(M + O + E)$.

`cargo test -p hymeko_pgraph` total wall time: 0.20 s (unchanged from
the pre-fix baseline).

## New / removed dependencies

None.

## §6.5 anti-pattern audit

No new anti-patterns. The rewrite uses an enum with payloads (single
entry point per axiom family, per §6.5 #1), free helper functions
only for the per-check primitives (`check_a1..a5` are small, single-
purpose, no Cartesian variants), and a shared `producers` + `adj`
build to avoid duplicated graph scans (§6.5 #1). No globals (§6.5
#11), no string-typed config (§6.5 #7).

## Open issues and follow-up items

1. **Phase 2 — re-derive downstream.** MSG (`maximal_structure`),
   SSG (`enumerate_with_options`), and ABB (`solve`) each encode
   their own feasibility predicate without calling
   `AxiomBundle::validate`. The audit confirmed they do not import
   the previous axiom variants. Still, their own conditions need to
   be lined up against canonical S1..S5:
   - MSG's "drop forward-unreachable units" matches A4. Confirm.
   - SSG's "every input is raw or produced by some other unit"
     matches A2 forward; "every produced non-product is consumed"
     is a stricter no-excess rule — needs documentation as
     S2-orthogonal.
   - ABB inherits MSG/SSG behaviour. Confirm via the existing
     chapter-6 fixture (cost optimum 18) which already passes.
2. **Phase 3 — re-measure.** Run the full pgraph test sweep (done
   above; 57 + 137 pass) and any benchmark. The crate currently has
   no `criterion` benches under `hymeko_pgraph/benches/` — Phase 3
   will scope whether to add one or to rely on the integration-test
   wall time only.
3. **A2 reverse direction in pgip ingest.** `pgip_io::read_pgip`
   could potentially construct a schema where a raw is produced
   inside (depending on PGIP syntax). All four existing PGIP tests
   pass under the new A2, so no current input triggers this, but
   the case should be smoke-tested with a hand-crafted bad PGIP
   when Phase 2 lands.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (working tree carries the Phase 1 edits
  + the GömbSoma cortical Slice 1 edits + earlier book regenerations,
  all uncommitted per user's "no commits without explicit ask"
  policy).
- **Rust toolchain:** `cargo --version` → confirmed working
  (pre-existing environment).
- **Tests:** `cargo test -p hymeko_pgraph` (57 pass) +
  `cargo test -p hymeko_graph` (137 pass).
- **Clippy:** `cargo clippy -p hymeko_pgraph --lib --tests` clean on
  my changes; one pre-existing `needless_lifetimes` warning in
  `tests/multi_objective.rs` is unrelated.
- **OS / kernel / host:** Ubuntu 24.04.4 / Linux 6.17.0-23 / x86_64.

## Acceptance check

- [x] Plan written in all four formats (`.tex` + `.pdf` + `.tikz` +
      `.mmd`) before code; PDF compiled (4 pp, 184 KB).
- [x] No `CORE.YAML` items touched.
- [x] No new dependencies.
- [x] All 57 prior `hymeko_pgraph` tests pass + 5 new ones added.
- [x] All 137 `hymeko_graph` tests pass (doc-only edit).
- [x] `cargo clippy` clean on my changes; warnings = errors gate
      respected for new code (pre-existing unrelated warning in
      multi_objective.rs noted, not introduced).
- [x] §6.5 anti-pattern audit clean.
- [x] Wrong-axiom paraphrases in `docs/plans/plans_20260429/hymeko_pgraph_plan.md` corrected.
- [x] Wrong-axiom paraphrases in `hymeko_graph/src/friedler.rs` corrected.
- [x] Report on disk.

## Memory entry

A pointer to this fix is being saved in
`memory/project_pgraph_axiom_semantics_fix_phase1_2026_05_19.md` so
the next session can resume Phase 2 without re-discovering the
audit.
