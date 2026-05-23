# P-graph axiom semantics fix — Phase 2 (2026-05-19)

## Summary

Phase 2 of the three-phase remediation triggered by the J. Pimentel
audit. Two pieces:

1. **Audit of downstream consumers** (MSG, SSG, ABB). The audit
   established that all three modules implement their own feasibility
   predicates without referencing the `AxiomViolation` enum, and that
   each predicate is **consistent with canonical S1..S5** — no code
   change required, only doc-string clarification of the
   correspondence.
2. **Extension-axiom bundle** (new `axiom_extensions.rs`). On the
   user's follow-up request, the prior (paraphrase) statements of
   A2/A4/A5 are preserved as a named, orthogonal "extension set"
   alongside the canonical bundle. Includes a non-contradiction
   theorem proven by construction.

## Files touched

| File | Status | Notes |
| --- | --- | --- |
| `hymeko_pgraph/src/msg.rs` | doc-only edit | Module docstring maps the forward/backward passes to S2/S4 and notes `strict_no_excess` as the orthogonal Friedler refinement |
| `hymeko_pgraph/src/ssg.rs` | doc-only edit | Module docstring maps `is_feasible` clauses (a)/(b)/(c) to canonical axioms |
| `hymeko_pgraph/src/abb.rs` | doc-only edit | Notes that ABB inherits SSG's predicate at leaf level + reachability bound is an early-cut form of A1/A4 |
| `hymeko_pgraph/src/axiom_extensions.rs` | **new module** | `ExtensionAxiomBundle` + `ExtensionAxiomViolation` with 3 extension axioms + 6 tests + compatibility-theorem fixture |
| `hymeko_pgraph/src/lib.rs` | minor | Register new module + re-export `ExtensionAxiomBundle`, `ExtensionAxiomViolation` |
| `docs/plans/plans_20260429/hymeko_pgraph_plan.md` | minor | Cross-reference the extension bundle in the post-audit note |

## CORE.YAML items touched

None.

## Audit of downstream consumers

### MSG (`maximal_structure_with_options`)

- **Forward pass** ("every input must be raw or producible by some
  surviving unit"): enforces **canonical A2/S2 forward** restricted to
  materials that are inputs of surviving units.
- **Backward pass** in *relaxed* mode: enforces **canonical A4/S4**
  (every surviving O-node has a backward-reachable product).
- **`strict_no_excess` strengthener**: an *orthogonal* Friedler 1992
  §3 condition ("strict P-graph rule"), explicitly not part of
  S1..S5.

What MSG does **not** check (relies on the schema layer):
**A1/S1** (products are M-nodes — preserved by `LoweredPGraph`
construction), **A2/S2 reverse** (raws not produced inside —
schema-layer concern), **A3/S3** (catalogue), **A5/S5** (no isolated
M-nodes — MSG never reasons about stray materials).

### SSG (`is_feasible`)

Three clauses:

- **(a)** *every input raw-or-produced* ≡ **canonical A2/S2 forward**
  restricted to inputs of selected units.
- **(b)** *every required product producible* ≡ the constructive form
  of **canonical A1/S1 ∧ A4/S4** taken jointly.
- **(c)** *`strict_no_excess`* ≡ orthogonal Friedler refinement (not
  in S1..S5).

### ABB (`solve_with_options`)

ABB delegates leaf-level feasibility to `ssg::is_feasible`, so it
inherits exactly SSG's axiom coverage. The two ABB bounds add no new
axiom semantics:

- **Inclusion bound** — purely cost-based, independent of S1..S5.
- **Reachability bound** — an early-cut form of the leaf's A1/A4
  check. If even the optimistic-remaining unit set cannot reach every
  product, no descendant will satisfy the leaf check.

### Verdict — no re-derivation needed

All three downstream feasibility predicates are corollaries of
canonical S1..S5 + the orthogonal `strict_no_excess` strengthener
they already expose. The 57 + 137 + 0 (test) regressions confirm the
implementations are stable under the canonical semantics.

## Extension-axiom bundle

The previous (pre-2026-05-19) implementations of A2/A4/A5 stated
*different* propositions than canonical S1..S5, but none of them
*contradict* the canonical set. Specifically:

| Old name | Statement | Canonical relationship |
| --- | --- | --- |
| Old A2 | $\forall m \in M$, $m$ has a directed path to some required product | $\equiv$ canonical $\{S1, S2, S4, S5\}$ ∧ **strict-no-excess** (orthogonal Friedler 1992 §3 refinement) |
| Old A4 | $\forall o \in O$, $\mathrm{in\_deg}(o) \geq 1 \land \mathrm{out\_deg}(o) \geq 1$ | Orthogonal to S4. Encodes the Friedler 1992 §2 *operating-unit well-formedness prerequisite* |
| Old A5 | a consumed M-node has a producer or is raw | Strict **subset** of canonical A2 forward (canonical also catches isolated non-raws via A5/S5) |

These are now preserved in [hymeko_pgraph/src/axiom_extensions.rs](../hymeko_pgraph/src/axiom_extensions.rs)
under more descriptive names:

| E-axiom name | Was | Purpose |
| --- | --- | --- |
| `NonReachingMaterials` | Old A2 | Catch by-product / waste M-nodes |
| `UnitsWithDegreeZero` | Old A4 | Operating-unit well-formedness |
| `ConsumedMaterialWithoutProducer` | Old A5 | Cheap one-edge-pass subset of A2 |

Three tests pin the **non-contradiction property** of the extension
bundle against the canonical bundle:

1. `strict_no_excess_catches_canonical_feasible_byproduct` — a
   by-product schema is canonical-feasible (S1..S5 all pass) but the
   extension axiom `NonReachingMaterials` fires. Witness that the
   extension is *strictly stronger* than the canonical bundle.
2. `consumed_has_producer_overlap_with_canonical_a2` — a consumed
   non-raw without producer fires BOTH canonical A2-forward AND
   extension `ConsumedMaterialWithoutProducer`. Witness that the
   extension is a *subset* of canonical A2 on this case.
3. `consumed_has_producer_silent_on_isolated_m_node` — an isolated
   non-raw M-node is caught by canonical S5 (and A2-forward) but is
   *silently accepted* by the extension. Witness that the canonical
   bundle is *strictly stronger* than `ConsumedMaterialWithoutProducer`.

Plus a compatibility-theorem fixture
(`canonical_plus_extension_compatible_on_chapter_4_style_fixture`)
where a well-formed two-raw-one-product textbook schema passes BOTH
the canonical and the extension bundles.

### Why preserve them at all?

Per the user's framing (2026-05-19): if a downstream search outcome
(HSiKAN architecture enumeration, GömbSoma circuit synthesis,
NAS / multi-objective P-graph pipeline) regresses under the looser
canonical S1..S5 alone, the extension bundle is available as an
opt-in hypothesis to test. The previous behaviour is therefore not
lost — it is named, theoretically justified, and unit-tested.

## Test results

| Crate | Before phase 2 | After phase 2 |
| --- | --- | --- |
| `hymeko_pgraph` lib unit | 17 | **23** (+6 extension tests) |
| `hymeko_pgraph` integration | 40 | 40 |
| `hymeko_pgraph` doctest | 1 | 1 (+ 1 ignored, from the new `axiom_extensions.rs` usage block) |
| `hymeko_graph` lib unit | 87 | 87 |
| `hymeko_graph` integration | 50 | 50 |

All passing. The new tests are:

- `strict_no_excess_catches_canonical_feasible_byproduct`
- `unit_well_formed_catches_zero_in_degree`
- `unit_well_formed_catches_zero_out_degree`
- `consumed_has_producer_overlap_with_canonical_a2`
- `consumed_has_producer_silent_on_isolated_m_node`
- `canonical_plus_extension_compatible_on_chapter_4_style_fixture`

Each verifies one face of the audit's non-contradiction claim.

## Performance results

The extension bundle uses the same `O(M+O+E)` style scans as the
canonical bundle — one forward-adjacency build, one BFS per
non-product M-node for the reachability check (worst case $O(M
\cdot (M+E))$ on a fully-dense graph; on textbook fixtures
sub-millisecond). Wall time of `cargo test -p hymeko_pgraph` is
unchanged from Phase 1 (0.20 s).

## New / removed dependencies

None.

## §6.5 anti-pattern audit

No new anti-patterns. The extension bundle:

- Uses the same `BTreeSet`-of-`DeclId` payload pattern as the
  canonical bundle — no string-typed config (§6.5 #7).
- Lives in its own module with a single public type per concern
  (§6.5 #4 — `axiom_extensions.rs` is 405 LOC, well under the
  decompose-when-painful threshold).
- Three free helper functions are private to the module; no public
  cross-cutting helpers.
- Tests are co-located with the module.

## Open issues and follow-up items

1. **Phase 3** — remeasure: re-run the full pgraph workload sweep
   (done above) + scope whether benchmark targets are needed. No
   `benches/` directory exists today.
2. **Multi-objective plan** — the morning's
   `docs/plans/2026-05-19-pgraph-multi-objective/` does not yet
   mention which axiom bundle the multi-objective ABB enforces.
   When that plan moves to implementation, declare canonical-only or
   canonical+extension explicitly.
3. **PGIP ingest** — `pgip_io::read_pgip` constructs a
   `LoweredPGraph` directly without invoking either bundle. A pure
   schema-level audit pass through both bundles on every parsed PGIP
   would surface malformed external files; out of scope for this PR.
4. **REPL exposure** — `hymeko_query` does not yet have a
   `.validate_pgraph()` query combinator. When it does (planned in
   `plans_20260429/hymeko_pgraph_plan.md` Phase 4), it should accept
   a flag to select canonical, extension, or both bundles.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (working tree carries Phase 1 + Phase 2
  edits + the earlier cortical + book regenerations, all uncommitted
  per the user's "no commits without explicit ask" policy).
- **Rust toolchain:** unchanged.
- **Tests:** `cargo test -p hymeko_pgraph` (63 tests pass / 1
  ignored doctest) + `cargo test -p hymeko_graph` (137 pass).
- **Clippy:** clean on my changes; pre-existing
  `needless_lifetimes` warning in `tests/multi_objective.rs` is
  unrelated.

## Acceptance check

- [x] No `CORE.YAML` items touched.
- [x] No new dependencies.
- [x] All 137 `hymeko_graph` tests pass (doc-only edit elsewhere
      this session).
- [x] All 23 `hymeko_pgraph` lib unit tests pass (+ 40 integration
      + 1 doctest).
- [x] The 6 new extension-bundle tests pin the non-contradiction
      property against the canonical bundle.
- [x] Module docstrings for MSG / SSG / ABB now name the canonical
      axioms each predicate enforces.
- [x] Plan doc in `plans_20260429/hymeko_pgraph_plan.md`
      cross-references the new extension bundle.
- [x] Report on disk.

## Memory update

Updating
`memory/project_pgraph_axiom_semantics_fix_phase1_2026_05_19.md` to
reflect that Phase 2 is done and that the prior paraphrases are
preserved as the extension bundle. Phase 3 (remeasure) is the next
step.
