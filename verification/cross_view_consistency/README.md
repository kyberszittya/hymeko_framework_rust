# Cross-view consistency: the extraction-function commuting square

Machine-verification of the HyMeKo T-SMC article's central but previously-unproved claim — *cross-view
consistency*: one canonical IR projects, without drift, into many target views that share non-trivial invariants.

The article asserts this (§codegen, "a cross-view discrepancy is impossible") but pins it on Proposition 3,
which is only about **cost factorization** (compile-once-emit-many) and explicitly "not a commuting diagram (the
codomains differ)". The **value-level** claim was never formalized. This folder closes that gap with an
*extraction-function* diagram: for each format `f`, an extractor `X_f` parses the **emitted** text back into a
structural invariant, and we verify

```
X_f(ε_f(H)) = X_g(ε_g(H))   for all view pairs   (mutual; each X_f parses a different concrete syntax)
X_f(ε_f(H)) = Q(H)          anchored to the IR query where available
```

Unlike Prop. 3 this square has a **common codomain** (the invariant `Q`), so it genuinely commutes.

## Scripts (run from the repo root)

| script | what | result |
|---|---|---|
| `cross_view.py` | drive the **real CLI** emitters (`hymeko emit --format …`) over the robot fixtures × 5 views (URDF/SDF/MJCF/DOT/Mermaid); apply `X_f`; test the square | **16/16 EXACT** (urdf/sdf/mjcf, full numeric invariant) and **16/16 TOPOLOGICAL** (all 5 views) |
| `extract.py` | `ViewExtractor` ABC + one Strategy impl per format → a frozen `KinematicInvariant` (links, mass, actuated joints, parent/child, axis) | — |
| `trace_witness.py` | the **second, non-robotics domain** (requirements traceability): `X_sysml = X_dot = Q` + a requirement-**coverage** invariant, over **two** fixtures (`traceability_smc`, `sysml_cell`) via the live CLI | **2/2** views+coverage agree; coverage 4/4 and 2/2 |
| `drift_demo.py` | **drift prevention**: one semantic edit, single-source emission vs. multi-file maintenance | HyMeKo drift **0**, pair-wise drift **1** |
| `commute_z3.py` | **Z3 proof**: shared-query architecture ⇒ agreement (T1, unsat negation); an untethered view *can* drift (T2, sat) | reduces consistency to per-view faithfulness |
| `storage_regime.py` | **sympy** reframe of Prop. 4: ρ is controlled, vanishing for high arity; robotics (binary joints, d̄≈2) gives ρ≈2, **not** ρ→1 | honest regime table |
| `plot.py` | the report figure (consistency grid + storage-regime curve) | `reports/figures/cross_view_consistency.{png,svg}` |

```
python verification/cross_view_consistency/cross_view.py     # the square, writes reports/cross_view_consistency.json
python verification/cross_view_consistency/commute_z3.py
python verification/cross_view_consistency/storage_regime.py
python verification/cross_view_consistency/trace_witness.py
python verification/cross_view_consistency/plot.py
pytest -p no:randomly verification/cross_view_consistency/tests/
```

## Substrate (measured, not assumed)

The Python binding resolves only **one** import level (26/35 robot fixtures fail to load) and its joint
extraction yields zero joints — so it is unusable here. The **CLI** uses `ModuleStore` (transitive imports) and
the model-based emitter (joints, axes, origins, limits) across all six formats; it is the emitter the article
benchmarks. The verification drives the CLI as a black box (no core edits).

## Two normalized conventions (named, not drift)

- **(W)** URDF emits a synthetic `world` ground anchor as a `<link>` with no inertial; the comparable link set is
  the **mass-bearing** links.
- **(F)** the fixed root weld (world→base) is an explicit joint in URDF/SDF/DOT/Mermaid but **implicit** in MJCF
  (a root body with no joint is welded to the world); the comparable joint set is the **actuated** joints, with
  fixed welds reported separately.

## Two genuine findings (the diagram views are lossy *by design*)

- **DOT** rounds the mass label to 1 decimal (`0.02 kg → 0.0`) and encodes the axis as an **unsigned** letter
  (`(Z)`), so `robot_4wh`'s real `(0,0,-1)` axis reads as `(0,0,1)`. **Mermaid** uses 2 decimals.
- Therefore the precise numeric invariants live in the **data-interchange** formats (URDF/SDF/MJCF), which are
  **exactly** consistent; the **diagram** formats (DOT/Mermaid) carry a topologically faithful projection whose
  numeric labels are a documented, reduced-resolution rendering. The checker tests EXACT across data formats and
  TOPOLOGICAL across all views accordingly.

## Fixed this session — `requirements_sysml`/`requirements_dot` dispatch

The live `requirements_sysml` / `requirements_dot` CLI emitters returned "model extraction failed": the CLI routed
these template-only transforms (`emit()`→`None`, declared `template_dir`) through the kinematic emit branch. Fixed
in non-core `hymeko_cli/src/main.rs` (fall through to the template path when `emit()` yields nothing and a
`template_dir()` is declared; `ModelKind` in core `hymeko_query` untouched). `trace_witness.py` now drives the
**live** emitter (committed-file fallback); live output is byte-identical to the committed witness. Regression
tests in `tests/test_cross_view.py`.

See `reports/2026-06-28-cross-view-consistency.md` and the plan at
`docs/plans/2026-06-28-cross-view-consistency/`. Companion to `verification/propositions/` (P1–P4).
