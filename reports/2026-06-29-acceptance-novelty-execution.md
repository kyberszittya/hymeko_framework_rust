# Acceptance-and-novelty plan — execution (WS1 + WS2 + WS3 + WS4)

**Date:** 2026-06-29 (JST) · **Author:** Aiko (Claude Code), for Dr. Csaba Hajdu & Ádám Csapó
**Plan:** `docs/plans/2026-06-28-acceptance-novelty/` · **Decisions:** second domain = SysML systems engineering; scope = Full; WS4 (formal lift) added on request (Csapó review pending).

## Summary

Executed the "Full" path of the acceptance/novelty plan to address the two remaining T-SMC swing factors —
*novelty reads as incremental* and *generality thinly evidenced*. The second (systems-engineering) domain is now a
first-class, machine-verified result with a real systems invariant; a drift-prevention demonstration makes the
contribution visceral; and the novelty framing is sharpened to *consistency by convention → machine-verified
property*. No central result changed; its evidence and framing did.

## WS1 — Second domain as a first-class result (DONE)

Generalised `verification/cross_view_consistency/trace_witness.py` from a one-fixture witness into a
requirements-domain cross-view checker over **two fixtures** (`data/paper/traceability_smc.hymeko`,
`data/profiles/sysml_cell.hymeko`) × **two independently-parsed views** (`requirements_sysml`, `requirements_dot`)
driven by the live CLI. Beyond entity/relation counts it recovers a genuine systems-engineering invariant —
requirement **coverage** (a requirement is covered iff directly satisfied or allocated to a component) — from each
view and checks agreement.

**Result (measured):**
```
pick_place_cell  req=4 blocks=4 edges=7 coverage=4/4 | views-agree=T coverage-agree=T Q-anchored=T
sysml_cell       req=2 blocks=2 edges=2 coverage=2/2 | views-agree=T coverage-agree=T Q-anchored=T
```
`X_sysml(ε_sysml(H)) = X_dot(ε_dot(H)) = Q(H)` on 2/2 fixtures; coverage agrees across views on 2/2. A coverage
discrepancy across views is impossible by Prop. cross-view — the same proposition that rules out a link-count
mismatch across robot formats.

## WS2 — Drift-prevention demonstration (DONE)

New `drift_demo.py`: one semantic edit (add a requirement + its `satisfies` edge), two regimes.
```
(1) HyMeKo single-source : edit the .hymeko, re-emit both views -> drift = 0 (new req in both views)
(2) Pairwise multi-file  : edit the SysML file only            -> drift = 1 (req in SysML, absent from DOT)
```
The test **asserts the baseline drifts** (≥1), so the demo is real, not a strawman: the same edit is benign under
single-source emission and silently divergent under pair-wise maintenance.

## WS3 — Novelty framing (DONE)

- **Abstract:** "cross-view invariants a machine-verified property rather than a tooling convention"; added the
  second-domain + drift sentence.
- **Introduction:** five properties (was four), each machine-verified; new contribution **C5** (machine-verification
  across two domains + drift demo); novelty paragraph rewritten to the explicit delta — prior art (EMF, LLVM/MLIR,
  CWL; the hypergraph precursor, HyperGraphOS) enforces consistency *by convention*, this work makes it a
  *machine-verified commuting diagram over a canonical hypergraph IR* ("to our knowledge the first…").
- **Evaluation §VII-A:** the thin second-domain paragraph replaced by the first-class WS1 result + the WS2 drift
  paragraph.

## WS4 — Formal lift + global invariants (DONE)

- **Generalised the theorem (Appendix A5).** Added a *faithful extraction function* definition (a left inverse of
  the emitter's encoding on the $Q$-fields) and restated the proof to fix **no** particular $Q$ — it holds for any
  invariant with a family of faithful extractors over a shared-query dispatcher (kinematics and requirements are
  instances). Added **Corollary (closure under derived invariants)**: if the per-element square holds, then for any
  $g$, $g\circ Q$ agrees across views too — so *global* properties agree, not just counts.
- **Verified two global invariants** (extending "invariant" beyond per-element counts), via a shared `is_acyclic`
  helper in `extract.py`:
  - robotics: the kinematic structure is an **acyclic single-parent forest** — **16/16** fixtures (`is_forest`);
  - systems engineering: the **requirement-derivation graph is a DAG** — both fixtures.
  Both recovered identically from every view (follows from chain/relation agreement) **and** checked to hold.
- Article: §VII-A gains a "Beyond counts: global invariants" paragraph citing the corollary; A5 generalised.

## Tests / quality

`pytest -p no:randomly verification/cross_view_consistency/tests/` — **22 passed** (added
`test_non_robotics_trace_domain` over both fixtures + coverage; `test_drift_prevention_demo` asserting
0-vs-≥1 drift; and WS4 `test_is_acyclic_dag_vs_cycle` / `test_is_forest_tree_vs_violations` with cyclic/multi-parent
negatives, plus forest/DAG assertions in the integration tests). `ruff` clean; new helpers `is_acyclic`/`is_forest`
at CC 4/5. No new Rust/Python code idioms in the article (the verification is Python tooling, not described in the
paper as code).

## Article state

Compiles clean, **15 pages** (was 14; +1 as the plan budgeted), no undefined refs/citations. Contributions now
C1–C5; properties five and machine-verified; two witnessed domains.

## Files touched

- `verification/cross_view_consistency/trace_witness.py` (generalised: multi-fixture + coverage), `drift_demo.py`
  (new), `tests/test_cross_view.py` (+2 tests), `README.md`.
- Article (`01_analysis/.../smc_02`): `bare_jrnl_new_sample4.tex` (abstract), `sections/01_introduction.tex`
  (C5 + novelty), `sections/07_evaluation.tex` (§VII-A result + drift).
- Plan: `docs/plans/2026-06-28-acceptance-novelty/{plan.tex,pdf,tikz,mmd}`.

## Review-ready artifacts (for Csapó-sensei)

- Main paper: `01_analysis/articles/superweapon/smc_02/bare_jrnl_new_sample4.pdf` (15 pp, compiles clean).
- Supplement: `.../smc_02/supplementary.pdf` (2 pp: head-to-head + morphology tables, full alias-invariance proof).
- Proof companion: `docs/proofs/hymeko_machine_verified_proofs.pdf` (26 pp, all 5 props + WS4 outputs + scripts).
- Verification: `verification/cross_view_consistency/` (22 pytest pass, ruff clean).

## Estimated effect

Moves the two swing factors: generality is now demonstrated (two domains, a systems invariant), and the
contribution is visceral (drift caught by construction). Consistent with the plan's ~50% → ~60–65% eventual-
acceptance target, contingent on writing quality and reviewer assignment (unmodelled).
