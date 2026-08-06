---
name: project-hymeko-cross-view-consistency
description: "2026-06-28 machine-verified the T-SMC article's cross-view-consistency commuting square X_f(eps_f(H))=Q(H) against the REAL emitters; the review's"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5adcad7b-0b61-4a80-84d9-d1dd4c4e2f3c
---

2026-06-28: closed the review's #1 gap for the T-SMC article (the [[project-hymeko-props-machine-verified]]
companion). The article ASSERTED cross-view consistency (§codegen "a cross-view discrepancy is impossible") but
pinned it on Prop.3, which is only COST factorization ("not a commuting diagram, codomains differ"). The
value-level claim was never proved. Built `verification/cross_view_consistency/` (sibling of
`verification/propositions/`): an extraction-function `X_f` per format parses the EMITTED text back into a
`KinematicInvariant`, proving the square `X_f(eps_f(H)) = X_g(eps_g(H)) = Q(H)` — common codomain, genuinely
commutes.

**Result (real CLI emitters, 16 robot fixtures × 5 views URDF/SDF/MJCF/DOT/Mermaid): 16/16 EXACT** (data formats
agree on full numeric invariant: links, mass, joints, parent/child, signed axis) **+ 16/16 TOPOLOGICAL** (all 5
views). Plus Z3 proof (T1 shared-query⇒agreement unsat; T2 untethered-view drift sat), sympy storage reframe
(ρ=1+2/d̄: robotics d̄≈2→ρ≈2 NOT →1; "controlled, vanishing for high arity"), non-robotics witness (SysML=DOT, 4
req/4 blocks/7 edges). 17 pytest pass, ruff clean, CC≤5 after refactor. Report
`reports/2026-06-24...` → actually `reports/2026-06-28-cross-view-consistency.md` + figure; plan in
`docs/plans/2026-06-28-cross-view-consistency/`.

**Why:** answers the reviewer attack the ChatGPT review predicted (cross-view counts), turns "proof sketches by
inspection" into machine-checked.

**How to apply / gotchas:**
- MUST drive the **CLI** (`target/release/hymeko emit --format f`), NOT the Python binding: Python resolves only
  1 import level so 26/35 robot fixtures fail to load and extract 0 joints ([[project-engine-transitive-imports]]).
- Two NAMED conventions (not drift): (W) URDF synthetic `world` link = no inertial → compare mass-bearing links;
  (F) MJCF root body without a joint = implicit world weld → compare ACTUATED joints.
- Two genuine FINDINGS: DOT rounds mass to 1dp (0.02→0.0) and axis is an unsigned letter (loses sign, robot_4wh
  real axis (0,0,-1)); Mermaid 2dp. So data formats = exact, diagram formats = topological-only by design.
- ElementTree gotcha that bit me: `node.find('parent') or default` is WRONG — a childless Element is falsy; test
  `is not None`.
- FIXED (non-core hymeko_cli) the `requirements_sysml`/`requirements_dot` "model extraction failed": CLI Emit
  dispatched `if accepts()==Kinematic {emit} else {template}`; these are template-only (emit()→None, declared
  template_dir) but declared Kinematic → hit emit branch. Fix in hymeko_cli/src/main.rs: fall through to template
  path when emit() yields None AND template_dir() is Some (ModelKind in CORE hymeko_query left untouched; user
  OK'd a core edit but none needed). Live output byte-identical to committed witness. Windows gotcha: subprocess
  needs encoding="utf-8" (SysML has « » guillemets). Regression tests added.
- Article edits done in `01_analysis/.../smc_02` (compiles clean): TITLE retargeted to "...Cross-View Consistency
  in Cyber-Physical Model Generation"; abstract→"Five algebraic properties...machine-verified"; new
  `prop:crossview` + appendix `A5_proof_cross_view.tex`; storage wording softened; §codegen re-anchored
  prop:commute→prop:crossview. Propositions renumber: alias1/content2/commute3/crossview4/storage5.
- PROOFS BUNDLE at `docs/proofs/`: `hymeko_machine_verified_proofs.pdf` (25pp, all 5 props + Z3 + storage + fix +
  full script listings via \lstinputlisting, needs listingsutf8 + literate map for unicode), MANIFEST.md,
  results/*.txt (capture with PYTHONIOENCODING=utf-8 then ASCII-transliterate or listings breaks).
- ACCEPTANCE/NOVELTY plan EXECUTED 2026-06-29 (plan: docs/plans/2026-06-28-acceptance-novelty/, 4-artifact;
  decisions: 2nd domain=SysML, scope=Full=WS1+WS2+WS3, WS4 deferred). WS1: trace_witness.py generalised to a
  requirements-domain checker over 2 fixtures (traceability_smc, sysml_cell) × 2 views + a COVERAGE invariant
  (4/4, 2/2, agree 2/2) → 2nd domain now a first-class result. WS2: drift_demo.py — semantic edit gives HyMeKo
  drift 0 vs pairwise-multifile drift 1 (test ASSERTS baseline drifts so it's not a strawman). WS3: abstract+intro
  reframed "consistency by convention→machine-verified property", added contribution C5, 5 props. Article 14→15pp,
  clean; 20 pytest pass. Report reports/2026-06-29-acceptance-novelty-execution.md. Est. ~50%→~60-65% accept.
- NOVELTY/prior-art (search, not assertion): NO concrete precedent for proved cross-view consistency over a signed
  hypergraph IR. Closest = the user's OWN precursor Hajdu&Hegyi 2025 Applied Mechanics (modeling/expressiveness,
  NOT proved consistency) + HyperGraphOS Ceravola 2024 (hypergraph codegen, no proofs); both now cited+differentiated.
  CLAUDE.md gained a "search before claiming novelty/prior-art" operating principle after I asserted "incremental"
  without searching. BLAKE3 kept+cited (has a spec). Article impl-cleanup rule: keep AVX/SIMD/COO/Arrow/design+memory
  patterns (contributions); remove only Rust/Python CODE idioms (crates/PyO3/Rust-string-builder/code symbols).
- WS4 DONE too (Csapo-sensei reviewing): generalised A5 to a "faithful extraction function" class + Corollary
  (closure under derived invariants g of Q) so the theorem isn't kinematics-specific; verified 2 GLOBAL invariants
  via is_acyclic/is_forest in extract.py — robotics acyclic single-parent forest 16/16, requirement-derivation DAG
  both fixtures; §VII-A "Beyond counts: global invariants" para. 22 pytest pass. Review-ready PDFs: main 15pp,
  supplement 2pp, proofs companion docs/proofs/hymeko_machine_verified_proofs.pdf 26pp — all compile clean.
