---
name: project-hymeko-props-machine-verified
description: Machine-verified the 4 T-SMC HyMeKo propositions (sympy proofs + property-tests vs real compiler); FOUND a real gap — P1 byte-equal-emit fails under sibling reordering
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-06-28: machine-verified the HyMeKo (IEEE T-SMC) article's four propositions — turning the review's #1
weakness (proof-sketches-by-inspection) into checked results. Layered by mathematical character:

- **P4 (storage overhead** ρ-1 = O(log n / d̄) → 1**) + P2 (collision/birthday bound** on the 256-bit digest**)**:
  full **sympy SYMBOLIC PROOFS** (derivation + bound + limit + monotonicity + witness ρ=2.00@d̄=2→1.01@d̄=200; and
  N=2⁶⁴→2⁻¹²⁹). Real proofs, not sketches.
- **P1 (alias invariance)**: **property-tested vs the REAL compiler** — `hymeko.PyHypergraphEngine().parse_dsl(src)`
  → `ir.canonical_hash`; 800 sibling-reorderings × 4 self-contained fixtures (data/typical_graphs/*) → **canonical
  hash 800/800 invariant** (distinct fixtures hash distinctly, so non-degenerate). **P3** emitter determinism 4/4.
- **FINDING (a real, fixable gap)**: P1's STRONGER claim — denotation-equal sources emit BYTE-IDENTICAL text —
  **FAILS under sibling reordering (0/400)**. The emitters assign internal IDs (n2, e5, …) in **source-declaration
  order**; only the canonical **HASH** is order-canonicalized, not the stored IR or its emit. Fix for the article:
  (i) scope byte-equality to alias/import rewrites, (ii) canonicalize emit IDs before emitting, or (iii) weaken to
  "equal canonical hash / up to isomorphism." The article's existing prop1_alias test uses alias-RENAMING pairs
  (which canonicalize), never sibling reordering — why the gap was invisible.

**CRITICAL engine gotcha**: `parse_dsl` MUTATES/ACCUMULATES engine state — a shared `PyHypergraphEngine` makes
repeated parses return the SAME (corrupted) hash. **Use a FRESH engine per parse.** (Caught a spurious "800/800"
where 4 different graphs all hashed identically — the tell.)

Tools (uv-pinned in pyproject: `z3-solver`, `pygraphviz`; sympy/hypothesis/networkx already present). z3 doesn't
naturally add over sympy+property-testing for THESE props (future use: gauge/signed-balance, resolver BMC).
Scripts: `verification/propositions/{p1_alias_invariance,p2_content_birthday,p3_emit_purity,p4_storage_overhead}.py`.
Report+PDF: `reports/2026-06-28-proposition-verification.{tex,pdf}`. Article reviewed in
`reports/2026-06-28-hymeko-article-review.pdf`. Possible new contribution: a machine-checked canonical-IR appendix.
Engine transitive-import limit still bites (galambos_task.hymeko won't load standalone) — see
[[project-engine-transitive-imports]].
