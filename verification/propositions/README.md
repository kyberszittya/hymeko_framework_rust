# Machine-verification of the HyMeKo (T-SMC) propositions

Layered verification of the article's four propositions. Run each from the repo root with the project env
(`PYTHONPATH=.`):

- `p4_storage_overhead.py`  — sympy SYMBOLIC PROOF of Prop. 4 (rho-1 = O(log n / d-bar) -> 1). Full proof.
- `p2_content_birthday.py`  — sympy proof of Prop. 2's collision bound (birthday bound on the 256-bit digest);
   the h-purity half is the same mechanism as P1. Random-oracle premise is a stated assumption.
- `p1_alias_invariance.py`  — property-test of Prop. 1 vs the REAL compiler: 800 sibling-reorderings x 4
   self-contained fixtures, canonical_hash invariant 800/800 (verified). snapshot_json reorders (only the hash
   is canonicalised).
- `p3_emit_purity.py`       — emitter determinism (4/4) and the FINDING: emitted DOT is NOT byte-equal under
   sibling reordering (0/400) because internal IDs are assigned in source order. Prop. 1's byte-equal-emit claim
   needs scoping to alias rewrites, or the emitter must canonicalise IDs.

Report: `reports/2026-06-28-proposition-verification.{tex,pdf}`. Tooling (uv-pinned): sympy, z3, hypothesis,
networkx, pygraphviz.
