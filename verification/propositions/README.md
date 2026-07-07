# Machine verification of the HyMeKo (T-SMC) propositions

Layered verification of the article's five propositions. Run each command from the repository root.

The current paper numbering is:

- Proposition 1: alias/source-invariance of the canonical content hash.
- Proposition 2: content-addressability and collision bound.
- Proposition 3: projection-emission independence / emitter purity.
- Proposition 4: cross-view consistency; see `verification/cross_view_consistency/`.
- Proposition 5: storage overhead.

## Scripts

- `p1_alias_invariance.py` - property-test of Proposition 1 against the real compiler:
  800 sibling reorderings x 4 self-contained fixtures. The canonical hash is invariant
  800/800. `snapshot_json` remains source-order sensitive, so the hash is the canonical
  artifact checked here.
- `p2_content_birthday.py` - `sympy` proof of Proposition 2's birthday bound on the
  256-bit digest. The BLAKE3 random-oracle premise is an explicit cryptographic
  assumption, not a theorem proved by the script.
- `p3_emit_purity.py` - property-test of Proposition 3's emitter determinism. Re-emitting
  the same source in a fresh engine is byte-identical (4/4). Sibling-reordered sources
  are also tested as a diagnostic and currently are not byte-identical in DOT (0/400),
  because DOT internal IDs are source-order assigned. This is evidence for scoping the
  byte-equality claim, or for canonicalizing emitted IDs in a future implementation.
- `p4_storage_overhead.py` - historical filename; verifies the current paper's
  Proposition 5 with `sympy`: `rho - 1 = O(log n / d_bar)`, monotone decrease in
  mean arity, and the witness `rho: 2.00 at d_bar=2 -> 1.01 at d_bar=200`.

Cross-view consistency, the current Proposition 4, is verified separately in
`verification/cross_view_consistency/`:

- `commute_z3.py` proves the shared-query dispatcher entails view agreement and that
  an untethered view can drift.
- `cross_view.py`, `trace_witness.py`, and `drift_demo.py` drive the real CLI emitters
  and extractors over the robot and requirements-traceability witnesses.
- `tests/test_cross_view.py` is the executable regression suite for the extraction
  functions, real CLI integration, trace domain, drift prevention, Z3 proof, storage
  regime check, and performance budget.

Historical report: `reports/2026-06-28-proposition-verification.{tex,pdf}` records the
earlier four-proposition audit and the DOT byte-equality caveat. Treat that report as
historical context; the README above is the current proposition mapping.
