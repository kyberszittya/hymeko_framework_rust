# Inductive transfer test — the rotor transfers where transductive can't; learned increment is modest

**Date:** 2026-06-18
**Plan:** [docs/plans/2026-06-18-inductive-transfer-test](../docs/plans/2026-06-18-inductive-transfer-test/) (4 artifacts; PDF compiles).
**Status:** ✅ driver + train-loop refactor + tests; 5-seed decomposition. **Nuanced positive:** the rotor's inductive claim holds *mechanically* (cross-graph transfer where the transductive baseline structurally cannot), but the *learned*-transfer increment is modest and the naive gate was confounded — caught and corrected with a random-init control.

## Summary

The rotor line's distinctive, untested claim is **inductive**: structural-feature
embeddings, no per-node identity table, so a model trained on one graph should
transfer. Tested by cross-graph transfer (train all weights on graph A, evaluate the
frozen model on graph B's strict deduped held-out edges), 5 seeds, both directions
(alpha↔otc).

- **Mechanism + discrimination (clean win):** the rotor transfers — 0.81–0.86 AUROC on
  the *unseen* graph — while the transductive control **`dadsgnn` cannot transfer at
  all** (`nn.Embedding(n_A)` cannot index B's nodes → "cannot transfer", recorded, not
  masked). This is the rotor's structural advantage made concrete.
- **The naive shuffle-A gate is confounded.** Training on shuffled-A signs still
  transfers at 0.75–0.82 — far above chance — because the eval builds B's *real*
  signed adjacency/features, so signed message-passing encodes B's signs into the
  embeddings regardless of A-training. A **random-init control** (no A-training) is the
  correct floor: it sits **below chance** (0.38–0.47, a pathologically anti-aligned
  readout). So the decomposition is: random-init (below chance) → any training warms
  the net so B's real features express (0.75–0.82) → **real-A adds a genuine
  learned-from-A increment of +0.038 to +0.063** (4/4 positive).
- **Honest significance:** the learned increment is consistent in direction (all four
  cells) but only ~1–1.5× the shuffle arm's σ (σ_shuffle ≈ 0.024–0.046), so
  *suggestive*, not strongly significant. The bulk of cross-graph AUROC is the eval
  graph's own structural signal, not transported A-knowledge.

## Results (5-seed mean ± pstdev, deduped) — `inductive_transfer_decomp_ab.jsonl`

| model | pair | real-A | shuffled-A | random-init | learned (real−shuf) |
|---|---|---|---|---|---|
| cayley_rotor | otc→alpha | 0.8126±0.025 | 0.7686±0.039 | 0.4747±0.075 | +0.044 |
| cayley_rotor | alpha→otc | 0.8539±0.008 | 0.8019±0.042 | 0.4631±0.092 | +0.052 |
| cayley_rotor_walk | otc→alpha | 0.8128±0.023 | 0.7495±0.046 | 0.4187±0.049 | +0.063 |
| cayley_rotor_walk | alpha→otc | 0.8603±0.007 | 0.8220±0.024 | 0.3840±0.082 | +0.038 |

Transductive control (`dadsgnn`, seed 0, `inductive_transfer_smoke1seed.jsonl`):
**cannot transfer** both directions (table index mismatch) — the intended discrimination.

**Measured / inferred / rejected (CLAUDE.md).** *Measured:* the table + the
dadsgnn failure. *Inferred:* random-init below chance ⇒ the cross-graph signal is B's
own structural features, exposed by training but not A-specific; real−shuffle is the
learned-from-A increment. *Rejected as superstition:* the naive "shuffle-A gate →
chance" expectation (it does not, because B's real adjacency carries the signal) and
the first-pass "structural prior ≈ 0.75 at random-init" hypothesis (random-init is
*below* chance, not 0.75).

## Files touched

**New (2):**
- `hymeko_neuro/experiments/runs/run_inductive_transfer.py` (+200) — cross-graph
  transfer driver; trains on A and evaluates the frozen model on B; reuses
  `run_baseline_audit._train` + `_evaluate` + dedup (§6.5 #3, no train-loop dup);
  `train=False` (random-init) and `shuffle_train_signs` controls; transductive failure
  caught as "cannot transfer" via a *specific* `(IndexError, RuntimeError)` catch (§6.4).
- `hymeko_neuro/tests/test_inductive_transfer.py` (+90) — 5 tests: rotor transfers
  across distinct graphs, eval set is from the eval graph (exact count), shuffle/walk
  arms, unknown-model failure.

**Modified (mine):**
- `hymeko_neuro/experiments/runs/run_baseline_audit.py` — extracted the BCE+early-stop
  loop into a reusable `_train(model, hp, ctx, e_tr, s_tr, e_va, s_va, device)`;
  `run_audit` calls it. **Behaviour-preserving** — the determinism regression test is the
  oracle and passes unchanged.

**Artifacts:** `inductive_transfer_decomp_ab.jsonl` (60 rows), `inductive_transfer_smoke1seed.jsonl` (12).
**CORE.YAML items touched:** none.

## Test results
- `test_inductive_transfer.py` 5 ✓; `test_baseline_audit.py` 16 ✓ (refactor oracle).
  (`pytest -p no:randomly`.)
- `ruff check`: clean on all touched files. `mypy --strict`: clean on the new driver.

## Performance
- Per cell ~3 s (transfer eval reuses audit-speed training); random-init arm faster.
  GPU (cu132). Peak RSS ≪ 16 GB. No regression claim.

## §6.5 anti-patterns
None. Train loop single-sourced (`_train` reused, §6.5 #3); transductive failure caught
with a *specific* exception type + reason, not a bare except (§6.4). No new globals.

## Experiment provenance
- Git SHA `7d16ad0` (tree dirty from session; touched files above). Datasets: cached
  SNAP `bitcoin_alpha`, `bitcoin_otc`. Seeds 0–4. Device CUDA (torch 2.12.0+cu132).
  Recipe: each baseline's `default_hparams`, deduped held-out on the eval graph.

## Open issues / follow-ups
- **The learned increment (+0.04–0.06) deserves a stronger test:** more seeds, or
  larger/more-distinct graph pairs (slashdot/epinions), to push it past the shuffle σ.
- **Node-holdout (within-graph) inductive** — the harder secondary in the plan
  (per-node support/target partition); not run (the clean cross-graph result lands first).
- **The transfer story for a paper:** "the rotor is the only model here that can be
  applied to an unseen graph at all (transductive baselines cannot), and it carries a
  small but consistent learned-from-source increment over its structural-feature prior."
  That is a *distinctiveness* claim the AUROC race cannot make.
