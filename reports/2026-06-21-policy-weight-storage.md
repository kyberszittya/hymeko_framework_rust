# Storing a trained policy AS a HyMeKo hypergraph (P4 prototype)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** [docs/plans/2026-06-21-rl-algorithm-architecture/](../docs/plans/2026-06-21-rl-algorithm-architecture/) (phase P4)

## Summary
The storage thesis is now an **artifact**, not a claim. A trained cart-pole HSiKAN actor-critic is written to
a **332 KB valid HyMeKo file** ([data/nn/cartpole_hsikan_policy.hymeko](../data/nn/cartpole_hsikan_policy.hymeko)),
read back **bit-exact**, and the reconstructed policy reproduces behaviour identically.

The identity it rests on: a weight matrix $W\in\mathbb{R}^{m\times n}$ is the weighted biadjacency of a
bipartite graph, which is the **star expansion** of a hypergraph ($m$ vertices, $n$ hyperedges, incidence
$B_{ij}=W_{ij}$). So every learned tensor round-trips `state_dict ⇄ .hymeko`. For the cart-pole the structural
`a_pos/a_neg` literally *are* the robot's signed kinematic incidence (the star expansion of the cart-pole);
the learned matrices are stored as incidence tensors beside them.

**Verified end-to-end:**
- `hymeko validate` → ✅ valid HyMeKo (self-contained; inline tensor vocab, no external import).
- round-trip → max $|\Delta|$ = **0.00e+00** over all 26 259 weights (bit-exact).
- equivalence → reconstructed policy `eval_balance` = **56.80** = original 56.80.

Two issues found and fixed along the way (recorded for the next person): `repr()` floats emit scientific
notation (`1.5e-05`) which HyMeKo's number lexer **rejects** → switched to shortest positional formatting
(`np.format_float_positional`, still round-trips float32); and `@"meta_nn.hymeko"` resolves relative to the
file, so the artifact is made **self-contained** (inline `nn.tensors` vocab) to validate anywhere.

## Figures
[docs/figures/2026-06-21-policy-storage/](../docs/figures/2026-06-21-policy-storage/) (PNG + SVG):
1. `01-weight-matrix-star-expansion` — a weight (sub)matrix heatmap beside its weighted star-expansion
   bipartite graph (the core identity).
2. `02-kinematic-incidence` — the cart-pole hypergraph + `a_pos/a_neg`: the robot's signed incidence = the
   fixed structural part of the policy.
3. `03-actor-critic-dataflow` — the HSiKAN actor-critic as a layered dataflow hypergraph.
4. `04-weight-gallery` — every learned weight matrix of the trained policy.
5. `05-results` — the structure-vs-capacity ablation + the vectorization speedup.
6. `06-roundtrip-verification` — all 26 259 weights on the diagonal, error histogram at zero.

## Files touched
| File | Δ |
|---|---|
| `hymeko_rl/policy_store.py` | new (+120): `weight_to_hypergraph`/`hypergraph_to_weight`, `policy_to_hymeko`/`hymeko_to_policy`, positional float fmt |
| `hymeko_rl/tests/test_policy_store.py` | new (+78): identity round-trip, no-exponent fmt, bit-exact state_dict ⇄ .hymeko, `hymeko validate` |
| `scripts/render_policy_storage_figures.py` | new (+210): the 6-figure battery |
| `data/nn/cartpole_hsikan_policy.hymeko` | new artifact (332 KB, the stored policy) |
| `reports/cartpole_hsikan_policy.pt` | the trained checkpoint (source of the artifact) |
| `docs/plans/2026-06-21-rl-algorithm-architecture/plan.*` | P4 phase added |

**CORE.YAML items touched:** none. **New deps:** none (matplotlib/networkx already present).

## Test results
- `test_policy_store.py` — **11 passed** (identity round-trip; non-2D rejected; 7 number-format cases
  round-trip without `e`; bit-exact state_dict round-trip; `hymeko validate` accepts the file).
- `ruff` clean; `mypy --strict` clean (no `mujoco` import here, so fully clean).

## Scope & honesty
- **What's demonstrated:** small/medium policies (26k params) store fully **inline** as valid HyMeKo and
  round-trip bit-exact. The `meta_nn`-style typing makes it a *description*, not an opaque blob.
- **Three weight kinds** (per the design): the cart-pole's learned weights are dense `SignedConv` matrices +
  heads, stored as incidence tensors; the structural `a_pos/a_neg` are the genuine signed star-expansion.
  The *fully* native case — learned incidence weights ($M_e$ per signed cycle) — needs a `signedkan_layer`
  policy (the cart-pole's adjacency is fixed), and would close `meta_nn`'s "M_e sparse-mm not yet in pure
  HyMeKo" gap.
- **For scale:** inlining 26k floats is 332 KB; a large net would store structure inline + bulk in a
  content-addressed `.safetensors` blob keyed by the canonical-IR hash (planned, not built here).
- **Dense-layer caveat:** a dense layer's "hypergraph" is complete-bipartite — the representation earns its
  keep on signed/sparse incidence, not on dense matrices.

## Provenance
Git SHA `292388b` (dirty). torch 2.12.0+cu132 (CPU), matplotlib 3.11.0, networkx 3.6.1. The stored policy was
trained 80 vec-iters (seed 0), upright 56.8/200 — a checkpoint for the round-trip, not a performance result
(the baseline is the 5-seed 192±15). Artifact hashes are reproducible from `cartpole_hsikan_policy.pt`.

## Open issues / follow-ups
1. A `signedkan_layer` cart-pole policy to demonstrate **learned incidence** as star edges (the deepest form).
2. The content-addressed blob path for large nets.
3. Round-trip *through the IR* (compile → snapshot → reconstruct) as an even stronger "stored in HyMeKo" proof
   than text parse (the file already `validate`s; this would read values back via `snapshot_json`).
