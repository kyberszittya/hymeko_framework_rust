# Research: HSiKAN architecture

For runnable architecture authoring → [Quickstart: Build an HSiKAN architecture](../quickstart/08-hsikan-architecture.md).

## Summary

HSiKAN — Hypergraph Signed Kolmogorov–Arnold Network — replaces a GNN's message-passing edges with **signed k-cycles** as hyperedges, and replaces fixed activations with **per-channel splines** trained jointly with the rest of the network.

Per signed cycle `c = (v_1, …, v_k)` with edge signs `σ ∈ {+, -}^k`:

$$h_c = \sum_{s \in \{+, -\}} \phi_e^s\!\left(\sum_{i: \sigma_i = s} \phi_v^s(h_{v_i})\right)$$

- `φ_v^s` and `φ_e^s` are batched Catmull–Rom (or B-spline / Kochanek–Bartels) splines, one set per sign branch
- `M_e^{(k)}` is a sparse signed-incidence matrix mapping per-cycle embeddings to per-edge (test-edge) embeddings
- `arity_mixer` softmax-blends per-arity outputs: `h_e = Σₖ softmax(α)_k · M_e^{(k)} h_c^{(k)}`

## Key results (locked-in as of 2026-05-06)

- **Bitcoin Alpha**: m=128 + balance pruner: AUC 0.9136 ± 0.020 (5-seed); single-seed best 0.9329 beats full enumeration
- **Slashdot**: 5-seed unbalanced: 0.8562 ± 0.010 — central 4.4σ regime split (unbalanced > balance)
- **Mixed-arity dominance**: k=4+k=5 mixed beats k=3+k=4 mixed across all signed-graph benchmarks
- **Cycle quality**: balance pruner closes ~5pp on top-K — axiom-conditioned cycle subset matches full enumeration at ~50× memory savings

## Key results (negative / falsified)

- **Single-seed claims dangerous**: "axiom replaces attention" was wrong at 5-seed (axiom is the bigger lever, but doesn't displace attention)
- **Entropy reg on Slashdot**: +0.006 mean at n=3 collapsed to +0.0007 (~0.1σ, NULL) at n=5 — falsified
- **Walks on Slashdot**: c3,c4,c5,w2 → 0.807 (worse than no-walks ~0.86); walks net-negative even with k=5 retained
- **K-B presets at m=32 BA**: smooth/skew/cusp tied within noise; tense/sharp/flat hurt; "K-B sharp activations help" not supported
- **Vision transfer**: HGNN (Feng 2019) and HSiKAN-style both score 0.34–0.62 on MNIST/Fashion vs CNN's 0.99/0.89; signed-cycle inductive bias does not transfer to image regimes regardless of which hypergraph variant

## See also

- [NN variants & layer geometry](./nn-architectures-and-layer-geometry.md) — how `SignedKANLayer`, `MultiLayerSignedKAN`, and `MixedAritySignedKAN` stack in code
- `signedkan_wip/HSIKAN_FINAL_RESULTS_2026_05_03.md` — the published 5-seed numbers
- `signedkan_wip/HSIKAN_STATE_2026_05_04.md` — current research state
- [HyMeKo-driven training](./hymeko-driven.md) — how to drive HSiKAN from .hymeko configs
