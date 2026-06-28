# StructuralActor — walks/cycles straight to the actuators (no message-passing)

**Sketched:** 2026-06-28 01:50 JST · for Dr. Csaba Hajdu (his idea: "avoid the signed message-passing and use
cycle/walk points directly to joint actuator points"). Dual of `structural_critic.py`; prototype pattern proven
by `spike_probe.py`.

## The idea in one line

Replace HSiKAN's **iterative** signed message-passing (launch-bound: spline-per-edge × layers) with a **static**
gather along the topology's **precomputed** walks/cycles, mapped directly to the actuators. Same structural prior
(walks = transport paths, cycles = holonomy loops), a fraction of the ops.

## Why it is faster (the launch-bound fix made structural)

- **HSiKAN forward** = L layers × (signed-adjacency propagate + Catmull-Rom spline per edge) = dozens of tiny ops
  → dominated by dispatch overhead (~2 ms measured, CPU eager).
- **StructuralActor forward** = one **gather** `x[motif_vid] · motif_sgn` (a precomputed index op) + one dense
  map over the motif set → a handful of large ops. The motif set is enumerated **once** at construction (the graph
  is fixed), via the same Rust `enumerate_top_k_{cycles,walks}_rs` bindings the critic already uses.

## Architecture (forward)

Input: raw per-vertex features `x ∈ (B, N, F)`. Buffers (from `enumerate_motifs(hg, cfg)`, registered once):
`motif_vid (M, Lmax)`, `motif_sgn (M, Lmax)`, `motif_len (M,)`, and a derived `vtx_motifs` incidence (which motifs
touch each vertex).

1. **Gather along motifs (signed transport):** `g = x[:, motif_vid, :] * motif_sgn[..., None]` → `(B, M, Lmax, F)`.
   This is the parallel-transport of node features along each walk/cycle — the gauge reading, done as one index op.
2. **Per-motif aggregate (the holonomy/nonlinearity):** reduce over `Lmax` (masked by `motif_len`) and apply the
   activation → per-motif embedding `e ∈ (B, M, d)`. Reuse the critic's `AGGREGATIONS` (`sign_mean` / `mlp` /
   `fir` = a learned signed FIR along walk position — the closest to a spline-along-walk). The Catmull-Rom spline
   can live *here*, applied to the `M` motif features instead of every edge — far fewer evaluations.
3. **Motif → actuator readout (local, identity-preserving):** for each actuated vertex `v`, aggregate the
   embeddings of the motifs incident to `v` (`vtx_motifs`) → `(B, N_act, d)` → `Linear(d, 1)` per actuator. Each
   joint's action is read from the walks/cycles that pass through it — the structural analogue of the per-node
   actor head (which the multidim-readout finding showed beats pooling).

No layer-to-layer propagation anywhere. The only learnable parts are the per-motif aggregate and the readout.

## What it unifies (why it's the session's keystone)

- **Gauge/holonomy** (`project-gauge-holonomy-signed-hsikan`): a walk *is* a transport path; the signed gather *is*
  the holonomy. `spike_probe.py` already showed walk-gather→output works *and generalizes* — this is that, wired
  to actuators.
- **No alpha-mixing** (Hajdu): the walk/cycle set is the feature basis, *given* by the (designed) topology — not a
  learned softmax over arities. A Steiner/sunflower augmentation fixes the motif set.
- **DTC made cheap:** "controller = topology" becomes a static gather → inference drops toward the LQR end of the
  latency scale, while staying inspectable (you can read the motifs).

## Honest trade-offs

- **Frozen structure:** the motif set is fixed at enumeration — you lose message-passing's ability to *learn which*
  structure matters. For a fixed/designed topology that is exactly the point (the design chooses the structure);
  for a topology you'd want to *adapt*, it's a real limitation.
- **Coverage:** top-K walks/cycles may miss long-range structure a deep message-passing net would reach. Mitigate
  with `walk_len`/`cycle_len`/`keep`, or a residual raw-feature path.
- **Expressivity vs HSiKAN is an empirical question** — hence the toy below before any RL claim.

## Toy (run now, CPU-light, alongside the A/B)

Supervised, fixed signed graph, a **structural per-node target** (signed 2-hop — exactly what walks capture).
Compare **StructuralActor** (motif-gather, no MP) vs **HSiKAN** (message-passing) vs **MLP**, on **test MSE** and
**forward latency**. Success = StructuralActor matches HSiKAN's MSE at materially lower forward time (the structural
prior preserved, the launch-bound cost removed). Reuses `enumerate_motifs`, `structural_probe`'s dataset/HSiKAN/MLP,
and the latency-measure pattern. Falls back to a tiny pure-Python enumerator if the `hymeko` binding isn't built.
