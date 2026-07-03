# Post-SOTA Plan — 2026-05-03

Sequential execution under auto-mode. Paper draft (Phase E) grows in
parallel with the experiment phases.

---

## Phase A — Honesty + statistical firmness (target: ~2 hours)

### A1. Per-query σ-masking (strict no-leak protocol)

Currently the cycle σ assignment on cycles incident to a query edge
encodes that edge's sign through Davis parity. `vertex_adjacency` M_e
removes the structural leak (cycle never *contains* query edge), but a
residual leak remains: cycles that share a *vertex* with the query
still have σ patterns that can be back-correlated to nearby edges in
the same neighbourhood, including indirect leakage paths.

Per-query σ-masking is the strictest fix:

- For each query edge `(u, v)` evaluated at test time:
  - Identify all cycles whose σ pattern depends on the sign of any
    held-out test edge incident to `u` or `v`.
  - Recompute σ for those cycles excluding the contribution of the
    held-out edge (treat that edge's sign as 0/unknown via the
    `use_zero_branch=True` path in `SignedKANConfig`).
- Implementation: per-query σ recomputation at evaluation. Expensive
  per-query, but only matters at test time (training uses train-only
  σ). Roughly 50–100 LOC.

**Deliverable:** `m_e_mode="vertex_adjacency_strict"` flag that
combines vertex-adjacency M_e with per-query σ-masking. Compare
honestly to `vertex_adjacency` alone on Bitcoin Alpha (current honest
number ~0.81; expected drop of ~0.01–0.03 if the residual leak is
real). If the drop is larger than the methodology paper budget, we
keep `vertex_adjacency` as the recommended protocol and document the
residual leak quantitatively.

### A2. 5-seed expansion

Bitcoin Alpha is at 5 seeds (0.940 ± 0.009).
Bitcoin OTC and Slashdot at 3 seeds. Add seeds 3 and 4 each.

- Bitcoin OTC k345+balance λ=1.0 (~30 s/seed on GPU): 1 min total
- Slashdot k34+balance λ=0.05 max_k4=3M (~1000 s/seed on GPU): 35 min total

**Deliverable:** All three datasets at 5 seeds with std reported.

---

## Phase B — Per-edge continuous features (target: ~1 day)

### B1. Architecture extension

Extend `MixedAritySignedKAN.encode_edges` to accept an optional
`edge_features: torch.Tensor` of shape `(E, d_edge_feat)`. Project via
a learned `Linear(d_edge_feat, d_hidden)` and add to the per-edge sign
contribution at the cycle σ-assignment stage:

```
σ_v_contribution_from_edge_e = σ_v_e_default + project_edge_features(edge_features[e])
```

Backwards compatible (`edge_features=None` → original behaviour).

### B2. Three downstream task hooks

- **MuJoCo forward kinematics**: edge features = joint angles + joint
  velocities (3-D per edge). Re-run phase 13 with HSiKAN consuming
  these alongside cycle structure. Compare to the MLP baseline (0.054 m
  RMSE on 4-DOF arm).
- **Scene graph relation prediction**: edge features = spatial-overlap
  fractions (IoU between bounding boxes), confidence scores, semantic
  embedding (small text-encoded vector). HSiKAN consumes structural
  cycles + per-edge continuous context.
- **Context graphs (knowledge-graph style)**: edge features = relation-
  type embedding + temporal stamp + source confidence.

### B3. Synthetic scene graph dataset

Generate small scene graphs with known relation labels (above/below/
beside/inside) and continuous spatial features (position offsets,
overlap fractions). Train HSiKAN to predict relation type given graph
+ features. This is the proof-of-concept that per-edge continuous
features generalize beyond MuJoCo.

---

## Phase C — SGCN baseline reproduction in our codebase (target: ~3 hours)

Implement SGCN's recursive sign-conditional message passing as a
reference baseline inside the same training/eval pipeline as HSiKAN:
same `SignedGraph` data structure, same train/val/test split, same
optional protocol flags (`feature_edges`, `dedupe_pairs`,
`m_e_mode`).

Run SGCN on Bitcoin Alpha, Bitcoin OTC, Slashdot under both:
- Published-paper protocol (`feature_edges="all"`, no dedupe) — should
  reproduce ~0.91/0.93/0.91 from the literature.
- Honest protocol (`feature_edges="train_val"` + `dedupe_pairs=True`)
  — gives the apples-to-apples baseline against HSiKAN.

**Deliverable:** Calibrated comparison table:

| dataset | HSiKAN (leaky) | HSiKAN (honest) | SGCN (leaky) | SGCN (honest) |
|---|---|---|---|---|

If SGCN drops more than HSiKAN under the honest protocol, the gap
narrows further or reverses. Either way, the comparison becomes
defensible.

---

## Phase D — Real-data adapter scaffolds (target: ~half day each)

### D1. NTU RGB+D action recognition (skeleton graphs)

- Skeleton graph: 25 body joints, fixed bone topology
- Sign assignment: joint-active (extended) vs joint-passive (flexed)
  via knee/elbow angle thresholds, OR pose-direction binary
- Per-vertex continuous features: 3D joint position
- Per-edge continuous features: bone length, joint-angle magnitude
- Task: action class classification (60 classes in NTU RGB+D 60)

Adapter scaffold + data download instructions; defer full training to
a dedicated session.

### D2. Visual Genome scene graphs

- Scene graph: ~25-50 objects per image, ~10-30 relations
- Sign assignment: spatial-positive (above/in/on) vs spatial-negative
  (below/under) for spatial relations
- Per-edge continuous features: bounding box IoU, relative position,
  relative scale
- Tasks: relation-type prediction, scene-consistency scoring

Adapter scaffold + download / preprocessing notes.

---

## Phase E — Paper draft (parallel, target: continuous through phases A-D)

Section outline as results land:

1. **Introduction / motivation**: signed graphs, HSiKAN's structural
   prior framing, k-uniform hyperedges generalize triadic balance.
2. **Related work**: SGCN, SiGAT, KAN, hypergraph NNs, signed-graph
   theory (Heider, Cartwright-Harary, Davis).
3. **Architecture**: SignedKANLayer (per-σ inner+outer splines), αₖ
   mixing, cycle-pool aggregation, optional balance loss.
4. **Methodology**: σ-as-label leak identification, vertex-adjacency
   M_e, pair-deduplicated splits, αₖ-mask B&B for arity selection.
5. **Empirical results**:
   - SOTA table (3 signed-link benchmarks, 5 seeds each)
   - αₖ patterns (k=4/5 dominant on real signed networks)
   - Cycle-budget scaling on Slashdot
   - Ablations: balance loss, attention M_e, direct messaging
6. **Cross-domain extension**: kinematic graphs (mechanism
   classification + DOF), scene graphs (relation prediction), MuJoCo
   physics integration.
7. **Discussion / future work**: Berge cycles, per-query σ-masking,
   real-data adapter follow-ups.

Target: full draft (10-12 pages incl. tables) by end of next session.

---

## Execution order (under auto mode)

```
[A1 ───── A2 ───── B (B1 → B2 → B3) ───── C ───── D ──── ]
                            │
                            └── E (paper draft) runs in parallel
                                writing sections as data appears
```

Phase A unblocks everything (honest baseline numbers). Phase B unlocks
cross-domain experiments. Phase C calibrates the comparison. Phase D
is paper-future-work. Phase E is the deliverable.
