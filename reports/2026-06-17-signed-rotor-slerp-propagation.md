# Signed slerp/nlerp rotor propagation — the input fix that finally lifts the ceiling

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-signed-rotor-slerp-propagation](../docs/plans/2026-06-17-signed-rotor-slerp-propagation/) (4 artifacts; PDF compiles).
**Status:** ✅ implemented + tested + 5-seed A/B + gate. **Positive result** — the first confirmed lever of the session, leakage-clean, both datasets. Computer-graphics interpolation × signed-graph AI.

## Why
Root cause (`reports/2026-06-17-*` chain): four readout/comparison levers landed flat because the rotor *input* is `_structural_features` = **6 degree-only numbers/node**; `val` was pinned at ~0.857/0.876 regardless of readout. The fix had to enrich the **input**. User insight: this is the shader interpolation problem — a signed triad is a triangle (barycentric), rotors are quaternions (slerp), and linear pooling shrinks off the sphere ("interpolate normals then renormalise"). Propagating each node's rotor over the signed graph, on S³, gives a neighbourhood-aware input *and* keeps it on the manifold.

## Mechanism
`SignedRotorPropagation` — signed **nlerp** aggregation, per node/block, K rounds:
`q_i' = normalize( w_self·q_i + (1/deg_i) Σ_{j∼i} s_ij · align(q_j, q_i) )`, with
`align(q_j,q_i)=sign⟨q_j,q_i⟩·q_j` (double-cover hemisphere fix) and `s_ij∈{±1}` the edge sign (signed propagation). nlerp = the cheap, autograd-clean slerp that stays on S³. Built from **train** edges/signs → leakage-safe; the shuffle gate must hit chance. Parameter-free (no weights).

## Files touched
- `signedkan_wip/src/embeddings/signed_rotor_propagation.py` (new, ~70 LOC)
- `signedkan_wip/src/embeddings/cayley_rotor.py` (factor `forward` → `rotors()` + `embed_rotors()`; parity-tested)
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` (`RotorInjector` propagation + `propagated_rotors()`; `--rotor-prop-rounds`, `--rotor-prop-self-weight`; provenance)
- `signedkan_wip/tests/test_signed_rotor_propagation.py` (new, 7), `test_multidim_preservation.py` (new, 3 — guards no scalar-collapse in propagation/pool), `test_cayley_rotor.py` (+embed_rotors parity, +algebra/`n_refs`), `test_hsikan_rotor.py` (+prop smoke)
- CORE.YAML items touched: **none**.

## Tests
`ruff`: PASS. `pytest -p no:randomly` (cayley/propagation/multidim/driver suites): all green (49+ across suites). Unit: nlerp output on S³; hemisphere-alignment invariance to `q→−q`; isolated node unchanged; signed aggregation; gradient flow; `embed_rotors(rotors(x))==forward(x)`.

## Results — 5-seed A/B (tuned recipe, dedup, head bilinear)
`reports/rotorprop_ab_20260617.jsonl` (+ `rotorprop_smoke_`, `rotorprop_selfweight_`).

| dataset | baseline (r0) | r1 sw2 | **r2 sw4 (winner)** | Δ test | Δ val | gate (r2 sw4) |
|---|---:|---:|---:|---:|---:|---:|
| bitcoin_alpha | 0.8455 | 0.8482 | **0.8500** | +0.0045 | +0.0036 | 0.530 [0.48–0.60] |
| bitcoin_otc | 0.8685 | 0.8782 | **0.8790** | +0.0105 | +0.0078 | 0.521 [0.50–0.55] |

**Over-smoothing** (the planned risk) appears with too many rounds / low self-retention: `self_weight=1` drops alpha (rounds 3 worst); `self_weight=4` curbs it and lifts both. **Leakage gate** ≈ chance → clean (the lift uses the real balance signal, destroyed under shuffle).

## Honest read
First lever to move both `val` and test, leakage-clean, on both datasets. Confirms the diagnosis: the bottleneck was the input, and manifold-aware signed propagation addresses it. Modest on alpha (+0.0045), clear on otc (+0.0105). The win is in **how rotations are interpolated on the manifold** (input), not the readout (kept plain bilinear; see the head-ablation report).

## Performance
Each round = one scatter-add over train edges × n_blocks × 4 + a normalise — negligible; parameter-free. Wall ~10–12 s/cell. Peak RSS ~1.7 GB scale (10.5 % of the 16 GB cap), no graph-sized allocation added.

## §6.5 anti-patterns
None. SGCN exists but propagates Euclidean features; this is manifold (quaternion) propagation — distinct, reuses the `cayley_rotor` quaternion algebra. `--rotor-prop-rounds 0` reproduces the prior line.

## Follow-ups
Learnable `self_weight` / per-block gate; the head ablation (separate report) shows the readout is best left real. Memory: `project-hsikan-geometric-attention-berge`.
