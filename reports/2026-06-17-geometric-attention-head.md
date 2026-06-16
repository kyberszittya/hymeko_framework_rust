# Geometric (quaternion + Clifford) attention head — implemented, ties/regresses vs bilinear

**Date:** 2026-06-17 (paused ~04:20 for sleep; resume tomorrow)
**Plan:** [docs/plans/2026-06-17-geometric-attention-head](../docs/plans/2026-06-17-geometric-attention-head/) (4 artifacts)
**Status:** ✅ implemented + tested + gated; ⚠️ **negative result** — does not yet beat the bilinear baseline.

## Goal
Close the ~0.04 leakage-free AUROC gap to SiGAT (alpha 0.884 / otc 0.902) on rotor-HSiKAN signed-link prediction. The gap is "partly attention, not cycles", so the lever tried here is a stronger attention *readout*.

## What was built
- New [signedkan_wip/src/core/geometric_triad_attention.py](../signedkan_wip/src/core/geometric_triad_attention.py): `GeometricTriadAttentionPool` — pools per-triad embeddings into per-vertex embeddings by **signed attention** whose score is a learned-gate mix of two geometric channels over the same projected features:
  - **quaternion**: Hamilton-product real part, signature (+,−,−,−);
  - **Clifford**: Cl(2,0) geometric-product scalar part, signature (+,+,+,−), reusing `sequence/clifford.py::geometric_product`.
  - `tanh` signed weights (balanced triads vote +, unbalanced −), magnitude-normalized `index_add` scatter; vertices in no triad get a zero row. Reuses the existing Clifford primitive — no rebuild.
- Wired into [run_hsikan_rotor.py](../signedkan_wip/experiments/runs/run_hsikan_rotor.py) as `--head geom_attn` (`--geom-channel both|quaternion|clifford`, `--geom-temperature`, `--geom-replace`). Endpoint head reuses the existing bilinear predictor (reaches held-out edges). **No edits to `core/signedkan.py`** — driver-level readout over `encode_triads`' existing outputs.

## Files touched
- `signedkan_wip/src/core/geometric_triad_attention.py` (new, ~135 LOC)
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` (+geom_attn head, residual/replace, geom CLI)
- `signedkan_wip/tests/test_geometric_triad_attention.py` (new, 13 tests)
- `signedkan_wip/tests/test_hsikan_rotor.py` (+ geom_attn integration test)
- `docs/plans/2026-06-17-geometric-attention-head/plan.{tex,pdf,tikz,mmd}` (new)
- CORE.YAML items touched: **none** (only `hymeko_core/` Rust is protected).

## Tests
- ruff: **PASS**. pytest: **24 passed** (both suites). Peak RSS at this scale ~1.7 GB (measured prior cell), ≪ 16 GB cap.

## Results (5 seeds, tuned recipe wd=1e-4/clip=1.0/class-wt/early-stop, strict triads)
`reports/hsikan_geom_attn_20260617.jsonl`

| dataset | bilinear | geom_attn (residual) | Δ mean | SiGAT | gap (geom) |
|---|---:|---:|---:|---:|---:|
| bitcoin_alpha | 0.8455 | 0.8470 | **+0.0015** (tie) | 0.8844 | −0.037 |
| bitcoin_otc | 0.8685 | **0.8518** | **−0.0166** (regress) | 0.9019 | −0.050 |

Leakage gates (geom_attn + shuffle): alpha **0.530**, otc **0.486** ≈ chance → leakage-clean.

## Honest read
The geometric attention readout **does not beat the plain bilinear head** — a wash on alpha, a regression on otc. The SiGAT gap is not closed; on otc it widened. The head is correct and leakage-clean, but the geometry-as-readout hypothesis, in this first form, did not lift HSiKAN.

### The replace→residual diagnosis (recorded; the one real finding)
The first cut **replaced** `h_v` with the triad pool → seed-0 collapse (alpha 0.737 / otc 0.717, −0.13/−0.17) **and** an alpha gate of 0.61 (leak). Diagnosed: replacing erases the rotor node signal, and the magnitude-normalized pool encodes a degree channel that correlates with sign. The fix — **residual** `node = h_v + pool` (now default; `--geom-replace` ablates) — recovered accuracy to ≈ bilinear and restored the gate. So the residual is load-bearing, but it makes the pool a small refinement of `h_v` that, so far, adds nothing net.

## Next (tomorrow)
1. (Hypotheses, gate each) **sign-aware values** — the value `W_v h_t` currently ignores the triad sign σ; a σ-weighted value is the most likely missing ingredient. Then: multi-head geometric attention, per-channel gate, attention-only lr / lower temperature. Inspect the learned gate σ(γ) (quaternion vs Clifford share) — is either channel even used?
2. If the readout stays flat, pivot to the **Berge cycles/walks** substrate (richer neighborhoods than simple triads; `hymeko_hre/src/traversal/berge.rs` foundation; new work = closed-Berge-cycle enumerator + binding). Separate plan; don't combine with readout changes.

## Open
SiGAT gap (alpha −0.037 / otc −0.050) still open. Memory: `project-hsikan-geometric-attention-berge`.
