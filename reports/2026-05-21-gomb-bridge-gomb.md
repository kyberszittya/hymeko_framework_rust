# Gömb → HSIKAN bridge → Gömb (two-stage cortex) — 2026-05-21

## Summary

User's architectural intuition: connect two Gömb cortices
through an HSIKAN bridge. Biological analogue — bilateral
cortices joined by a fibre bundle; the HSIKAN bridge is
the corpus-callosum analogue, a signed-cycle-aware inter-
cortex communication layer.

Architecture:
```
cycles+signs → Gömb_1 (V1→V2, CPML bypassed)
            → x_for_core_1 (per-vertex)
            → Linear projection + HSIKAN bridge (L layers)
            → bridge_h_v (per-vertex)
            → highway-gated mix with Gömb_2's base embedding
              (g init ≈ 0.05 per channel)
            → Gömb_2 (V1→V2→V4, full cascade)
            → edge logits
```

**Headline result (Bitcoin Alpha, 5-seed paired vs plain
Gömb 0.9034):**

| arch | mean ± σ | paired Δ | σ_d | wins |
| --- | --- | --- | --- | --- |
| bridge L=2 | 0.9079 ± 0.0087 | +0.0045 | +2.58 | 5/5 |
| bridge L=4 | 0.9057 ± 0.0075 | +0.0023 | +2.34 | 5/5 |
| **single Gömb + outer HSIKAN d=4 cr** (reference) | **0.9100 ± 0.0096** | **+0.0066** | **+5.68** | **5/5** |

**The bridge IS positive vs plain Gömb** (5/5 wins at both
depths, paired σ_d 2.3-2.6) — but it **underperforms the
simpler single-Gömb-with-outer-HSIKAN** by 0.0021-0.0043 AUC.
The intuition composes cleanly but doesn't beat the simpler
architecture on Bitcoin Alpha.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/hymeko_gomb/cascade.py` | extended | +180 (`HymeKoGomb.encode_per_vertex` + new `GombBridgeGomb` class) |
| `signedkan_wip/src/hymeko_gomb/__init__.py` | extended | +2 (re-export) |
| `signedkan_wip/experiments/runs/run_gomb_smoke.py` | extended | +1 (`_MODELS["gomb_bridge_gomb"] = GombBridgeGomb`) |
| `signedkan_wip/tests/test_gomb_bridge_gomb.py` | new | 144 (8 unit tests) |
| `docs/plans/2026-05-21-gomb-bridge-gomb/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4-format plan |
| `reports/2026-05-21-gomb-bridge-gomb.md` | new | this file |

## CORE.YAML items touched

None.

## Architecture details

Param breakdown at the BA strict-bench config (d_embed=32,
M_outer=8, d_outer=20, d_middle=24, d_core=48, n_tiers=4):

```
g1_node_embed       : 121,056   (Gömb_1's nn.Embedding)
g1_outer            :   5,488   (Clifford-FIR)
g1_middle           :   5,112   (HSIKAN aggregator)
bridge_pre          :   6,944   (Linear: g1.x_for_core → d_embed)
bridge_hsikan       : 129,345   (MultiLayerSignedKAN, L=4)
g2_base_node_embed  : 121,056   (Gömb_2's learned embedding for residual)
g2_outer            :   5,488   (Clifford-FIR)
g2_middle           :   5,112   (HSIKAN aggregator)
g2_core             : 128,129   (CPML — the only CPML active in this design)
─────────────────────────────────
total               : 527,762
```

vs.\ single Gömb + outer HSIKAN d=4 ($\sim$280k params): bridge
has ~1.9× the parameters.

Key design points:
- **Gömb_1's CPML is BYPASSED**. The cascade is cut after
  `x_for_core_1` (the pre-CPML per-vertex feature). Saves
  Gömb_1's CPML parameters; means Gömb_1's cortical-routing
  capacity is unused.
- **Highway-gated residual mix** into Gömb_2's input embedding
  — same productive pattern as the single-Gömb-outer-HSIKAN
  win. Init g ≈ 0.05 per channel.
- **M_vt cache** in the bridge HSIKAN forward (same pattern
  as today's other caching optimisations).

## Why the bridge underperforms the single-Gömb variant

Most likely overfitting at 24k-edge Bitcoin Alpha:
- Bridge has 528k params vs $\sim$280k for the
  single-Gömb-with-outer-HSIKAN variant (1.9× more).
- Train edges: 19,349 → ~36 params per train edge for the
  bridge, vs ~14 params/edge for the single-cortex variant.
- Val AUC σ across seeds is similar (~0.008), suggesting the
  problem isn't seed instability but a true ceiling.

Possible architectural cause: the bridge's input is
**Gömb_1's post-Clifford-FIR + post-HSIKAN-middle feature**.
Gömb_1's outer FIR layer was tuned to produce vertex
features ready for the *CPML routing* — not for an HSIKAN
bridge. The bridge's CR-highway spline may be processing
features that already carry "Clifford-FIR-shaped" inductive
bias, which conflicts with the HSIKAN's own signed-cycle
processing.

The single-Gömb-with-outer-HSIKAN variant feeds the HSIKAN
**RAW vertex embeddings** (just `nn.Embedding.weight`) — a
cleaner upstream interface.

## What worked anyway

1. **End-to-end training**: gradient flows correctly through
   all 4 parameter groups (Gömb_1 shells, bridge HSIKAN,
   Gömb_2 base + gate, Gömb_2 shells). Pinned by
   `test_backward_reaches_all_four_param_groups`.
2. **Numerical stability**: 5/5 seeds converged without
   NaN/Inf at both bridge depths.
3. **Memory**: fits the 7.6 GiB GPU comfortably (~3.5 GiB
   peak at L=4 on BA strict-bench config).
4. **Composition is paired-positive vs plain Gömb** at both
   depths — the new architecture DOES learn something useful,
   it's just not BETTER than the simpler alternative.

## When this architecture might still win

The two-cortex bridge has more capacity. On a dataset where:
- **Overfitting isn't the binding constraint** (larger
  edges-per-param ratio, e.g., Epinions's 640k train edges).
- **The bridge has truly distinct inductive bias** vs.\ both
  cortices (e.g., a bridge with arc-weight CR-highway on a
  weighted graph).
- **The composite captures multi-scale structure plain Gömb
  misses** (e.g., a multi-resolution signed-network with
  hierarchical signed-cycle structure).

…the bridge could outperform. Not on Bitcoin Alpha at the
current configs.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_gomb_bridge_gomb.py` | **8 / 8 pass** |
| All prior suites | no regression |
| Bitcoin Alpha 5-seed × 2 depth grid | 10 / 10 cells succeeded |

## §6.5 anti-pattern audit

- One new class `GombBridgeGomb` (per §6.5 #8 —
  structural difference → class, not forward-time toggle).
- Reuses existing `GombConfig.outer_hsikan_*` fields for the
  bridge HSIKAN config (no Cartesian-product API).
- Pure additive: `HymeKoGomb.encode_per_vertex` is a new
  method, doesn't modify the existing forward path.
- M_vt cache pattern mirrors the existing
  GombWithOuterHSIKAN cache (no new caching machinery).

Clean.

## Open follow-ups

1. **Try the bridge on Epinions** (640k train edges, less
   overfitting risk). Would require the smaller-Gömb config
   we used today + edge-batched _edge_logits.
2. **Asymmetric depths**: bridge L=1 + larger Gömb_2 config.
   Less capacity in the bridge, more in the second cortex.
3. **Skip Gömb_1's middle shell**: feed the bridge from
   Gömb_1's `x_outer` directly (cuts $\sim$5k params, lets
   the bridge work on raw FIR output).
4. **Include arc weights through the bridge**: combine the
   `cr_highway` arc-weight lever with the bridge composition
   on weighted graphs (Bitcoin's trust scores).

## Experiment provenance

- **Git SHA:** uncommitted.
- **GPU:** RTX 2070 SUPER 8 GiB.
- **5-seed BA grid:** L ∈ {2, 4} × seeds {0, 1, 2, 3, 4} =
  10 cells, ~12 s/cell, ~2 min total.
- **JSONL:** `signedkan_wip/experiments/results/gomb_bridge_gomb_5seed_ba_2026_05_21.jsonl`
- **Baseline:** plain Gömb 5-seed BA at strict-bench config
  (0.9034 mean, mixed from earlier files).

## Acceptance check

- [x] Plan in 4 formats on disk.
- [x] CORE.YAML items touched = 0.
- [x] 8 / 8 unit tests pass; no regression.
- [x] 5-seed BA bridge grid complete (10/10 cells).
- [x] Paired Δ vs plain Gömb reported.
- [x] **Honest framing**: bridge is positive vs plain Gömb
      (5/5 wins) but underperforms the single-Gömb-with-
      outer-HSIKAN by 0.002-0.004 AUC. The user's intuition
      composes architecturally but doesn't beat the simpler
      lever on BA at this scale.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
