# Waking the dead geometric-attention score — alive + leakage-clean, but still loses to bilinear (readout line falsified)

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-geom-attention-wake-score](../docs/plans/2026-06-17-geom-attention-wake-score/) (4 artifacts; PDF compiles).
**Status:** ✅ implemented + tested + planned + 5-seed A/B + leakage gate. ⚠️ **negative result** — the woken readout does not beat the bilinear baseline. Decision: **the geometric-attention readout line is exhausted; pivot to Berge cycles** (the plan's own falsification rule).

## Why this ran
The gate-inspection diagnostic ([2026-06-17-geom-gate-inspection.md](2026-06-17-geom-gate-inspection.md)) found the geom_attn readout's "adds nothing" was **score collapse**: 100 % dead weights, gate frozen at init, `W_q`/`W_k` stuck at the `0.1×` init. The fork was: wake the score (and falsify the readout properly) vs pivot to Berge now. User chose to wake it first, then Berge. This is the wake.

## What was built
Three opt-in knobs on the existing `GeometricTriadAttentionPool` (config fields, **not** new functions/classes — §6.5 #1/#5/#8); defaults reproduce the legacy numbers exactly:
- `score_init_scale` (default `0.1` = legacy; `1.0` = no init suppression → score gets dynamic range).
- `learn_scale` — a learnable `log`-scale on the score (`exp(0)=1` at init); the direct amplitude knob the diagnostic showed was missing.
- `sign_aware` — `w = b_t · σ(s̃)`: the per-incidence triad **balance sign** `b_t∈{+1,−1}` sets the vote direction, `σ(s̃)∈(0,1)` the learned geometric relevance. At init `s̃≈0 ⇒` relevance ≈0.5 ⇒ the pool is the (uniform) balance-signed value mean — a warm start geometry refines, with monotone gradient pressure (the sign is correct *a priori*, so growing relevance on a useful triad reduces loss). `b_t` is built from **train** triads only → leakage-safe.

`summarise_gate`, `run_hsikan_rotor.run()` (`--geom-init-scale/-learn-scale/-sign-aware`, per-incidence `inc_balance`, provenance fields) and the `GeomTrainedState` snapshot were extended; `_score` refactored into `_raw_score`/`_score`/`_weights` (no duplication).

## Files touched
- `signedkan_wip/src/core/geometric_triad_attention.py` (+~55 LOC: knobs, `_raw_score`/`_weights`, `summarise_gate` inc_balance)
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` (+~25 LOC: `triad_balance`/`inc_balance`, 3 CLI knobs, run() params, provenance)
- `signedkan_wip/experiments/runs/inspect_geom_gate.py` (inc_balance pass-through)
- `signedkan_wip/tests/test_geometric_triad_attention.py` (+7 tests), `signedkan_wip/tests/test_hsikan_rotor.py` (+1 woken integration test)
- `docs/plans/2026-06-17-geom-attention-wake-score/plan.{tex,pdf,tikz,mmd}` (new)
- CORE.YAML items touched: **none** (only `hymeko_core/` Rust is protected).

## Tests
- `ruff check` (5 files): **PASS**. `pytest -p no:randomly` (both suites): **37 passed in ~19 s** (2 pre-existing torch sparse-CSR warnings, not this change). New regression tests pin: defaults bit-identical to legacy `_score`; `score_init_scale` 1.0 vs 0.1 = 10× projection norm; `learn_scale` adds a live grad'd param; `sign_aware` requires `inc_balance` (raises) and `w = b_t·σ(s̃)`; flipping `b_t` flips the pool; woken pool has live weights (`w_frac_dead = 0`, vs the dead legacy's `1.0`).

## Results — 5-seed A/B (tuned recipe, strict triads, dedup)
`reports/woken_ab_20260617.jsonl` (30 cells) + `reports/woken_smoke_20260617.jsonl` (seed-0 smoke incl. legacy anchor).

| dataset | bilinear | legacy geom (dead) | **woken geom** | Δ woken−bilinear | seed wins | SiGAT | gap (woken) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bitcoin_alpha | 0.8455 | 0.8470 | 0.8455 | **−0.0000** (tie) | 2/5 | 0.884 | −0.039 |
| bitcoin_otc | 0.8685 | 0.8518 | 0.8563 | **−0.0122** (regress) | 0/5 | 0.902 | −0.046 |

**Leakage gate (woken + `--shuffle-train-signs`, 5 seeds):** alpha 0.532 (0.454–0.557), otc 0.504 (0.480–0.561) ≈ chance → **leakage-clean**. The alpha 0.532 mean equals the legacy alpha gate (0.530) — a property of the rotor line, not introduced by the woken change.

### Measured / inferred
**Measured:** the woken score is alive — the live-weights unit test plus the shuffle gate collapsing to chance prove the readout now uses the *real* balance signal. Waking the score recovered part of the legacy otc regression (0.8518 → 0.8563, +0.0045). But woken loses to bilinear: alpha exact tie, otc −0.0122, 0/5 winning seeds on otc.

**Inferred:** the geometric **readout** does not carry information the bilinear endpoint head lacks, even with a functioning, balance-warm-started, dynamically-scaled score. The signed-triad neighbourhood, pooled any way we have tried (mean → dead attention → woken balance-relevance attention), is not the missing ingredient for the SiGAT gap on this line.

## Honest read
This is the **clean falsification** the plan set up: *"if a score with real dynamic range still adds nothing, the readout is genuinely dead and we pivot with confidence."* It does, and it is. Two generations of readout (dead score, woken score) are both negative vs bilinear. The geometry-as-readout hypothesis on simple triads is closed. The woken machinery is correct and leakage-clean; it is kept (defaults off) for the record and for reuse on a richer neighbourhood.

## Performance
- Woken adds 1 scalar param (`log_scale`): 128 213 vs 128 212 (alpha), 195 349 vs 195 348 (otc) — negligible. One σ, one elementwise multiply, one `exp(scalar)` per forward; no new graph-sized allocation.
- Wall: ~10–12 s/cell, 30-cell grid ≈ 6 min. Peak RSS not separately polled; same scale/config as the prior run measured at **1724 MB** (10.5 % of the 16 GB cap).

## §6.5 anti-patterns
None. Knobs are config fields on the existing module (no `geom_attn_v2`, no new function-per-axis, no class-per-variant since the difference is parametric); defaults preserve the old contract (back-compat test pins it); no globals; `_raw_score`/`_weights` factor the shared score path (no duplication). One pre-existing convention reused: the `0.1×` init idiom (now parameterised away from the hardcode).

## Decision / next step
**Pivot to Berge cycles** (the user's stated plan: "option 1, then Berge"). Richer neighbourhoods than simple signed triads; `hymeko_hre/src/traversal/berge.rs` is the foundation. New effort, separate plan dir (don't combine with readout work — confounds attribution). First action there: discovery pass + confirm CORE.YAML scope for `hymeko_hre` before any new enumerator.

Memory: `project-hsikan-geometric-attention-berge` (updated).
