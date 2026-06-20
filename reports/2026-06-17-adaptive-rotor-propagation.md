# Adaptive rotor propagation — learnable retention (parity) + depth (no gain)

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-adaptive-rotor-propagation](../docs/plans/2026-06-17-adaptive-rotor-propagation/) (4 artifacts; PDF compiles).
**Status:** ✅ Phase 1 implemented + tested + 5-seed A/B; Phase 2 (depth) seed-0 scan. **Neutral result:** adaptive propagation removes a hyperparameter at no cost but does **not** push AUROC — the ceiling is input-bounded (third independent confirmation).

## Summary

Two levers from the adaptive-propagation plan, both now characterised:

- **Phase 1 — learnable per-block self-retention.** Replaced the fixed scalar
  `self_weight` in `SignedRotorPropagation` with a per-block learnable retention
  `α_b = retention_floor + exp(θ_b)`, initialised at the fixed sw=4 behaviour
  (bit-parity before training). **5-seed A/B = parity:** alpha 0.8500±0.0128
  (fixed) vs 0.8502±0.0123 (learn); otc 0.8790±0.0102 vs 0.8780±0.0107; val
  identical (0.861/0.884); gates clean (0.53/0.52). Value: it reproduces the
  tuned operating point automatically, **removing the per-dataset `sw` knob at no
  AUROC cost** — not an AUROC lift. Kept default-off (fixed sw reproduces prior).
- **Phase 2 — propagation depth / propagation-as-encoder.** Seed-0 rounds scan
  {2,3,4,6} × {fixed sw4, learnable}: depth does **not** help. alpha **over-smooths**
  past r2 (val 0.874→0.857, test 0.866→0.861); otc is flat (r6 test +0.002, inside
  the 5-seed σ≈0.01). Learnable retention curbs the alpha over-smoothing slightly
  at r6 (val 0.863 vs 0.857) but it does not reach test. **rounds=2 sw=4 remains
  the operating point**; the propagation-as-encoder rewrite was *not* built — the
  scan answered its premise negatively first (CLAUDE.md: discriminating test before
  concluding).

**Design note (softplus → exp).** Dr. Hajdu flagged softplus as the wrong tool
here. Correct: the self-weight is a ratio the S³ `nlerp` normalisation defines
only up to scale → a multiplicative quantity, learned in log-space `exp(θ)` — which
is also exactly a per-block sigmoid residual gate reparameterised (`g/(1−g)=exp θ`),
unifying the plan's two listed options into one parameter. softplus belongs on
*heads* (positivity on an emitted scalar), not inside manifold interpolation.

**Measured / inferred.** *Measured:* the A/B means/stds and the rounds-scan val
trend. *Inferred:* the val ceiling is **input-bounded** — neither retention
schedule nor propagation depth raises the fit, the third independent confirmation
(after geom-attn / rotor-rel / k-cycles) that the degree-only `STRUCT_DIM=6` node
feature is the bottleneck. *Next lever (hypothesis):* enrich the leakage-free input
features beyond degree (signed triangle participation, local balance/frustration,
2-hop signed degree) — the only axis that attacks equal-degree-node indistinguishability.

## Files touched

**Modified (mine):**
- `signedkan_wip/src/embeddings/signed_rotor_propagation.py` — learnable per-block
  retention `α_b = retention_floor + exp(θ_b)` (`learnable`, `n_blocks`,
  `retention_floor` ctor args; `self_retention()`); init-at-`self_weight` parity;
  fixed-scalar path unchanged (default).
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` — `RotorInjector` +
  `run()` thread `prop_learnable`/`rotor_prop_learnable`; `--rotor-prop-learnable`
  CLI flag; provenance field.
- `signedkan_wip/tests/test_signed_rotor_propagation.py` — 6 new: init parity,
  on-sphere, gradient-to-retention, n_blocks-required, block-mismatch, floor bound.
- `signedkan_wip/tests/test_hsikan_rotor.py` — learnable-path run smoke + provenance.

**Artifacts:** `signedkan_wip/experiments/results/rotorprop_learnable_ab.jsonl`
(40 rows, 5-seed Phase-1 A/B). Phase-2 rounds scan is seed-0 diagnostic (stdout).

**CORE.YAML items touched:** none.

## Test results

- `test_signed_rotor_propagation.py`: 14 passed (6 new). `test_hsikan_rotor.py`:
  32 passed (1 new learnable smoke). (`pytest -p no:randomly`.)
- `ruff check`: clean on all touched files. `mypy --strict`: clean on all my added
  code. The module retains **1 pre-existing** strict error (`_propagate`'s
  `acc / acc.norm(...)` → `Any`, a torch-stub deficiency on the *published*
  rotor-propagation numeric path) — not introduced here, left untouched to preserve
  the measured numerics.

## Performance

- Per-cell wall ≈ 10–12 s (GPU, cu132); learnable adds `O(n_blocks)` params
  (negligible). 5-seed A/B (40 cells): peak RSS ≈ 1.77 GB (11 % of cap). Rounds
  scan: 16 cells. No regression claim (measurement task).

## §6.5 anti-patterns

None. The learnable retention is a parametric generalisation of the existing
scalar (config flag, default reproduces prior — not a new structural class).
softplus→exp removed a helper and the typing friction. No new globals.

## Experiment provenance

- Git SHA `7d16ad0` (working tree dirty from the session; touched files above).
- Datasets: cached SNAP `bitcoin_alpha`, `bitcoin_otc`. Seeds 0–4 (A/B) / 0 (scan).
  Device CUDA (torch 2.12.0+cu132). Recipe: rotor embed, bilinear head, dedup,
  tuned BCE (wd 1e-4, grad-clip 1.0, class-weighted, early-stop), r2 sw4.

## Open issues / follow-ups

- **The numbers lever is input enrichment, not propagation.** Leakage-free
  structural features beyond degree (signed triangle counts, local balance,
  2-hop signed degree) — needs its own plan; attacks the confirmed root cause.
- Fuzzy-defuzzification head unification ([[project-fuzzy-defuzzification-heads]])
  is a design lens, not an AUROC lever (the head ablation already showed readout
  algebra is not the bottleneck).
