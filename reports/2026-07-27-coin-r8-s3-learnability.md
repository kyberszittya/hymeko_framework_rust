# Coin-R8 · S3 — development-only residual-learnability audit

**Date:** 2026-07-27 · **Branch:** `recovery/coin-r8-tip-referenced-residual-rl` (worktree `hymeko_coin_r8_wt`)
**Gate result:** `RESIDUAL_LEARNABILITY_SIGNAL_PASS` → **RL (S4) AUTHORISED**
**Preceding gates:** S0 physical feasibility PASS · S1 safe scaffold PASS · S2 update-zero identity PASS (`a7bd37a8`, bit-exact, max diff `0.00e+00`).

## Scope

S3 is a **development-only** audit (dev cradles `s1`, `s3`; held-out `s4`, `s7` are **not touched**). It answers a single
question before any SAC/TD3 is allowed to run: *does the bounded residual interface over the frozen tip-referenced scaffold
carry a learnable, safe, rankable signal?* No teacher actions are used as labels. The corrected methodology is binding — a
safe deterministic scaffold is required; deterministic 4/4 delivery is **not** an RL prerequisite.

## S3.1 — residual sensitivity (does the interface do anything?)

Per-role ±0.6 perturbation of the tanh emission over the dev cradles, classified EFFECTIVE / WEAK / INERT / UNSAFE:

| role | class |
|---|---|
| `d_fwd_vel` (forward joint-velocity reference) | **EFFECTIVE** |
| `d_squeeze` | WEAK |
| `d_stop_gain` (servo/decel gain) | WEAK |

No role is INERT or UNSAFE → the interface **has effect** and is safe to actuate. The dominant lever is the forward
velocity reference, as expected for a transport task.

## S3.2 — safe-positive candidate existence

40 samples × {constant, piecewise, coherent} × 2 dev cradles = **240 candidates**, scored by option-return vs the
update-zero scaffold (K6-independent return; safety-gated on `peak_coin_speed`/`peak_qdot`):

- **safe rate = 1.00** (every sampled residual stays inside the physical envelope)
- **safe-positive rate = 0.508** (over half improve on the scaffold, all while safe)
- improvement: median **5.8 mm**, max **254 mm** (best single candidate)
- by kind: coherent **0.55** > constant 0.513 > piecewise 0.463 → `temporal_coherence_helps = true`

A safe, positive, non-trivial region of the residual space exists, and temporally-coherent residuals help — consistent with
a transport task where a sustained push/brake schedule beats i.i.d. jitter.

## S3.3 — rankability (can a value estimator prefer the good ones?)

Cross-validated ridge difference-predictor `Δreturn(state, residual)` over the 240 safe rows, using **sign-preserving**
features (the signed candidate direction `a`, kind, cradle) — the earlier magnitude-only feature set was a bug in this
diagnostic (it could not separate +δ_fwd from −δ_fwd, opposite-effect residuals). Robustness by **median + IQR + hit-fraction
over 25 random half-splits** (§3 multi-seed discipline — a single split is a point estimate, not a verdict):

| metric | median | IQR (p25–p75) | frac. splits crossing threshold |
|---|---|---|---|
| Spearman ρ | 0.129 | [0.077, 0.191] | 0.20 (> 0.2) |
| pairwise acc. | 0.528 | [0.487, 0.55] | 0.00 (> 0.6) |
| top-10% enrichment | **1.379** | **[1.304, 1.525]** | **0.76 (> 1.3)** |

**Robust criterion (fixed before the result): rankable ⟺ a MAJORITY of resampled splits cross a single-metric threshold.**
Met by top-decile enrichment (0.76 of splits, IQR entirely above 1.3). Spearman is weak but **consistently positive**
(IQR > 0); pairwise ordering is near chance.

## Honest interpretation (carried into S4)

The predictor **cannot finely order** candidates (pairwise ≈ chance, ρ ≈ 0.13) but **reliably concentrates positive
candidates in its top decile** (enrichment ≈ 1.38, robust across splits). That is precisely the signal a value-based critic
exploits: a policy climbs the top of the value ranking, it does not need the full order. The learnability signal is therefore
**real and robust but shallow** — S4 should expect a critic to find *good directions*, not *large margins*. This tempers the
S4 success bar and is a discriminating expectation, not a promise.

## Verdict

`has_effect = true` ∧ `has_safe_positive = true` ∧ `rankable = true` → **`RESIDUAL_LEARNABILITY_SIGNAL_PASS`**.
With S2 (update-zero identity) also PASS, the corrected gate authorises **S4 (matched SAC/TD3 on dev)**. All integrity
constraints remain hard; held-out `s4`/`s7` remain excluded from every training, replay, hyperparameter and threshold
decision through S4.

## Artifacts

`reports/2026-07-27-coin-r8-residual-rl/{s3_sensitivity,s3_candidate_search,s3_rankability,s3_learnability}.json`
· harness `hymeko_rl/experiments/coin_r8_residual_rl.py` (`--s3`) · adapter `hymeko_rl/coin_delivery/theta_option/residual_adapter.py`.
Wall: 99.8 s (panel build dominated). No CORE.YAML items touched.
