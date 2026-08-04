# Limit-cycle-aware certificate via the Poincaré map (the L-coupled running regime)

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `ed0fe2b6`)
**Follow-up (1)** of the viability arc — certifying the running gait where `L` genuinely matters.

---

## Summary

M2⁺ found that a point-Lyapunov certificate collapses on the running gait because the target is a **limit cycle**,
and that this only bit once the regulation was soft enough for `L` to matter. This resolves it with the standard
tool: the **Poincaré map**. Taking a section at gait phase 0, the limit cycle becomes a **fixed point `x*` of the
one-stride map `P`**, so a quadratic Lyapunov `V(x) = eᵀPe` (with `e = (L,pitch) − (L*,pitch*)`) certifies it by
the stride-to-stride decrease `V(Px) ≤ V(x)` — reducing limit-cycle stability to point stability.

- **`limit_cycle.py`** — `soft_running_config` (the `L`-coupled regime, `pitch_gain 2.5 / l_damp 4 / torque_bias 3`);
  `gait_fixed_point` (iterate the stride map to its `(L,pitch)` fixed point); `stride_map` (one phase-aligned
  period, with a fall mask); `PoincareLyapunovCertificate` — a PSD quadratic on the section fit by the analytic
  stride-decrease gradient, with two verifications:
  - **exact LMI** on the numerically-linearised stride map `DP` at `x*`: `Q = DPᵀP DP − P ⪯ 0` **and** spectral
    radius of `DP` < 1 (a formal guarantee, plus a stability certificate for the gait itself);
  - **empirical** rollout-consistency: section states inside `{V≤c}` converge to the gait over several strides
    without falling.

## Results

- **Genuine fixed point:** iterating the Poincaré map gives `x* = (L*=0.395, pitch*=0.125)` with a `(L,pitch)`
  stride residual of **0** (machine precision) — fixing the M2⁺ collapse (the point target's residual had made
  `x*` itself "increasing").
- **Stable gait, formal guarantee holds:** `DP` spectral radius **0.47 < 1** (the running gait is stably
  attracting), and `Q ⪯ 0` (max eig **−0.77**) — `V` provably decreases stride-to-stride. This is an **exact**
  guarantee, not a sample.
- **Conservative + L matters:** `certified_level = 1.36`, **fall-violation 0.0** (no certified section state
  falls), and the certified set spans **L ∈ [−0.6, 1.4]** (width 2.0) — `L` is genuinely a live coordinate here,
  which was the entire point of the soft regime. IoU vs the multi-stride-recoverable set is **0.13**: a *conservative
  one-stride-capturable inner approximation*, honestly — a state that needs several strides to converge (with `V`
  transiently rising) is recoverable but outside the one-stride-decrease set. Tightening it needs a higher-order /
  neural `V` on the section or a multi-step decrease; flagged, not claimed.

## Files touched

| File | LOC | notes |
|---|---|---|
| `scenarios/humanoid/limit_cycle.py` | +142 (new) | Poincaré section, stride map, `PoincareLyapunovCertificate` (fit + LMI + rollout verify) |
| `tests/test_limit_cycle.py` | +89 (new) | 8 tests (fixed point, contraction, PSD, formal LMI, conservative, L-dependence, determinism, perf) |
| `reports/2026-08-05-limit-cycle-poincare-certificate.md` | new | this report |

## CORE.YAML items touched
None. Non-core Python; numpy-only (no torch needed — the quadratic-on-section certificate has an analytic
gradient). No dependency change.

## Test results
- `pytest tests/test_limit_cycle.py -p no:randomly` → **8 passed in 2.0 s**.
- `ruff check` → clean. No regression risk (new module + `soft_running_config`; the default `CentroidalConfig`
  and all prior suites are untouched).

## Performance
Fit (625 section samples, 3000 analytic-gradient steps) + `gait_fixed_point` (200 stride iterations) + `verify`
(multi-stride rollout): the full 8-test suite runs in **2.0 s**. numpy-only, RSS negligible.

## §6.5 anti-patterns
None. Reuses the single shared `centroidal_step` (via `stride_map`); `PoincareLyapunovCertificate` bundles
state+behaviour; `soft_running_config` is a thin config helper (parametric, not a new axis); no globals.

## Open issues / follow-up
- **Coverage:** the quadratic one-stride certificate is conservative (IoU 0.13). A neural `V` on the section, or a
  multi-step (`V(Pᵏx) ≤ V(x)`) decrease, would enlarge it.
- **Robustness of the section fixed point:** `(z,ż)` are not certified here (decoupled, and the `z`-bounce need
  not be `TC`-periodic); the certificate is purely transverse in `(L,pitch)`. A full-section (4-D) transverse
  certificate is the next step if `(z,ż)` coupling is ever introduced.

## Provenance
Git SHA at start `ed0fe2b6`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2), macOS (darwin 25.5). Deterministic:
fixed-point iteration + analytic-gradient fit have no RNG; pinned `dt = 4 ms`. No GPU, no dataset.
