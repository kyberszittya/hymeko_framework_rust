# M1 — verified Lyapunov certificate of the viability boundary

**Date:** 2026-08-03
**Plan:** `docs/plans/2026-08-02-learned-viability-boundary/` (M1 addendum added; on disk, gitignored per repo convention)
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `1662160d`)

---

## Summary

Implemented milestone **M1**: a **verified Lyapunov certificate** of the viability boundary — the M1 substance
is *certificate vs. classifier*, not "neural". Delivered **numpy-only, no new dependency**:

- **`LyapunovCertificate`** (added to `viability.py`) — a learnable PSD quadratic `V(x) = zᵀ(LLᵀ+εI)z` over
  `z = (wrap(θ−θ*), θ̇)`. PSD by construction (`V ⪰ 0`, `V(x*) = 0`); **fit** by gradient descent on `L` with the
  **analytic** gradient of the discrete-decrease hinge `Σ relu(V(x⁺) − V(x))` on the non-crossing region (no
  autodiff — `∂(zᵀLLᵀz)/∂L = 2(zzᵀ)L`); **certified_level** = the largest `c` with decrease-and-no-saddle-cross
  on `{V ≤ c}`; **verify** = dense-sample the violation rate + IoU vs the analytic ROA. Seeded from `H_d`
  (`P₀ = diag(½k, ½I)`).
- **Refactor** — extracted one shared `closed_loop_step` used by both the rollout labeller and the certificate's
  H-step lookahead (the "one shared step" contract; a regression test pins it to the manual integrator, and the
  M0 suite confirms the labels are unchanged).

**The torch/§1 decision, made honestly:** the *neural* certificate would add torch — a §1 core change. It is not
needed on the pendulum (a linear closed loop ⇒ the exact certificate is quadratic), so M1 ships dependency-free.
The neural certificate is deferred to **M2** (a genuinely nonlinear runner basin), which is the honest trigger
for the §1 escalation — not a default.

## Results (self-validating on the pendulum)

| | seeded by H_d | unseeded (identity) |
|---|---|---|
| IoU (certified set vs analytic ROA) | **0.937** | 0.599 |
| verified violation rate on `{V≤c}` | **7e-4** | 1e-3 |
| certified level vs barrier `c*=½kπ²=118.4` | **107.6** (0.91·c*, strictly inside) | 11.1 |

- **Verified + conservative:** the certified sublevel set has a near-zero decrease/no-cross violation rate and
  its level stays **strictly below the true barrier** `c*` (0.91·c*) — it never over-certifies unsafe states.
- **Honest scoping:** the **H_d seed is load-bearing** (IoU 0.94 seeded vs 0.60 unseeded) — the fit *verifies*
  decrease but does not, from identity, recover the anisotropic H_d shape. On the pendulum `H_d` is already a
  valid Lyapunov function, so the fit is near-degenerate: this is the *self-validation* (it recovers the known
  answer), not a strength claim. The certificate does non-trivial learning only at M2.

## Files touched (full list)

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/viability.py` | +85 | `LyapunovCertificate`; refactor to shared `closed_loop_step` |
| `tests/test_viability_certificate.py` | +100 (new) | 6 tests (PSD, recovers-ROA, conservative, seed-load-bearing, shared-step, perf) |
| `reports/2026-08-03-learned-viability-boundary-m1.md` | new | this report |

Pre-existing `scenarios/humanoid/certificate.py` and `lyapunov.py` are **unrelated CIP-control modules** — read,
identified as different concerns, and **left untouched** (the certificate lives with the viability machinery
instead, one cohesive module ~280 LOC < the 400 heuristic).

## CORE.YAML items touched
None. **No new dependency** (numpy-only). The neural/torch certificate is explicitly deferred to M2 as the §1
escalation point.

## Test results
- `pytest tests/test_viability_certificate.py -p no:randomly` → **6 passed in 0.22 s**.
- Regression: `tests/test_viability_boundary.py` (M0) → **14 passed in 56.5 s** (labels unchanged after the refactor).
- `ruff check` → clean.

## Performance
- `LyapunovCertificate.fit` (3000 GD steps on ~1600 2-D samples) + `verify`: **< 0.2 s** (the full 6-test suite,
  including the 5-iteration perf test, runs in 0.22 s) — budget 10 s, met by ~50×. RSS negligible (numpy 2-D).

## §6.5 anti-patterns
None. `LyapunovCertificate` bundles state+behaviour; one shared `closed_loop_step` (removed a duplicated
integrator, §6.1); no globals; explicit `ValueError` on empty fit. Did **not** clobber the same-named
pre-existing modules — chose cohesion (certificate beside `viability.py`) over a colliding filename.

## Open issues / follow-up
- **M2:** centroidal capturability of the runner — a nonlinear basin where the quadratic class is insufficient,
  so a neural certificate (torch) is warranted → the **§1 escalation** (written justification + migration plan)
  is due there, not before. The `L`-ports (foot placement, arm swing, shoulder + pitch hold) become the inputs.
- The finite lookahead (0.1 s) makes `certified_level` slightly conservative vs `c*`; a longer lookahead or a
  continuous decrease certificate would tighten it.

## Provenance
Git SHA at start `1662160d`. Env: HyMeKo `.venv` (Python 3.11, NumPy, SymPy 1.14; torch/MuJoCo present but
unused here), macOS (darwin 25.5). Deterministic: fixed `seed=0`, pinned `dt=4 ms`, no RNG in the certificate;
GD is full-batch. No GPU, no dataset.
