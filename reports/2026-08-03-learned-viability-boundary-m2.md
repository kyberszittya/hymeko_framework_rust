# M2 — neural Lyapunov certificate of the centroidal capturability basin

**Date:** 2026-08-03
**Plan:** `docs/plans/2026-08-02-learned-viability-boundary/` (M2 addendum; on disk, gitignored per repo convention)
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `8d5ebadf`)

---

## Summary

Implemented milestone **M2**: a **neural Lyapunov certificate** of the runner's capturability basin — the
nonlinear case M1's quadratic could not reach — reusing the M1 verification idea one dimension up.

- **`centroidal.py`** — the L-regulated centroidal closed loop as a vectorised numpy dynamics over the reduced
  state `x = (z, ż, L, pitch)` (mirrors the visualization: SLIP bounce + the angular-momentum ports that damp `L`
  and hold the pitch upright; "fall" = pitch crosses `fall_pitch`). One shared `centroidal_step`, and a rollout
  labeller `centroidal_rollout` (recover/fall ground truth).
- **`neural_certificate.py`** — `NeuralLyapunovCertificate` (torch MLP): `V_θ(x) = ‖φ_θ(x) − φ_θ(x*)‖²`
  (`V ⪰ 0`, `V(x*) = 0` by construction), fit with the discrete-decrease hinge on non-falling rollouts plus a
  term that pushes `V` high on falling states so `{V ≤ c}` excludes them; `verify` by dense sampling on a
  held-out grid.

**The §1 status, resolved by inspection:** `torch==2.12.0` is **already pinned** in `CORE.YAML` and
`scenarios/humanoid/*.py` is non-core, so *using* torch is **not** a dependency edit — **no §1 escalation**
(my earlier "M2 needs §1" claim was withdrawn after checking the manifest). The version is left at `==2.12.0`.

## Results (production-scale smoke + verification)

- **§3 production-scale smoke** (before any training): 28 561 states, real horizon 1.5 s (375 steps) →
  **0.11 s** wall, both classes present (recover 0.85 / fall 0.15). The basin is real and cheap to sample.
- **Certificate** (grid_n=9, 6561 states, 400 Adam iters, torch CPU): fit **2.6 s**; verify on a held-out grid →
  **fall_violation_rate 4e-4**, IoU-vs-recoverable **0.45**, certified 2633/6561. `V(x*)=0`, `V>0` elsewhere.

**Honest reading:** the certificate is **verified and conservative** — a state inside `{V ≤ c}` falls only ~0.04 %
of the time (the safety property) — but it is a **loose inner approximation**: it certifies ~45 % of the
recoverable set. More training / a larger grid makes it *more* conservative (IoU 0.45 → 0.33), not tighter — the
"exclude falling states" term shrinks the sublevel. Tightening the certificate (architecture/loss, or a formal
bound) is the honest follow-up; the claim here is a **verified safe inner set**, not a complete basin.

**Verification is empirical, not a proof.** Higher-dim ⇒ sampling can miss violations between samples; `verify`
reports the sampled fall-violation rate, and a formal SMT/Lipschitz guarantee is flagged as a separate later
step, not claimed.

## Files touched (full list)

| File | LOC | notes |
|---|---|---|
| `scenarios/humanoid/centroidal.py` | +106 (new) | vectorised centroidal closed loop + rollout labeller |
| `scenarios/humanoid/neural_certificate.py` | +107 (new) | torch MLP Lyapunov certificate + sampling verify |
| `tests/test_centroidal_certificate.py` | +88 (new) | 7 tests (dynamics, PSD-analog, conservative, coverage, determinism, perf) |
| `reports/2026-08-03-learned-viability-boundary-m2.md` | new | this report |

## CORE.YAML items touched
None. **No dependency change** — torch used at the pinned `==2.12.0`; numpy present. `scenarios/humanoid/*.py` is
non-core. §1 not triggered.

## Test results
- `pytest tests/test_centroidal_certificate.py -p no:randomly` → **7 passed in 9.0 s** (torch CPU, 4 threads).
- No regression risk to M0/M1 (only new files added; `viability.py` unchanged this milestone).
- `ruff check` → clean.

Coverage: **dynamics** (deterministic shared step; near-nominal recovers; basin has both classes) ·
**certificate** (V(x*)=0, V>0; conservative fall-violation ≤ 0.02; non-trivial coverage IoU ≥ 0.35;
deterministic fit) · **performance** (fit+verify median < 30 s).

## Performance
- Smoke: 28 561 states × 375 steps = 0.11 s (vectorised numpy). Fit (6561 states, 400 iters, torch CPU): 2.6 s;
  full 7-test suite 9.0 s. Peak RSS well under the 2 GB budget (torch CPU small MLP + numpy). No GPU.

## §6.5 anti-patterns
None. `centroidal_step` is the single shared integrator (no duplication); `NeuralLyapunovCertificate` is one class
(state+behaviour); `CentroidalConfig` is a typed config; no globals (a module logger only). Explicit `ValueError`
on empty fit. Determinism via seeded init; no RNG in the loop.

## Open issues / follow-up
- **Tighten the certificate:** the inner approximation is loose (IoU ~0.45). Options: a level-set/coverage-aware
  loss, a larger/So­bol sample, or annealing the exclusion term. Track coverage vs. violation as the trade-off.
- **Formal guarantee:** replace sampling `verify` with an SMT / Lipschitz-bounded check for a real certificate
  (currently empirical). Flagged, not claimed.
- **Close the loop to the policy:** M2 certifies the *scripted* L-regulator's basin; co-learning a policy inside
  the certified set (a shield) is the next arc — reusing `centroidal_step` so the certificate tracks the deployed
  controller.

## Provenance
Git SHA at start `8d5ebadf`. Env: HyMeKo `.venv` (Python 3.11, torch 2.12.0 CPU, NumPy 2, SymPy 1.14), macOS
(darwin 25.5), 4 CPU threads. Deterministic: fixed `seed=0` (torch + numpy), pinned `dt=4 ms`, no RNG in rollouts.
No GPU, no external dataset (analytic sampling + simulation).
