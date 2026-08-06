# Lipschitz-sound formal verification of the M2 neural certificate (honest: sound but conservative)

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `88fccb44`)
**Follow-up (2)** — a formal (sound) guarantee for the M2 neural certificate, replacing the empirical sampling.

---

## Summary

The M2 neural `V_θ` decrease was only checked by sampling. This adds a **sound Lipschitz-certified** verification —
a real per-cell guarantee, not sampling — and reports honestly what it can and cannot deliver.

- **`NeuralLyapunovCertificate.lipschitz_formal_verify`** (numpy + torch, no new dep): the `(L,pitch)` flow is
  **linear**, so the H-step flow matrix `M_H` and its norm are exact; `V` is Lipschitz with
  `L_V = 2·max‖φ−φ*‖·∏‖W_i‖₂` (each Tanh is 1-Lipschitz); the decrease residual `g = V(x⁺)−V(x)` then has
  `L_g = L_V(‖M_H‖+1)`. On a grid with cell half-diagonal `r`, a cell where `g(centre) ≤ −L_g·r` **provably**
  decreases over its whole extent, and `maxpitch(centre) ≤ fall_pitch − L_fall·r` **provably** does not fall — a
  sound guarantee for that cell. `spectral_lipschitz()` exposes the network's Lipschitz bound.

## Results — sound, refining, but conservative (the honest finding)

The guarantee is real and refines as the grid tightens (`r → 0`):

| grid_n | cell radius `r` | **sound coverage** | uncertifiable core `L_g·r` |
|---|---|---|---|
| 81  | 0.076 | 0.243 | 5.01 |
| 161 | 0.038 | 0.585 | 2.52 |
| 241 | 0.026 | **0.713** | 1.68 |

- **Sound coverage → 0.71**: up to 71 % of the box is *provably* decrease-and-no-fall (a genuine formal
  guarantee, monotone-improving with resolution).
- **But no clean Lyapunov sublevel** (`max_sound_sublevel ≈ 0`), for two honest reasons: (i) the naive
  spectral-norm bound is **loose** — `L_g ≈ 65.8`, so the required margin `L_g·r` (≈ 1.7 even at grid 241) exceeds
  the actual decrease, leaving an **uncertifiable core** around the gait (where `g → 0`) that *contains the entire
  empirical certified set* `{V ≤ 3.998}`; and (ii) soundness is **not monotone in `V`**, so a `{V ≤ c}` sublevel
  mixes sound and unsound cells.
- **Conclusion:** naive Lipschitz gives a *sound but conservative coverage* guarantee here, not a tight sublevel
  certificate. A tight sublevel certificate for a neural `V` needs a **tighter Lipschitz bound (LipSDP / interval
  bound propagation)** or an **SMT / reachability** tool (dReal, Marabou) — the latter would be a §1 dependency;
  flagged, not added. The exact route already exists for the *linear-flow / quadratic-V* cases (M1 and the
  Poincaré certificate carry an **exact LMI**), so the neural certificate is the one case still on sampling +
  this conservative sound bound.

## Files touched

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/neural_certificate.py` | +45 | `spectral_lipschitz`, `lipschitz_formal_verify` (sound coverage + core) |
| `tests/test_centroidal_certificate.py` | +11 | sound-coverage refines with resolution; core shrinks; Lipschitz finite |
| `reports/2026-08-05-lipschitz-formal-verify-neural.md` | new | this report |

## CORE.YAML items touched
None. numpy + the already-pinned torch (`==2.12.0`); no new dependency. SMT tools deliberately **not** added (§1).

## Test results
- `pytest tests/test_centroidal_certificate.py -p no:randomly` → **8 passed in 6.4 s** (incl. the Lipschitz test).
- `ruff check` → clean. Additive method; no other suite affected.

## §6.5 anti-patterns
None. Additive methods on the existing class; reuses the shared `centroidal_step`; no globals; honest reporting
of the conservative result rather than dressing it as a tight certificate.

## Open issues / follow-up
- **Tight neural certificate:** LipSDP / IBP for a much smaller `L_g`, or SMT/reachability (§1) for an exact
  neural sublevel — either would turn the 0.71 sound *coverage* into a genuine sound *sublevel* certificate.
- The **exact** formal route is already in place for the linear/quadratic certificates (M1 LMI, Poincaré LMI); the
  neural case is the remaining gap.

## Provenance
Git SHA at start `88fccb44`. Env: HyMeKo `.venv` (Python 3.11, torch 2.12.0 CPU, NumPy 2), macOS (darwin 25.5),
4 CPU threads. Deterministic (seeded fit; the verify is a deterministic grid sweep). No GPU, no dataset.
