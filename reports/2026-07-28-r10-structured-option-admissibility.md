# R10.2 Stage 2 (Boundary 3) — structured-option coordinate conditioning + episodic-exploration admissibility

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · parent commit `c8e90e11` (Boundary 2) · dev s1 · downstream/transit/Stage-1B scaffold/physics/safety FROZEN · s4/s7 untouched · f1–f4 SEALED · NO training / reward / actor update · no tag moved**

## Summary

Boundary 3 conditions the frozen structured-option torque-path coordinate *before* any RL: it proves the non-zero coordinate is controllable and well-conditioned (not "an elegantly-named swamp"), freezes a per-dimension exploration normalization `D`, and measures episodic-exploration admissibility over the three pre-registered scales `σ∈{0.05,0.10,0.20}`. No policy is trained; the σ freeze is a **separate review decision**. Ordering executed exactly as pre-registered: **3A** non-zero terminal-offset audit → 15-D `±ε` local sensitivity/SVD → **3B** freeze `D` → `θ=σDz` admissibility → verdicts → STOP.

A small **coordinate refinement** landed first: the transient torque-path basis was upgraded from C⁰ linear-interp to a **C¹ clamped Catmull-Rom** spline (smooth first derivative at the knots and zero slope at the endpoints, still interpolating the knots). This is identity-preserving — the three Boundary-2 identity gates re-run **byte-identical** (`identity_gates.json` unchanged: θ=0 bit-exact, strict K6 2.79 mm) because `k₁=k₂=0 ⇒ transient≡0` for any basis — so the coordinate freeze holds while the non-zero path is now physically smooth.

## Verdicts

| gate | result | evidence |
|---|---|---|
| `TERMINAL_OFFSET_TRACKING` | **PASS** | isolated Δτ_T (only the terminal block active), per-joint ±, monotone/sign-correct on all 4 joints, executed offset tracked-or-mask-explained; no preload random walk |
| `LOCAL_THETA_SENSITIVITY` | **PASS** | task-aware: **no task-dead dims** (all 15 move `min_dtz`, |13.3|–|159.8| mm per unit z), effective rank **12/12**, **no task-redundant pairs**. One terminal-state collinear pair (k₂[j1]↔Δτ_T[j1], cos 0.996) is **advisory only** — task-distinct (−153.6 vs −49.2 mm/z), reported not failed |

## The frozen exploration normalization `D` (proposed)

Equalises the physical (terminal-state) effect per unit `z`; `D_i = median_effect/effect_i`, clamped to `[0.5, 2.0]`. Frozen once; identical across all three σ.

```
dim:   ds    dp    db   k1_0  k1_1  k1_2  k1_3  k2_0  k2_1  k2_2  k2_3  dT_0  dT_1  dT_2  dT_3
D  : 1.000 0.583 0.500 2.000 2.000 1.229 1.717 1.542 2.000 0.500 0.500 1.095 0.975 0.771 0.888
```

## σ result table (pre-registered contract: 3 seeds × 32 = 96 episodes/σ, frozen dev panel, single θ at READY, no per-step noise, frozen downstream + K6 monitor)

| σ | k6_seeds | info_seeds | **safety viol.** | boundary viol. | degen | clip-dom | min_dtz bands `<10 / 10–25 / 25–50 / >50` | admissible (strict) |
|---|---|---|---|---|---|---|---|---|
| **0.05** | 3/3 | 3/3 | **0** | 1/96 | 0 | no | **6** / 1 / 86 / 3 | no¹ |
| 0.10 | 3/3 | 3/3 | **0** | 2/96 | 0 | no | 5 / 1 / 85 / 5 | no¹ |
| 0.20 | 1/3 | 3/3 | **0** | 3/96 | 0 | no | 1 / 0 / 79 / 16 | no¹ |

¹ The **only** unmet strict criterion is zero boundary-route regressions. **There are zero safety violations at every σ** (peak q̇ / coin-speed within limits — the governed stack holds). The "boundary violations" are episodes where the downstream took a different HANDOFF_RESET count (nominal = exactly 1) while remaining **physically safe** — a safe alternate route, not a safety failure. Nominal θ=0 remains bit-exact strict K6 (2.79 mm, 1 reset).

**Reading.** Exploration produces both **positive (K6) and safe distinct negative** samples at every σ — the learnable ranking signal the critic needs — with **no safety violations, no degenerate/duplicate trajectories, and clipping never dominating**. The distribution is positive-sparse (most perturbations land safely in 25–50 mm); a smaller σ would raise positive density.

## Recommendation (for the separate σ-freeze approval)

**σ = 0.05** is the recommended freeze: it is clean on safety (0/96), K6 (3/3 seeds, 6 deliveries), informative negatives (3/3 seeds), degeneracy (0), and clipping. Its single residual is **1/96 (~1%) safe boundary-route variation**. Two review options:
1. **Freeze σ=0.05** and let the TD3 boundary-correctness reward term penalize the rare reset≠1 route (recommended — the sample is safe and rare, and a boundary reward is already planned).
2. **Authorize probing σ<0.05** (e.g. 0.03) for a zero-boundary-regression run with denser positives (not pre-registered here; a small extra measurement).

I did **not** unilaterally relax the pre-registered strict criteria; `proposed_smallest_admissible_sigma` stays `null` and the recommendation is surfaced separately for your decision. Machine-readable: `reports/2026-07-28-r10-structured-option-torque-path-td3/admissibility.json`.

## Files touched

| file | role | Δ |
|---|---|---|
| `torque_path_option.py` | **modified** — transient basis C⁰→**C¹** clamped Catmull-Rom (`_catmull_rom_clamped`); identity-preserving | +32/−5 |
| `torque_path_conditioning.py` | **new** — `axis_sensitivity` (±ε Jacobian + SVD, effective/stable rank, dead/collinear), `freeze_normalization` (`D`), `sample_theta` (`θ=σDz`) | +129 |
| `coin_kinetic_structured_option_admissibility.py` | **new** — 3A terminal-offset audit + 15-D sensitivity, 3B normalized ≤3-scale admissibility, verdicts + recommendation; imports no trainer | +262 |
| `test_torque_path_option.py` | **modified** — added C¹ derivative-continuity test | +13 |
| `test_torque_path_conditioning.py` | **new** — 7 tests (sampler, normalization, physics sensitivity) | +76 |
| `admissibility.json` | **new** — machine-readable Boundary-3 result | — |

**CORE.YAML items touched: none.** Frozen scaffold/downstream/transit/physics/safety untouched; s4/s7 untouched; f1–f4 sealed.

## Test results

- **`pytest -p no:randomly`: 31 passed** (17 torque_path incl. the new C¹ test, 7 conditioning, 8 existing capture_rl) in ~72 s.
- Unit: C¹ derivative continuity at knots + zero-slope endpoints; `sample_theta` (σ=0→exact zero, per-dim `D`, clipped); `freeze_normalization` (bounded, inverse-to-effect, no unbounded dead-dim amplification). Physics: `axis_sensitivity` every dim moves the task, terminal-offset dims move the preload, sane SVD rank.
- Coverage: every new function exercised. The C¹ test would fail against the prior linear-interp basis (knot slope discontinuity).

## Static analysis & performance

- **`ruff check`: All checks passed.** **`radon cc -a -nc`: no blocks at C or worse** (`_seed_scale` folded to B; `run`/`_scale_admissibility` B; conditioning all A/B). Module lengths 129 / 262 (< 400 heuristic).
- **§6.5 anti-patterns: none introduced.** No global mutable state; `D`/σ passed explicitly; one `rollout` entry; the rig reused from the audit `_rig` (no new copy). Env-free; deterministic given seeds.
- Admissibility run: **wall 42.5 s, peak RSS 0.25 GB** (`/usr/bin/time -l`) — under the 2 GB plan budget and 16 GB cap. Production-scale (real mujoco 3.10, full downstream horizon 80, the 16-member frozen dev panel × 3 seeds × 32 episodes × 3 σ = 288 episodes).

## Provenance

- Parent `c8e90e11`; this boundary committed as its own commit (files above). No tag moved. Unrelated untracked worktree artifacts not staged.
- Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU).
- Seeds: exploration seeds `{0,1,2}`; panel seed `90210`; sensitivity central-difference `eps=0.10` (a local linearisation — a large-ish step; the Jacobian is a local read, not a global claim). Deterministic.

## Confirmation of scope (what did NOT happen)

- **No training, no reward optimization, no actor update, no SAC/PPO, no geometry generalization.** The admissibility experiment imports no trainer.
- No persistent state mutated beyond the read-only result JSON + report.
- The σ freeze is **not** finalized here — it is proposed for separate approval.

## Stop condition (Boundary 3) — STOP for review

- `TERMINAL_OFFSET_TRACKING` = **PASS**
- `LOCAL_THETA_SENSITIVITY` = **PASS** (coordinate well-conditioned; one advisory terminal-state collinear pair)
- σ result table = above (0.05 / 0.10 / 0.20)
- proposed smallest admissible σ (strict) = **none**; **recommended σ = 0.05** (residual 1/96 safe boundary variation)

**Awaiting your decision on the σ freeze** (option 1: freeze 0.05; option 2: probe σ<0.05). TD3 / reward / actor update do **not** start until the σ is frozen.
