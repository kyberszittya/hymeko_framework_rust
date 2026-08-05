# R11.5R — Density-ablation learning curve (density vs descriptor)

**Date:** 2026-08-05
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA:** `498b20ff`
**Verdict (parametric):** `R11_5R_DESCRIPTOR_LIMITED_DENSIFY_UNLIKELY_TO_HELP`
**Actionable finding:** the **retrieval** policy is density-responsive — densify *for retrieval*, not for the regressor.

---

## Question

The B0-vs-B1 A/B showed robust targets fix narrow-basin learnability (train fit + crude-policy
held-out) but leave the **parametric** ridge/mlp held-out flat. A free `nn_distance` diagnostic hinted
at a coverage signature (held-out misses farther from the train manifold). Coverage and descriptor
are different levers with very different cost, so this settles it directly: subsample the 38
WIDE-recertified train scenarios to k ∈ {10, 20, 30, 38} × 3 seeds, refit the **same** BC policies,
and measure the **fixed** 12 held-out (dev+test) closed-loop strict-K6 + delivered dtz.

- held-out **rises** with k → density-limited (densify helps);
- held-out **flat** → descriptor-limited (more demos at this descriptor won't help).

Held-out capture snapshots reconstructed once; only the training subset varies. Harness unchanged.

---

## Curve (per-k mean over 3 seeds; held-out = 12 dev+test)

| k | mean_theta K6 / dtz | nearest_schedule K6 / dtz | ridge K6 / dtz | mlp_bc K6 / dtz |
|---|---|---|---|---|
| 10 | 0.33 / 27.7 | 0.33 / 23.6 | 0.25 / 47.7 | 0.22 / 64.7 |
| 20 | 0.33 / 29.6 | 0.36 / 22.9 | 0.20 / 100.4 | 0.25 / 98.8 |
| 30 | 0.33 / 29.7 | 0.36 / 23.0 | 0.17 / 64.1 | 0.22 / 56.2 |
| 38 | 0.33 / 29.0 | **0.42 / 21.5** | 0.17 / 67.8 | 0.17 / 80.2 |

(dtz in mm; K6 rate over 12 held-out. k=38 = full wide pool → identical across seeds, zero variance.)

---

## Findings

**1. The parametric map is DESCRIPTOR-limited, not density-limited.**
Ridge held-out K6 goes 0.25 → 0.20 → 0.17 → 0.17 as training grows; mlp 0.22 → 0.25 → 0.22 → 0.17.
Adding wide-basin training scenarios does **not** improve — and for ridge slightly **degrades** —
held-out generalization (more data pulls the global linear fit toward the bulk, away from the
held-out regions). Parametric held-out **dtz stays 48–100 mm** (far off strict-K6) at every k. A
smooth regressor cannot generalize this descriptor→θ map regardless of how many wide demos it sees.

**2. The RETRIEVAL policy is DENSITY-responsive — and is the best held-out policy.**
`nearest_schedule` held-out K6 **rises 0.33 → 0.36 → 0.36 → 0.42** with k, its dtz **drops**
23.6 → 21.5 mm, and its predictions stay **close** (~22 mm) at every k while the regressors are 50–100 mm
off. Retrieval benefits from a denser table (nearer neighbour → better robust-θ). `mean_theta` is a
constant policy, flat by construction (0.33).

**3. The two A/B levers are now disambiguated.**
The A/B left "densify demos" and "retrieval policy" as parallel options. The curve resolves them into
**one** path: densifying *for the parametric regressor* is a dead end (descriptor-limited);
densifying *for a retrieval policy* is the lever (retrieval rises with k and is already the best,
CEM/oracle-free, held-out policy). The `nn_distance` coverage hint was a weak correlate the direct
ablation refutes **for the regressor** but corroborates **for retrieval**.

---

## Interpretation & recommendation (HALT for review)

The residual R11.4B/R11.5R bottleneck is **not** reward (R11.6B), **not** basin width (fixed by
R11.5R), **not** parametric-map data density (refuted here) — it is the **policy form**. The
deployable, teacher-free (no CEM / no oracle) form for the seen + near distribution is
**nearest-robust-basin retrieval**, and **densification helps retrieval** (unlike the regressor).

**Recommended next lever:** a **retrieval deployment policy over a densified robust-demo table** —
combine R11.5R's wide-basin targets with more scenarios on a finer grid, deployed via descriptor →
nearest robust-θ. This is the coherent path and stays within the R11 teacher-free-deployment goal.

**Honest bound:** retrieval held-out is **0.42 at k=38 — still below the 0.50 bar.** The curve shows
retrieval *rising* in [10, 38] but cannot prove it clears 0.50 with more demos (k > 38 needs the
densification itself). So densification-for-retrieval is a **hypothesis the curve supports, not
proves**; it is the test worth running, where densification-for-regressor is not.

**RL:** stays off the table for held-out generalization — a warm-start from the descriptor-limited
regressor inherits its ceiling. RL re-enters only as residual refinement on top of a retrieval
policy, or after a fundamentally different (non-smooth) map.

---

## Caveats

- **Held-out is 12 scenarios**; K6 moves in 1/12 steps. The single-policy deltas are ±1 scenario
  (ridge −1, retrieval +1) — small. The **qualitative pattern** (parametric flat/declining vs
  retrieval rising) is consistent across k and **reinforced by the continuous dtz** signal
  (parametric 48–100 mm vs retrieval ~22 mm at every k), which is not subject to the 1/12 quantum.
- The verdict label is **parametric-scoped** (computed from ridge/mlp, the "will densify help the
  regressor?" question). It is accurate for that; the retrieval-positive is the actionable half and
  is foregrounded above, not in the label.

---

## Files / provenance

| File | Δ |
|---|---|
| `hymeko_rl/experiments/r11_5r_density_curve.py` | curve driver (reconstruct-once, sweep k×seed) |
| `hymeko_rl/tests/test_r11_5r_density_curve.py` | 5 tests (pool / subsample / summ / aggregate / verdict) |
| `reports/2026-08-05-r11-5r-density-curve/curve.json` | full curve + per-point records |

- **CORE.YAML:** none. **Deps:** none. Harness `r11_4b_conditioned_bc` reused read-only (no CEM, no
  controller change).
- **Env:** framework `.venv`, torch 2.12.0, macOS, CPU; single process, `OMP_NUM_THREADS=2`,
  ~14 min wall, ~320 MB RSS ≪ 16 GB. Deterministic (`MlpBcPolicy.fit(seed=0)`; subsample
  `np.random.default_rng(seed)`; per-scenario fresh rig).
- **Inputs:** B1 dataset `reports/2026-08-05-r11-5r-robust-teacher/dataset_b1`; wide-train pool from
  `merged.json` (38 WIDE-recertified train scenarios); 12 held-out reconstructed.

## Boundary

Still certified-grasp → delivery → K6. Exact-zero-home composition remains **R11.6C**.
