# R11.5R — Robust Delivery Teacher Re-certification

**Date:** 2026-08-05
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA (pre-merge-commit):** `3678b094`
**Status:** `R11_5R_ROBUST_TEACHER_RECERTIFICATION_PASS`

---

## Summary

R11.6B concluded that TD3, warm-started from the narrow-basin teacher demonstrations, did **not**
find a wider-basin delivery policy in that action space and budget. That is *not* a proof that
no reachable wide-basin policy exists — the correct falsification is to search for wide-basin
delivery **θ directly**, with a teacher, rather than hoping RL rediscovers them from a narrow
warm-start. R11.5R does exactly that: it moves robustness into the **demonstration generator**
(the teacher's scoring), re-certifies the 56-scenario delivery bank, and asks whether wider-basin
K6 deliveries exist for the *same* scenarios.

**They do.** The robust teacher re-certifies **49/56** scenarios as `WIDE_RECERTIFIED`
(train_wide_frac 0.864), lifts mean local-K6 survival under 1 % θ-perturbation from **0.708 → 0.958**
(+0.25), and does so on held-out **dev (+0.343) and test (+0.400)** splits by *more* than on train.
This is the clean falsification R11.6B could not provide: **the narrow-basin bottleneck was a
property of the demonstrations-as-produced-by-nominal-CEM, not of the delivery problem.** Wide-basin
θ are reachable — a direct teacher search finds them where warm-started TD3 did not.

The only change versus the R11.4B teacher is the **scoring** (lexicographic robust key instead of
nominal K6). Reconstruction, action-coordinate (θ ∈ ℝ⁶), physics, strict K6, safety bounds, the
scenario split, and the CEM structure are all frozen.

---

## Method

### Robust teacher (only scoring changes)

Per scenario, T0 = the nominal teacher θ (from the R11.4B bank); T1 = a CEM **warm-started at the
nominal θ** and ranked by a lexicographic robust key:

```
key = (safe, nominal_K6, survival@0.5%, survival@1%, −CVaR(dtz), survival@2%, −release_step)
```

- A candidate is **survival-eligible only if it is nominal-K6 and safe** — a lucky perturbation can
  never rescue a failed nominal.
- Survival is measured over a **shared, reproducible perturbation seed-bank** (seed 12345), so T0 and
  T1 see identical δ-directions at each scale and are directly comparable.
- A **progressive funnel** (screen 0.5 % → refine 1 % → stress 2 %) keeps it economical: only screen
  survivors pay for the 1 % rollouts, only 1 %-survivors pay for 2 %.
- `WIDE_RECERTIFIED` requires survival@1 % ≥ 0.75; else `NARROW_ONLY`; `NO_NOMINAL_K6` if the nominal
  itself misses K6 under the closed-loop eval config.

### Warm-start fix (the decisive correction)

The **first** re-cert run seeded the CEM mean at the **box centre**. On the hard/chaotic scenarios a
single box-centre CEM run under-found K6 versus the nominal teacher's R = 11 restarts, producing **21
`NO_NOMINAL_K6`** — a *budget artifact*, since every scenario provably has a K6 nominal θ (it is in
the bank). That depressed `train_wide_frac` to 0.595 → `WIDE_BASIN_SUPPORT_LIMITED`.

Fix: `robust_cem(init_theta=nominal_θ)` warm-starts the CEM mean at the nominal θ, so K6 is
guaranteed in the neighbourhood and the CEM **refines it toward wider basins**. This is also the
*literal* hypothesis ("find a wider θ for the **same** scenario") and the fairer A/B. The box-centre
result is archived as `merged-boxcenter.json`.

---

## Results

### Teacher gate (all 56 scenarios, warm-start)

| Metric | Box-centre (1st run) | **Warm-start** | Gate |
|---|---|---|---|
| `train_wide_frac` | 0.595 | **0.864** | ≥ 0.70 ✓ |
| mean survival gain T1−T0 | +0.164 | **+0.250** | ≥ 0.20 ✓ |
| nominal coverage | — | **63/64** | ≥ 45 ✓ |
| `WIDE_RECERTIFIED` | 30 | **49** | — |
| `NARROW_ONLY` | 3 | 3 | — |
| `NO_NOMINAL_K6` | 21 | **4** | — |
| mean survival@1 % T0 → T1 | — | **0.708 → 0.958** | — |
| **verdict** | `WIDE_BASIN_SUPPORT_LIMITED` | **`RECERTIFICATION_PASS`** | — |

Warm-start recovered **17 of the 21** box-centre `NO_NOMINAL_K6` artifacts.

### Survival gain by split (nominal-K6 scenarios only)

| Split | n (K6) | mean survival gain T1 − T0 |
|---|---|---|
| train | 40 | +0.215 |
| dev | 7 | **+0.343** |
| test | 5 | **+0.400** |

The gain is **larger on the held-out dev/test splits** than on train — the robust θ found by the
teacher are not train-overfit; wider basins exist across the scenario space.

### Narrow → wide recoveries (survival < 0.75 → ≥ 0.75): 18

Including two **held-out test** scenarios:

| Scenario | Split | survival@1 %: T0 → T1 |
|---|---|---|
| bank_c3_r7_a+15 | train | 0.00 → 1.00 |
| bank_c2_+0.015_-0.025 | train | 0.00 → 1.00 |
| bank_c1_+0.03_+0.00 | **test** | 0.20 → 1.00 |
| bank_c2_+0.025_+0.000 | **test** | 0.60 → 1.00 |
| bank_c3_r7_a+45 | dev | 0.40 → 1.00 |
| bank_c2_+0.025_-0.025 | dev | 0.40 → 1.00 |
| … (12 more train/dev) | | ≥ +0.20 each |

A θ perturbation of 1e-5 flipping delivery 7.99 → 54.27 mm (the R11.4B chaotic-basin measurement)
is exactly the fragility these robust θ remove: at 1 % perturbation, survival goes from ~0 to 1.0.

### Compute

Mean **2 690** sim-calls / scenario (T1, no early-exit — keeps searching for higher survival after
K6), **150 651** rollouts total across the 56-scenario re-certification. Parallelised 9×6 + a 2-wide
tail; wall ≈ 30 min on the Mac (CPU). Peak RSS per worker well under the 16 GB cap (single-scenario
CEM, no torch training).

---

## B1 dataset (for the same-size BC re-run)

`--merge` emits `dataset_b1/extract_000.jsonl` in the R11.4B `BcSample` shard format — the **same 56
scenarios, same 44/7/5 split**, the robust θ where `WIDE_RECERTIFIED`, the nominal fallback (== the
B0 θ) elsewhere. **49/56 θ differ from B0**; 7 are identical (3 NARROW + 4 NO_K6 fallbacks). It loads
cleanly through the unchanged BC harness `_load_dataset`, so B0 vs B1 is a single-variable A/B:
**only the target θ changes.**

---

## Interpretation

- **R11.6B is now correctly bounded.** "RL did not find a wider policy" ≠ "no wider policy exists."
  A direct teacher search finds wide-basin θ for 49/56 scenarios, including held-out test. The
  R11.6B negative was a limitation of TD3 from a narrow warm-start in that action space/budget.
- **The bottleneck was the demonstrations, not the delivery problem.** R11.4B/6A/6B were all capped
  by narrow-basin demonstrations produced by the nominal CEM. Re-scoring the teacher for local
  survival produces demonstrably wider demonstrations for the same scenarios.
- **Not yet proven:** that a conditioned BC policy *generalises better* from wider-basin targets
  (that is B1), nor that RL from a wider warm-start improves delivery (gated on B1). Wide θ existing
  is necessary, not sufficient.

---

## Files touched (this change)

| File | Δ |
|---|---|
| `hymeko_rl/experiments/r11_5r_robust_teacher.py` | +`_merge_rows`, `_write_b1_dataset`, `_run_merge`, `--merge` flag |
| `hymeko_rl/tests/test_r11_5r_recert.py` | +`test_merge_and_b1_dataset_roundtrip` |
| `hymeko_rl/coin_delivery/delivery_teacher/robust_teacher.py` | `init_theta` warm-start (prior commit `3678b094`) |
| `reports/2026-08-05-r11-5r-robust-teacher/merged.json` | 56-row re-cert + gate (new) |
| `reports/2026-08-05-r11-5r-robust-teacher/dataset_b1/extract_000.jsonl` | B1 dataset (new) |
| `reports/2026-08-05-r11-5r-robust-teacher/merged-boxcenter.json` | archived 1st run (new) |

**CORE.YAML items touched:** none.

## Test results

- `pytest hymeko_rl/tests/test_r11_5r_recert.py test_r11_5r_robust_teacher.py` → **11 passed**, 0.55 s.
- `ruff check` on both changed files → clean.
- The merge (`_merge_rows`/`_write_b1_dataset`) is exercised by `test_merge_and_b1_dataset_roundtrip`
  (WIDE/NARROW/NO_K6/NO_CAPTURE row handling; NO_CAPTURE excluded from B1).

## Provenance

- **Git SHA:** `3678b094` (working tree dirty: the merge code + report + `merged.json`/`dataset_b1`,
  committed in the follow-up).
- **Python:** framework `.venv`, torch 2.12.0, macOS (darwin 25.5.0), CPU.
- **Perturbation seed-bank:** 12345 (shared T0/T1). **CEM seed:** 0. Scales (0.005, 0.01, 0.02).
- **Config:** `RobustTeacherConfig(k_screen=3, k_refine=5, k_stress=6, screen_min=0.34,
  cvar_alpha=0.5, robust_min_survival=0.75)`.
- **Dataset in:** `reports/2026-08-03-r11-4b-bc/dataset` (56 non-omitted BcSamples).

## Open issues / follow-up

- **4 `NO_NOMINAL_K6` + 3 `NARROW_ONLY`** remain genuinely narrow/chaotic (e.g.
  `bank_c2_-0.015_-0.025`: T0 = T1 = 0.0). They fall back to the nominal θ in B1 (identical to B0),
  so the A/B is unaffected; they mark the scenarios where even direct search cannot widen the basin
  in this action space — candidates for the R11.6C exact-zero / controller-expressivity axis.
- **Next (per the R11.5R plan):** BC **B0 (nominal R11.4B dataset) vs B1 (robust dataset)** through
  the *same* `r11_4b_conditioned_bc` harness and size — does closed-loop ridge/MLP generalisation
  improve from wider-basin targets? Only if B1 improves do we RL from the wider warm-start.
- **Decision tree:** teacher finds wide (✓) but BC still fails → descriptor / coverage problem;
  teacher finds wide and BC improves → the demonstrations were the bottleneck (thesis confirmed).

## Boundary

R11.5R is **certified-grasp → delivery → K6 with wider basins**. Exact-zero-home composition remains
**R11.6C** and is untouched here.
