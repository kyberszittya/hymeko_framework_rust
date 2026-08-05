# R11.5R — Nearest-Robust-Basin Retrieval Delivery Policy (build + characterization)

**Date:** 2026-08-06
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA:** `4e2af83d`
**Verdict:** `R11_5R_RETRIEVAL_DEPLOYABLE_BELOW_050` — a deployable teacher-free policy; best held-out 0.417 < 0.50.

---

## What this is

The R11.5R density curve showed the smooth ridge/mlp map is descriptor-limited while **retrieval** is
the density-responsive, teacher-free (no CEM / no oracle) deployment form. Per the user's decision
("build retrieval policy first"), this formalizes it and characterizes its two design axes on the
existing robust bank (no new demos — densification is the separate, subsequent lever).

`RetrievalDeliveryPolicy` (new, non-core): descriptor → k nearest robust demonstrations (in
standardized descriptor space) → one θ by a `SelectRule` → clip to the certified box. At run time it
uses only a stored `(descriptor, robust θ, survival)` table and a nearest lookup. It is a strict
generalization of `NearestSchedulePolicy` (`RetrievalConfig(standardize=True, k=1, NEAREST)`
reproduces it — pinned by a parity test) and reuses `Standardizer` / `clip_theta`.

---

## Characterization (closed-loop strict-K6, train = leave-one-out, held-out = full-train table)

| cell (metric × rule) | train-LOO | dev | test | **held-out** | dtz (held) |
|---|---|---|---|---|---|
| **std_nearest** (deploy default) | **0.591** | 0.571 | 0.00 | 0.333 | 23.0 mm |
| std_widest3 (k=3) | 0.591 | 0.571 | 0.00 | 0.333 | 23.0 mm |
| **std_weighted3** (k=3, best held) | 0.432 | 0.571 | **0.20** | **0.417** | 22.6 mm |
| raw_nearest | 0.273 | 0.429 | 0.00 | 0.250 | 47.2 mm |
| raw_widest3 | 0.295 | 0.429 | 0.00 | 0.250 | 34.6 mm |
| raw_weighted3 | 0.227 | 0.286 | 0.20 | 0.250 | 31.1 mm |

`certificate = {teacher_free, cem_free, oracle_free} = True`; `is_deployable = True`.

---

## Findings

**1. Standardization of the descriptor metric is load-bearing.**
std_nearest vs raw_nearest: held-out 0.333 vs 0.250 (+0.083) and mean held-out dtz **23.0 vs 47.2 mm**
(halved). Every std cell dominates its raw counterpart on both train-LOO and dtz. The 30-D descriptor
has components spanning very different scales; z-scoring is what makes nearest-neighbour meaningful.

**2. Distance-weighted blending (k=3) is marginally the best held-out cell; widest-basin does nothing.**
- `std_weighted3` held-out 0.417 vs `std_nearest` 0.333 — a **single test scenario** (test 0.00 → 0.20),
  with dtz 22.6 vs 23.0 mm. Blending the 3 nearest robust θ trades **train-LOO precision**
  (0.591 → 0.432 — nearer neighbours available in-distribution favour a single copy) for **held-out
  robustness** (farther queries benefit from averaging). A textbook bias/variance trade.
- `std_widest3` is **identical** to `std_nearest` — choosing the widest-basin θ among the 3 nearest
  changes no held-out outcome. Basin-width-based tie-breaking is inert here (the robust bank already
  made every demo wide; the nearest is already good enough or the neighbourhood is homogeneous).

**3. The retrieval ceiling is 0.417 held-out — below 0.50, and the test split is the wall.**
dev is 0.571 (clears 0.50), but test is 0.00 (nearest) / 0.20 (weighted3). The 5 test scenarios are
the far ones (`c3_r9_a±` angles, `+/+` offsets) whose nearest robust neighbour is beyond the working
radius. No metric or rule reaches them from the current 44-demo table — this is the honest ceiling
and the concrete target for densification.

**Deployment choice (a real trade, not a single winner):**
- **`std_nearest`** — best in-distribution (train-LOO 0.591, dev 0.571), simplest (k=1); use when the
  operating distribution stays near the demo grid.
- **`std_weighted3`** — best held-out generalization (0.417, reaches a test scenario); use when
  robustness to unseen scenarios matters more than in-distribution precision.
Both are `is_deployable = True`, teacher-free, CEM-free, oracle-free.

---

## Interpretation

This confirms and packages the density-curve conclusion: **retrieval is the deployable teacher-free
form**, standardization is essential, and mild neighbour-blending helps generalization at the margin.
The residual is exactly what the curve predicted — the table is too sparse to cover the far test
scenarios. **Densify-for-retrieval** (more robust demos on a finer grid, especially the
`c3_r9_a±`/`+/+` regions) is the well-motivated next lever, and it now has a concrete acceptance
target: lift test-split coverage off 0.00–0.20 and the held-out ceiling past 0.50.

RL stays off this path (a warm-start from the descriptor-limited regressor inherits its ceiling; a
retrieval policy is non-parametric and has no warm-start to hand RL). RL re-enters only as residual
refinement on top of a retrieval deployment, if at all.

---

## Caveats

- **Held-out is 12 scenarios (dev 7, test 5)**; the weighted3 > nearest held-out gain is **one test
  scenario** — real but small. It is corroborated only weakly by dtz (22.6 vs 23.0 mm). The
  **standardization** effect is the robust one (holds on train-LOO n=44 and halves dtz).
- Train-LOO is the honest train number (self-retrieval, trivially 1.0, is excluded via `exclude_idx`).

---

## Files / tests / provenance

| File | Δ |
|---|---|
| `hymeko_rl/coin_delivery/delivery_bc/retrieval.py` | policy + config + `SelectRule` enum + certificate (new) |
| `hymeko_rl/experiments/r11_5r_retrieval_policy.py` | build / eval-fanout / merge / certify (new) |
| `hymeko_rl/tests/test_r11_5r_retrieval.py` | 6 policy tests (parity, rules, LOO, box-clip, cert) |
| `hymeko_rl/tests/test_r11_5r_retrieval_policy.py` | 3 experiment tests (survival join, aggregate, verdict) |
| `reports/2026-08-06-r11-5r-retrieval/retrieval.json` | per-cell coverage + certificate (new) |
| `docs/plans/2026-08-06-r11-5r-retrieval-policy/` | §2 plan, 4-format, tectonic PDF (gitignored) |

- **Tests:** 9 pass; ruff + radon clean. Parity test pins the strict generalization of
  `NearestSchedulePolicy`.
- **CORE.YAML:** none. **Deps:** none. Harness (`reconstruct_capture`, `rollout_theta`,
  `_load_dataset`) reused read-only; no CEM, no controller change.
- **Env:** framework `.venv`, torch 2.12.0, macOS, CPU; 8-way fanout, `OMP_NUM_THREADS=1`, ~11 min
  wall, ~0.35 GB RSS/worker ≪ 16 GB. Deterministic (standardizer fit on train; per-scenario fresh
  rig; stable train order for the LOO index).
- **Inputs:** robust bank `dataset_b1` (44 train + 12 held-out); survival from
  `merged.json` (WIDE → robust t1, else nominal t0).

## Boundary

Certified-grasp → delivery → K6. Exact-zero-home composition remains **R11.6C**.
