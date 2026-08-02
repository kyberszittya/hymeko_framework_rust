# R11.4B — Conditioned Delivery BC: `BC_REPRESENTATION_INSUFFICIENT` (chaotic teacher θ)

**Date:** 2026-08-03 (Mac)
**Verdict:** **`BC_REPRESENTATION_INSUFFICIENT`** — the certified scenario-specific teacher θ-schedules do **not** assimilate
into one smooth coin/target/handoff-conditioned policy. Two upstream causes (measured, below): **(1)** the descriptor→θ map
does not fit or generalize from 44 demonstrations — smooth regressors fail even in-distribution and 1-NN retrieval reaches
only 25 % held-out; **(2)** ~1/3 of the certified deliveries are narrow-basin (chaotically sensitive), unlearnable by any
approximate policy. It is **not** an MLP tuning bug (MLP ties ridge) and **not** fixable by a bigger network (44 demos).
**No RL** — an RL branch would only mask a failed assimilation whose cause is demonstration density + target robustness.

## Load-bearing question and answer
*Can the 49 certified teacher θ (+7 frozen-R2 anchors) be assimilated into one policy π: descriptor → θ ∈ ℝ⁶ that
delivers strict K6 with no per-scenario CEM and no oracle?* **Answer: not as a smooth θ-regression.** Exact-θ retrieval
(1-NN) reproduces K6 on training scenarios (it copies its own θ) but generalizes to only 25 % held-out; every *smooth*
regressor (mean, ridge, MLP) fails to reproduce K6 even **in-distribution** (≤ 39 % on train), because a small deviation
from the exact teacher θ misses the delivery basin.

## The closed-loop competition (physical strict-K6, no CEM / oracle / lookup)
44 train / 7 dev / 5 test scenarios (whole coin/target scenario in one split; 2 parked cases excluded).

| policy | train | dev | test | held-out |
|---|---|---|---|---|
| mean-θ | 0.205 | 0.286 | 0.00 | 0.167 |
| **nearest_schedule** (1-NN, exact θ copy) | **1.00** | 0.429 | 0.00 | **0.25** |
| ridge | 0.386 | 0.143 | 0.20 | 0.167 |
| mlp_bc (small MLP) | 0.386 | 0.143 | 0.20 | 0.167 |
| frozen R2 (incumbent) | 0.136 | 0.00 | 0.00 | 0.00 |

- `teacher_theta_reproduces_k6 = true`, `bc_safe = true` (0 safety regression on BC deliveries).
- **MLP exactly ties ridge on train (0.386)** ⇒ not an MLP fit/tuning bug; the smooth target itself is not reproducible.
- **1-NN train = 1.0 is trivial memorization** (nn distance 0 → its own certified θ). It is excluded from the
  optimization-failure comparison; the fair parametric comparator is ridge.
- **Honest baseline read (as requested):** the *best* held-out method is 1-NN retrieval at 0.25 — not the neural policy.
  There is no neural win to claim here.

## Mechanism — two measured causes (basin diagnostic REFINED the first hypothesis)
A pilot measurement on `bank_c1_-0.03_+0.00` first suggested pure chaos: the certified θ delivers **7.99 mm** K6, but a
**4-dp rounding** perturbation (~1e-5, integer step indices unchanged) sends the coin to **54.27 mm**. The basin-robustness
diagnostic (perturb each certified θ by Gaussian noise at relative box scales, K6-survival fraction, `k=6`/scale, **all 56
scenarios**, `base_k6_all=true`) **partly refuted** that — the population is *mixed*, not uniformly chaotic:

| perturbation scale (of box range) | 0.5 % | 1 % | 2 % | 5 % |
|---|---|---|---|---|
| mean K6 survival | 0.783 | 0.637 | 0.524 | 0.527 |

- **38 / 56 wide-basin** (≥ 0.5 survival at 1 %); many fully robust (survival 1.0 at every scale). **18 / 56 (32 %)
  narrow-basin** — e.g. `bank_c1_-0.03_+0.00` and `bank_c2_-0.025_-0.015` at 0.0 across all scales.

**Cause 1 (dominant) — the descriptor→θ map does not fit/generalize from 44 demos.** If chaos were the whole story, the
~68 % wide-basin scenarios should be reachable; they are not. Smooth regressors score only **0.386 train** (well below the
~68 % wide-basin ceiling) and **0.167 held-out**; even for wide-basin held-out scenarios the predicted θ lands outside the
basin. 1-NN retrieval reaches only 0.25 held-out — the certified θ vary too much between scenarios to interpolate or
retrieve from 44 examples. This is an under-determined map, not a tuning bug.

**Cause 2 — ~1/3 of certified deliveries are narrow-basin**, which caps even a perfect regressor and blocks any approximate
policy on those scenarios regardless of demonstration density.

## Interpretation (does NOT trigger RL)
Per the pre-registered rule, the negative is classified, not papered over:
- Not `BC_OPTIMIZATION_FAILURE`: the MLP matches closed-form ridge on train.
- Not purely `BC_DATA_COVERAGE_INSUFFICIENT` in the nn-distance sense: smooth regressors also fail **in-distribution**
  (train nn-distance ≈ 0). But the *root* is demonstration **density/complexity** — 44 (input, θ) pairs under-determine a
  30→6 map whose targets vary sharply and are 32 % non-robust.
- `BC_REPRESENTATION_INSUFFICIENT`: no smooth descriptor→θ map reproduces the certified deliveries at this demonstration
  density. (The user's 3-way taxonomy has no "targets-non-robust" bucket, which is the second, upstream cause.)

**Two forward levers, neither is RL nor a bigger net:**
1. **Densify the demonstrations** — extract *multiple* K6 θ per scenario (the certification kept only the ranked winner),
   and add more coin/target grid points, so the descriptor→θ map is over- rather than under-determined.
2. **Robustness-aware delivery teacher** — a CEM objective that rewards a θ-*neighborhood* of K6 (basin width / tolerance),
   not a single point; re-certify deliveries whose K6 survives perturbation (the 18 narrow ones especially), then re-run
   this exact BC competition against the robust+dense bank. A retrieval + local-refinement deployment policy is the fallback.

## R11.4B2 (handoff → deliverability surrogate) — remains GATED
Not started: it is gated behind a BC closed-loop PASS, and BC did not pass. A surrogate would only pick a better grasp;
there is no working policy to use that grasp yet.

## Files touched (all non-core)
- `hymeko_rl/coin_delivery/delivery_bc/` — `dataset.py`, `models.py`, `evaluate.py`, `__init__.py` (new package).
- `hymeko_rl/experiments/r11_4b_conditioned_bc.py`, `hymeko_rl/experiments/r11_4b_basin_robustness.py` (new).
- `hymeko_rl/tests/test_r11_4b_{dataset,models,gate,evaluate}.py` (new; 19 tests).
- `reports/2026-08-03-r11-4b-bc/` — dataset (56 verified θ + descriptors), eval (per-scenario K6), basin, gate.
- **No source outside this package changed**; transport primitive, solver, ranking, demonstration bank read-only.

## CORE.YAML items touched
**None.** Only new Python under `hymeko_rl/`; no dependency added (numpy + existing torch; ridge/kNN are numpy closed-form).

## Test results
19 tests pass (`test_r11_4b_{dataset,models,gate,evaluate}.py`): descriptor assembly + geometry, model fit/predict + box
clip, gate PASS + all three negative classifications, basin-survival monotonicity. ruff / mypy --strict / radon (no C+)
clean.

## Data fidelity (two bugs the smoke caught before the full run)
1. **Shared-rig reconstruction** carried MuJoCo state across scenarios (order-dependent handoff) → **fresh rig per
   scenario**; reconstruction now depends only on (scenario, seed), verified deterministic.
2. **`DeliveryResult.theta` is 4-dp-rounded**, which breaks chaotic deliveries → store **full-precision θ verified through
   the exact `rollout_theta` eval harness**; every label is guaranteed to deliver K6 at eval (`teacher_theta_reproduces_k6
   = true`). Dataset: 56/56 kept, 0 omitted (7 anchors converted to the θ-schema), dtz 1.26–19.98 mm.

## Performance / provenance
- Extract 56 (8-way, R=11) + eval + basin: wall ~1 h; per-process RSS ≪ 16 GB (same footprint as the certification).
- Mac, 48 GB, Apple Silicon; `OMP_NUM_THREADS=2`. Deterministic (fixed seeds; fresh-rig reconstruction reproducible).
- Bank `reports/2026-07-30-r11-4a-bank/bank.jsonl` md5 `473244de4795254f5de99f4ca7732714`; demonstration source
  `reports/2026-08-02-r11-5ppp-cert-mac/merged.json`. §2 plan `docs/plans/2026-08-03-r11-4b-conditioned-delivery-bc/`
  (4-format). Energy diagnostic-only. No deps.

## Next
1. **Densify demonstrations (dominant lever)** — extract multiple K6 θ per scenario + more coin/target grid points so the
   descriptor→θ map is over-determined; re-run this exact BC competition (the harness is reusable as-is).
2. **Robustness-aware delivery teacher** — CEM objective rewarding basin width / θ-tolerance; re-certify the 18 narrow-basin
   deliveries whose K6 does not survive a θ-neighborhood. The per-scenario basin summary already labels robust vs knife-edge.
3. R11.4B2 (handoff→deliverability surrogate) stays **gated** until BC PASS — it cannot help while no policy can use the
   grasp it would select.
4. Not RL: the gap is demonstration density + target robustness, which RL would mask, not resolve.
