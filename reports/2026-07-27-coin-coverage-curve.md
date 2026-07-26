# Coverage-only causal curve — does dev-cradle coverage ALONE close the held-out update-0 regression?

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-teacher-to-rl` · **Base:** `a3459629` (tag
`coin-physical-feasibility-closed`) · physics contract `4c71f12f` (per-side collision masks + `CONTROLLED_INSERTION`).

## Question (one independent variable)

After the frozen update-0 gate regressed the held-out cradles at N=2 (`UPDATE_ZERO_REGRESSES_HELD_OUT`, blocker
`INSUFFICIENT_DATASET_COVERAGE_2_DEV_CRADLES`, oracle 4/4), the single question:

> **Is growing the development-cradle set alone enough to remove the held-out regression?**

This is a coverage-only causal experiment. **Only N changes.** No architecture change, no multimodal proposal, no new
loss, no hyperparameter tuning.

## Design — what is held identical across N (only N changes)

| Quantity | Value (frozen across N=2,4,6) |
|---|---|
| Model | `B0` (features-only flat-obs `DetActor` — the frozen update-0 variant) |
| Feature set | structured 42-D causal state + θ-independent causal history (frozen) |
| Optimiser / epochs / lr / init seed | Adam / 1200 / 1e-3 / **seed 0 (fresh matched init each N)** |
| Search budget / semantics | **8** / centre-inclusive fixed bounded search (frozen) |
| Evaluation panel | **frozen 4-state {s1, s3, s4, s7}** — same evaluation seeds at every N |
| Held-out | **always {s4, s7}** — never trained on, at any N |
| K6 / motion / collision / task contract | frozen monitor + `4c71f12f` per-side masks + `CONTROLLED_INSERTION` |

Every N is trained **from a fresh matched initialisation** (no N=2→N=4 continuation), so the curve is a genuine coverage
curve, not a fine-tuning trajectory.

### Nested development sets

- **N = 2** → s1 (14250), s3 (14750)
- **N = 4** → s1, s3 + **two additional cradles selected and FROZEN before any training** by a non-outcome rule
- **N = 6** → all six usable K6-deliverable development cradles {14250, 14750, 16500, 17750, 19500, 24000}

Set-nested: N=2 ⊂ N=4 ⊂ N=6.

### The N=4 selection rule (non-outcome, frozen before results)

Greedy **farthest-point sampling in the frozen 42-D geometry-fingerprint space** — it reads *only* the cradle geometry
(never delivery / held-out K6), so it maximises geometric coverage gain and cannot be tuned to flatter the result.
Deterministic; ties break to the lower seed.

- Candidate min-distance to the frozen dev set {s1,s3}: `{16500: 1.84, 17750: 2.58, 19500: 1.18, 24000: 1.15}`
- **SELECTED (recorded before any deploy is read):** `[17750, 16500]` — the two geometrically-farthest new cradles.

## Method (reuses the frozen machinery — no duplication)

- The 4 new dev cradles are reproduced with the **same** frozen CEM + basin-augmentation machinery as the teacher bank
  (`reproduce_state`, extended with `tag`/`split`/`basin_seed` overrides; defaults reproduce the frozen-panel behaviour
  bit-identically). Canonical θ verified against the delivery-pass inventory.
- Per N: build the structured dataset from the N-dev coverage bank → fit B0 fresh → deploy on the **fixed** 4-state panel
  via the unchanged `update_zero_eval` (informed BC θ₀ vs uninformed box-centre vs oracle teacher-θ, budgets {0,4,8}).
- Harness: `hymeko_rl.experiments.coin_theta_rl_benchmark --coverage-curve`; pure logic in
  `hymeko_rl/coin_delivery/theta_option/coverage_curve.py`.

## Results

Artifact `reports/2026-07-27-coin-teacher-to-rl/coverage_curve.json`; figure `coverage_curve.png`; deploy GIF
`coverage_N6_deploy_s4.gif`. Wall **647.3 s** (~10.8 min), peak RSS **0.319 GB** (« the 16 GB cap). All 4 new dev cradles
reproduced K6-deliverable with canonical θ matching the frozen delivery-pass inventory (`matches_delivery_pass=True`).

| N | train cradles | dev K6 | held s4 | held s7 | total K6/4 | proposal-only (b0) | search-8 | actor→teacher θ (held, norm) | held dtz_end (mm) | failure mode |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **2** | s1, s3 | 2/2 | 0 | 0 | **2/4** | 1/4 (held 0/2) | 2/4 (held 0/2) | **0.649** | s4 33.7, s7 50.9 | s4 REACHED_BUT_NO_SETTLE, s7 NEVER_REACHED_ZONE |
| **4** | s1, s3, 17750, 16500 | 2/2 | 0 | 0 | **2/4** | 1/4 (held 0/2) | 2/4 (held 0/2) | **1.534** | s4 77.6, s7 114.9 | both NEVER_REACHED_ZONE |
| **6** | s1, s3, 16500, 17750, 19500, 24000 | 2/2 | 0 | 0 | **2/4** | 1/4 (held 0/2) | 2/4 (held 0/2) | **1.521** | s4 76.6, s7 111.1 | both NEVER_REACHED_ZONE |

- **Oracle (teacher θ + the same fixed budget-8 search) = 4/4 at every N** — a delivering θ exists for s4, s7 and the
  search finds it *from the teacher θ*; the search and physics are not the blocker.
- Dev θ error is flat at ≈0.23 across N (s1, s3 are always trained and B0 fits them: s1 0.14, s3 0.32). Only the held-out
  extrapolation moves.

**Held-out K6 by N: {2: 0, 4: 0, 6: 0}.** Adding development cradles does **not** deliver a single held-out cradle at
any N.

## Verdict

**COVERAGE_ALONE_INSUFFICIENT** — `authorise_sac_td3 = False`. Growing the development set 2 → 4 → 6 does **not** remove
the held-out regression; held-out delivery stays flatly 0/2. Per the decision tree: **stop before changing the model** →
next is an acceptable-set / multimodal proposal → update-0 retry → only then RL. **SAC/TD3 is not run.**

### Mechanistic diagnosis (why coverage fails here — observed on this matched-init curve)

The failure is **not** "too few points to interpolate." It is **out-of-distribution extrapolation of a single-θ
regressor**:

- The held-out actor→teacher θ distance does **not** shrink with N — it *grows* (0.65 → 1.53), and the held-out coin
  ends **progressively further** from the target zone (s4 34 → 78 mm, s7 51 → 115 mm). s4 degrades from
  `REACHED_BUT_NO_SETTLE` (N=2) to `NEVER_REACHED_ZONE` (N=4,6).
- The two N=4 additions were the geometrically-farthest cradles (FPS min-dist 2.58, 1.84). Anchoring a features-only B0
  regressor on more *spread* dev geometry pulls its held-out prediction into a worse extrapolation regime — s4, s7 are
  outside the manifold the dev cradles span, so more dev anchors do not bracket them, they lever the fit away.
- The delivering θ exists (oracle 4/4) but the actor's θ₀ sits outside the budget-8 search basin, *increasingly* so with
  N. A single regressed θ₀ from features does not generalise to held-out geometry, and coverage does not repair it.

This is the opposite of the `HELD_OUT_GENERALISATION_IMPROVES_WITH_CRADLE_COVERAGE` sub-finding — it is a clean negative
that both rules out coverage-alone **and** diagnoses the structural cause, directly motivating the pre-specified next step
(a proposal that emits a *set/mixture* rather than one interpolated θ, so the fixed search can reach a delivering mode).

**Caveat (measured vs inferred):** the primary verdict — held-out 0/2 at every N — is the robust answer to the question.
The θ-distance-grows-with-N trend is *observed* on this single matched-init curve (init seed 0, the design that isolates
N); it is consistent with OOD extrapolation but is not asserted as a multi-seed monotone law.

## Tests

- `hymeko_rl/tests/test_coin_coverage_curve.py` — 8 fast pure-logic tests (FPS non-outcome + deterministic + tie-break;
  nested sets; θ-distance; per-N record extraction; the 3 verdict branches) + 1 slow physics test (`reproduce_state`
  dev-override yields a K6-deliverable dev entry with basin for a new seed). All pass; ruff clean.

## Files touched

- `hymeko_rl/coin_delivery/theta_option/coverage_curve.py` (new, pure coverage logic).
- `hymeko_rl/coin_delivery/theta_option/teacher_bank.py` (`reproduce_state`: +`tag`/`split`/`basin_seed` overrides;
  defaults unchanged).
- `hymeko_rl/experiments/coin_theta_rl_benchmark.py` (`--coverage-curve` mode + viz).
- `hymeko_rl/tests/test_coin_coverage_curve.py` (new).

**CORE.YAML items touched:** none. **Performance:** peak RSS 0.319 GB (« 16 GB cap); wall 647.3 s. Single-threaded
(`torch.set_num_threads(1)`), as the other benchmark modes.

## Follow-up

- **Next (pre-specified):** acceptable-set / multimodal proposal → update-0 retry → only then RL. The mechanistic
  diagnosis says the proposal head must emit a *set/mixture* of θ (or a held-out-aware acceptable region) so the fixed
  budget-8 search can reach a delivering mode the single regressed θ₀ misses.
- **Not run:** SAC/TD3 (gate not reached — held-out 0/2 at every N).
- **Optional robustness:** repeat the curve at 2–3 additional init seeds to confirm the θ-distance-grows trend as more
  than an init artifact (the primary 0/2-at-every-N verdict already does not depend on it).
