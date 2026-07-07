# DAgger as a declared HyMeKo training strategy — build & result (2026-07-06)

**Author:** Aiko (agent) · **Host:** kato15 (RTX 6000 Ada, torch 2.11+cu128, `MUJOCO_GL=egl`) ·
**Plan:** `docs/plans/2026-07-06-dagger-declarative-strategy/plan.md` · **Config:** `data/robotics/pick_place_dagger.hymeko`

## ❄️ FROZEN RESULT (canonical, 2026-07-06) — pick-place DAgger branch closed

**This result is frozen. No further pick-place DAgger variants are to be run unless explicitly requested.**
The numbers below are the canonical, corrected evidence; the earlier n_eval=8 figures are superseded.

| quantity | value (LiftPlaceMetric, place) | source |
|---|---|---|
| expert ceiling | **0.875 place / 0.958 lift** | scripted IK, n=24 |
| BC0 floor | **median 0.542** | n_eval=24, 3 seeds |
| **DAgger best-checkpoint** (deployable) | **median 0.792 / mean 0.833** | n_eval=24, 3 seeds |
| DAgger final (reported separately) | **median 0.792 / mean 0.736** | n_eval=24, 3 seeds |
| TD3+BC | **0.0 (value-drift collapse)** | reports/2026-07-06-fanuc-pick-td3bc.md |

**Correct claim (use this wording):** *DAgger substantially closes the pick-place cloning / covariate-shift gap
and approaches the expert ceiling, but does not cleanly match the ceiling under n_eval=24. Best-checkpoint
selection remains necessary because the per-round curve is non-monotonic.*

**Explicitly on record:**
- **n_eval=8 was optimistic/noisy** — its "best-checkpoint 0.875 = matches ceiling" is a **superseded provisional
  estimate only**; do not use that wording as the result.
- **n_eval=24 is the corrected evidence** (resolution ±0.042) and supersedes it.
- **Warm-started per-round re-BC did NOT fully smooth the curve** — the D1 dip was mixed and new mid/late dips
  appeared; round-to-round variance is inherent.
- **The selected / best checkpoint (`_best.pt`) is the deployable policy**; the **final checkpoint is reported
  separately** (median 0.792 / mean 0.736) and is *not* the deployable artifact.
- **DAgger avoids the TD3+BC value-drift collapse because it stays in the imitation regime** — it queries the
  expert and re-clones (no critic, no Q), so there is no value function to drift.

Artifacts (evidence-complete): `experiments/2026_07_06_19_22_pick_place_dagger_hsikan/` — 21 checkpoints
(bc0·d1–d4·best·final ×3 seeds), gif, `results.json`, `run.log`. Config: `data/robotics/pick_place_dagger.hymeko`.

## Summary

Made the **training strategy a declared field** (`algorithm "bc" | "td3_bc" | "dagger"`) and implemented DAgger
as one strategy — the HyMeKo thesis applied to the training loop. On FANUC pick-place, DAgger **substantially
closes the BC cloning gap** and — unlike TD3+BC — **never collapses**. Under the strengthened metric (n_eval=24,
3 seeds) the honest best-checkpoint estimate is **median 0.792 / mean 0.833** (one seed 0.917) — **approaching,
not cleanly matching**, the 0.875 expert ceiling; a coarse n_eval=8 first run had appeared to hit 0.875 exactly
(partly eval-noise). The right lever past the BC ceiling is imitation (no critic, no value-drift). See
**"Strengthened re-run"** for the corrected numbers.

## The framework (non-core: grammar is generic, verified)

- **Declaration** — `data/robotics/pick_place_dagger.hymeko` declares `algorithm "dagger"` + a `@dagger` knob
  block (`dagger_iters`, `rollouts_per_iter`, `beta`, `beta_decay`, `expert_replay_ratio`, `n_eval`), in the
  same generic `experiment_spec` grammar as `galambos_ab_deliver.hymeko`. `meta_experiment.hymeko` gained one
  additive `@dagger` param type. **No Rust grammar / `CORE.YAML` change** — the `.hymeko` grammar has no
  per-tag rules, so a new field/block is pure data + Python parse.
- **Loader / dispatch** — `hymeko_rl/experiments/training_spec.py` (`TrainingSpec.from_hymeko`, generic over the
  `_profile` shim) → `hymeko_rl/experiments/pick_place_dagger.py` dispatches on `algorithm`: `dagger` → the new
  loop; `bc`/`td3_bc` delegate to the existing runners **unchanged** (the layer selects, it does not
  re-implement).
- **Loop** — `hymeko_rl/train/dagger.py` (`Dagger`, composing — not forking — the `Campaign` scaffolding:
  reuses `behaviour_clone`, `experiment_dir`, `tee_stdout`, `render_actor_gif`). Lock-step expert labelling:
  the FANUC expert has a stateful `_lift_xy` latch, so the expert is queried **every** learner step (not
  sparsely) — a `label_sanity` gate verifies this before any run.

## Staged execution (per directive)

| stage | result |
|---|---|
| import/compile | PASS |
| config parse (`test_training_spec`, 3) | PASS (galambos unaffected — regression clean) |
| env reset/step | PASS |
| **label-sanity** | PASS — `max_over_box_ranges 0.4` (raw IK overshoot the env clips; a strict box check would false-fail), `committed_steps 456` (latch genuinely exercised), coherent/finite/deterministic |
| tiny DAgger plumbing (`test_dagger`, 2) | PASS — evidence-complete artifacts (BC0/D1/best/final) |
| 1-seed production (seed 1) | **place 0.75 → 0.875** (see below) |
| multi-seed (seeds 0, 2) | PASS — best-checkpoint median **0.875** (see 3-seed verdict) |

## 1-seed production result (seed 1) — `experiments/2026_07_06_19_03_pick_place_dagger_hsikan/`

| round | agg size | β | lift | place |
|---|---|---|---|---|
| BC0 | 8404 | — | 0.75 | 0.75 |
| D1 | 14 569 | 0.500 | 0.625 | 0.375 |
| D2 | 21 314 | 0.250 | 0.625 | 0.625 |
| D3 | 27 670 | 0.125 | 0.75 | 0.75 |
| **D4** | 34 072 | 0.062 | **1.0** | **0.875** |

- **selected (best-checkpoint):** `policies/pick_place_dagger_hsikan_s1_best.pt` — place **0.875** (= D4).
- **final:** `policies/pick_place_dagger_hsikan_s1_final.pt` — place **0.875** (best = final; no late collapse).
- per-round checkpoints saved (`_bc0/_d1/_d2/_d3/_d4`), gif of the best policy, `run.log`, `results.json`
  (evidence-complete, per `feedback-evidence-complete-learned-policy`).

## Comparison (seed 1, identical env + metric)

| method | place | note |
|---|---|---|
| expert (scripted IK) | 0.875 | the ceiling |
| BC (clone) | 0.625–0.75 | covariate-shift gap |
| **TD3+BC** | **→ 0.0** | value-drift collapse (reports/2026-07-06-fanuc-pick-td3bc.md) |
| **DAgger** (best-ckpt) | **~0.79–0.83** | closes the gap, no collapse (n_eval=24 median 0.792 / mean 0.833; the 0.875 at n_eval=8 was partly eval-noise) |

## 3-seed verdict (n_eval=8 — superseded by the strengthened run below) — `experiments/2026_07_06_19_07_pick_place_dagger_hsikan/` (seeds 0, 2) + seed 1

| seed | BC0 floor | DAgger best (best-checkpoint) | DAgger final (D4) | best round |
|---|---|---|---|---|
| 0 | 0.625 | **0.75** | 0.625 | D3 |
| 1 | 0.75 | **0.875** | 0.875 | D4 |
| 2 | 0.50 | **0.875** | 0.75 | D2 |
| **median** | **0.625** | **0.875** | **0.75** | — |

**Measured:** DAgger's **best-checkpoint median is 0.875 = the expert ceiling** — a clear multi-seed win over
BC (0.625) and over TD3+BC (0.0 collapse). All three seeds' best-checkpoint reached ≥ 0.75; two hit 0.875. The
deployable artifact is the best-checkpoint (`_best.pt`), so what you ship is at the ceiling.

**Honest caveats (measured vs inferred):**
- The DAgger curve is **noisy round-to-round**, not monotone: **D1 dips hard on every seed** (place 0.375 /
  0.0 / 0.125) then recovers by D2–D4. The `final` (D4) is therefore below the peak on 2/3 seeds (final median
  0.75 < best median 0.875). Best-checkpoint is what protects the result; the last round is not the artifact.
- `n_eval = 8` gives **coarse resolution (±0.125)**, so the per-round/per-seed scalars are noisy point
  estimates; the robust, qualitative claim is "DAgger reaches the expert ceiling at its best and never
  permanently collapses," not a tight 0.875 ± ε.
- The **D1 collapse-then-recover** (fresh re-BC each round + first learner-labelled batch under β=0.5) is a
  candidate to smooth later (warm-start the re-BC, or larger `n_eval`) — deferred (this branch is narrow).

## Strengthened re-run — n_eval=24, warm-start re-BC, 3 seeds (REVISES the estimate)

`experiments/2026_07_06_19_22_pick_place_dagger_hsikan/`. Same declared config, with `warm_start 1.0` and
`n_eval 24` (resolution ±0.042). Per-seed place:

| seed | BC0 | D1 | D2 | D3 | D4 (final) | best-checkpoint |
|---|---|---|---|---|---|---|
| 0 | 0.542 | 0.500 | 0.792 | 0.708 | 0.500 | **0.792** (D2) |
| 1 | 0.667 | 0.833 | 0.417 | 0.417 | 0.917 | **0.917** (D4) |
| 2 | 0.417 | 0.167 | 0.625 | 0.708 | 0.792 | **0.792** (D4) |
| **median** | 0.542 | | | | 0.792 | **0.792** |
| **mean** | 0.542 | | | | 0.736 | **0.833** |

**Honest revision (measured).** The tighter n_eval=24 gives a more reliable — and slightly LOWER — estimate than
the n_eval=8 run: **best-checkpoint median 0.792 (mean 0.833)**, not 0.875. The earlier "0.875 = ceiling" was
partly eval-noise at n_eval=8 (the BC0 floor also reads lower here, 0.542 vs 0.625 median). Corrected claim:
**DAgger substantially closes the gap (BC0 ~0.54 → best-checkpoint ~0.79 median / 0.83 mean, one seed 0.917),
approaching but NOT cleanly matching the 0.875 expert ceiling.** Still a decisive contrast with TD3+BC (0.0).

**warm-reBC did NOT reliably smooth the curve** (round headers confirm `warm-reBC` was engaged). The D1 dip was
mixed (s1 *rose*, s0 mild, s2 still crashed to 0.167), and new mid/late dips appeared (s1 D2/D3 → 0.417, s0 D4 →
0.500). The round-to-round variance is inherent (each round's fresh learner rollouts shift the aggregate), so
best-checkpoint selection stays mandatory. Warm vs fresh is confounded with the n_eval change here, so there is
**no clean "warm helps" verdict** — the noise persists either way.

**Corrected verdict:** DAgger reaches **~0.79–0.83 place (best-checkpoint)** on FANUC pick — a real, multi-seed
gain over BC and a decisive win over the TD3+BC collapse — with best-checkpoint selection mandatory and the
expert 0.875 an upper bound not yet cleanly matched. Evidence-complete artifacts saved (BC0/D1–D4/best/final +
gif + run.log + results.json) in the run dir above.

## Exact metric

`LiftPlaceMetric` (`hymeko_rl/eval/evaluate.py`): per episode, `lift` = object rose ≥ `lift_thresh=0.035` m and
`place` = object within `place_radius` of the target at ≥ lift height, with a divergence guard
(`diverge_qacc=5000`, NaN/|qacc| → the episode is a FAILURE, never a counted success). Rates over `n_eval=8`
episodes (seed0 20000). Expert ceiling / BC floor measured 2026-07-06 on the same metric.

## Open items

- Smooth the noisy DAgger curve (D1 dip; final < peak): warm-start the per-round re-BC and/or raise `n_eval`
  from 8 to tighten the estimate — deferred (narrow branch).
- DAgger cannot exceed the expert 0.875; to go higher needs a better expert (out of scope for this branch).
- The declared-strategy dispatch now covers `bc`/`td3_bc`/`dagger`; a follow-up could route the galambos
  experiment through `TrainingSpec` too (one loader instead of per-family `from_hymeko`).

CORE.YAML touched: none. New deps: none. New files: `pick_place_dagger.hymeko`, `training_spec.py`,
`dagger.py`, `pick_place_dagger.py`, `test_training_spec.py`, `test_dagger.py` (+ additive `@dagger` in
`meta_experiment.hymeko`). No TD3/SAC/residual run in this branch (per directive).
