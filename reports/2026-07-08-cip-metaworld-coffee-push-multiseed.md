# MetaWorld coffee-push CIP — multi-seed real-env aggregation

**Date:** 2026-07-08 16:44 JST
**Author:** Aiko
**Status:** done. 5 independent real-env batches, read-only rollouts, **no training**. Aggregation **corrects** the
single-run claim (below). Prior single-run artifacts untouched.

---

## Goal

Turn the single-run MetaWorld coffee-push CIP result into a robust multi-seed estimate — because MetaWorld's env
randomization is not controlled by the seed I pass (two identical runs gave different orders), so a single run's
DAG order is a point estimate.

## Setup

- **Exact command:**
  ```
  PYTHONIOENCODING=utf-8 python -m hymeko_rl.eval.cip.metaworld_cip --multiseed 5 --n 80 --task coffee_push --out reports/figures
  ```
- **Versions:** `metaworld==3.0.0`, `mujoco==3.10.0`, `gymnasium==1.3.0` (mujoco pinned to avoid the
  metaworld-3.1.1 → mujoco-3.3 downgrade; keeps the coin/pick-place envs on 3.10).
- **Batches × rollouts:** 5 batches × N=80 (400 real scripted-policy rollouts total). Each batch is an independent
  run with its own seed (`seed0 + 1000·b`), same `SawyerCoffeePushV3Policy`, same per-episode action-noise protocol
  (`noise ~ U(0, 0.7)` = the observed exogenous input). Read-only; no training.
- **Artifacts:** `reports/figures/2026_07_08_16_39_cip_metaworld_multiseed/` — `coffee_push_multiseed_summary.json`
  + `batch_{0..4}/` (each with its `RolloutFrame` summary, DAG png, and cross-view-verified `.hymeko`).

## Per-batch results

| Batch | Seed | Monitor pass | Reward↔monitor disagreement | Cross-view | `near–reward` \|w\| | `prog→reward` | `noise→reward` |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.512 | 0.364 | ✅ | +0.962 | **+0.805** | — |
| 1 | 1000 | 0.450 | 0.273 | ✅ | +0.973 | — | −0.079 |
| 2 | 2000 | 0.588 | 0.409 | ✅ | +0.981 | — | −2.942 |
| 3 | 3000 | 0.550 | 0.299 | ✅ | +0.949 | — | −0.112 |
| 4 | 4000 | 0.438 | 0.354 | ✅ | +1.005 | — | — |

`prog→reward` is present in **only batch 0**; `near–reward` is present in **all 5** (its LiNGAM direction flips
batch-to-batch — see below).

## Aggregate (median / IQR over 5 batches)

| Quantity | Median | IQR |
|---|---|---|
| Monitor pass rate | 0.513 | [0.450, 0.550] |
| Reward↔monitor disagreement | 0.354 | [0.299, 0.364] |
| Cross-view verification | **passed in all 5 / 5 batches** | — |

## Edge presence + stability

Decision rule: an edge is **stable** iff it appears in ≥ 60 % of batches, its weight IQR excludes zero (sign
clear), and cross-view passes for every batch (it does, 5/5).

| Variable pair | Presence | Median \|w\| | Weight IQR | Dominant direction (count) | Sign-consistent | **Stable?** |
|---|---|---|---|---|---|---|
| `near_fraction – total_reward` | **5/5 (1.00)** | +0.973 | [0.962, 0.981] | `near→reward` (3/5) | ✅ | **✅ STABLE** |
| `progress_score – near_fraction` | 4/5 (0.80) | +0.870 | [0.851, 0.880] | `near→progress` (3/4) | ✅ | **✅ STABLE** |
| `action_noise – total_reward` | 3/5 (0.60) | −0.112 | [−1.527, −0.095] | `noise→reward` (2/3) | ✅ (neg) | ⚠️ sign-stable only |
| `progress_score – total_reward` | **1/5 (0.20)** | +0.805 | — | `progress→reward` | — | **❌ NOT stable** |
| `action_noise – near_fraction` | 3/5 (0.60) | +0.081 | [−0.119, 1.350] | mixed | ❌ | ❌ |
| `action_noise – progress_score` | 0/5 (0.00) | — | — | — | — | ❌ |

## Verdict on the single-run claim

The single run reported **two** stable signatures: `near_fraction → total_reward` **and**
`progress_score → total_reward`. Multi-seed splits them:

- **`near_fraction → total_reward`: CONFIRMED.** Present in 5/5 batches, |w| median +0.973, IQR [0.962, 0.981]
  (extremely tight, excludes zero), sign-consistent, cross-view passes every batch. The real MetaWorld coffee-push
  reward is **proximity/contact-shaped** — robust. *Caveat (honest):* DirectLiNGAM assigns the edge
  `near→reward` in 3 batches and `reward→near` in 2 — the **coupling** (~0.97) is rock-solid, the **causal
  direction** is not uniquely resolved by LiNGAM (expected under the near/progress collinearity below).
- **`progress_score → total_reward`: WEAKENED → REJECTED as a direct edge.** Present in only 1/5 batches. Its
  single-run appearance did not survive replication. The reason is visible in the aggregate: `near_fraction` and
  `progress_score` are themselves a stable strong pair (`near–progress` +0.870, 4/5), so `near_fraction` **mediates**
  the reward coupling and absorbs progress's direct link in most batches. Reward tracks progress *through*
  proximity, not as an independent parent.
- **`action_noise → total_reward`: weak and unreliable.** Present 3/5, sign-consistently negative (more noise →
  less reward, sensible) but the magnitude IQR spans [−1.53, −0.10] (a batch hit −2.94). The scripted policy is
  noise-robust, so the intervention's causal influence is small and poorly estimated — not a dependable edge.

**Net:** the multi-seed pass **confirms** the reward is proximity/contact-shaped (`near_fraction ↔ total_reward`,
5/5, ~0.97) and **rejects** the single-run co-claim that `progress_score` directly drives reward — that was a
point-estimate artifact; `near_fraction` mediates it. Every emitted DAG cross-view-verified.

## Representative DAG

`reports/figures/2026_07_08_16_39_cip_metaworld_multiseed/batch_0/dag_coffee_push_real.png` — shows
`near_fraction →+0.96 total_reward` (the stable edge) and, in this batch only, `progress_score →+0.80 total_reward`
(the edge that did **not** replicate). `action_noise` is isolated, as in most batches.

## Tests + static

- **14** `test_metaworld_cip.py` tests pass (incl. 4 new: `_edge_in`, two `_aggregate_batches` cases, a 2×16
  real-env multi-seed integration). Full CIP suite (90) green. Real-env / multi-seed tests skip if `metaworld`
  is absent.
- `ruff` clean · `radon cc` no block ≥ C (`_aggregate_edge`/`main` split via `_collect_edge`/`_run_single`) ·
  `mypy --strict` clean on the module.
- **No §6.5 anti-patterns.** Multi-seed reuses the single-batch `run_metaworld_cip_real` per batch (no duplicated
  rollout loop); aggregation is pure and unit-tested; monitors used read-only.

## Constraints honored

No training; no reward ablation; no dial-turn; FANUC v2 / coin-collab v2b / CORE.YAML untouched; `pyproject.toml`
**not** edited (its pre-existing CRLF whole-file diff); prior single-run artifacts (`…16_00_cip_metaworld_real/`)
not overwritten.

## Provenance

- Git SHA at start: `d9a436f` (branch `hymeko-neuro-migration`, working tree dirty).
- Seeds 0/1000/2000/3000/4000; deterministic action-noise draws (but MetaWorld env physics is seed-uncontrolled →
  batches are the replication unit, which is the point of this pass).
- No persistent state mutated. Wall ≈ 6 min for 5×80 on CPU.

**Related:** `2026-07-08-cip-metaworld-templates.md` (single-run); memory `project-cip-lingam-rl-diagnostics`.
