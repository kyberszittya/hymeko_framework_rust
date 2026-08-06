# CIP scenario — standing status (Ito + Kato)

**Date:** 2026-07-08 · Aiko · branch `hymeko-neuro-migration`
**Purpose:** durable status anchor for the Ito + Kato CIP scenario after the full Phase-2 / MetaWorld / Kato-brief
work. Task of record: `docs/task/20260702_task_ito_kato`.

---

## 1. Executive summary

- The CIP scenario is **built end-to-end** and committed (`a39acbb`→`49b8785` on `hymeko-neuro-migration`).
- **No RL training was run** — every result is from read-only rollouts of cached / scripted policies.
- The pipeline is **validated on two substrates**: coin-toss (de-risk PoC) and MetaWorld coffee-push (the Ito+Kato
  scenario, items 2–4).
- **Cross-view HyMeKo verification passed in 100% of runs** — every declared causal DAG re-derived identically
  through the engine (declared ≡ tensor ≡ Blake3 hash).
- **All discovered DAGs remain PROPOSED causal hypotheses** unless intervention-backed. The cross-view proves
  *representation* consistency, not causal truth.

## 2. Pipeline

```
rollouts → HyMeKo runtime monitor → CIP variables → DirectLiNGAM → causal DAG
  → .hymeko declaration → HyMeKo engine tensor (star/clique) + Blake3 hash verification
```

## 3. Built components

| Component | Path | Status |
|---|---|---|
| Reward-independent runtime monitors (coin, coffee-push, dial-turn) | `hymeko_rl/eval/task_monitor/` | built (prior), used **read-only** |
| CIP-export bridge (verdict → CIP variables) | `hymeko_rl/eval/task_monitor/cip_export.py` | built |
| DirectLiNGAM consumer (numpy/scipy, **no `lingam` dep**) | `hymeko_rl/eval/causal/` | built; recovers synthetic ground truth (recall 1.00) |
| `.hymeko` emit + cross-view verification | `hymeko_rl/eval/causal/hymeko_emit.py` | built; cross-view passed every run |
| Contact-reward ablation (Stage A) | `hymeko_rl/eval/cip/contact_reward_ablation.py` | built |
| MetaWorld CIP (synthetic + real-env + multi-seed) | `hymeko_rl/eval/cip/metaworld_cip.py` | built |
| GIF / meeting-brief deliverables | `hymeko_rl/eval/cip/metaworld_gifs.py` | built |

**Tests:** 91 CIP tests pass; ruff / radon / mypy(strict) clean.

## 4. Results

**A. Coin-toss PoC (de-risk) — `a39acbb`.** DirectLiNGAM over real coin rollouts: `contact_score → total_reward`
(+0.69) strongest; `total_reward` disconnected from `delivery_score` → top intervention `reward_farming_candidate`.

**B. Contact-reward ablation Stage A (no training) — `a39acbb`.** The coin reward is `Σ weight·term`, recomputable
offline **bit-exactly**. Removing the contact terms **collapses the `contact_score → total_reward` edge to 0.0** and
re-parents the reward onto `delivery_score` (+0.88); reward↔monitor disagreement drops (mlp 0.59→0.43, hsikan
0.57→0.28). Downweight ×0.25 does **not** collapse the mlp edge (removal is required).

**C. MetaWorld synthetic templates (coffee-push, dial-turn) — `b3d56d3`.** Synthetic trajectories to each monitor
story (acyclic non-Gaussian SEM). Both recover the chain (`approach/engage → contact/rotation → progress →
reward-sink`); every DAG cross-view-verified. The MetaWorld **Phase-1 analog**.

**D. MetaWorld coffee-push real-env — `d9a436f`.** Real `SawyerCoffeePushV3Policy` + per-episode action noise
(observed exogenous input) through the HyMeKo monitor; obs → monitor-schema mapping. Pipeline runs on real
physics/reward and cross-view-verifies (the Phase-2 analog).

**E. MetaWorld coffee-push multi-seed aggregation (5×80, no training) — `52e3af9`.**
- `near_fraction ↔ total_reward`: present 5/5, |w| median **0.973**, IQR [0.962, 0.981], sign-consistent — the
  real reward is **proximity/contact-shaped**. (Direction flips 3/2 across batches — coupling stable, direction
  unresolved.)
- `progress_score → total_reward`: present **1/5** — `near_fraction` mediates it (`near↔progress` +0.870, 4/5).
- Cross-view passed **5/5**; monitor pass-rate median 0.51, disagreement median 0.35.

**F. Kato deliverables — `49b8785`.** Coffee-push GIFs (success / failure / compare) and the consolidated meeting
brief as `.md` / `.tex` / **PDF** (compiled with tectonic; embeds the coin, ablation, and real-env DAGs plus
coffee-push stills).

## 5. Robust verdicts

- **Coin reward-farming hypothesis: SUPPORTED at the reward-computation level** (removing contact terms collapses
  the edge and re-parents reward onto delivery).
- **MetaWorld coffee-push `near_fraction ↔ total_reward`: STABLE** across 5 seeds (~0.97) — reward is
  proximity/contact-shaped.
- **`progress_score → total_reward`: REJECTED as a direct edge** after multi-seed aggregation (was a single-run
  artifact; mediated by `near_fraction`).
- **Cross-view verification: PASSED** for every emitted DAG across all runs.
- **Policy-learning consequences: NOT yet tested** (would require retraining under a modified reward — deliberately
  not run).

## 6. Honest scope / limitations

- **No coin-toss solution claimed** — CIP *diagnosed* the reward; a physics wall and an imitation gap remain
  (per the fair-vector-critic thread, the lever is better imitation, not local RL refinement).
- **No training run** anywhere in this work.
- **DAGs are PROPOSED, not truth** — controlled ablation decides; cross-view proves representation, not causality.
- **MetaWorld randomization caveat** — the env's randomization is **not controlled by the passed seed**; single-run
  DAG order is a point estimate, so the coffee-push claim rests on the 5-batch aggregate.
- **Dial-turn real-env not done** — needs a dial-angle extraction from the obs (its synthetic template is done).
- **Real-env MetaWorld reward ablation not done** — the coin Stage-A method has not yet been transferred to a real
  MetaWorld reward decomposition.
- **LiNGAM-SH not implemented** — the signed-hypergraph LiNGAM science contribution remains a separate thread.

## 7. Environment

- `metaworld` **3.0.0**, `mujoco` **3.10.0** (pinned to block the metaworld-3.1.1 → mujoco-3.3 downgrade that would
  affect the coin/pick-place envs), `gymnasium` **1.3.0**, `scipy` **1.17.1**.
- Native `hymeko` engine **built with maturin** into the Mac venv (`maturin develop --release` in `hymeko_py/`).
- **`pyproject.toml` not edited** because of a pre-existing whole-file CRLF diff on that file — declaring
  `metaworld` in the `ml` group (with the `mujoco==3.10` pin) is a clean follow-up.

## 8. Artifact index

**Reports** (`reports/`):
- `2026-07-08-cip-phase2-coin-poc.md`
- `2026-07-08-cip-contact-reward-ablation-setup.md`
- `2026-07-08-cip-metaworld-templates.md`
- `2026-07-08-cip-metaworld-coffee-push-multiseed.md`
- `2026-07-08-cip-kato-meeting-brief.{md,tex,pdf}`
- `2026-07-08-cip-status.md` (this file)

**Figures** (`reports/figures/`):
- `2026_07_08_13_32_cip_lingam_coin/` — coin causal DAGs (mlp, hsikan)
- `2026_07_08_15_12_cip_contact_ablation_stageA/` — contact-reward ablation (edge collapse)
- `2026_07_08_15_45_cip_metaworld_synthetic/` — MetaWorld templates
- `2026_07_08_16_00_cip_metaworld_real/` — MetaWorld coffee-push real-env
- `2026_07_08_16_39_cip_metaworld_multiseed/` — multi-seed (per-batch DAGs + `.hymeko`)

**GIFs** (`reports/gifs/metaworld_coffee_push/`): `coffee_push_{success,failure,compare}.gif`

**Code** (`hymeko_rl/eval/`):
- `causal/hymeko_emit.py` — signed-DAG → `.hymeko` + `cross_view_verify`
- `cip/contact_reward_ablation.py` — Stage-A reward ablation
- `cip/metaworld_cip.py` — synthetic templates + real-env + multi-seed
- `cip/metaworld_gifs.py` — coffee-push GIF rendering

## 9. Open decisions for Kato

1. **LiNGAM-SH** — the science contribution: constrain LiNGAM's `B = A_out Σ A_inᵀ` so non-Gaussianity identifies
   the *grouping* (the theorem). Nearest prior art arXiv:2511.03831 validates the direction, does not pre-empt.
2. **Real-env reward ablation on MetaWorld** — transfer the coin Stage-A method (decompose the reward, recompute
   contact-off offline, check the `→ total_reward` edge collapses).
3. **Multi-seed everywhere + dial-turn real-env** for a publishable ranking claim.
4. **`pyproject.toml` housekeeping** — declare `metaworld` (ml group, `mujoco==3.10` pin) once the pre-existing
   CRLF diff is resolved.

## 10. Constraints honored

- **No training** run.
- **FANUC v2 untouched.**
- **coin-collab v2b untouched.**
- **CORE.YAML untouched.**
- **`pyproject.toml` not edited.**
- **Prior artifacts not overwritten** (each run writes a fresh timestamped dir).
- MetaWorld monitors used **read-only**.
