# Session handoff — CIP / DirectLiNGAM diagnostic consumer, Phase 1 (2026-07-08)

**Purpose:** catch-up doc for resuming the Hymeko-CIP scenario. One page tying the session together; the full
report (`reports/2026-07-08-cip-directlingam-diagnostic.md`) + the plan bundle
(`docs/plans/2026-07-07-cip-directlingam-diagnostic/`) hold the depth. **Base SHA:** `03b01c3` (working tree
dirty — everything below is uncommitted). Nothing is running; no RL training was launched.

## The arc in one line

The CIP scenario's next brick — the **causal experiment-prioritization layer above the runtime monitor** — is
built and validated: per-rollout CIP variables → `RolloutFrame` → **DirectLiNGAM (numpy/scipy, no `lingam` dep)**
over continuous vars only → `CausalDiagnosis` that ranks reward–monitor disagreement and proposes the next
intervention + ablation. On a synthetic **ground-truth** SEM it recovers the imitation failure chain exactly
(recall 1.00, 0 spurious). **Phase 2 (coin PoC) is written but gated — held per user.**

## Where this sits in the stack (unchanged from the roadmap)

| Layer | Status |
|---|---|
| Reward-independent runtime monitors (`task_monitor/`) | built (prior sessions) |
| CIP-export bridge — verdict → 8 CIP scalars (`cip_export.py`) | built (prior sessions) |
| **DirectLiNGAM diagnostic consumer (`eval/causal/`)** | **built this session (Phase 1)** |
| Coin-toss PoC → `.hymeko` → cross-view verify | **written, GATED (Phase 2, not run)** |
| LiNGAM-SH (signed-hypergraph LiNGAM) — the *science contribution* | separate thread, not this layer |

## What was built (all non-core; `CORE.YAML` untouched, no dependency added)

`hymeko_rl/eval/causal/`:
- **`lingam.py`** — `DirectLiNGAM` (Shimizu 2011). Strategy `IndependenceMeasure` = `PairwiseEntropyMeasure`
  (Hyvärinen max-entropy approx). Adjacency by **backward-elimination-by-significance** OLS (scipy t-test), *not*
  naive lstsq → transitive/indirect edges don't leak. Also `sample_linear_sem` (ground-truth generator, shared
  with the tests).
- **`frame.py`** — `RolloutFrame` (struct-of-arrays: continuous / categorical / missing). `VarKind` enum splits
  columns; **a categorical routed into the linear model RAISES**. `group_by`/`subset` for stratification.
- **`prioritize.py`** — `CipPrioritizer`: ranks `reward_progress_disagreement` (+ `expert_vs_policy_monitor_gap`,
  `critic_vs_monitor_disagreement` once monitors emit them). **Reads** the monitor's disagreement, never re-derives.
- **`diagnose.py`** — `CausalDiagnosis` → `DiagnosisReport` (order, strongest edges, failure ranking,
  next-intervention template, ablation plan; every element stamped **PROPOSED, not proof**). `run_stratified`
  runs LiNGAM per categorical stratum (categoricals stratify, never mix in).
- **`experiments/cip_lingam_demo.py`** — one file, `--mode synthetic|coin` (§6.5 #13). Synthetic runs now; coin is
  the gated Phase 2.

Tests: `tests/test_causal_lingam.py`, `tests/test_causal_diagnose.py`. **35 pass** (incl. 8 cip_export
regression). ruff clean · mypy `--strict` clean (changed files) · radon clean.

## Headline result (synthetic ground truth, seed 0, n=600)

Recovered order `[reward_progress_disagreement, approach_error, both_contact, dist_reduction, delivery]`; edges
`approach_error →(−0.64) both_contact →(+0.62) dist_reduction →(+0.75) delivery`. `reward_progress_disagreement`
correctly **isolated** (a reward with no causal path to delivery = the farming candidate). **Edge recall 1.00
(3/3, signs correct), 0 spurious.** Top intervention surfaced: `reward_farming_candidate`. DirectLiNGAM N=200×8
median **87.9 ms** (budget 150). Artifacts: `reports/figures/2026_07_08_00_31_cip_lingam_synthetic/`
(`summary.json`, `discovered_dag.png`, `adjacency_true_vs_recovered.png`).

## How to resume

Re-run the synthetic demonstrator (fast, deterministic):
```
PYTHONIOENCODING=utf-8 python -m hymeko_rl.experiments.cip_lingam_demo --mode synthetic --n 600 --seed 0
```
Re-run the suite:
```
python -m pytest hymeko_rl/tests/test_causal_lingam.py hymeko_rl/tests/test_causal_diagnose.py -p no:randomly -q
```

**Phase 2 (coin PoC) — only if the user explicitly asks** (standing rule: do not run CIP over real rollouts
unprompted). Build-out for `cip_lingam_demo.py::run_coin`:
1. Roll out the cached policy `checkpoints/galambos/...` / `experiments/2026_07_05_18_34_coin_arch_ab_mlp/policies/coin_arch_ab_mlp_s0.pt`
   for N seeded episodes through `TaskMonitor` → `TaskVerdict` per episode (no new training).
2. `RolloutFrame.from_verdicts(verdicts, extra_continuous={final_dist, total_reward, ...}, categoricals={method, arch, seed})`.
3. `CausalDiagnosis().run_stratified(frame, stratify_by=["architecture"])` — categoricals stratify.
4. Emit the DAG figure (reuse `render_dag`), **declare the discovered graph as a `.hymeko` signed hypergraph**,
   and cross-view verify (DOT + tensor) via the existing hymeko emitters.
5. Report Phase 2; every proposed edge remains a hypothesis to confirm by controlled ablation.

## Cautions

- **Disk `D:` was 100 % full (0 bytes)** at session start; writes failed until **1.6 GB was reclaimed by
  deleting 651 regenerable `__pycache__` dirs** (no user data touched). The volume is effectively full — clear
  space before the next artifact-heavy run.
- **Doctrine (do not weaken):** DirectLiNGAM *proposes*; controlled ablations *decide*. Only continuous rollout
  variables enter the linear model; categoricals stratify. Windows console is cp1250 — keep stdout ASCII (the
  demo already does; a stray `→` crashed the first run).

**Related:** report `2026-07-08-cip-directlingam-diagnostic.md`; plan `docs/plans/2026-07-07-cip-directlingam-diagnostic/`;
memory `project-cip-lingam-rl-diagnostics` (updated: Phase 1 BUILT), `project-kato-lingam-cip-hymeko` (LiNGAM-SH
sibling), `project-hymeko-planner-roadmap`.
