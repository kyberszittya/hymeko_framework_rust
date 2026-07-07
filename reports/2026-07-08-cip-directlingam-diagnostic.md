# CIP / DirectLiNGAM Diagnostic Consumer — Phase 1

**Date:** 2026-07-08 00:35 CEST
**Plan:** `docs/plans/2026-07-07-cip-directlingam-diagnostic/` (plan.tex/pdf/tikz/mmd)
**Author:** Aiko
**Status:** Phase 1 complete (module + synthetic ground-truth demonstrator). Phase 2 (coin PoC) gated — see Open items.

---

## Summary

Built the **CIP / DirectLiNGAM diagnostic consumer** — the causal experiment-prioritization layer that sits
*above* the already-built CIP-export bridge (`hymeko_rl/eval/task_monitor/cip_export.py`). It aggregates
per-rollout CIP + continuous variables into a data-oriented frame, ranks failures by **reward–monitor
disagreement** (reading, never re-deriving, the monitor's disagreement), and runs a **numpy/scipy-only
DirectLiNGAM** over the *continuous variables only*, stratifying categoricals. The output — causal order,
strongest edges, failure ranking, next-intervention template, ablation plan — is stamped throughout as a
**proposal to test by controlled ablation, never proof** (framework doctrine: LiNGAM proposes, ablation decides).

The scenario's `DO NOT run CIP/DirectLiNGAM unless explicitly requested` gate was lifted by the user's request
to continue the CIP scenario. No RL training was launched.

**Key design decisions.**
- **No new dependency.** DirectLiNGAM is implemented in ~300 LOC over numpy + scipy (both already pinned in
  `pyproject.toml`); the heavyweight `lingam` package is deliberately *not* pulled in, avoiding the §1 dependency
  gate and a large transitive tree.
- **Continuous-only is structural, not hoped-for.** The frame splits columns by kind (`VarKind` enum); a
  categorical routed into the linear model **raises**. Categoricals stratify via `run_stratified`.
- **Adjacency by significance-pruned OLS.** Naive least-squares over correlated ancestors leaks transitive
  edges (why the reference uses adaptive lasso). Instead, backward elimination by regression significance
  (scipy t-test) recovers *direct* parents only — an ancestor is insignificant once the true direct parent is
  conditioned on. This is dependency-free and principled.

---

## Files touched

**New (additive; no existing module modified):**

| File | LOC | Role |
| --- | --- | --- |
| `hymeko_rl/eval/causal/__init__.py` | 42 | package re-exports |
| `hymeko_rl/eval/causal/lingam.py` | 296 | DirectLiNGAM (Strategy measure), significance-pruned adjacency, `sample_linear_sem` |
| `hymeko_rl/eval/causal/frame.py` | 182 | `RolloutFrame` (SoA; continuous/categorical/missing split) |
| `hymeko_rl/eval/causal/prioritize.py` | 81 | `CipPrioritizer` (reward–monitor disagreement ranking) |
| `hymeko_rl/eval/causal/diagnose.py` | 141 | `CausalDiagnosis` orchestrator + intervention templates |
| `hymeko_rl/tests/test_causal_lingam.py` | 143 | ground-truth recovery + significance-pruning regression |
| `hymeko_rl/tests/test_causal_diagnose.py` | 178 | frame/prioritizer/orchestrator + perf |
| `hymeko_rl/experiments/cip_lingam_demo.py` | 179 | demonstrator (`--mode synthetic\|coin`, one file per §6.5 #13) |

**No files modified** in existing modules. `cip_export.py` and the monitor were left untouched.

## CORE.YAML items touched

**None.** No dependency added (scipy/numpy already pinned; `lingam` intentionally avoided). No core file edited.

---

## Test results

Runner: `pytest -p no:randomly`. **35 passed** (10.3 s combined incl. cip_export regression).

| Layer | Tests | Notes |
| --- | --- | --- |
| Unit / correctness (discriminating test) | `test_causal_lingam.py` (13) | ground-truth topological order + signs (uniform & Laplace), scrambled-label root, **transitive-edge pruning regression**, `_ols_with_pvalues` guard, input contract, determinism |
| Integration + contract | `test_causal_diagnose.py` (12 non-perf) | frame split, missing→dropped, **categorical-into-model raises** (frame & orchestrator), stratification, disagreement ranking, well-formed report, `none`-intervention path, graceful LiNGAM skip |
| Performance | `test_directlingam_perf_budget` | numeric assertion (see below) |
| Regression (unchanged) | `test_cip_export.py` (8) | export bridge still green |

Every new public and private function is exercised (`_ols_with_pvalues`, `_backward_eliminate`,
`_split_cip_variables`, `_checked_column`, all measures, frame/prioritizer/diagnose paths). The significance
change carries a regression test (`test_significance_pruning_removes_transitive_edges`) that fails against the
prior naive-lstsq implementation.

## Performance results

DirectLiNGAM, N=200 × d=8, 7 iterations after warm-up (host: this Windows 11 box; numpy 2.4.6, scipy 1.17.1):

| Metric | Value | Budget (plan) | Verdict |
| --- | --- | --- | --- |
| median wall | **87.9 ms** | < 150 ms | ✅ |
| IQR | [83.7, 101.9] ms | — | |
| worst | 139.1 ms | — | |
| peak Python alloc | 0.1 MB | < 300 MB RSS | ✅ (numpy over ≤10³×≤32 matrices; no torch) |

No regression baseline exists (new module). No profile needed — under budget, no optimization performed.

---

## Graphical output (§9)

Synthetic demonstrator (`--mode synthetic --n 600 --seed 0`) emitted all three forms into
`reports/figures/2026_07_08_00_31_cip_lingam_synthetic/`:

1. **Numerical** — `summary.json`: recovered order, true-vs-recovered edge sets, edge recall, full `DiagnosisReport`.
2. **Plotted** — `discovered_dag.png` (causal DAG, cause→effect, signed/weighted) and
   `adjacency_true_vs_recovered.png` (side-by-side B heatmaps).
3. **Animated** — N/A (a causal graph has no temporal/control character; the GIF clause does not apply).

**Ground-truth validation result:** the imitation chain
`approach_error →(−0.64) both_contact →(+0.62) dist_reduction →(+0.75) delivery` was recovered exactly;
`reward_progress_disagreement` was correctly left **isolated** (a reward signal with no causal path to delivery —
the reward-farming candidate). **Edge recall 1.00 (3/3, correct signs); 0 spurious edges.** The diagnosis
surfaced `reward_farming_candidate` as the top intervention ("audit reward vs task BEFORE spending RL budget").

---

## New / removed dependencies

None. (scipy `>=1.11,<2` and numpy already pinned in `pyproject.toml`; a scoped `# type: ignore[import-untyped]`
on the scipy import documents the missing-stubs situation, §6.3.)

## Static analysis & anti-patterns

- `ruff check` — clean. `mypy --strict` (changed files) — clean. `radon cc -nc` — nothing flagged (the one
  block initially at C=14, `from_verdicts`, was refactored below threshold via `_split_cip_variables` /
  `_checked_column`).
- **No §6.5 anti-patterns introduced.** Strategy trait for the measure (not a Cartesian API); `VarKind` enum,
  not string config; config threaded, no globals; discovery pass ran before the package was created; one demo
  file with a `--mode` flag (no v2/v3 proliferation); shared `sample_linear_sem` and `experiment_dir` reused,
  not duplicated.

---

## Experiment provenance

- Git SHA at start: `03b01c3` (working tree dirty; this change adds the files above — see git status).
- Seed: 0 (synthetic demonstrator). Deterministic (`np.random.default_rng`).
- No persistent state mutated; no checkpoints/datasets touched; no background processes launched.
- **Host note:** disk `D:` was found **100% full (0 bytes free)** at session start — writes failed until 1.6 GB
  was reclaimed by clearing 651 regenerable `__pycache__` dirs (no user data touched). Flag for the user: the
  volume is effectively full and will block further artifact writes.

---

## Open items / follow-ups

1. **Phase 2 — coin PoC (gated).** `cip_lingam_demo.py --mode coin` is stubbed with a pointer. It should build a
   `RolloutFrame` from real coin-toss rollouts of the cached `coin_arch_ab_mlp_s0.pt` policy (via the monitor →
   `export_cip_variables` path — no new training), run the diagnosis, then declare the discovered graph as a
   `.hymeko` signed hypergraph and cross-view verify (DOT + tensor). Gated behind this report per the plan.
2. **Real disagreement channels.** `expert_vs_policy_monitor_gap` and `critic_vs_monitor_disagreement` are
   recognised by the prioritizer but not yet produced by a monitor (audit gap C.2 in the runtime-monitor spec);
   they default-and-flag until the violation submonitors land — the bridge picks them up unchanged once they do.
3. **Science sibling (distinct concern).** LiNGAM-SH (signed-hypergraph LiNGAM) remains a separate contribution
   thread (`project-kato-lingam-cip-hymeko`), not this diagnostic layer.
