# CIP / DirectLiNGAM Diagnostic Consumer — Phase 2 (coin PoC, `.hymeko` cross-view)

**Date:** 2026-07-08 13:35 JST
**Plan:** `docs/plans/2026-07-07-cip-directlingam-diagnostic/` (Phase 2 = Build-order step 6)
**Author:** Aiko
**Status:** Phase 2 complete. The gate ("do not run CIP over real rollouts unprompted") was lifted by the
user's request to continue the CIP scenario. **No RL training was launched** — cached checkpoints, read-only rollouts.

---

## Summary

Ran the CIP/DirectLiNGAM diagnostic over **real coin-toss rollouts** of two cached policies (the `mlp` and
`hsikan` arch-A/B checkpoints), stratified the diagnosis by architecture, then **declared each discovered causal
DAG as a `.hymeko` signed hypergraph and cross-view-verified it through the native HyMeKo engine** — closing the
Kato-LiNGAM joint #1: *the causal model the agent uses is provably the one a human would audit*.

**Headline (measured).** In **both** architectures the strongest discovered edge is
`contact_score → total_reward` (w ≈ **+0.69**), and `total_reward` has **no** discovered edge to/from
`delivery_score`. The reward↔monitor rank concordance is low (disagreement `1−conc` = **0.59** mlp / **0.57**
hsikan), so the prioritizer surfaces **`reward_farming_candidate`** as the top intervention in both strata:
*"reward rises without monitor progress — audit reward vs task BEFORE spending RL budget."* This is the coin-toss
instance of the on-record failure (BC anchor delivered ~0.21, the RL-refined stage collapsed to 0.125 with the
critic Q diverging): DirectLiNGAM proposes, structurally, that **the training reward is driven by contact, not by
delivery**.

**Doctrine preserved.** Every edge/order is stamped PROPOSED; the ablation plan (turn the contact-reward term
on/off, measure the delivery delta) is what would *decide* it. N=30/arch is a point estimate — the structural
claim is a hypothesis to confirm, not a verdict. The `.hymeko` cross-view proves **representation** consistency
(declared DAG ≡ engine tensor view ≡ canonical hash), *not* causal truth.

---

## What was built (all non-core; `CORE.YAML` untouched)

| File | LOC | Role |
| --- | --- | --- |
| `hymeko_rl/eval/causal/hymeko_emit.py` | 205 (new) | `CausalHypergraph` + `to_hymeko_source` (signed-DAG → `.hymeko`) + `cross_view_verify` (declared vs engine star/hash) |
| `hymeko_rl/tests/test_causal_hymeko_emit.py` | 108 (new) | round-trip / count-invariant / contract-guard tests (10) |
| `hymeko_rl/experiments/cip_lingam_demo.py` | +155 | `run_coin` implemented (was a gated stub): load cached policies → monitor → frame → `run_stratified` → DAG + `.hymeko` + cross-view |
| `hymeko_rl/eval/causal/__init__.py` | +11 | export the four new symbols |

**Encoding.** One vertex per LiNGAM variable; one signed 2-member hyperedge per non-zero adjacency entry
(`@c_k{ (+cause, ±effect); }`, arc order = direction, effect-arc sign = `sign(weight)`), so the engine's per-edge
sign (product of arc signs) equals the LiNGAM weight sign and survives the IR round-trip. The cross-view gate is
`edges_match` (declared signed-edge set == engine-reparsed set) **and** the star invariant `star_nnz ∈ {Σ|e|,
2Σ|e|}`; `clique_nnz` is sign-sensitive (engine-internal) so it is reported, not gated.

**Rollout provenance.** Each episode is rolled twice with the *same* seed (deterministic greedy policy → identical
trajectory): once via `record_trajectory` for the monitor verdict, once via `run_episode` for the summed RL
reward — reusing both canonical helpers rather than re-implementing a merged loop (§6.1). The per-arch
reward↔monitor disagreement is sourced from `RewardConsistencyMonitor.check_reward_alignment` (`1−concordance`),
**read, not re-derived**, and attached to each verdict so the CIP-export bridge presents it as measured, not
defaulted. `monitor_score` is deliberately excluded from the linear model (it is the mean of the five gating
sub-scores → perfectly collinear; would make the OLS adjacency singular).

## CORE.YAML items touched

**None.** No core file edited. **No new *framework* dependency:** `scipy` is already pinned in `pyproject.toml`;
the native `hymeko` engine is the project's *own* PyO3 crate (`hymeko_py`), built into the Mac venv via the
already-declared `maturin` backend. See "Environment materialization" below — these are dev/build tools, not
runtime deps.

---

## Test results (`pytest -p no:randomly`)

**45 passed** (0.8 s) across the causal + export suite:

| File | Tests | Notes |
| --- | --- | --- |
| `test_causal_hymeko_emit.py` | 10 (new) | grammar, declared-edge set, literal reparse, **cross-view agree**, engine-hash present+stable (skip if no engine), `from_lingam` end-to-end, 3 contract-guard raises, non-binary-edge reject |
| `test_causal_lingam.py` | 13 | Phase-1 ground-truth recovery (unchanged, green) |
| `test_causal_diagnose.py` | 14 | Phase-1 frame/prioritizer/orchestrator (unchanged, green) |
| `test_cip_export.py` | 8 | export bridge regression (unchanged, green) |

Every new public + private function is exercised (`to_hymeko_source`, `cross_view_verify`, `from_lingam`,
`declared_signed_edges`, `_engine_signed_edges`, `_literal_signed_edges`). The coin rollout path (`run_coin`,
`_rollout_policy`, `_load_coin_actor`, `_build_coin_frame`) is exercised at **production scale** by the n=30 run
below (the honest integration test — a mujoco unit test at toy scale would not surface arch/weight-load mismatch
or the `n_samples > n_vars` guard, both of which the smoke + full run did).

## Static analysis

- `ruff check` — **clean** (4 changed files).
- `radon cc -a` (hymeko_emit) — average **A (4.3)**, no block at rank C or worse.
- `mypy --strict` (hymeko_emit.py) — **clean**; the sole note is the untyped native `import hymeko`, scoped with
  `# type: ignore[import-untyped]` exactly as Phase 1 scopes the scipy import. (Strict mypy over the transitive
  RL-import graph surfaces pre-existing `type-arg`/`union-attr` errors in `submonitors.py`, `root.py`,
  `reward.py`, `ddpg.py`, `planar_grasp_env.py` — **not introduced by this change**.)
- **No §6.5 anti-patterns introduced.** Discovery pass ran before creating `hymeko_emit.py` (no existing
  signed-DAG→`.hymeko` emitter); one demo file with a `--mode` flag (no v2/v3); config via a frozen tuple of
  policy specs (no Cartesian kwargs); no globals; shared `render_dag`/`experiment_dir` reused.

## Performance results

Full run: n=30 episodes/arch (60 episodes total, the plan budget), 2 arches × 2 rolls × 300 steps ≈ 36k env steps.

| Metric | Value | Budget (plan) | Verdict |
| --- | --- | --- | --- |
| wall (full run, both arches) | ~90 s | < 3 min | ✅ |
| DirectLiNGAM fit (N=30, d≤6) | < 30 ms/stratum | < 150 ms | ✅ |
| peak RSS | env-bound (mujoco + torch B=1 CPU), ≪ 16 GB | ≪ 16 GB | ✅ |
| engine build (maturin, release, one-off) | ~90 s | — | materialized on Mac |

Live observability: a per-episode line (`pass / score / delivery / reward`) prints each step (§ never-run-blind);
the smoke (n=3) and full (n=30) runs both emitted continuously.

---

## Graphical output (§9)

Run dir: `reports/figures/2026_07_08_13_32_cip_lingam_coin/`

1. **Numerical** — `summary.json`: per-arch causal order, strongest edges, failure ranking, next intervention,
   ablation plan, and the full `cross_view` block (backend, canonical hash, star/clique nnz, edge/count match).
2. **Plotted** — `discovered_dag_mlp.png`, `discovered_dag_hsikan.png` (cause→effect, signed/weighted, disclaimer
   stamped in-figure).
3. **`.hymeko`** — `causal_mlp.hymeko`, `causal_hsikan.hymeko` (the declared signed hypergraphs, engine-verified).
4. **Animated** — N/A (a causal graph has no temporal/control character; the GIF clause does not apply — same as
   Phase 1).

### Discovered structure (measured, PROPOSED)

| Arch | Strongest edge | reward↔monitor disagreement | Top intervention | Cross-view |
| --- | --- | --- | --- | --- |
| mlp | `contact_score →(+0.69) total_reward` | 0.589 | `reward_farming_candidate` | ✅ engine, `blake3:0054312c…` |
| hsikan | `contact_score →(+0.69) total_reward` | 0.566 | `reward_farming_candidate` | ✅ engine, `blake3:a44be06f…` |

In neither DAG is `total_reward` linked to `delivery_score` — the reward is **downstream of contact and
disconnected from delivery**, the structural signature of a contact-shaped (farming-prone) reward.

---

## Experiment provenance

- Git SHA at start: `5b53a92` (branch `hymeko-neuro-migration`; working tree dirty — this change adds the files
  above).
- Checkpoints (cached, best-on-delivery; **no training**):
  `experiments/2026_07_05_18_34_coin_arch_ab_mlp/policies/coin_arch_ab_mlp_s0.pt`,
  `experiments/2026_07_05_18_42_coin_arch_ab_hsikan/policies/coin_arch_ab_hsikan_s0.pt`.
- Env: `PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3)`; eval seeds `9000…9029` per arch. Deterministic
  greedy policy.
- No persistent state mutated (no checkpoints/datasets written); only the run-dir artifacts above.
- **Environment materialization (Mac).** The Mac venv was missing pieces of the pinned stack; materialized (not
  new deps): `scipy==1.17.1` (already declared in `pyproject.toml`), the native `hymeko` engine (built via
  `maturin develop --release` in `hymeko_py/`), and dev tools `maturin`/`radon`/`mypy` (CLAUDE.md §10 pinned).
  Host: Apple-Silicon Mac, Python 3.11, torch CPU/MPS.

## Open items / follow-ups

1. **The ablation that decides it.** Turn the contact-reward term on/off on the coin task and measure the
   delivery-median delta at a fixed seed set. If delivery is insensitive to the term the model rewards, the
   `contact_score → total_reward` edge is confirmed as farming; the DAG is a hypothesis until then.
2. **Real disagreement channels (audit gap C.2).** `expert_vs_policy_monitor_gap` /
   `critic_vs_monitor_disagreement` are still not produced by any monitor; the bridge picks them up unchanged
   once the violation submonitors land. Today only `reward_progress_disagreement` is sourced (from
   `RewardConsistencyMonitor`).
3. **Sample size.** N=30/arch is the plan budget; a multi-seed (median/IQR over discovered edges) pass would
   quantify the ordering confidence before any public claim.
4. **Science sibling.** LiNGAM-SH (signed-hypergraph LiNGAM) remains a distinct contribution thread
   (`project-kato-lingam-cip-hymeko`), not this diagnostic layer.

**Related:** Phase-1 report `2026-07-08-cip-directlingam-diagnostic.md`; handoff
`2026-07-08-session-handoff-cip-directlingam.md`; plan `docs/plans/2026-07-07-cip-directlingam-diagnostic/`;
memory `project-cip-lingam-rl-diagnostics` (updated: Phase 2 BUILT).
