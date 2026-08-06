# Session handoff — Aibo standing substrate, DAgger-as-hypergraph, Gazebo plan (2026-07-07)

**Purpose:** catch-up doc for resuming. One page tying the session's arc together; the two detailed reports + the
plan bundles hold the depth. **Base SHA:** `f43f501` (working tree dirty — everything below is uncommitted).

## The arc in one line

A faithful **22-DOF Sony Aibo (ERS-1000)** declared in HyMeKo → the whole **standing MDP in one `.hymeko`** →
**BC 0.458 → TD3+BC collapses to 0.0 → DAgger recovers 0.958** — and the *training algorithm itself* is now a
declared dataflow hypergraph (bc / dagger / td3_bc told apart by topology). Plus a **plan** for Gazebo output.

## Headline results (measured, kato15 RTX 6000 Ada, 3 seeds, standing dwell ≥200/250)

| method | median | per-seed | note |
|---|---|---|---|
| scripted PD-hold (demonstrator) | 1.0 | — | the imitation ceiling |
| BC clone (from `.hymeko`, DART demos) | 0.458 | [0.29, 0.46, 0.79] | covariate shift limits it |
| **TD3+BC (refine)** | **0.0** | collapse on every seed | off-policy value drift |
| **DAgger (declared as a hypergraph)** | **0.958** | [0.958, 0.958, 1.0] | recovers the ceiling; worst BC seed 0.167→1.0 |

**Verdict:** the lever past a BC ceiling is **imitation (DAgger)**, not off-policy RL. DAgger's round curve is
noisy (fresh-reBC variance) — **best-checkpoint** is the robust deliverable; `warm_start` AND `aggregate_cap` were
both tested and are **no-help** (over-aggregation hypothesis falsified).

## What was built (three pillars)

1. **The plant** — `data/robotics/quadruped.hymeko` rewritten to the faithful Aibo ERS-1000 (legs 3×4, head 3,
   neck 1, waist 1, mouth 1, ears 2, tail 2 = 22 DOF), rounded dog body, cute face. The env **holds the 10
   expressive DOF at neutral (passive spring + armature)** and RL drives the **12 legs**.
2. **The single-source substrate** — `quadruped_stand.hymeko` declares plant + tuning + reward + strategy;
   `QuadrupedGoalEnv.from_hymeko` + `TrainingSpec.from_hymeko` build it. RL: **DART noisy demos** fixed the BC
   covariate-shift trap (0.0 → 0.79 floor); `n_demos` matters (a one-field graph edit 200→40 moved DAgger
   0.792→0.958). Detail: `reports/2026-07-07-aibo-quadruped-hymeko-substrate.md`.
3. **The strategy as a dataflow hypergraph** — `meta_strategy_graph.hymeko` (stages=nodes, signed flows=edges);
   `StrategyGraph.classify()` reads the algorithm from **topology**: acyclic⇒`bc`, relabel-cycle⇒`dagger`,
   critic-node⇒`td3_bc`. Three graphs exist; `quadruped_stand_strategy.py` is one entry that classifies + dispatches
   to the existing `Dagger`/`Campaign` (reuse, not reimpl). Detail: `reports/2026-07-07-dagger-strategy-hypergraph.md`.

## File map (all non-core; CORE.YAML untouched)

- **Plant/scenario:** `data/robotics/{quadruped.hymeko, quadruped_stand.hymeko}`; strategy graphs
  `quadruped_stand_{dagger,bc_graph,td3bc_graph}.hymeko` + vocab `meta_strategy_graph.hymeko`; `meta_reward.hymeko`
  (+4 stand terms); `meta_experiment.hymeko` (BOM stripped — was FABLE-era damage).
- **Code:** `hymeko_rl/env/quadruped_env.py` (held-joint mechanism, `from_hymeko`, PD-hold `expert_action`),
  `train/strategy_graph.py` (reader/classify/runners), `train/dagger.py` (stateless-expert `label_sanity`,
  `aggregate_cap`), `experiments/{quadruped_stand_train,quadruped_stand_dagger,quadruped_stand_strategy}.py`,
  `eval/tasks.py`.
- **Tests (green):** `test_quadruped_aibo_plant`, `test_quadruped_from_hymeko`, `test_strategy_graph` (12),
  `test_quadruped_standing`/`_env`, `test_scenario_sanity`; ruff + mypy clean.
- **Artifacts:** `experiments/2026_07_07_18_49_quadruped_stand_dagger/` (DAgger 0.958 + gif),
  `.../17_32_quadruped_stand/` (BC/TD3+BC 0.458/0.0 + gif); plots `reports/figures/{quadruped_stand_result,
  dagger_standing_result}.png`.
- **Plans:** `docs/plans/2026-07-07-{aibo-quadruped-hymeko-substrate, dagger-strategy-hypergraph, gazebo-output}/`
  (each tex/pdf/tikz/mmd).

## kato15 (to resume compute)

`ssh kato15` (passwordless). RTX 6000 Ada 48 GB, driver CUDA 12.8. **The pinned torch 2.12.0 only ships a CUDA-13.2
GPU build → cannot use kato15's GPU;** the runs used **torch 2.11.0+cu128 in the `.venv_stand` uv scratch venv**
(repo `CORE.YAML` pin untouched; §3 RL carve-out; user-approved venv-only). Repo already cloned at
`~/hymeko_framework_rust`, Linux `hymeko` CLI built. Relaunch scripts in `~`: `launch_dagger.sh`, `launch_stand.sh`.
Sync changed files with `tar czf - <files> | ssh kato15 'bash -lc "cd ~/hymeko_framework_rust && tar xzf -"'`.
GPU rate ~1135 steps/s (~40× the local RTX 3070).

## Open threads / next steps

- **Gazebo output — PLANNED, not built** (`docs/plans/2026-07-07-gazebo-output/`). Key finding: SDF/URDF/gazebo
  emitters already exist; `emit -f sdf` on the Aibo **works** (full model); `emit -f gazebo` is a **stub**. Primary
  path = the proven `dual_fanuc` pattern (emit `sim/aibo/model.sdf` + author `world.sdf` + `gz sim`); the **free
  base** (Aibo `@base` → freejoint in MuJoCo; in SDF = no world weld) is the load-bearing risk. Secondary = wire the
  stubbed `generate_gazebo_world`. `hymeko_formats` is non-core.
- **DAgger clean curve** — do NOT chase further; best-checkpoint 0.958 is robust (warm_start + aggregate_cap both
  measured no-help; noise is fresh-reBC variance).
- **A fourth topology** — a `td3_bc` graph *with* a relabel branch (critic + cycle) would be a novel hybrid for
  `classify()` to name; bounded, no compute.
- **Nothing running on kato15** (verified: no orphaned processes). Working tree dirty; nothing committed this
  session (per policy — commit on request).

## Reports index (this session)

1. `reports/2026-07-07-aibo-quadruped-hymeko-substrate.md` — plant + substrate + BC/TD3+BC.
2. `reports/2026-07-07-dagger-strategy-hypergraph.md` — strategy-as-hypergraph + DAgger 0.958 + 3-way classify.
3. `reports/2026-07-07-session-handoff-aibo-standing-dagger.md` — this catch-up doc.
4. Plans: `docs/plans/2026-07-07-{aibo-quadruped-hymeko-substrate,dagger-strategy-hypergraph,gazebo-output}/`.
