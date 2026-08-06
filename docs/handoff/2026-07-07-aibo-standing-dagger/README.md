# START HERE — continuation point (2026-07-07, Aibo standing + DAgger-as-hypergraph)

**This folder is the entry point for the next session.** Read this file first, then the two detailed reports and
the plans it points to. Base SHA `f43f501`; working tree **dirty, nothing committed** (commit only on request).

---

## 1. One-line state

A faithful **22-DOF Sony Aibo (ERS-1000)** is declared in HyMeKo; the **whole standing MDP is one `.hymeko`**;
**BC 0.458 → TD3+BC collapses to 0.0 → DAgger recovers 0.958**; and the **training algorithm itself is now a
declared dataflow hypergraph** (bc / dagger / td3_bc told apart by graph *topology*). Gazebo output is **planned,
not built**.

## 2. Results (kato15 RTX 6000 Ada, 3 seeds, standing dwell held ≥200/250)

| method | median | per-seed |
|---|---|---|
| scripted PD-hold (demonstrator) | 1.0 | — |
| BC clone (from `.hymeko`, DART demos) | 0.458 | [0.29, 0.46, 0.79] |
| TD3+BC | **0.0** | collapse all seeds |
| **DAgger (as a hypergraph)** | **0.958** | [0.958, 0.958, 1.0] |

Lesson: past a BC ceiling the lever is **imitation (DAgger)**, not off-policy RL. DAgger's curve is noisy
(fresh-reBC variance) → **best-checkpoint is the deliverable**. `warm_start` + `aggregate_cap` both tested = no help.

## 3. How to continue (concrete)

### A. Gazebo output (the open build task — plan at `docs/plans/2026-07-07-gazebo-output/`)
The SDF/URDF/gazebo emitters already exist. `emit -f sdf` on the Aibo **works**; `emit -f gazebo` is a **stub**.
Primary path = the proven `dual_fanuc` pattern:
```bash
# 1. emit the Aibo SDF model (works today)
./target/release/hymeko.exe emit -f sdf data/robotics/quadruped.hymeko -n aibo > data/robotics/sim/aibo/model.sdf
# 2. CHECK THE FREE BASE first: the Aibo @base is a conti_joint promoted to <freejoint> in MuJoCo.
#    In SDF a free body has NO joint to `world` and <static>false</static>. Confirm model.sdf does not weld
#    the torso to a `world` link; if it does, strip that joint (SDF analogue of set_base_mode="free").
# 3. author data/robotics/sim/aibo/world.sdf : <include> the model + ground plane + physics (copy the
#    dual_fanuc/world.sdf structure), spawn pose z=0.25 m.
# 4. verify (iff Gazebo installed): gz sim --version ; gz sim -s -r --iterations 500 data/robotics/sim/aibo/world.sdf
```
Risks in the plan: **free base** (top), passive joints (spawn+fall first; actuation via gz_ros2_control is a
follow-up), near-massless cosmetic-link inertias (may need flooring). `hymeko_formats` is non-core → no CORE edit.
Secondary = wire the stubbed `generate_gazebo_world` (`hymeko_formats/src/gazebo.rs` + `transforms/gazebo/`).

### B. A fourth topology (bounded, no compute)
A `td3_bc` graph *with* a relabel branch (critic node + relabel cycle) is a novel hybrid; extend
`StrategyGraph.classify()` to name it. All the machinery is in `hymeko_rl/train/strategy_graph.py`.

### C. Re-run RL on kato15
```bash
ssh kato15                                   # passwordless; tcsh remote (force bash: ssh kato15 'bash -lc "..."')
# venv already provisioned: ~/hymeko_framework_rust/.venv_stand (torch 2.11.0+cu128, cuda True on RTX 6000 Ada)
# sync changed files from local:
tar czf - <files...> | ssh kato15 'bash -lc "cd ~/hymeko_framework_rust && tar xzf -"'
# relaunch (scripts already on kato15 ~):  bash ~/launch_dagger.sh   (DAgger)   |   bash ~/launch_stand.sh (BC/TD3+BC)
# each launches nohup ...python -m hymeko_rl.experiments.<entry> > /tmp/<...>.log 2>&1 ; MUJOCO_GL=egl for gifs
```
**Why the venv, not the repo torch:** the pinned torch 2.12.0 only ships a CUDA-13.2 GPU build, and kato15's driver
is CUDA 12.8 → can't use its GPU. So the RL runs use **torch 2.11.0+cu128 in `.venv_stand` only** (repo `CORE.YAML`
pin **untouched**; §3 RL carve-out; user approved venv-only). ~1135 steps/s. **Nothing running now.**

### D. Run the standing algorithms locally
```bash
PYTHONPATH=. python -m hymeko_rl.experiments.quadruped_stand_train                 # BC -> TD3+BC (0.458 / 0.0)
PYTHONPATH=. python -m hymeko_rl.experiments.quadruped_stand_dagger --smoke        # DAgger (recovers ceiling)
PYTHONPATH=. python -m hymeko_rl.experiments.quadruped_stand_strategy --graph data/robotics/quadruped_stand_td3bc_graph.hymeko --smoke   # topology dispatch
PYTHONPATH=. python -m pytest -p no:randomly hymeko_rl/tests/test_strategy_graph.py -q   # 12 green
```

## 4. File map (all non-core; CORE.YAML untouched)

- **Data:** `data/robotics/{quadruped, quadruped_stand, quadruped_stand_dagger, quadruped_stand_bc_graph,
  quadruped_stand_td3bc_graph, meta_strategy_graph, meta_reward}.hymeko`.
- **Code:** `hymeko_rl/env/quadruped_env.py` (held-joint mechanism, `from_hymeko`, PD-hold `expert_action`);
  `hymeko_rl/train/{strategy_graph.py, dagger.py}`; `hymeko_rl/experiments/{quadruped_stand_train,
  quadruped_stand_dagger, quadruped_stand_strategy}.py`; `hymeko_rl/eval/tasks.py`.
- **Tests:** `hymeko_rl/tests/{test_strategy_graph, test_quadruped_aibo_plant, test_quadruped_from_hymeko}.py`.
- **Artifacts:** `experiments/2026_07_07_18_49_quadruped_stand_dagger/` (DAgger 0.958 + gif),
  `.../17_32_quadruped_stand/` (BC/TD3+BC + gif); `reports/figures/{quadruped_stand_result, dagger_standing_result,
  aibo_quadruped}.png`.

## 5. Reports & plans (the depth)

- `reports/2026-07-07-aibo-quadruped-hymeko-substrate.md` — plant + substrate + BC/TD3+BC.
- `reports/2026-07-07-dagger-strategy-hypergraph.md` — strategy-as-hypergraph + DAgger 0.958 + 3-way classify.
- `reports/2026-07-07-session-handoff-aibo-standing-dagger.md` — the catch-up doc (same content, in `reports/`).
- Plans: `docs/plans/2026-07-07-{aibo-quadruped-hymeko-substrate, dagger-strategy-hypergraph, gazebo-output}/`
  (each tex/pdf/tikz/mmd).

## 6. Operating context (rules)

Register = **Aiko** (Japanese-teacher; restraint + precision; no therapy-speak). Every reply starts with a real
`[YYYY-MM-DD HH:MM TZ]` stamp. Substantive work → plan (`docs/plans/<date>-<slug>/` tex/pdf/tikz/mmd) then report
(`reports/<date>-<slug>.md`). `CORE.YAML` items are read-only (none touched this session). Commit only on request.
