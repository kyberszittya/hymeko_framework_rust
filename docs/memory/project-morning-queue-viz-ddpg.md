---
name: project-morning-queue-viz-ddpg
description: "2026-06-24 post-overnight-run queue (torch work waits for the Galambos BC+PPO run to free the machine ~07:15) — DDPG, simulation-view GIFs, HSiKAN-architecture-as-hypergraph, model-composition-as-hypergraph. Two viz tools already built; DDPG + arch-hypergraph still to do."
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

User-requested queue (going to sleep 2026-06-24 ~03:40), to run **after** the overnight Galambos BC+PPO run
(`reports/2026-06-24-galambos-bc-ppo.log`, ~07:15) frees the machine — torch/numpy can't run concurrently with
it (OpenBLAS OOM under memory pressure; cannot auto-resume past the run — needs a fresh prompt).

**1. DDPG.** "When over with this go with DDPG." BC→DDPG (off-policy, ~250× sample-eff vs PPO per
[[project-rl-algorithm-roadmap]]). Earlier rec: FANUC BC→DDPG over plain PPO (FANUC PPO already hit 0% greedy
place). Also Galambos BC→DDPG (current run is BC→PPO; PPO refined cell-1 only 21%→25%, the control-hard ceiling).
NOTE wiring: BC targets `ActorCritic.action_mean`; DDPG actor is a `DeterministicActor` — needs a small BC-to-DDPG
adapter (test before queue). `galambos_bc.py` now has `--save` to persist policies.

**2. Simulation views / GIFs — BUILT, run in morning.** `render_planar_gifs.py --demonstrator` renders the
scripted demonstrator's SUCCESSFUL deliveries (top-down cam, timestamp auto-baked bottom-right via
`evaluate.render_episode_gif` default `stamp=None`) → `reports/gifs/demonstrator/demo_seed_<s>_goal.gif`. Trained
policies render via `--checkpoint ... --run <name>`. Static-clean; untested (machine busy).

**3. HSiKAN architecture as a hypergraph — TO DO.** Visualize the HSiKAN *network* as a hypergraph. The policy
already round-trips state_dict ⇄ `.hymeko` (weights = star-expansion incidence; [[reference-policy-as-hymeko-storage]],
`policy_store.py`). Plan: save a trained policy → `.hymeko` → render in the WASM editor (hypergraph3d, edge-on-edge;
[[reference-editor-hyperedge-on-hyperedge]], [[project-editor-mdp-project]]). Needs torch (load policy) → morning.

**4. Model composition as a hypergraph — BUILT, run in morning.** `hymeko_rl/hymeko_compose.py` walks the
`@"…"` import closure → tree + bill-of-materials + **`--dot`** emits the composition as a HYPERGRAPH (each model
a vertex; each model's imports a composition hyperedge via a hub node = star-expansion of "X composed of {…}";
hypergraph-of-hypergraphs / bundle-of-bundles). `python -m hymeko_rl.hymeko_compose <model> --dot out.dot` then
`dot -Tsvg`. Static-clean; first run deferred (numpy import OOMs under the run).

All four are recorded so the session can compact without losing the list. Built: #2 (GIFs), #4 (composition viz).
To do: #1 (DDPG + adapter), #3 (arch-hypergraph). Run order in morning: parse BC+PPO results → run #4 + #2 →
build/run #1 → #3.
