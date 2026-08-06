---
name: project-galambos-hsikan-tie-rootcause
description: "Why HSiKAN ties a params-matched MLP on Galambos — root-caused: the coin+zone aren't graph nodes (so structure can't engage the objective) AND neither policy delivers the coin (tie at failure). Fix = put coin/zone in the graph as a grasp/goal HYPEREDGE; discriminating A/B defined."
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

**⏳ THE A/B IS NOW RUNNING (2026-06-27, user: "coin as grasp, shared resource/channel = coin position itself").**
The fix is BUILT (not just defined): `PlanarGraspEnv(task_graph=True)` adds coin+zone+grasp_hub+goal_hub (6→10
vtx), the COIN VERTEX CARRIES ITS XY (the shared coordination channel = coin position, exactly the user's idea),
grasp_hub ties both arms↔coin; `collaborative.py` does the CTDE shared-backbone+per-arm-heads version;
`structural_critic.py` routes value along the arm-coin-zone cycle; test_galambos_task_graph asserts the 10-vtx
shape. WHAT WAS MISSING = the decisive RL RESULT under the honest grasp-gate. Running overnight (bhv0l8f7x):
diag_contact HSiKAN **task_graph vs baseline**, 80k each, grasp-gated reward, metric = grasp_fraction (vs MLP
baseline 0.056). ISOLATES whether coin-as-shared-resource UNLOCKS grasping where reward-shaping was FALSIFIED
([[project-galambos-reward-shaping]] grasp-gate 0.056→0.056). If task_graph grasps and baseline doesn't → STRUCTURE
is the lever (confirms this root-cause + the DTC thesis [[project-hymeko-as-control-substrate]]). If both still
~0.056 → grasping is exploration/scale, need attractor-field/BC/20M. diag_contact now takes --kind/--task-graph.

**Measured (2026-06-23, PPO journal `reports/2026-06-23-stage2-arch-algo.jsonl`, 10 cells).** HSiKAN vs
params-matched MLP on Galambos, PPO, 5 seeds: **tie** (curve-max HSiKAN −224.6±11.2 vs MLP −223.4±13.3, gap
−1.2 ≪ IQR). ALL 10 runs (both backbones, every seed) sit at −215…−233 with **no positive point in any curve** —
a delivered coin spikes the return positive (in_zone +10/step, center +5). So **nothing delivers the coin**: it is
a **tie at FAILURE**, not HSiKAN failing to beat a strong MLP. Both stall in the approach regime (~−1.4/step).

**Root cause (code-grounded, two parts):**
1. **The coin and target are NOT in the graph HSiKAN reasons over.** `PlanarGraspEnv.hg` is built from the ARM
   MJCF only (`planar_grasp_env.py:189`) — 6 vertices = the two arms' links. The coin/zone are composed into the
   scene afterward and never become nodes; they enter as flat per-vertex features, and the decisive **coin→zone
   vector is broadcast IDENTICALLY to every vertex** (`planar_grasp_env.py:268-269`). HSiKAN's only edge is
   message-passing over *structure*, but its structure is robot kinematics, which holds no coin/goal-relational
   signal the MLP doesn't read flat. HSiKAN ≈ "MLP on the same features diffused through a fixed robot graph" →
   zero structural leverage on the objective. No arm–coin–zone relation exists in the topology.
2. **Long-horizon structure (reach→grasp→transport→place) is not given** — flat RL (either backbone) must
   discover the phase sequence from one scalar, the same wall as the FANUC pick-place (0% place). Stalls at
   "approach the coin."

**Fix = put the task objects in the graph as a HYPEREDGE** (the canonical true >2 hyperedge): a **grasp
hyperedge** {fingertip_left, fingertip_right, coin} + a **goal hyperedge** {coin, zone}. Then HSiKAN
message-passes over arm–coin–zone and structure finally engages the objective. Directly uses the
`HypergraphState.star_expansion()` hub-node machinery ([[project-rl-algorithm-roadmap]] Stage 3) and motivates
[[project-fsm-structured-rl]] (decompose the long-horizon credit).

**Discriminating A/B (next experiment):** add coin+zone as graph entities, re-run HSiKAN vs params-matched MLP on
Galambos. HSiKAN beats MLP ⇒ "structure pays off only when task-relevant entities are in the graph" (confirmed).
Still ties ⇒ bottleneck is long-horizon credit (→ FSM), not representation. Caveat: "nothing delivers" is
measured for PPO; SAC (stronger, still running in the overnight campaign) may deliver on some seeds — but the
root cause holds regardless of learner. Connects to [[project-engine-transitive-imports]] (true hyperedges want
the canonical `.hymeko` star-expansion).

**UPDATE 2026-06-25 — task_graph under BC FALSIFIES the structural-leverage hypothesis (it HURTS HSiKAN).**
Ran the not-previously-tested regime: BC warm-start (delivers, unlike PPO-from-scratch) × task_graph=True.
Delivery (3 seeds, 24 eps): baseline N=6 → mlp 0.139 / hsikan 0.111 (tie). taskgraph N=10 → **mlp 0.250 (UP),
hsikan 0.042 (DOWN, flat across all 3 seeds), dual 0.13–0.17**. So putting coin/zone in the graph helps the MLP
(richer flat obs) and DEGRADES HSiKAN — opposite of the prediction. MEASURED: the flip. INFERRED (not isolated):
the coin/zone/hub vertices dilute the arm message-passing under row-normalized signed adjacency (hub rows mix
unrelated nodes); 0.042=1/24 ⇒ near-constant policy collapse. CONCLUSION: hyperedge representation is
necessary-NOT-sufficient AND the current signed-adjacency aggregation doesn't exploit it — it dilutes. Report
`reports/2026-06-25-dual-rate-taskgraph.md`. FOLLOW-UP = isolate the collapse (ablate hub rows / incidence=weighted)
+ the structural critic [[project-actor-critic-shared-reasoning]] attacks it from the value side (per-cycle value
over the arm-coin-zone cycle that exists only with task_graph). Don't re-run plain HSiKAN-vs-MLP on task_graph BC
expecting a win; the open question is WHY HSiKAN dilutes, not whether structure helps as-wired (it doesn't).

**UPDATE 2026-06-24 — coin-placement RULED OUT; the real wall is hard-exploration (→ FSM/demos).** Built the
`task_graph=True` env (coin/zone in graph) + lowered the spawn (the env's `difficulty` knob). Measured PPO-from-
scratch curve-max: difficulty 1.0 → −224, 0.3 → −214, **0.0 (coin 0.071 m from zone center, 3 cm outside the
4 cm zone) → −198 with a FLAT curve** (`-200.8…-198.0` over 18k steps, never positive). So moving the coin into
the lap barely helps and it **never delivers / never even starts learning**. Conclusion: the binding constraint
is NOT reach, backbone, or representation — it is the **two-arm grasp hard-exploration**: pure PPO-from-scratch
(no demos, no task structure) can't discover approach→pinch→pull→place, the SAME wall as the FANUC pick-place
(0% place). The HSiKAN-vs-MLP A/B is therefore uninformative under failing PPO (tie at failure either way).
**Productive next move = declared STRUCTURE ([[project-fsm-structured-rl]]) or DEMONSTRATIONS (BC warm-start, as
FANUC has), NOT more coin-placement / backbone / hyperedge work.** The hyperedge graph (`with_task_hyperedges`,
`PlanarGraspEnv(task_graph=True)`) is built + tested and remains the right representation for WHEN a learner
actually engages the task — it's necessary, not sufficient. (Caveat: prior `project-galambos-reward-shaping`
reached 5/8 goals at some config, so the task IS achievable with the right setup/budget; these PPO-from-scratch
runs at 18–30k steps are not it.)
