---
name: project-hsikan-loses-possible-bug
description: "OPEN #1 (2026-06-26): the systematic 'HSiKAN loses to MLP' RL pattern (quadruped C3 -30 vs MLP -16.5, + ties everywhere) is SUSPECT — likely an obs/hypergraph WIRING bug, not an architecture verdict. User read the GIFs as HSiKAN 'fighting its own structure'. Audit before trusting any HSiKAN-vs-MLP result. Full handoff on disk."
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

**The suspicion (user, watching the overnight GIFs 2026-06-26):** HSiKAN underperforms a params-matched MLP on
*every* RL task measured (cartpole tie, galambos tie/MLP-better, task-graph MLP-better, **quadruped C3: HSiKAN
−30.1 vs MLP −16.5**, 2 tight non-overlapping seeds). A real architecture should win *somewhere*; a uniform loss
+ the GIF symptom of the policy "fighting its own structure" points at an **implementation bug in the HSiKAN
obs/hypergraph path**, not at the architecture being worse. **Do NOT report any HSiKAN-vs-MLP verdict as settled
until this is audited.**

**Bug-hunt (priority order), epicentre = the structure→obs mapping:**
1. obs↔structure alignment: `hymeko_rl/env/*_env.py::node_features()` vs `hymeko_rl/hypergraph_state.py`
   (`dense_signed_adj`, star_expansion, **vertex ordering**). Misaligned vertices↔feature-rows = HSiKAN gets
   garbage structure while the flat-obs MLP is unaffected = the exact "fights structure, MLP wins" signature.
2. degenerate/empty adjacency rows: `_row_normalise` clamps degree≥1; isolated/hub vertices (already flagged for
   task_graph all-zero hubs) pass features unchanged/zeroed. Check the quadruped 14-vtx graph.
3. quadruped graph encoding: `_quadruped_env` in `offpolicy_eval.py` (~L78) — is the branching topology (the
   thing C3 rewards) actually encoded?
4. HSiKAN forward: `signed_kan/backbone.py` + `policy.py` — behavioural check vs hand-calc (hsikan_diagnose finds
   blow-ups, NOT silent-wrong-but-finite).
**Discriminating test:** a toy task where structure MUST help (cycle parity / hidden path the MLP can't see). If
HSiKAN can't win there → bug is in the HSiKAN path, not the tasks.

**AUDIT PROGRESS (2026-06-26, task-by-task order cartpole→collaborative→arm→pick-place→quadruped):**
- **Cartpole CLEARED — no universal backbone bug.** SAC 15k, params-matched: HSiKAN **200/200**, MLP **200/200**
  (RSS 660MB). Wiring statically clean (signed adj + feature↔vertex alignment + healthy forward). The HSiKAN path
  learns end-to-end → any §0 problem is **per-task wiring**, not the backbone. (Cartpole is the negative control;
  a tie/ceiling is the CORRECT null, never bug evidence.)
- **Collaborative (Galambos): a REWARD-wiring bug found+FIXED** (separate from the HSiKAN-vs-MLP question — it
  shapes both backbones equally, explains the low 0.208 delivery, NOT the gap). The dense `grasp_approach` term
  measured the **nearest arm BODY ORIGIN** (`compute_planar_metrics._nearest` min over base/upper/lower), not the
  fingertip: emitted arm ships NO tip site → used the elbow (~0.14m short); coin near centre → nearest body is the
  **immovable base** → zero gradient (user: "robot position rewarded, not the tip"). FIX = inject massless
  `tip_{side}` sites (`with_fingertip_sites`, planar_grasp_env.py) + metric = tip-dominant blend 0.75·tip+0.25·elbow;
  also repairs the BC demonstrator's `_extract_arms` tip_site=-1 → read target_zone. Dynamics bit-identical (sites
  massless). Plan `docs/plans/2026-06-26-galambos-fingertip-reward/`, report same-date slug, 26+74 tests pass.
- **Arm-movement (arm6dof) CLEARED.** 7-vtx chain base_link→link_0..4→tool, correct +1/-1 signed adj,
  joint features scatter onto child vertices exactly (`_scatter_joint_scalar` uses jnt_bodyid-1, matches
  hg convention), NO zero-degree/all-zero rows, forward finite, diagnose healthy.
- **Quadruped CLEARED — and the suspicion's premise was WRONG.** It is **9 vertices not 14** (handoff
  error): torso + 4 thighs + 4 shins. Branching topology **correctly encoded**: torso has exactly 4
  down-chain children (thigh_fl/fr/bl/br), each thigh→shin — the precise structure C3 rewards. NO
  isolated/zero-degree vertices, NO all-zero feature rows (the §0 #2 "degenerate hub" suspect is ABSENT),
  feature↔vertex aligned (torso=[dx_to_goal,vx], legs=[qpos,qvel]), forward finite, diagnose healthy.
- **Pick-and-place CLEARED** (PickPlaceEnv + fanuc_pick_env): 9-vtx base→link_0..4→tool→{finger_l,finger_r},
  no isolated/all-zero rows, healthy forward.

**STRUCTURAL PROBE VERDICT (2026-06-26, `hymeko_rl/structural_probe.py`, report `2026-06-26-structural-probe`):
backbone is NOT broken — hypothesis (b) FALSIFIED.** Supervised RL-free probe on a fixed signed graph, two
targets, params-matched HSiKAN vs MLP, data-scaling sweep. Findings (SURPRISING, the prediction was wrong):
HSiKAN's advantage is REPRESENTATIONAL and GROWS with data (not sample-efficiency). It is **POOLING-FIRST**:
on a structure-free separable target `Σtanh(x_v)` HSiKAN beats MLP up to **52×** (its per-node-activation +
mean-pool = a Deep-Sets prior); the SIGNED-2-hop advantage is real but secondary (**3.17× @ n=512**, tie/loss
below ~256 because a FIXED small graph's B² operator is absorbable by the MLP's first layer). So the robot tie
is NOT incompetence — it's a **mismatch**: robot value/policy under current rewards isn't a pooled/signed-
structural function, OR (NEW HYPOTHESIS) the **mean-pool readout discards cross-joint coordination control
needs** (pooling helps separable aggregates, hurts coordination; ties [[project-fuzzy-defuzzification-heads]]).
USER'S REFRAME VINDICATED: don't drop HSiKAN, give it structure to cash in. **NEXT robot-side tests:** (1)
structural reward (galambos_taskgraph hyperedges / HTL structural predicates) — `2026-06-26-htl-reward-poc`;
(2) **readout ablation** — swap mean-pool for concat/attention/per-node, re-run a robot task; if the tie flips,
pooling (not the backbone) was the bottleneck; (3) vary-the-graph probe (incidence=learned) to expose
structural value the fixed-graph ceiling hides.

**READOUT ABLATION DONE — HYPOTHESIS CONFIRMED (2026-06-26, `2026-06-26-readout-ablation`, supervised):** same
backbone, mean-pool vs concat readout, +a node-specific `local` target mean-pool can't isolate. On `local`:
mean-pool MSE 0.0267 vs concat **0.0024 (10.9× better)** vs MLP 0.0021 — **the mean-pool READOUT (not the
backbone, +5% params only) collapses node identity**; on pooled targets (structural/bag) mean-pool is BEST
(it matches aggregates). So the §0 chain CLOSES: wiring clean → backbone correct → **mean-pool readout is the
bottleneck for the node-specific/coordinated info robot control needs** (MLP reads flat per-node obs, never
loses identity). Trade-off: naive concat is WORSE on pooled targets (0.10 vs 0.03) → fix = identity-preserving
+ aggregation-capable readout (ATTENTION pool / concat+pool hybrid), NOT naive concat.

**MULTIDIM READOUT — USER'S INSIGHT, DRAMATIC WIN (2026-06-26, `2026-06-26-multidim-readout`):** user said
"global pooling AND multidimensional output, we have multiple joints anyway" — i.e. the action is per-joint and
the graph is per-joint, so DON'T collapse: each joint's output from its OWN message-passed node embedding.
Tested on a VECTOR target y∈R^N: **pool_expand (collapse→expand) MSE 0.484 vs per_node 0.0001 = 3723× worse**;
**per_node BEATS the flat MLP 33× (0.0001 vs 0.0033) with FEWER params**; per_node_global (per-node + broadcast
mean-pool context = the user's exact phrase) ties for the win, global context ~free. SO: the robot HSiKAN-vs-MLP
TIES were the ACTOR's pool-then-expand readout throwing away HSiKAN's per-node structural signal — give each
joint its own node embedding and HSiKAN WINS outright. Caveat: probe target = HSiKAN's own per-node computation
(B²x), so near-perfect fit partly by construction; architecture claim (per_node ≫ collapse, global ~free) is
target-agnostic; RL run is the real validation. **THE FIX (next):** per-node multidim actor head in
`hymeko_rl.policy.ActorCritic` — replace `actor_mean = Linear(feat_dim, action_dim)` over the POOLED backbone
with a per-node head over `[h_node ; mean-pool(h)]` (`node_activations`), joint→vertex via `_jnt_vtx`; critic
keeps the global pool (scalar V wants an aggregate). Re-run galambos/quadruped vs MLP, params-matched →
prediction: tie flips to HSiKAN win. That completes the architecture story.

**RL PAYOFF RESULT (2026-06-26, `2026-06-26-pernode-actor`, galambos SAC 20k, SINGLE seed, params~matched):**
delivery mlp **0.45** / hsikan_pooled **0.00** / hsikan_pernode **0.20**. **The per-node readout fix is
RL-LOAD-BEARING: 0%→20%** (pooled actor is INERT — collapses the per-joint signal, return −177 = just
penalties; per-node controls + delivers 20% CLEANLY, 0 deaths) — confirms the multidim probe in RL. **BUT
per-node does NOT beat the MLP yet** (0.20 vs 0.45; MLP aggressive=45% but 4 deaths/fumbles, HSiKAN
conservative=0 deaths). So readout UNLOCKS HSiKAN (useless→functional), does NOT make it win. CAVEATS: single
seed (not a verdict per new §3 policy — need multi-seed median/IQR); **highway α-gate was OFF (skip=none)** —
the HSiKAN feature-collection "H" not engaged. NEXT (4-config run IN FLIGHT, exp_pernode_actor_ab now has
pooled/pernode/**pernode_hw**(per-node+highway,~20.4k params)/mlp parallel): does highway close the gap? Then
multi-seed the winner. If still short: exploration (target_entropy/start_steps) + structural reward
(task_graph/HTL). Infra built this session: per-node actor (policy.PerNodeActionHead +
sac.PerNodeSquashedGaussianActor, actor-only, critic keeps pool), actuator_vertices, PARALLEL multi-process A/B
(per new §3 policy, 1 thread/worker), timestamped experiments/<ts>_name/ dirs (experiment_dir+results_to_csv),
skip axis plumbed through build_sac.

**MULTI-SEED VERDICT (2026-06-27, galambos coin-toss, 3 configs × 3 seeds × 30k, the ROBUST confirmation):**
mlp delivery median **0.45** [0.25,0.45,0.45] > hsikan_pernode **0.15** [0.05,0.15,0.15] > hsikan_pooled **0.0**
[0,0,0.2]. Confirms the single-seed: per-node readout UNLOCKS HSiKAN (pooled 0→pernode 0.15) but does NOT close
the gap to MLP. NOT a backbone bug (the toy graph-property tests — structural-probe/chain/coupling-order — HSiKAN
wins by 100×, falsifying hypothesis b). It's hypothesis (a): this control objective isn't a graph property (coin/
zone not graph nodes, [[project-galambos-hsikan-tie-rootcause]]) so the structural prior has nothing to leverage.
Settled honestly: HSiKAN's edge = REPRESENTATION, not easy robot control. Pivot = DTC + DESIGNED structure
[[project-hymeko-as-control-substrate]], the real-plant test uses Steiner-augmented controllers, not vanilla HSiKAN.

**EXPLORATION CONFOUND (user-flagged 2026-06-26, watch the payoff run):** SAC exploration is MODEST —
`SACConfig` `start_steps=1000` (~3% of a 30k run) + `target_entropy=-action_dim` (default, not high) + α
auto-tuned DOWN from there, no α floor. On the hard galambos grasp this can collapse before the two-finger
contact is found. **INTERPRETATION RULE for the per-node payoff A/B:** if all 3 configs (pooled/pernode/mlp)
land at ~0 delivery, that is a tie-AT-FAILURE from under-exploration, NOT evidence about the actor readout —
re-run with more exploration before concluding. Levers: target_entropy less negative (e.g. -0.5·action_dim or
a small positive floor), longer start_steps, an α floor. difficulty=0.3 (coin in easy 2-arm reach) partly
mitigates. Ties to the 2026-06-20 galambos finding ("ent_coef=0 → no exploration pressure", PPO).

**AUDIT VERDICT (2026-06-26): the wiring-bug hypothesis is FALSIFIED across EVERY task.** No obs/hypergraph
misalignment, no degenerate rows, no NaN/inf, healthy backbone forward everywhere; graphs faithfully encode
the kinematics. So "HSiKAN loses/ties to MLP" is NOT a wiring bug. Two hypotheses remain, and they split
cleanly: **(a) genuine parity / structure-not-load-bearing** — the robot OBJECTIVES (reach, balance, deliver)
are not graph-structural properties, so message-passing has no leverage a flat MLP lacks (aligns with
[[project-galambos-hsikan-tie-rootcause]]); **(b) silent-wrong-but-finite backbone forward bug** that
hsikan_diagnose cannot catch. **THE DISCRIMINATING TEST (do this next):** a TOY task whose answer IS a graph
property the MLP structurally cannot see (cycle parity / hidden path over the hypergraph). HSiKAN wins there
→ backbone correct, robot-task ties are REAL (verdict a). HSiKAN loses there → backbone forward buggy
(verdict b). One test collapses the question. The user resisted the architecture-verdict reading ("a real
arch wins somewhere") — exactly right: test "somewhere" = a structure-load-bearing toy, not a robot task
where structure isn't load-bearing.

**Banked this session (DONE, tested, all non-core, CPU):** divergence bug FIXED+confirmed (grad-clip +
reward-norm `hymeko_rl/normalize.py`; 8h/0 divergences); diagnostic-in-loop (`offpolicy_eval` aborts+localises,
`--eval-every`); reward refinement (`coin_pregrasp_still` + `settle` + `pull`1→2, 35 tests); GPU support (additive
`device`, but HSiKAN **launch-bound** → needs `torch.compile`, CPU used); **arm joint limits ±2.8→±4.0**
(collaborative mobility fix). in_zone sweep DECISIVE: bigger success reward HURTS (10→0.25, 50/200→0.04) — keep 10.

**How to apply:** full handoff `reports/2026-06-26-session-handoff.md` (read first in the new chat). Relates to
[[project-galambos-hsikan-tie-rootcause]] (the tie may be the same bug), [[project-unify-hsikan-core]],
[[project-gauge-holonomy-signed-hsikan]] (debuggability is the demonstrated win; architecture superiority is NOT).
Pending: arm-fix validation (`exp_reward_weight_sweep --weights 10` on ±4.0 arm), multi-coin (active-coin design),
fanuc resume (1/4 done).
