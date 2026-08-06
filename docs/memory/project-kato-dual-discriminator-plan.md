---
name: project-kato-dual-discriminator-plan
description: "Kato's next direction — collaborative k-agent Galambos + dual-discriminator (HSiKAN deliberative / MLP+CliffordFIR reflexive) policy + edge-weighted hyperedges; 4-artifact plan on disk"
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

Kato (2026-06-24) proposed three coupled changes; the 4-artifact plan is at
`docs/plans/2026-06-24-kato-collab-dual-discriminator/` (plan.tex/.pdf/.tikz/.mmd, all compile/validate):

1. **Collaborative k-agent Galambos** — each finger an agent; CTDE (shared centralized critic on the coin
   shared-hyperedge + parameter-shared per-finger actors so it scales to k). User directive: define the arm
   **once as a template** (`planar_finger.hymeko`) and instantiate k times — zero duplication, real k-scaling.
2. **Dual-discriminator backbone** — HSiKAN = deliberative ("complex") discriminator; MLP = reflexive
   discriminator on a short temporal window with a **CliffordFIR** geometric filter; coupled by a message-passing
   sync layer. System-1/System-2.
3. **Edge-weighted hyperedges** — user directive: weights live on the hyperedge **arcs** (incidence), NOT as node
   attributes. Reward bundle arcs carry term weights; HSiKAN A± becomes weighted (not binary 0/1).

**Non-obvious facts that shaped the plan (don't re-derive):**
- **CliffordFIR already exists** at `signedkan_wip/src/sequence/clifford_fir.py` and a dual-stream prototype at
  `dual_router.py` (`DualPathSeqBlock`) — port, don't rebuild. Same for AgentSpec, affect.py, the dual-context
  `hymeko_robot_dual.hymeko` pattern, `_BACKBONES` registry, policy_store (weighted incidence ⇄ .hymeko).
- **Arc weights ALREADY parse (no CORE edit):** confirmed in `parser/src/hymeko.lalrpop` — a signed arc ref's
  optional annotation value (`OptValue`) admits any `Value` incl. `Value::Num`, so `(+ approach 4.0, + pull 1.0)`
  parses today (weight → `RefAtom.anno.value`); same mechanism as the joint transform `[[pos],[rpy]]` (a List anno).
  Only the non-core readers (reward lowering, HSiKAN incidence builder, Python bridge) must consume the value. The
  earlier "parser-CORE risk" was unfounded — user corrected it 2026-06-24.
- **CliffordFIR needs a temporal window** → frame-stack the obs (a contract change to the env/obs space).

**RATE-ASYMMETRY (measured 2026-06-25, folded into the plan):** single-step CPU latency HSiKAN ~2.4ms/420Hz vs
MLP ~0.16ms/6400Hz = ~15× → the two discriminators run at **two timescales**: fast reflex every step + slow
HSiKAN every N steps, held as a conditioning signal. Revised fusion decision to "rate-asymmetric async coupling"
+ a new decision: cadence N (4–16). The biological motor hierarchy. Plan PDF rebuilt.

**PROTOTYPE BUILT + RUN (2026-06-25):** `hymeko_rl/dual_rate.py` (`DualRateController` reflex⊕held-deliberation,
context-optional so drop-in for `behaviour_clone`; `RateAsymmetricLoop` cadence; `build_dual_rate`); 5 tests, ruff/
mypy clean. Discriminating experiment `hymeko_rl/exp_dual_rate.py` (BC dual vs mlp vs hsikan on Galambos, 3 seeds,
dual at N=1/4/8) → report `reports/2026-06-25-dual-rate-galambos.md`. **RESULT = TIE on the architecture** (means
mlp 0.139 / hsikan 0.111 / dual_N1 0.139, stds 0.04–0.11 swamp all gaps) — **reproduces the Galambos
no-structural-leverage tie** [[project-galambos-hsikan-tie-rootcause]]; Galambos CANNOT discriminate the
dual-discriminator. Rate-asymmetry signal survives: N=4≈N=1 (compute saving ~free), N=8 degrades (0.069). **Lesson:
the architecture benefit is UNTESTED until a structural-leverage task exists** — do the coin/zone-as-hyperedge A/B
(from the tie memory) or an FSM-structured task [[project-fsm-structured-rl]] BEFORE reading any dual-loop verdict.
Mechanics (fusion/cadence/compute saving) confirmed; benefit not.

**COLLABORATIVE REFRAME PROTOTYPE BUILT (2026-06-25 overnight, Kato item #1):** `hymeko_rl/collaborative.py` —
`CollaborativeGalambos` (cooperative multi-agent VIEW over PlanarGraspEnv: one agent per arm, per-arm action
split via `arm_action_partition` by actuator `_left`/`_right` suffix → `(slice(0,2),slice(2,4))`, shared graph
obs, **team reward** = the existing global reward — NO env clone) + `CTDEActorCritic` (one shared HSiKAN backbone
→ per-arm action heads + 1 centralized critic; `action_mean(obs)`=concat full action so it's drop-in for
`behaviour_clone`/`eval_delivery`) + `build_collaborative`. 5 tests pass, ruff/mypy clean. Realizes
[[project-actor-critic-shared-reasoning]] (shared reasoning + per-agent heads). Discovery confirmed ZERO prior
multi-agent scaffolding (no duplication). BC functional-sanity harness `hymeko_rl/exp_collaborative.py` (single
HSiKAN vs collab CTDE) staged, run pending. **NOT YET BUILT = a real CTDE TRAINER** (train_ppo is single-agent;
per-agent rollouts/advantages needed) — that's the next piece to make multi-agent training actually exercise the
structure; BC only proves the reframe is sound/drop-in. Also in flight: task_graph (coin/zone-as-hyperedge)
dual-rate A/B under BC `reports/2026-06-25-dual-rate-taskgraph.log` (the NOT-ruled-out structural-leverage test —
BC delivers, unlike the ruled-out PPO-from-scratch case [[project-galambos-hsikan-tie-rootcause]]).

**FLAT-PPO REFINE COLLAPSES THE CTDE (confirmed 2026-06-25, 3 seeds).** BC collab CTDE = 0.208 delivery (good,
⅓ fewer params than single). PPO-refine via the standard `train_ppo` (treats the per-arm-head action as one flat
vector) COLLAPSED it: [0.0, 0.125, 0.0] mean 0.042; single-HSiKAN held (~0.236 ≈ its BC 0.25). So flat PPO/off-
policy refine corrupts structured multi-head policies (same fragility as the off-policy divergence on ddpg/SAC).
DON'T re-try flat refine on the CTDE — the **BC collaborative (0.208) is the keeper/demo**; strengthening it needs a
REAL CTDE trainer with PER-AGENT advantages (not flat PPO). Same gap quadruped needs (a stable structure-aware
learner). Demo GIF = the BC version (`reports/gifs/collaborative/coin_toss_collab_ctde.gif`), NOT the refined one.

**3 open decisions flagged for Kato (recommendations):** MARL scheme = CTDE (prototype uses it); fusion =
rate-asymmetric async coupling (revised); CliffordFIR = on the reflexive branch (+frame-stack); + cadence N.

**Why:** Kato's ideas land almost entirely on existing scaffolding; the value is reuse + the arm-template/edge-weight
DOD refactors. **How to apply:** plan is written but NOT approved-to-code; gate only on Kato's 3 decisions
(arc-weight syntax already confirmed, no parse-test gate). Builds on [[project-kato-collaboration-grasping]], [[project-actor-critic-shared-reasoning]],
[[project-hsikan-geometric-attention-berge]], [[project-cayley-rotor-idea]], [[project-galambos-hsikan-tie-rootcause]]
(coin-as-grasp-hyperedge is the tie fix), and the FANUC lesson [[project-fanuc-offpolicy-collapse]].
