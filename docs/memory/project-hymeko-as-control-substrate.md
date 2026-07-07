---
name: project-hymeko-as-control-substrate
description: "BIG pivot (Hajdu+Kato 2026-06-27): HyMeKo's real contribution is a DECLARATIVE CONTROL SUBSTRATE (beyond RL, into control theory) — not 'HSiKAN beats MLP' (control results don't support that). Two siblings: (1) reward/task as a runtime-tunable algorithm-agnostic .hymeko DSL; (2) Kato's idea — GENERATE hypergraphs → isomorphic controller configs → benchmark which topology controls which plant best. Plans on disk."
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

**THE SCHOOL, properly stated (2026-06-27, user: "this could be a new school if properly stated"):**
`docs/theory/declarative_topological_control.md` — **Declarative Topological Control (DTC)**, working name. Thesis:
a controller IS a (signed-hypergraph) topology, declared; what it can represent AND control is governed by the
TOPOLOGICAL MATCH between controller and plant — generatable, measurable, gauge-grounded (signs/rotors=connections,
balance/holonomy=invariants), declarable (one DSL). 3 commitments: structure-first / match-is-the-lever / declarative-
&-accountable. The EMPIRICAL LAW = match is load-bearing IN PROPORTION TO TASK HARDNESS (the P1→P2→P2b→P3 gradient:
100× → 3% → 2-7% → predicted-decisive). 4 falsifiable predictions incl. "a topology invariant predicts the control
cost" (open) + refutation condition (collapses to 'use a big network' if capacity-matched perf is topology-independent).
HONEST: toy scale, a PROGRAM not a theorem; the 3 artifacts that earn "school" = Phase-3 decisive under-actuated result
+ one invariant→performance law + cross-optimizer parity figure. Synthesises SYSTEM_ENGINEERING_VIEW + gauge doc +
reward-DSL hypothesis + the isomorphic-controllers phases.

**REAL-PLANT TURN (2026-06-27, user: "maybe it's time to test it right? On MuJoCo?"):** plan
`docs/plans/2026-06-27-mujoco-designed-control/` (4 artifacts). User's thesis = a combinatorial-design (Steiner/
sunflower) HSiKAN controller needs NO alpha-mixing (the design GIVES the features/cycles/walks), more robust/
accurate, bounded-time control on LARGE systems. Precise: alpha-mixing = learned softmax-over-ARITIES (signedkan_wip/
ac_hsikan); a design FIXES the arity/coverage → α redundant. RL HSiKAN (hymeko_rl/policy.HSiKANBackbone) uses
arc-weights+aggregation (highway/residual) NOT arity-α, takes hg_state= explicitly. KEY DESIGN DECISION = AUGMENT
the plant's kinematic HypergraphState with design hyperedges over body indices (with_task_hyperedges) — base
vertices keep obs [q,qdot], design hubs add coverage; pad obs with zeros for hubs. WIRING SMOKE PASSED: design
augments inverted_pendulum kinematic graph (2 bodies→+1 hub), HSiKAN forwards finite. Infra all present:
inverted_pendulum_env (unstable, rung1 harness-validation), quadruped_env+quadruped.hymeko (rung2 STANDING =
headline, rich bodies = meaningful design), MuJoCo 3.9. Arms: kinematic/Steiner-aug/sunflower-aug/MLP/LQR. Metric
that matters = DISTURBANCE REJECTION on the unstable plant (not nominal return). HONEST: inverted pendulum too tiny
(2 bodies) for a real design test — rung1 validates harness, thesis tested at rung2 quadruped. NEXT = exp_designed_
control.py + training smoke beside baseline, then rung1, then quadruped standing.
**SMOKE PASSED (2026-06-27 21:58): harness VALIDATED end-to-end on real MuJoCo.** exp_designed_control.py built
(DesignAugmentedEnv wraps kinematic hg + pads obs; reuses run_config/build_sac/train_sac/evaluate — only the
wrapper is new). Pendulum smoke: kinematic ret 27.8, sunflower_aug ret 26.5 (nv 2→3, 100% success 0 deaths), mlp
27.3 — ~equal as EXPECTED (2-body pendulum = trivial design = harness-validation NOT thesis). Bug found+fixed:
this gymnasium Wrapper doesn't auto-forward base-env attrs (max_steps) → added guarded __getattr__ delegation.
**KEY: QuadrupedGoalEnv has exactly n_vertices=9 → Steiner S(2,3,9) FITS EXACTLY** (affine plane, every pair of
9 bodies coupled once), action=(8,). So quadruped = ideal rung-2 plant. ONE PIECE TO BUILD: QuadrupedGoalEnv is
LOCOMOTION (walk-to-goal); user wants STANDING → need a stay-upright reward (in .hymeko per the rule). Then rung2 =
kinematic vs Steiner-aug vs sunflower-aug vs MLP on quadruped standing, metric=disturbance rejection. Baseline at
8/9 → machine nearly free for the full run. Surrogate becomes meaningful here
(designs differ in achievable control on unstable plants). Machine: smoke fits beside coin-toss baseline, full
multi-seed waits (no 4-worker page wedge).

**The reframe (2026-06-27, after the control results showed HSiKAN ties/loses the MLP on robot delivery and is
noise single-seed):** stop selling HyMeKo as a better RL *optimizer/architecture*. Its defensible contribution is
as a **declarative substrate for control** — author/tune/generate the reward AND the controller structure as an
inspectable, runtime-tunable model, across optimizers and topologies, spanning RL *and* classical control theory.
Two complementary deliverables (both planned this session):

**(1) Reward-DSL hypothesis** (`docs/theory/hymeko_reward_dsl_hypothesis.md`). H1 algorithm-agnostic (one .hymeko
reward drives PPO/TD3/SAC/DDPG — ALREADY built: `offpolicy_eval._ALGOS` + shared `RewardSpec.from_hymeko`); H2
runtime-tunable (reward change = a `.hymeko` arc-weight edit, no code/recompile, git-tracked — the galambos grind
demonstrates it, rule `feedback-reward-definition-in-hymeko`); H3 zero perf cost (same Σwᵢtermᵢ); H4
inspectable/accountable. Experiment: E1 PPO/TD3/SAC on the same .hymeko reward = parity figure (0 reward LOC); E2
tuning-velocity (grind diffs); E3 hardcoded-Python counterfactual. It's an MDSD/software-engineering claim — keep
SEPARATE from the structural-prior claim.

**(2) Kato's isomorphic-controllers program** (`docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/`,
4 artifacts, compiles). A hypergraph H = a controller's interconnection structure. GENERATE typical topologies
{H_i} (chain/ring/star/tree/grid/small-world/random/kinematic) → INSTANTIATE isomorphic controllers C(H_i)
[learned = SignedKANBackbone over A±(H), ALREADY built; structured P2 = u=-Kx with gain sparsity = H, the
control-theory leg] → BENCHMARK across scenarios → a TOPOLOGY→PERFORMANCE map (which topology controls which plant
best). Load-bearing definition + guard: isomorphic H'=π(H) must give SAME performance (permutation-equivariance —
the well-definedness test); interesting variation is across NON-isomorphic topologies. Seed evidence (measured):
the chain probe — perf scaled 4.5→41× with chain length (`2026-06-27-chain-and-spiral-toys`). Phase 1 ETA ~1 day
(reuse HypergraphState + SignedKANBackbone + structural_probe harness; NEW topology_zoo.py + controller_bench.py;
no graph generators exist in hymeko_graph). P2 = structured-K/distributed control per topology (the "beyond RL"
core); P4 = generate H_i in HyMeKo + EMIT controllers (MDSD tie to deliverable 1). Risks: isomorphism must be
well-defined (equivariance test first); cheap supervised proxy must correlate with closed-loop (gate it).
**PHASE 1 DONE (2026-06-27, `reports/2026-06-27-topology-performance-map.md`):** topology_zoo.py (8 families
chain/ring/star/tree/grid/small-world/random/complete @ N=9) + controller_bench.py (plant×controller MSE map,
reuses structural_probe harness + SignedKANBackbone), 16 tests pass. RESULT: **matching topology wins 8/8** —
each plant best controlled by the controller whose topology mirrors it (diagonal 0.001-0.028 vs off-diag up to
1.6). KEY NUANCE: it's STRUCTURAL MATCH not capacity — `complete` (densest, 36 edges) is NOT a universal winner,
only best on itself; denser ≠ better, matching = better (strong form of Kato's hypothesis). WELL-DEFINEDNESS:
my first statistical invariance guard was CONFOUNDED (50% relative gap = near-zero-denominator artefact, didn't
permute input); replaced with EXACT equivariance check (identical weights, permute graph+input together) →
residual 1e-8 = controller exactly permutation-equivariant → map structure is real. HONEST LIMIT: supervised
structural proxy, NOT closed-loop control yet; the control-task-correlation gate is the next (machine-bound) step.
**PHASE 2 DONE (2026-06-27, `reports/2026-06-27-structured-control-phase2.md`):** structured_control.py — the
`u=-Kx` leg, gain-sparsity=H, structured-LQR via projected gradient (scipy CARE/CALE), 10 tests. RESULT: matched
topology = best sparse controller for ALL 7 plants (diagonal dominance like Phase 1) BUT margins TINY (2-5%
worst-penalty). HONEST CONTRAST = the real finding: topology governs REPRESENTATION strongly (Phase 1: 100×) but
control-of-BENIGN-plants weakly (Phase 2: ~3%) — because fully-actuated stable symmetric LTI barely needs
off-diagonal feedback (same lesson as RL: structure load-bearing only when the problem is hard). BUG found+fixed
per contract: projected-gradient from K=0 diverged near stability boundary (complete read rho 1.5-11.9, impossible
since superset) → warm-start from masked-CARE-optimum K*⊙mask → complete=1.000 exact (regression test). Oracle
full-mask→J* exact. PHASE 3 = the regime where topology STRONGLY gates control: under-actuation (sparse B, route
through topology) + open-loop instability (only covering topologies stabilize → rho=∞); needs stabilizing init
(Lin-Fardad-Jovanović augmented Lagrangian). Don't overclaim Phase 2 — it's the validated framework + honest
weak-dependence result.
**PHASE 2b DONE (2026-06-27, `reports/2026-06-27-structured-mpc-phase2b.md`):** structured_mpc.py — added MPC.
Key honesty: MPC≡LQR unconstrained, so added INPUT SATURATION (u_max=0.6, x0=3·1); topology enters as the MPC
PREDICTION MODEL coupling (MPC is model-predictive). RESULT: matched model = best for ALL 8 plants (CLEAN diagonal,
incl complete→complete — cleaner than Phase 2's near-ties); mismatch penalty 2-7% (modestly > Phase 2's 2-5% — the
constraint makes topology bite more, hypothesis confirmed DIRECTIONALLY but still modest on benign plant); MPC beats
sat-LQR ~1%. Oracle: unconstrained matched MPC = discrete-LQR exact. 7 tests. THE COHERENT GRADIENT for Kato:
representation strongly topology-dependent (P1 100×) → unconstrained control weakly (P2 ~3%) → constrained MPC
slightly more + clean diagonal (P2b 2-7%) = "structure matters more as the control problem gets harder." Phase 3
(under-actuation + instability) is where it should become decisive.

**How to apply:** both are the likely PAPER pivot (HyMeKo as accountable declarative control, beyond "another KAN
variant"). Ties [[project-hymeko-aggregation-semantics]], [[project-fsm-structured-rl]], [[project-mdsd-reuse-and-docs]],
[[feedback-reward-definition-in-hymeko]]. Next concrete: run E1 (reward parity across PPO/TD3/SAC) once the galambos
grind is going; build topology_zoo Phase 1 (CPU-light, parallel-safe). NOT yet built — plans only.
