---
name: project-fsm-structured-rl
description: "BIG idea — RL optimizes/fits an expert-designed DECLARATIVE (concurrent) state machine instead of learning flat; HyMeKo is the FSM language; resolves the \"full pick-place won't converge\" wall and is likely the article pivot"
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

User's fundamental reframe (2026-06-23): the core problem with flat RL is that it must discover BOTH the task
**structure** (phases: approach→grasp→lift→transport→release + transition guards) AND the continuous
**control**, from one scalar reward — the hard-exploration wall we watched fail (PPO learned approach+contact,
never the lift→place; we were patching the missing structural prior with reward hacks: bin walls, pre-grasp
disturbance penalty, curriculum — backwards). The fix: **give RL the expert-designed state machine and let it
optimize WITHIN it.** HyMeKo describes state machines — including **concurrent** ones (arm-motion FSM ∥ gripper
FSM ∥ safety monitor) — natively.

**Why it's the right call:** grounded in established paradigms — **reward machines** (Toro Icarte et al.,
ICML'18: an automaton over task phases exposes structure → per-state credit assignment) and **options /
hierarchical RL** (Sutton–Precup–Singh: FSM states = options with termination conditions, RL learns intra-option
control). The literature does single/sequential machines well and **concurrent** ones awkwardly (parallel reward
machines = active, messy). **HyMeKo as a declarative language for CONCURRENT task automata that structure RL is
the contribution** — and it's the user's SysML/state-machine wheelhouse.

**How to apply:**
- The scripted IK expert is ALREADY an implicit FSM (REACH→DESCEND→GRASP(dwell)→LIFT→TRANSPORT→RELEASE with
  guards grasp_hold/lift_xy/thresholds) hidden in Python if-ladders. Step 1: make it a declarative HyMeKo
  (concurrent) FSM (states, guards, per-state sub-goal). Step 2: RL optimizes the LEAF controllers (HSiKAN per
  state, or one HSiKAN conditioned on active state) to each state's dense sub-goal. Step 3 (optional): RL tunes
  transition guards. Start LEAF-ONLY (cleanest + most auditable).
- The place-won't-converge problem DISSOLVES: LIFT state reward = raise grasped object; TRANSPORT = reduce xy to
  target — each dense/local/learnable. No bin/disturbance bribes needed; structure forbids the cheat.
- Subsumes the env-design + HTL threads: bin / two-tables / disturbance become scene declarations OR states/guards
  in the machine. Same realization as the HTL declarative-env idea, arriving from two directions.
- **Thesis upgrade → likely article pivot:** "declarative concurrent state machines as accountable RL controllers
  with HSiKAN leaves." Makes "structurally accountable AI" concrete: a declared, inspectable task automaton
  (verify the safety monitor is a real concurrent state) with learned-only leaves. Auditable RL.

**Open design Qs for the plan:** (1) learned vs declared — leaf controllers only (start here) vs also transition
guards; (2) one state-conditioned policy (shares HSiKAN trunk, cf [[project-actor-critic-shared-reasoning]]) vs N
per-state policies (cleaner accountability); (3) reward-machine vs options framing (reward-machine = cleaner
accountability narrative). Needs a §2 4-artifact plan before code; fold in the HTL/declarative-env consolidation.

**HTL-robustness-as-reward PoC (2026-06-26, DONE+tested, non-core):** a step toward this — express the galambos
task as ONE geometric temporal-logic formula and use its robustness ρ as the reward. The thesis (durable): an
STL/HTL predicate `x≥θ` has robustness ρ=v(x)−θ = a SIGNED GEOMETRIC MARGIN, so one declared formula yields BOTH
the dense reward (ρ) AND the monitor verdict (sign ρ) — unifies reward.py + hymeko_monitor/HTL into one artifact
(ties [[project-hymeko-aggregation-semantics]], [[project-gauge-holonomy-signed-hsikan]]). Built: `hymeko_rl/htl_reward.py`
(`HtlRewardSpec` duck-types `RewardSpec.evaluate` → drops into the env `reward_spec=` seam, NO env change),
`data/robotics/galambos_spec.htl`, A/B harness `exp_htl_reward_ab.py`, 6 tests. **REUSES the existing non-core HTL
evaluator `signedkan_wip/src/htl/` (parse/robustness_at/HtlMonitor/HypergraphEvent) — do NOT rebuild it**; bridge =
the `dashboard_node.py` sys.path pattern; the global predicate `_REGISTRY` is BYPASSED (signals via
`event.scalar_signals`). Key honest scoping: per-step reward = `robustness_at` (instantaneous, MARKOVIAN — valid for
replay buffer); the temporal ρ over a rollout = NON-Markovian reward machine = THIS memory's scope, kept as the
per-episode VERDICT only. Dense leaves not binary (AND=min would let a binary in_zone constant mask the dense terms;
min auto-curricula's approach→deliver). Follow-up: smooth/soft-min (LSE/AGM) if hard-min bottlenecks; `.hymeko`
monitor{} block is the CORE-gated grammar step. Plan+report `2026-06-26-htl-reward-poc`.

Builds on [[project-kato-collaboration-grasping]] (declarative reward parity done), [[project-hymeko-rl-phase2-debug]],
[[project-hsikan-geometric-attention-berge]], [[project-rl-algorithm-roadmap]] (off-policy for the leaves). The
hand-tuned env thrash (bin/two-tables/disturbance, 2026-06-23) is what motivated this — patching structure with
reward is the anti-pattern this fixes.
