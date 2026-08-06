# Declarative Concurrent State Machines as Accountable RL Controllers

**Technical report — design + academic background**
2026-06-23 · hymeko_rl / hymeko_monitor · Kato collaboration, HSiKAN line
Companion plan: `docs/plans/2026-06-23-fsm-structured-rl/` (plan.tex/pdf/tikz/mmd)

---

## 1. Summary

We propose to stop training reinforcement-learning (RL) policies that must *discover* the structure of a
long-horizon manipulation task, and instead **declare that structure as a concurrent state machine, let RL
optimize only the leaf controllers, and declare what to monitor in Hypergraph Temporal Logic (HTL)**. The three
layers — structure, control, specification/monitoring — all live over the existing HyMeKo signed-incidence
hypergraph IR, and three of the four pieces (`hymeko_monitor`, `hymeko_neuro/eval/htl`, HSiKAN + off-policy RL)
already exist. The contribution is the *unification*: a single model from which task structure, learned control,
reward, and runtime-verifiable accountability properties are all read.

This report states the problem precisely, situates it in the literature (reward machines, hierarchical RL,
temporal-logic-guided RL, STL robustness, runtime verification, behavior trees), describes the proposed
architecture, maps it onto the codebase, and lays out the experimental design and the open questions.

---

## 2. The problem: flat RL co-discovers structure *and* control

A Markov policy trained on a scalar reward must simultaneously learn (a) the **task decomposition** — that
manipulation proceeds through *reach → descend → grasp → lift → transport → release*, each with its own
sub-goal and termination condition — and (b) the **continuous control** inside each phase. The first is a
combinatorial, sparse-feedback problem; the second is a smooth, dense one. Forcing one optimizer to solve both
from one scalar is the canonical hard-exploration failure.

We measured it directly. On the FANUC LR Mate-config top-down pick-and-place
(`reports/2026-06-23-fanuc-lrmate-top-down-grasp.md`), a behavior-cloning-warm-started, curriculum-shaped PPO
agent improved its return from −321 to +93 (at curriculum difficulty 0.87) yet **placed 0 % of objects under the
greedy policy at every difficulty**. Diagnostics showed the learned policy approaches and contacts the object
(the early, dense-reward phases) and then *pushes* it toward the target rather than lifting — it never discovers
the lift→transport→release sub-sequence. Our reflex was to *bribe* the missing structural prior with reward and
scene engineering: an open-box target so pushing fails, a pre-grasp disturbance penalty, a difficulty curriculum.
Each is a patch for the same underlying defect: **the structure is known to the engineer but withheld from the
learner.** The principled fix is to give the learner the structure and ask it to fill in only the control.

---

## 3. Academic background

The proposal sits at the intersection of four established lines. None of them, to our knowledge, has been
expressed over a signed-incidence *hypergraph* IR, nor with *native concurrency*, nor unified with a learned
*structural* leaf controller (HSiKAN). That intersection is the novelty; the individual ingredients are mature
and provide the formal footing.

### 3.1 Reward machines and temporal-logic task specification

A **reward machine** (Toro Icarte, Klassen, Valenzano & McIlraith, *ICML* 2018; extended *JAIR* 2022) is a
finite-state automaton whose transitions are labelled by environment events and emit reward. It exposes the
task's temporal structure to the agent, turning one opaque scalar into a sequence of per-state sub-rewards, and
admits algorithms (QRM, counterfactual experience for reward machines) that provably exploit the structure for
faster learning. Our "expert-designed state machine that RL fits" is precisely a reward machine with **learned
leaf controllers**.

Reward machines are commonly *compiled from* a temporal-logic specification. **Camacho et al.** (*LTL and Beyond:
Formal Languages for Reward Function Specification in RL*, IJCAI 2019) show LTL/LDL$_f$ formulas translate to
reward machines. The broader **temporal-logic-guided RL** literature includes **Li, Vasile & Belta**
(*RL with Temporal Logic Rewards*, IROS 2017), **Aksaray et al.** (*Q-Learning for Robust Satisfaction of STL
Specifications*, CDC 2016), **Hasanbeig, Abate & Kroening** (logically-constrained RL / LCRL), and
**Jothimurugan, Bansal, Bastani & Alur** (*Compositional RL from Logical Specifications*, DiRL, NeurIPS 2021),
which decomposes a spec into a DAG of sub-tasks and learns a policy per edge — close in spirit to per-state
leaves. The takeaway: *specifying the task in a temporal logic and learning control within the induced
structure is a known, effective recipe.*

### 3.2 Hierarchical RL and the options framework

The **options framework** (Sutton, Precup & Singh, *Between MDPs and semi-MDPs*, Artificial Intelligence 1999)
formalizes temporally extended actions: an option is a sub-policy with an initiation set and a termination
condition — exactly a state of our machine. **Option-critic** (Bacon, Harb & Precup, AAAI 2017) learns options
end-to-end. We deliberately take the *opposite* design point for accountability: the option set and termination
conditions are **declared, not discovered**, and only the intra-option control is learned. This trades some
autonomy for auditability — the central design choice of this work.

### 3.3 Signal Temporal Logic and quantitative robustness

**STL** (Maler & Nickovic, *Monitoring Temporal Properties of Continuous Signals*, FORMATS 2004) and its
**quantitative robustness** semantics (Fainekos & Pappas, TCS 2009; Donzé & Maler, *Robust Satisfaction of
Temporal Logic over Real-Valued Signals*, FORMATS 2010) give a real-valued degree of satisfaction $\rho$:
$\rho>0$ means satisfied with margin $\rho$; $\rho<0$ quantifies violation. This is what turns "the policy is
accountable" from a slogan into a measurement — we can report *which* property a learned policy violates and *by
how much*, and pick the worst-violating trace from a batch (the basis of our adversarial / "evil-env"
accountability sweep). `hymeko_monitor` already implements the bounded-horizon STL fragment with this robustness
(`robustness.rs`, `incremental.rs`).

### 3.4 Runtime verification and behavior trees

**Runtime verification** (RV; Bartocci et al., *Specification-based Monitoring of Cyber-Physical Systems*, a
standard RV survey) monitors a running system against a formal spec via an online monitor — our use of
`hymeko_monitor` over RL rollouts. In robotics, **behavior trees** (Colledanchise & Ögren, *Behavior Trees in
Robotics and AI*, 2018) are the practitioner's structuring tool for exactly this reach→grasp→lift→place
decomposition; a behavior tree is a (more restricted) cousin of the concurrent automaton we declare. We choose a
state machine over a behavior tree because HyMeKo already represents (and the team reasons in SysML/UML about)
**concurrent** state machines, and because the reward-machine formalism gives the cleaner RL-theoretic footing.

### 3.5 Where HyMeKo adds something new

1. **Temporal logic native to hypergraphs.** HTL (design sketch 2026-05-21) operates on signed-incidence
   hypergraph signals — per-cycle and aggregate predicates over the IR — not just scalar STL traces.
2. **Concurrency as a first-class citizen.** Parallel reward machines are an awkward, active research corner;
   HyMeKo describes arm $\parallel$ gripper $\parallel$ safety as concurrent regions natively.
3. **A structural leaf controller.** HSiKAN's signed-hypergraph message passing with per-channel Catmull-Rom
   activation is the per-state control — the structure-aware function approximator in the leaves of the
   structure-aware automaton.
4. **One IR for all four layers.** Structure, control, reward, and monitored properties are all read from the
   same HyMeKo model — the "one model describes structure + state + action + reward + *environment + task*"
   thesis, now with a runtime-verifiable accountability layer.

---

## 4. Proposed architecture

Three layers, pipeline `spec → automaton → learned leaves → monitor` (see `plan.tikz`):

- **Structure (`pick_place_machine.hymeko`).** A concurrent task automaton: an arm-motion region
  (Reach→Descend→Lift→Transport→Release), a gripper region (Open→Hold→Open), and a safety region, composed as a
  product. Transition guards are predicates over the hypergraph state (tool–object distance, grasp dwell,
  lift height). This *is* the reward machine.
- **Control (`fsm_policy.py`).** A state-conditioned HSiKAN trunk with per-state heads, trained off-policy
  (DDPG/SAC). Each state supplies a dense local sub-goal reward (Lift: raise the grasped object; Transport:
  reduce xy-distance to target), so credit assignment is per-state. The place-doesn't-converge wall dissolves
  because no single state requires discovering the whole sequence.
- **Specification / monitoring (`pick_place_spec.htl` + `htl_monitor.py`).** HTL formulas declare what must
  hold — safety $\square\,\neg\text{floor\_contact}$, bounded response
  $\square(\textsc{Grasp}\Rightarrow\Diamond_{[0,k]}\text{lifted})$, liveness $\Diamond\,\text{on\_target}$ —
  checked online by `hymeko_monitor`'s robustness. The robustness trace is the accountability artifact.

The guards and the safety properties are *the same temporal predicates* viewed two ways: as transition
conditions in the automaton and as monitored formulas in HTL. That is the conceptual economy of the design — one
declaration, used to *act* and to *verify*.

---

## 5. Mapping to the codebase

| Layer | Exists | New / to extend |
| --- | --- | --- |
| Structure | scripted expert = implicit FSM (if-ladder) | `pick_place_machine.hymeko`, `env/task_machine.py` (load + drive expert) |
| Control | HSiKAN backbone, DDPG/SAC, declarative `RewardSpec` (parity-tested), Markovian phase obs | `fsm_policy.py` (state-conditioned), per-state reward terms |
| Spec/monitor | `hymeko_monitor` (Rust robustness), `hymeko_neuro/eval/htl` (Python), HTL design sketch | `pick_place_spec.htl`, `htl_monitor.py` (rollout adapter) |

CORE.YAML: untouched (`hymeko_monitor`, `hymeko_rl`, `hymeko_neuro`, `data/` are non-core; core crates not
modified). The if-ladder refactor retires anti-pattern §6.5 #8 (forward-time flags for structural variants) into
a declared machine.

---

## 6. Experimental design

1. **Parity** — the FSM-driven scripted expert reproduces the current expert's place rate (regression test);
   the if-ladder is deleted only after parity holds.
2. **Baseline robustness** — monitor scripted-expert rollouts against `pick_place_spec.htl`; report per-property
   robustness (the reference the learned policy is measured against).
3. **Learned leaves** — state-conditioned HSiKAN, off-policy, per-state sub-goal reward. **Primary outcome:**
   greedy place rate $>0$ (the wall flat RL hit) without bin/disturbance bribes; median/IQR over $\geq5$ seeds.
4. **Accountability** — the difficulty / evil-env sweep reported as HTL robustness margins: which property breaks
   and by how much, at which difficulty. This is the quantitative, formal form of adversarial accountability.

The two earlier A/Bs fold in cleanly: A/B #1 (HSiKAN vs KAN vs MLP parameter efficiency) is run on the leaf
controllers; A/B #2 (serial vs branched morphology) becomes "does the structural leaf controller help more on
branched morphologies *within the declared machine*" — a sharper test than the flat-RL version.

---

## 7. Contributions claimed

1. A method for **accountable RL on robot hypergraphs**: declared concurrent task automaton (reward machine) +
   learned HSiKAN leaves + HTL runtime monitoring, all from one HyMeKo model.
2. **Concurrent** task automata for RL, native — addressing the awkward parallel-reward-machine corner.
3. **Hypergraph-native temporal logic** as both transition guards and monitored accountability properties.
4. An **empirical resolution** of a measured flat-RL failure (0 % greedy place) by structural decomposition
   rather than reward engineering.

---

## 8. Open questions and risks

- **Learned vs declared boundary.** Start leaf-only (guards declared, fixed) for auditability; learning guards
  is a later opt-in. Stated as a risk in the plan.
- **Concurrent-FSM semantics.** Begin with an explicit product of two regions + a safety monitor, not ad-hoc
  interleaving.
- **One conditioned policy vs N per-state policies.** Per-state is cleaner for accountability; conditioned shares
  the HSiKAN trunk (ties to the "shared structural reasoning" line). To be decided in Phase 3.
- **Reward-machine vs options framing** of the math — close; reward machines give the cleaner accountability
  narrative and the better RL-theoretic guarantees.
- **HTL predicate cost.** Pick-and-place uses scalar predicates (cheap); the design sketch's per-cycle-quantifier
  scaling risk (200k triads/step) does not bite this task but remains for graph-level monitoring.

---

## 9. Status and next step

- **Done:** problem measured (curriculum PPO, 0 % greedy place); existing HTL/monitor/HSiKAN/off-policy
  inventory confirmed; this report + the 4-artifact plan written; idea recorded in memory
  (`project-fsm-structured-rl`).
- **Not started:** any implementation (per §2, code begins only after the plan is on disk — it now is).
- **Next:** Phase 1 — declare `pick_place_machine.hymeko` and drive the scripted expert from it (parity test),
  retiring the if-ladder. This is the lowest-risk, highest-information first step and gates the rest.

---

## References (canonical, by concept)

- R. Toro Icarte, T. Klassen, R. Valenzano, S. McIlraith. *Using Reward Machines for High-Level Task
  Specification and Decomposition in RL.* ICML 2018; extended JAIR 2022.
- A. Camacho, R. Toro Icarte, T. Klassen, R. Valenzano, S. McIlraith. *LTL and Beyond: Formal Languages for
  Reward Function Specification in RL.* IJCAI 2019.
- X. Li, C. Vasile, C. Belta. *Reinforcement Learning with Temporal Logic Rewards.* IROS 2017.
- D. Aksaray, A. Jones, Z. Kong, M. Schwager, C. Belta. *Q-Learning for Robust Satisfaction of STL
  Specifications.* CDC 2016.
- M. Hasanbeig, A. Abate, D. Kroening. *Logically-Constrained Reinforcement Learning* (LCRL).
- K. Jothimurugan, S. Bansal, O. Bastani, R. Alur. *Compositional RL from Logical Specifications* (DiRL).
  NeurIPS 2021.
- R. Sutton, D. Precup, S. Singh. *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL.*
  Artificial Intelligence 112, 1999.
- P.-L. Bacon, J. Harb, D. Precup. *The Option-Critic Architecture.* AAAI 2017.
- O. Maler, D. Nickovic. *Monitoring Temporal Properties of Continuous Signals.* FORMATS 2004.
- G. Fainekos, G. Pappas. *Robustness of Temporal Logic Specifications for Continuous-Time Signals.* TCS 2009.
- A. Donzé, O. Maler. *Robust Satisfaction of Temporal Logic over Real-Valued Signals.* FORMATS 2010.
- E. Bartocci et al. *Specification-based Monitoring of Cyber-Physical Systems* (runtime-verification survey).
- M. Colledanchise, P. Ögren. *Behavior Trees in Robotics and AI: An Introduction.* 2018.

*(Bibliographic details by concept; verify exact venues/years before any external submission.)*
