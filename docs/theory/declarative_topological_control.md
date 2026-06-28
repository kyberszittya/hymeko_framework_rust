# Declarative Topological Control — a position statement

*A controller is a topology, declared; what it can represent and control is governed by the topological
relationship between controller and plant — a relationship one can generate, measure, ground geometrically, and
author as a runtime-tunable model.*

**Stated:** 2026-06-27 18:50 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu & Prof. Kato
**Working name:** *Declarative Topological Control* (DTC). **Status:** a research *program*, not a theorem — read
§8 before quoting. Synthesises and supersedes-in-scope: `SYSTEM_ENGINEERING_VIEW` (the MDSD substrate),
`gauge_holonomy_signed_hsikan` (the geometry), `hymeko_reward_dsl_hypothesis` (the reward leg), and the
isomorphic-controllers program (`docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/`, Phases 1–2b).

---

## 1. The central claim

A controller is not, in the first instance, a function or a gain vector. It is a **(signed) hypergraph of
couplings** among state, feature, and action nodes — a *topology*. Parameters are fitted *within* a declared
topology; the topology is the hypothesis. Two questions then have a single answer:

- *What can a controller represent?* and *What can it control?*

are both governed by the **topological relationship between the controller's hypergraph and the plant's**. That
relationship is (i) **generatable** — enumerate topology families; (ii) **measurable** — the topology→performance
map; (iii) **geometrically grounded** — signs and rotors are connections, balance and holonomy the invariants;
and (iv) **declarable** — authored as an inspectable, runtime-tunable model in one DSL (HyMeKo).

## 2. The three commitments (what makes it a school, not a trick)

1. **Structure-first.** The primary object of design is the interconnection topology, not the parameter vector.
   A flat network treats structure as incidental; DTC treats it as the hypothesis under test. "Which architecture"
   becomes "which *topology*," and that is a generatable, comparable space — not a craft.

2. **Match is the lever.** Performance is a function of topological *match* between controller and plant, and the
   *strength* of that dependence scales with task difficulty. This is a falsifiable empirical law (§4), not a
   slogan: match is load-bearing in proportion to how hard the control problem is.

3. **Declarative & accountable.** The reward, the task, and the controller topology are all declared in one
   model — making control algorithm-agnostic (one spec drives PPO/TD3/SAC/LQR/MPC), runtime-tunable (edit the
   model, not the code), inspectable, and auditable. *Structurally accountable control*: you can read, diff, and
   verify the controller's structure, not just its weights.

## 3. The method (one loop)

> **generate** topology families → **instantiate** isomorphic controllers (learned or structured) →
> **benchmark** across plants/scenarios → read the **topology→performance map** → **declare** the winner as a
> tunable, auditable model → iterate.

Each step is built and tested (`topology_zoo`, `controller_bench`, `structured_control`, `structured_mpc`); the
well-definedness of "isomorphic controller" is *proved*, not assumed (permutation-equivariance residual 1e-8).

## 4. The empirical law (measured, toy scale)

Topological match is **load-bearing in proportion to task hardness**:

| regime | what "match" buys | measured |
|---|---|---|
| **representation** of a graph-structured target (Phase 1) | matched topology vs mismatched | **~100×** lower error — decisive |
| **unconstrained control** of a benign plant (Phase 2) | matched gain-sparsity vs mismatched | **~3%** — marginal |
| **constrained MPC** of a benign plant (Phase 2b) | matched prediction-model vs mismatched | **2–7%, clean diagonal** — biting |
| **under-actuated / unstable control** (Phase 3) | covering topology vs non-covering | **predicted decisive** (ρ→∞) |

The gradient is the point: structure governs *representation* strongly and *easy control* weakly, with constrained
control in between — and the prediction is that *hard* control (under-actuation, instability) returns to the
Phase-1 strength. A coherent curve, not four disconnected results.

## 5. The geometric foundation (why "topology" is principled, not heuristic)

Signed graphs are **Z₂ connections**; balance is trivial holonomy (Zaslavsky). Rotors generalise the connection
to SO(2)/SO(3); a controller **parallel-transports information (fibers) along walks**, and the collected
**holonomy** is its computation (`gauge_holonomy_signed_hsikan`). So the structural invariants DTC measures —
balance, holonomy spectrum, signed connectivity — are *gauge-theoretic*, not ad-hoc graph statistics. This is
what separates DTC from "add a GNN": the topology carries a connection, and control is transport.

## 6. Falsifiable predictions (the program's spine)

1. **A topology invariant predicts the control cost.** Algebraic connectivity / diameter / signed balance /
   holonomy spectrum predicts the off-diagonal entries of the topology→performance map — turning the map into a
   *law*. [open]
2. **Hard control makes match decisive.** Under under-actuation and open-loop instability, only *covering*
   topologies stabilise (ρ→∞ otherwise). [Phase 3, predicted]
3. **Control is topology, not algorithm.** One declared topology controls comparably across PPO/TD3/SAC/LQR/MPC.
   [partly shown — shared `RewardSpec`/structure across optimizers]
4. **Refutation conditions.** If performance is independent of topology once *capacity* is matched, or if no
   invariant predicts the map, DTC collapses to "use a big enough network." The program lives or dies on §6.1–6.2.

## 7. Borrowed vs new (intellectual honesty about novelty)

- **Borrowed:** structured/distributed optimal control (Lin–Fardad–Jovanović), message-passing / GNNs, spectral
  graph theory, model-driven systems engineering, and the gauge theory of signed graphs (Zaslavsky).
- **New — the synthesis:** treating the controller topology as the **declared, generatable, gauge-grounded
  primitive** whose **match** to the plant is the *measured* lever across **representation *and* control *and*
  learning**, authored and audited in **one DSL**. No existing field unifies "generate controller topologies,
  measure structural match, ground it in a connection, declare and audit it" as a single program. DTC is the
  unification, not any one part.

## 8. Honest status (read before quoting)

The evidence is **toy scale**: N=9, linear plants, supervised + LQR/MPC, single graph seed per family. The
strong-control regime (Phase 3) is **predicted, not shown**. The invariant→performance *law* (§6.1) is **open**.
The cross-optimizer claim is **partial**. This document states a *program* — a thesis with a method and falsifiable
predictions — at the moment its first evidence gradient became coherent. It is deliberately ambitious in scope and
deliberately modest in claimed proof. The next three artifacts that would earn the word "school": (a) Phase 3's
decisive under-actuated result; (b) one topology *invariant* that predicts the map; (c) the cross-optimizer parity
figure on a shared declared spec.

---

*If DTC holds, "designing a controller" becomes "declaring and matching a topology," and the question shifts from
how many parameters to which structure — generatable, measurable, accountable.*
