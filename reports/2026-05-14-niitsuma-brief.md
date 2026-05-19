# A Note for Prof. Niitsuma

**HyMeKo-Gömb: signed-hypergraph cycle pools for embodied multi-agent
systems — architectural structure, hard benchmark validation, and
applications to human-robot collaboration**

**Date:** 2026-05-14
**Context:** This brief is sent in the spirit of your long-standing
collaboration with Prof. Korondi (BME) on etho-robotics and
spatial-intelligence systems. It summarizes a methodology and a
recent empirical result that sit naturally alongside the multi-agent
human-robot collaboration research direction in your laboratory.

---

## 1. The structure

### 1.1 The architectural primitive

A **signed hypergraph** `H = (V, E, σ)` where vertices are
entities (robots, humans, objects in a shared space), edges are
relationships, and `σ ∈ {+1, −1}` encodes the *valence* of that
relationship (trust / distrust, reliable / jammed communication,
cooperative / conflictual behaviour, rotational / translational
joint, attachment / avoidance).

**Cycle σ-products** `π(c) = Π σ(eᵢ)` over k-cycles in this graph
are the fundamental architectural feature. Cartwright-Harary
(1956) showed that `π(c) = +1` *for every cycle* iff the system is
**structurally balanced** — i.e. it admits a 2-colouring of the
vertices into two cooperative groups. This is the formal definition
of a *stable* multi-agent coalition.

### 1.2 The three-shell cascade ("Gömb")

The architecture composes three architecturally-orthogonal
inductive biases over the shared signed-hypergraph representation:

```
         OuterFIRShell   ──  Clifford-algebra graded multivector
                              filtering (geometric-algebra axis)
            │
         MiddleHSiKAN    ──  sign-branched Catmull-Rom spline
                              activations + α-mixer over arities
                              (learnable-nonlinearity axis)
            │
         InnerCPMLCore   ──  tier-stratified capsule routing
                              (routing-topology axis)
            │
            ▼
      σ-cycles → edge prediction head
```

The name *Gömb* (Hungarian: "sphere") refers to the concentric
nesting of the three shells. Each shell is independently variable;
their composition is constructive — no single shell achieves the
empirical results below alone.

### 1.3 Strict-by-construction protocol

The cycle pool enumerates over *training* edges only. Test edges
never participate in σ-products. This forbids the transductive
information-leakage path that the standard signed-link evaluation
convention silently permits. A label-shuffle audit confirms the
property: with shuffled training-edge signs, the architecture's
AUC drops to chance (0.54), indicating that the learned
predictions are not artefacts of test-set leakage.

---

## 2. The results

### 2.1 Headline — Epinions signed-trust benchmark

5-seed test ROC-AUC, strict protocol (no test-edge participation
in cycle features):

| Method                 | Protocol         | AUROC      |
|------------------------|------------------|------------|
| **HyMeKo-Gömb (ours)** | **strict**       | **0.9526 ± 0.0018** |
| SiGAT (Huang 2019)     | transductive     | ~0.95      |
| SDGNN (Huang 2021)     | transductive     | ~0.95–0.96 |
| SGCN  (Derr 2018)      | transductive     | ~0.93      |

The Epinions result exceeds the published SiGAT under a stricter
evaluation protocol than any prior work in the family. Each
individual seed exceeds 0.949; the result is not a lucky-seed
artefact.

### 2.2 Cross-dataset 5-seed table

| Dataset       | AUROC mean | ± pstd  |
|---------------|-----------:|--------:|
| Bitcoin Alpha | 0.8972     | 0.0079  |
| Bitcoin OTC   | 0.9145     | 0.0068  |
| Slashdot      | 0.9017     | 0.0008  |
| Epinions      | 0.9526     | 0.0018  |

### 2.3 Deployment footprint

| Hardware                          | NVIDIA RTX 2070 SUPER (2019, 8 GB VRAM) |
|-----------------------------------|------------------------------------------|
| Total wall time (5-seed Epinions) | ~33 minutes                              |
| Wall time per seed                | ~6.5 minutes                             |
| Peak GPU memory                   | ~5.5 GB                                  |
| Model parameter count             | 4.3 million                              |

A 4-million-parameter model is **directly compatible with an
NVIDIA Jetson Orin Nano** or equivalent embedded inference platform
of the kind your laboratory's mobile manipulators, wheelchairs, or
companion robots already host. The architecture was deliberately
designed not to require workstation-class GPUs.

---

## 3. Applications to your research domain

### 3.1 Multi-robot communication cliques in human-robot collaboration

Your recent work on **situation-based proactive human-robotic
systems in future convenience stores** (2024) describes a setting
in which multiple robots, customers, and staff form a dynamic
network of reliable / disrupted communication and cooperative /
conflicting intent. This is exactly a signed graph.

Our framework natively identifies **balanced cliques** as the
formal definition of a *stable communication team*:

- Each clique's σ-product = +1 certifies that every internal
  cycle of communication is consistent.
- The size and composition of balanced cliques dynamically updates
  as the environment changes (a customer enters, a robot loses
  connectivity, a staff member changes role).
- A working interactive demonstration (Gradio web GUI) generates
  synthetic robot communication networks and enumerates balanced
  cliques as shaded convex hulls in the spatial layout — directly
  visualising the "which agents currently form a stable team"
  question.

### 3.2 Spatial intelligence with σ-product belief calibration

Your foundational 2007 work on **spatial memory as an aid system
for human activity in intelligent space** maintains a structured
memory of agent-object-position relationships. The
signed-hypergraph representation extends this in two specific ways:

- **σ-product consistency check.** As new observations arrive, the
  cycle σ-products provide a fast invariant: an inconsistent
  σ-product on a recently-updated cycle is a flag that the spatial
  memory has been corrupted by a sensor anomaly or a stale belief.
- **Multi-scale abstraction.** Our *derivative-nodelet* operator
  contracts balanced sub-graphs (e.g. a stable agent-object
  coalition) into super-vertices. The spatial memory then operates
  at the appropriate level of abstraction for the current task:
  individual joint angles for fine manipulation; coalition-level
  reasoning for cooperative planning.

### 3.3 Etho-robotics — animal-inspired behaviour with signed-relational structure

The Korondi-Niitsuma 2015 etho-robotics work proposes animal-
inspired behaviour models for robots, with the dog-human
relationship as the canonical case. These models implicitly encode
*signed* relational structure: dominance vs submission, attachment
vs avoidance, cooperation vs competition.

Our framework gives a *learnable* form to these signed structures:

- **Behaviour-chain consistency.** A cycle of behaviour transitions
  (greet → engage → cooperate → disengage) has a σ-product that
  encodes whether the cycle is psychologically coherent under the
  agent's current internal state. Inconsistent σ-products predict
  the points where the agent should re-evaluate.
- **Trust calibration from observation.** Given a sequence of
  interactions, our architecture updates σ-values on the relevant
  relational edges and predicts whether the long-term relationship
  is converging to a balanced (stable) or unbalanced (volatile)
  configuration.
- **Multi-agent etho-robotics.** When several robots interact with
  the same human, the full system's stability is a balanced-
  clique property over the agent-human-object graph. This
  generalises the dyadic Korondi-Niitsuma 2015 dog-human case to
  team scenarios.

### 3.4 Kinematic family identification for mobile platforms

Every URDF (the standard ROS / MoveIt robot description format)
maps into a signed kinematic graph: link nodes connected by joint
edges with signs encoding joint type (+1 = rotational; −1 =
prismatic). Closed kinematic loops surface as k-cycles.

A trained classifier identifies mechanism families (4-bar
linkage, Stewart platform, Delta 3-RRR, serial chain) with 100%
test accuracy on 13 catalogued URDFs, discriminating Stewart from
Delta at the k=6 cycle level by cycle multiplicity.

In your context, this is an *automatic structural fingerprint*:
given a new mobile platform (wheelchair, mobile manipulator,
service robot), the system identifies its kinematic family and
the appropriate control / planning primitives without manual
configuration.

### 3.5 Hierarchical reasoning for multi-task scenarios

Your recent **feedback-driven adaptive task estimation in
human-robot collaboration** (2025) and **XR-based caregiver task
assistance** (2024) both involve a hierarchy of tasks at multiple
levels of granularity. Our *derivative-nodelet* multi-scale
operator naturally supports this hierarchy:

- Low-level task primitives are vertices of the fine-scale
  hypergraph.
- Mid-level task groupings emerge as balanced cliques (consistent
  sub-task coalitions).
- High-level strategic decisions operate on the contracted graph
  whose vertices are these cliques.

The single cycle-σ-product machinery applies at every scale of
the hierarchy. This is one architecture, several abstraction
levels — directly aligned with the layered task-estimation
direction your laboratory has taken.

---

## 4. Reproducibility

All artefacts are version-controlled and accessible on request:
benchmark JSONL outputs, trained model checkpoints, audit
experiments, reproducibility scripts, full forensic audit report.
Compilation tooling: standard TeX Live, PyTorch, Rust toolchains.
No specialised hardware or proprietary dependencies.

---

## 5. Bottom line

A state-of-the-art result on a public signed-link benchmark,
achieved on consumer hardware (RTX 2070 SUPER), under a stricter
evaluation protocol than any prior published work in the field —
on a representation that is *natively compatible* with the
multi-agent, spatial-intelligence, etho-robotics, and mobile-
manipulation problem classes your laboratory works on. The robot
communication cliques demo and the URDF kinematic-family
classifier are concrete examples that the framework is a general-
purpose representation for embodied multi-agent systems, not a
benchmark-specific construction.

I would be honoured by your reaction to the framing, and very
glad to share the demo + artefacts in any format useful to your
group.

With respectful regards,

\[name redacted\]
*Hungary, 2026-05-14*

---

**Supplementary artefacts available on request:**

- Per-seed JSONL benchmark output (Bitcoin Alpha, Bitcoin OTC,
  Slashdot, Epinions).
- Audit framework specification (label-shuffle test as a
  diagnostic for σ-leakage in signed-link benchmarks).
- Trained model checkpoints (Bitcoin runs).
- Gradio demo source code (robot communication cliques + kinematic
  graph parser).
- Forensic audit report including all sanity checks.
