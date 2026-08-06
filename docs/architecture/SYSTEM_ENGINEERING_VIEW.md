# The HyMeKo System-Engineering View — a manifesto

*Status: living document. Started 2026-06-20.*
*Companion: [HYMEKO_ECO_CAP.md](../HYMEKO_ECO_CAP.md) (what is already built, with evidence).*

---

## Thesis

> **One source, many views.** A robotic system is described **once**, as a signed
> hypergraph in HyMeKo. Every artifact a downstream tool needs — URDF, SDF, MJCF,
> a Gazebo launch bundle, a SysML block diagram, a Lean4 proof obligation, a DOT
> graph, the figure in a paper, an RL `AgentSpec` — is a **projection** (a *view*)
> of that single source, never a parallel hand-maintained copy.

This is **Model-Based Systems Engineering (MBSE)** taken literally. In INCOSE's terms,
MBSE is *the formalized application of modelling to support system requirements, design,
analysis, verification and validation, beginning in the conceptual design phase and
continuing throughout development and later life-cycle phases* (INCOSE SE Vision; the
direction of travel of the discipline through SE Vision 2035). HyMeKo is the
**authoritative source of truth** that vision calls for: the model is not documentation
*about* the system; the model **is** the system of record. Code, simulation assets, and
diagrams are generated downstream and are, by construction, consistent with each other
because they share one upstream cause — a single point on the **digital thread**.

The failure mode this exists to kill is what INCOSE calls the *document-centric* trap:
the same robot encoded three times — once in a URDF, once in a slide, once in a training
script — drifting out of sync the moment anyone edits one of them. Under MBSE there is
nothing to keep in sync, because there is only one editable artifact and everything else
is derived from it.

---

## Two layers, separated on purpose

Every HyMeKo description separates **vocabulary** from **instance**.

| Layer | What it is | Authored | Example file |
|------|------------|----------|--------------|
| **Vocabulary** (`meta_*`) | The *kinds* — link, joint, axis, sensor, context, signal, aggregation. Declared once per ecosystem. | rarely | [`data/robotics/meta_kinematics.hymeko`](../../data/robotics/meta_kinematics.hymeko), [`data/robotics/meta_context.hymeko`](../../data/robotics/meta_context.hymeko) |
| **Instance** | The *specific* robot — its links, its joints, its context graph. Imports a vocabulary; declares only what is unique to this system. | per system | [`data/robotics_imported/wam/wam.hymeko`](../../data/robotics_imported/wam/wam.hymeko), `scenarios/hymeko_robot_reuse.hymeko` |

The mechanism is HyMeKo's cross-file import:

```hymeko
robot_description {
    @"meta_kinematics.hymeko";          // pull in the vocabulary
    using kinematics.elements as el;     // alias the kinds you use
    using kinematics.rev_joint  as rj;
}
robot: el, rj {
    base:     el.link {}                 // declare only instances
    @j1: rj { (+ base, - shoulder, - ax.AXIS_Z); }
}
```

A vocabulary is amortised across **every** system that imports it. The cost of
defining "what a revolute joint is" is paid once, not once per robot.

---

## Why this is the selling point, not a refactor

Concrete, measured (2026-06-20, `reports/2026-06-20-mdsd-reuse-scenario.md`):

The ROS2 demo robot was authored both ways — bare inline types
(`scenarios/hymeko_robot.hymeko`, kept as the control) and imported vocabulary
(`scenarios/hymeko_robot_reuse.hymeko`):

- **Length:** 189 → 126 lines (−33 %); 108 → 76 code lines (−30 %) for **one**
  robot. The shared vocabulary (~55 lines) is paid once, so for the *N*-th robot in
  the fleet the marginal saving is the full per-file re-declaration.
- **Semantics, not just brevity:** the imported form carries *typed* joints
  (`rev_joint`) with real axis vectors and limits. The kinematic extractor recovers
  them and URDF/SDF emit valid `<joint>` elements. The bare baseline's generic
  `joint` type is **not** one of the four joint kinds the extractor knows, so it
  emits **zero** joints. Reuse is not cosmetic — it is the difference between a
  description that *projects to a working robot* and one that does not.

The length number is the headline; the semantic number is the argument.

---

## What counts as a "view"

The word *view* is used here in the precise sense of **ISO/IEC/IEEE 42010** (the
architecture-description standard INCOSE builds on): a **view** addresses a set of
**stakeholder concerns** from a **viewpoint** (the rules governing how that view is
constructed). In HyMeKo the architecture description *is* the signed hypergraph; each
emitter is a viewpoint, and each emitted artifact is a view answering one stakeholder's
concern:

| Stakeholder concern | Viewpoint (emitter) | View (artifact) |
|---|---|---|
| "Will it move correctly in sim?" | URDF / SDF / MJCF / Gazebo | simulation scene |
| "How does it decompose into blocks & ports?" | SysML | block / IBD diagram |
| "Is this property provable?" | Lean4 | proof obligation |
| "What is the structure?" | DOT / Mermaid / WASM editor | graph |
| "What is the learning problem?" | RL `AgentSpec` | obs + action + reward + vertex count |
| "Did the projection lose information?" | HyMeKo → HyMeKo | canonical round-trip |

All evidence anchors are in the capability ledger. The point of the 42010 framing is that
the views are **not** independent documents to be reconciled — they are, by construction,
**consistent** because they share one architecture description. Adding a view means
writing one viewpoint (emitter) against the IR. It never means asking the *author* to
maintain another copy.

---

## The boundary rules (so the principle does not rot)

These are the lines that keep "one source" honest. They mirror the project's
anti-pattern list (CLAUDE.md §6.5).

1. **Vocabulary is imported, never re-declared.** If a scenario writes `link {}`,
   `joint {}`, `aggregation {}` inline when a `meta_*` profile already declares
   them, that is duplication wearing the costume of self-containment. The only
   sanctioned exception is a **frozen reference artifact** — e.g. the paper-faithful
   baseline, kept verbatim *on purpose* as a control. Mark such files as frozen.

2. **Parametric difference → configuration. Structural difference → a different
   description.** A robot that differs only in link lengths is the same description
   with different field values. A robot with a different topology is a different
   instance graph. Do not encode a structural fork as a runtime `if`.

3. **A view is read-only downstream.** Never hand-edit a generated URDF and expect
   it to survive. If a generated artifact is wrong, the bug is in the source or the
   emitter, and the fix goes there.

4. **One source per system, even across tools.** The robot that drives the MuJoCo
   RL env, the Gazebo demo, and the SysML diagram is *one* `.hymeko` file. Three
   tools, three views, one cause.

---

## Relationship to SysML / UML

HyMeKo is not a replacement for SysML; it is the **single semantic core** SysML can
be a view *of*. The signed hypergraph (GGK 4-tuple `K = (B, G, μ, r)`) is strictly
more expressive than a block-and-port diagram for the things this project cares
about — higher-order (arity > 2) relations, signed incidence, cross-context
constraints — and SysML block/IBD diagrams are recovered by projection
(`sysml` emitter). The systems engineer keeps their SysML view; they just stop
*authoring* in it.

---

## Alignment with INCOSE systems-engineering practice

This manifesto is not a private convention; it is the project's reading of established
systems-engineering doctrine, applied to a hypergraph core. The mapping:

| HyMeKo | Systems-engineering concept | Source |
|---|---|---|
| The `.hymeko` source | Authoritative source of truth / single system model | INCOSE MBSE, SE Vision 2035 |
| Emitter + emitted artifact | Viewpoint + view addressing a concern | ISO/IEC/IEEE 42010 |
| Vocabulary (`meta_*`) vs instance | Separation of concerns; reference data vs system data | ISO/IEC/IEEE 15288 |
| Cross-file import + canonical hash | Traceability; configuration identity of a model element | INCOSE Handbook (traceability, config mgmt) |
| `.hymeko` → URDF/MJCF/SysML/Lean4/AgentSpec | The digital thread across life-cycle activities | Digital Engineering |
| Round-trip + validation-gated emission | Verification & Validation | ISO/IEC/IEEE 15288 §V&V |

**Where this sits on the Vee.** The classic SE *Vee* has a left leg (decomposition and
definition: stakeholder needs → requirements → architecture → design) and a right leg
(integration and verification: unit → subsystem → system V&V). HyMeKo's authoring sits on
the **left leg** (architecture and design definition, captured once as the hypergraph);
the emitters carry that single definition rightward into each V&V activity — the same
model that *defines* the robot is the one *verified* against each consumer.

**Verification vs validation, kept distinct (INCOSE's two questions).**

- *Verification — "did we build the system right?"* Here: are the views **faithful** to
  the source? This is mechanically checkable. The HyMeKo→HyMeKo canonical round-trip and
  the canonical hash are verification gates: a view-generator that drops information, or
  an edit that silently changes semantics, fails a test. The MDSD reuse result is a
  verification statement — the imported description *projects to the same kinematic
  structure*, hash-confirmed.
- *Validation — "did we build the right system?"* Here: does the model meet the actual
  need? This is **not** something the framework can assert for you — it requires running
  the robot, the sim, the experiment. The capability ledger is deliberately honest about
  this line: `PROVEN` rows are largely verification (a test passes); `SHIPPED` and
  `RESEARCH` rows are where validation is still open. Conflating the two is the failure the
  ledger's status legend exists to prevent.

**Why a hypergraph, in SE terms.** INCOSE and SysML model systems as blocks with ports
and pairwise connections. Many real engineering relations are **not** pairwise — a
constraint over three quantities, a context that fuses five signals, a signed
balance over a cycle. Forcing those into binary connectors loses information at authoring
time. The signed hypergraph (GGK 4-tuple `K = (B, G, μ, r)`) keeps higher-order, signed,
cross-context relations *as first-class*, and the pairwise SysML view is recovered by
projection — never the other way around. HyMeKo is MBSE with an expressive-enough core
that the standard views are lossy projections of it, not the source.

> If a fact about a system is true, it is stated **once** in HyMeKo and **derived
> everywhere else**. Any place a fact is restated by hand is a latent
> inconsistency, and is treated as a defect — not a convenience.

This is the single-authoritative-source principle of MBSE stated as an operational rule,
and it is what makes traceability mechanical rather than aspirational: every view traces
to exactly one source element, and the canonical hash detects when that link is broken.
Everything in the capability ledger is, ultimately, verification evidence for or against
this one sentence.
