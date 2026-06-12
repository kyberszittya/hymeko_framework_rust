# Cognitive Stack — Component Relationships

**Scope.** Top-level relationship map for the HyMeKo cognitive architecture: HIVE, AKOIRE, Gömb (Clifford-FIR / HSiKAN / CPML), HOTARU, Soma-Chordex, and the Clifford-FIR membrane. Complements `architecture/akoire/` (protocol detail) and `architecture/hsikan/` (model detail); vocabulary pinned in the analysis-phase `memory.md` (HIVE = canonical L0 data model).

**Last revised.** 2026-06-11.

**Files.** `overview.svg` (rendered), `overview.mermaid` (source-of-truth graph).

---

## 0. Theory layer — G-SPHF

Beneath every component sits one mathematical object: **G-SPHF (Generalized Signed Patch Hypergraph Fields)**, the 5-tuple $G=(V,E,B,\sigma,P)$ with the GGK 4-tuple axioms K1–K4 (`docs/spec/g_sphf_axioms.tex`, CORE-locked; reference impl `03_implementation/gsphf/`). Nothing in the stack is *not* a view of it: HIVE is its computational realization; the `project` views (signed incidence, star, clique expansion) are its three native sparse forms; Gömb computes functions on its fields; the spectral-entropy regularizer is a functional on its normalized Laplacian; the WL/patch tensor canonicalization is its isomorphism quotient; NAGARE is the calculus of its gradient flows (`hymeko_clifford` = Clifford autograd backend for G-SPHF); HSMM is its machine model. **Relationship kind 0: is-semantics-of.** When the unification paper is written, G-SPHF — not any product name — is the object in the title.

## 1. Components

| Component | Role | Layer | Status | Code anchor |
|---|---|---|---|---|
| **HIVE** | Canonical substrate: signed typed hypergraph IR + canonical hash; compile · project · emit | L0 | implemented | `hymeko_core/src/ir/ir.rs`, `module_store` |
| **AKOIRE** | Transaction protocol: render → propose (LLM) → parse/lower → evaluate ⟂ → commit. **Sole door for structural mutation.** | L0↔L2 bridge | spec v0.1, renderer ~10–15 % grammar | `01_analysis/akoire_prompt_template.md`, `akoire_ambience_renderer_spec.md`; `architecture/akoire/` |
| **Gömb** | Three-shell inference cascade (perception/judgement) | L2 perception | implemented | `signedkan_wip/src/hymeko_gomb/cascade.py` |
| — Clifford-FIR | Outer shell: grade assignment + FIR ring conv (edge decoration) | — | implemented | `signedkan_wip/src/hymeko_gomb/` |
| — HSiKAN | Middle shell: mixed-arity signed cycle pool, Catmull–Rom splines; standalone-capable | — | implemented | `signedkan_wip/src/` |
| — CPML | Inner core: grade-preserving polynomial layers, grade-0 readout ⟨·⟩₀ | — | implemented | `signedkan_wip/src/hymeko_gomb/` |
| **HOTARU** | Planner / graph sequencer: policy over HIVE-delta space | L2 deliberation | **planned — this doc is its first disk artifact** | — |
| **Soma-Chordex** | Reflex lane: fastest transport exteroception → actuators, ≤10 ms budget | L1 | Soma vision backbone implemented (quadtree, Hodge ∂₂, stim graphs); Chordex transport planned | `signedkan_wip/src/hymeko_gomb/soma/vision/` |
| **Clifford-FIR membrane** | Architecture-wide synaptic transport: every inter-component signal is a Cl(p,q) multivector through a learned per-synapse FIR bank | cross-cutting | planned (promotion of the shell to shared module) | must be its own module, imported by the Gömb shell — no duplication (§6.1) |
| **NAGARE (流れ)** | Dataflow execution substrate: SoA cycle pools as universal datatype, closed-form Clifford (fwd, bwd) operator pairs, no autograd, commutative MapReduce over cycles, lockless atomic grad accumulation | cross-cutting (runs-on) | skeleton: 579 LOC, ops for linear/scatter/BCE/Adam, Clifford-FIR stub; backends + tests empty | `hymeko_nagare/`; plan `docs/plans/2026-05-11-hymeko-nagare-flow/`; frozen compiler plan `docs/plans/CRITICAL-nagare-to-hsmm-compiler/` |

## 2. Relationship kinds

Four distinct kinds; do not conflate them.

1. **Composition (part-of).** Gömb ⊃ {Clifford-FIR, HSiKAN, CPML}. Strict shell order, grade structure preserved until ⟨·⟩₀. HSiKAN is the only shell that also exists standalone.
2. **Protocol (uses, gated).** AKOIRE mediates all structural mutation of HIVE. Its `evaluate` step is a Strategy slot; Gömb verdicts are the first plug-in (learned judge of proposed deltas).
3. **Policy (decides-over).** HOTARU plans in HIVE-delta space and acts **only through** AKOIRE, inheriting the commit gate's guarantees. Inputs: Gömb state utility + HIVE canonical hashes (free duplicate detection / transposition tables — one hash serving consistency *and* search).
4. **Transport (latency-bounded lane).** Soma-Chordex carries exteroception → actuators outside the deliberative loop. The Clifford-FIR membrane is the orthogonal transport concern: the wire type of *every* edge.
5. **Execution (runs-on / compiles-to).** NAGARE is the substrate the learned components execute on: Gömb shells lower to (fwd, bwd) op pairs; Chordex reflex kernels and membrane FIR banks are NAGARE ops — the ≤10 ms budget is only credible here (no autograd, no Python in the loop). Downstream: the frozen Nagare→HSMM compiler targets the HSMM abstract machine on Zynq FPGA (three-paper arc: HSMM theory → Nagare/MLSys → compiler/ASPLOS-FPGA).

## 3. Invariants (pin these before implementation)

- **Single mutation door.** Only AKOIRE commits structural deltas to HIVE. Soma-Chordex writes are append-only percept deltas — observations, never structure. Reflexes act; they don't redefine the body.
- **Single motor authority.** Actuators belong to Soma-Chordex alone. HOTARU influences the world only by reconfiguring Chordex set-points/thresholds via the slow loop (descending modulation).
- **Contract-bounded fast lane.** The reflex path replaces the per-action evaluate gate with deploy-time contracts: pre-verified reflex policies, hard latency and output-range bounds.
- **One wire type.** Inter-component messages are graded multivectors; grades are the type system (grade-0 magnitude/confidence, grade-1 sign/orientation, grade-2 relational). HIVE canonicalizes *state*; the membrane canonicalizes *signals* — the hub-and-spoke thesis applied at runtime.

## 4. Open questions / resolution criteria

- **Membrane must earn its weight.** The CogInfoCom ablation measured Clifford-FIR ≈ 0 contribution on the Bitcoin graphs. Before adopting it as universal transport: one cross-component task where grade-structured messages measurably beat flat vectors. Until then the membrane is hypothesis, not ABI.
- **HOTARU operators:** hand-written rewrite rules over HIVE vs learned proposals — undecided.
- **AKOIRE round-trip theorem** (rendered fragment re-lowers to structurally equal sub-HIVE) — specified in the renderer spec, unproven; mechanization candidate.
- **Implementation order (bottom-up dependency):** ambience renderer → AKOIRE loop closure → HOTARU; Gömb pluggable as evaluate-gate after loop closure; Chordex transport after L1 mutation API exists.
- **NAGARE parity gate before substrate claims:** Slashdot 5-seed AUC parity Nagare-CPU vs PyTorch+Triton (the frozen compiler plan's prerequisite #4) is also the gate for calling NAGARE the stack's execution substrate. Until parity, Gömb runs on PyTorch and NAGARE is a promising skeleton.
