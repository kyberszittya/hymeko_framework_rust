# HyMeKo: One Hypergraph Substrate for Representing and Learning from Structured Systems
*From declarative hypergraph structure to structural-prior learning.*

PhD seminar · Kato Laboratory · Csaba Hajdu · June 2026
Deck: `docs/seminar/HyMeKo_Seminar.pptx` (33 slides, self-contained — embedded video + figures).

---

## Abstract

Structured systems — robot mechanisms, signed trust and affect networks, chemical
processes, neural dataflow — are intrinsically n-ary, yet coercing their relations
into pairwise graphs erases each relation's identity and inflates the
representation to O(|E|·d²). HyMeKo addresses this with a single canonical-hypergraph
substrate that serves both representation and learning, so that the structure a
framework compiles is the same structure a model reasons over. A declarative
language compiles to a content-hashed hypergraph intermediate representation, from
which one query-and-template engine emits URDF, SDF, MJCF, SysML v2 and PyTorch with
cross-view consistency, while an O(|E|·d) star expansion streams zero-copy to the
GPU. Learning is driven by enumerated signed cycles and walks through a SignedKAN,
HSiKAN and Gömb family whose Kolmogorov–Arnold spline activations turn graph
structure into an inductive prior, and a vision variant further incorporates
Forman–Ricci curvature and Hodge decomposition, all evaluated under a
leakage-audited strict protocol. Signed-graph link prediction is competitive-to-
leading (Epinions 0.943 / 0.953; Bitcoin and Slashdot ≈ 0.90) at roughly a quarter
of the parameters and about eleven times faster inference, and the same primitive
transfers to vision (HyMeYOLO, mAP₅₀ ≈ 0.90, at parity with ReLU) and recovers
kinematics from topology alone to about five centimetres. Ongoing work pursues a
σ-masked strict protocol, round-trip `.hymeko`-to-`torch.nn` code generation,
broader transform targets, and larger-scale signed-graph and vision corpora.


---

## Section-by-section (33 slides)

**Framing (1–6).** Why hypergraphs (n-ary relations, clique blow-up); a hyperedge
and its signed Levi/Berge incidence (= the star expansion); the one-substrate /
two-payoffs thesis; what HyMeKo describes (robotics, MBSE/SysML, signed graphs,
neural dataflow); goals at two levels (language + infrastructure).

**Part I — the framework (7–13).** The declarative DSL; the SIMD-lexed, LALRPOP
compiler to a Blake3-hashed hypergraph IR; star vs clique (1,498 vs 10,991 NNZ;
79.7 vs 496 ms); the zero-copy Iceoryx2/Arrow/DLPack bridge with the topology-hash
gate (<100 µs); query-driven transforms (one IR → many targets, hub-and-spoke);
and the systems-engineering discipline (SysML 2, CORE.YAML, plan-first).

**Part II — the science (14–29).** Structure→prior (cycles as tuples, the
cycle-arity compass); the SignedKAN→HSiKAN→Gömb family; the strict Gömb cascade;
*what k-enumeration is and why HSiKAN needs it*; Inside HSiKAN (Catmull–Rom
activations, arity mixer, quaternion sign-attention); the **learned αₖ + ROC**
(real inference output, AUROC 0.9957, transductive convention); Clifford-FIR &
quaternion machinery (Forman curvature, Hodge, Bochner — the V1→V4→IT analogy);
link-prediction results; lean & fast (30 k params ≈ ¼ of joint; within-family
forward ~3.5× CPU at h4-vs-h16); *what leakage is and why it matters*; honesty as a protocol; how we know
it isn't leakage (label-shuffle, σ-leakage); and the **HyMeYOLO trio** —
hypergraph convolution in vision, detections on Cluttered MNIST, and the
Ricci-curvature / differential-topology backbone (Forman κ, Hodge ∂²=0 / 1-homology,
SDRF discrete Ricci flow).

**Close (30–33).** The bridge (one DSL describes the robot and the network that
reasons about it); contributions; reference diagrams (appendix); conclusion &
outlook.

---

## Key results (honest framing)
- **Signed-graph link prediction (Gömb-strict, honest):** Bitcoin α 0.897, OTC
  0.915, Slashdot 0.902, Epinions 0.943 (fine-tune 0.953); wiki_elec 0.911. Under
  the field's standard transductive convention, tuned optuna_best reaches 0.996 /
  0.993 on Bitcoin. **Framing:** competitive-to-leading at a fraction of the cost,
  with the genuine lead on Epinions — not a flat SOTA claim.
- **Efficiency:** ~30 k parameters (≈¼ of the joint baseline). Within-family
  forward gap (h4 lean vs h16 joint, **same device**) measured **~3.5× on CPU,
  ≈1× on CUDA** (`inference_bench.json`, 2026-06-11). SGCN is faster in absolute
  terms — the win is **accuracy-per-parameter**. (The older "11×" 30.5-vs-342
  figure is the optuna_best_otc-vs-joint result: real but OTC-specific and
  tuple-set-driven, not a general width claim.)
- **HyMeYOLO:** honest mAP_50 ≈ 0.90 (Stage-B HSiKAN-CR), a statistical tie with a
  ReLU ResNet-tiny backbone (+0.008, z≈0.6); FPN adds nothing. Presented as an
  *approach to hypergraph convolution*, not a benchmark win. (The corrected metric
  replaced a bug-inflated 0.723.)
- **Methodology:** label-shuffle leakage diagnostic (Gömb → 0.540 = chance);
  5-seed paired-Δ significance; every number maps to an on-disk artifact.

---

## Supporting artifacts (in the repo)
- Title & abstract — `docs/seminar/SEMINAR_TITLE_ABSTRACT.md`
- Figures — `docs/architecture/Diagrams/` (rendered article TikZ, drawio HSiKAN /
  Gömb, Catmull–Rom, Forman κ, learned αₖ + ROC; `yolo_panels/` detections)
- 3D star-expansion viewer — `demo_web/star_expansion_viewer.html`
- Demo build specs — `SEMINAR_DEMO_OUTLINE.md`, `signedkan_wip/demos/SEMINAR_DEMOS.md`,
  `docs/INFERENCE_DEMOS_OUTLINE.md`, `demo_web/STAR_EXPANSION_VIEWER_OUTLINE.md`

## Notes
- Header font is Futura (Century Gothic is the Windows-native fallback); body is
  Bahnschrift. Embedded figures are immune to font substitution.
- Stray LaTeX temp files remain under `paper/smc2026/figures/` and
  `paper/nature_comm_v1/figs/` (`__out_*` / `__wrap_*`) — remove with `git clean`.
