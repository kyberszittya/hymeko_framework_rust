# Seminar — References & Future Work (slide content)

Draft content for the two new slides. References are curated from the project's
own bib files (`paper/smc2026/references.bib`, `paper/arxiv_v1/references.bib`,
`docs/review/2026-06-06-ac-hsikan-elsevier/references.bib`) — so every entry maps
to a real, already-used citation. Exact formatting can be pulled from those
`.bib` keys (given in brackets).

---

## Slide A — References (selected)

**Hypergraphs & representation**
- Berge, *Graphs and Hypergraphs*, 1973 `[berge1973graphs]`
- Levi, *Finite geometrical systems*, 1942 (the Levi/star incidence) `[levi1942]`
- Gallo et al., *Directed hypergraphs and applications*, 1993 `[gallo1993directed]`
- Feng et al., *Hypergraph Neural Networks (HGNN)*, AAAI 2019 `[feng2019hypergraph]`
- Bai et al., *Hypergraph Convolution and Hypergraph Attention*, 2021 `[bai2021hypergraph]`
- Joslyn et al., *HyperNetX*, 2021 `[joslyn2021hypernetx]`

**Signed-graph learning**
- Cartwright & Harary, *Structural balance*, 1956 (σ-cycle balance) `[cartwright_harary]`
- Derr, Ma & Tang, *Signed Graph Convolutional Networks (SGCN)*, ICDM 2018 `[sgcn / derr2018sgcn]`
- Huang et al., *Signed Graph Attention Networks (SiGAT)*, 2019 `[sigat / huang2019sigat]`
- Huang et al., *SDGNN: node representation for signed directed graphs* `[sdgnn]`
- Kumar et al., *Edge weight prediction in weighted signed networks* (Bitcoin) `[bitcoin]`
- Leskovec et al., *Signed networks in social media* (Epinions/Slashdot) `[leskovec]`

**Kolmogorov–Arnold priors**
- Kolmogorov, *On the representation of continuous functions…*, 1957 `[kolmogorov]`
- Liu et al., *KAN: Kolmogorov–Arnold Networks*, 2024 `[kan / liu2024kan]`
- Parcollet et al., *Quaternion Convolutional Neural Networks*, 2019 `[parcollet2019quaternion]`
- Vaswani et al., *Attention Is All You Need*, 2017 `[vaswani2017attention]`

**Curvature & differential topology (vision)**
- Forman, *Bochner's method for cell complexes / combinatorial Ricci curvature*, 2003
- Topping et al., *Understanding over-squashing… (SDRF)*, ICLR 2022
- (Hodge decomposition / discrete exterior calculus on the patch graph)

**Systems modelling, robotics & infrastructure**
- Quigley et al., *ROS / URDF*, 2009 `[urdf2012]`; Macenski et al., *ROS 2*, 2022 `[macenski2022ros2]`
- Todorov et al., *MuJoCo*, IROS 2012 `[todorov2012mujoco]`; Koenig & Howard, *Gazebo*, 2004 `[koenig2004gazebo]`
- OMG, *SysML v2.0* `[sysmlv2]`; Friedenthal et al., *A Practical Guide to SysML* `[friedenthal2008sysml]`
- Fey & Lenssen, *PyTorch Geometric*, 2019 `[torchgeometric]`
- O'Connor et al., *BLAKE3*, 2020 `[blake3]`; Eclipse, *iceoryx2* `[iceoryx2]`
- Graph transformation: VIATRA `[varro2007viatra]`, EMF-IncQuery `[ujhelyi2015emfincquery]`, ATL `[jouault2008atl]`

**Author's prior & companion work**
- Hajdu et al., tensor / generative hypergraph papers, 2022 `[hajdu2022tensor, hajdu2022generative]`
- Hajdu, CogInfoCom 2024 `[hajdu2024coginfocom]`; HSMM, 2026 `[hajdu2026hsmm]`
- SignedKAN / HSiKAN / mixed-arity αₖ line `[signedkan2026, hsikan, …]`; Friedler P-graph axioms `[friedler]`

> Trim to ~16–20 lines for the slide; the rest can go to a backup/appendix slide.

---

## Slide B — Future Work

- **σ-masked strict protocol** — finish the sign-aware leakage audit (label- and
  σ-shuffle) and the 5-seed grid; lock the honest operating point.
- **Round-trip `.hymeko` → `torch.nn`** — structural parity already holds (the
  emitted module realises every declared layer); next is the *runnable* round-trip
  (numeric parity vs the reference cascade) and the faithful **Soma** vision path
  (Hodge / stim / patch internals, not just the skeleton).
- **Broader transform targets** — task-level emitters (BehaviorTree.CPP / PDDL /
  ROS 2 action servers) for the `.hymeko` task layer; more MBSE views.
- **Larger-scale corpora** — bigger signed-graph and vision datasets; multi-seed
  overnight runs under the 16 GB RSS discipline.
- **HSMM → FPGA path** — the Nagare dataflow substrate → HSMM abstract machine →
  Zynq, closing the theory → systems → compiler arc.
- **Authoring surface** — the in-browser editor (this period): multi-file imports
  & vocabulary profiles, parametric hypergraph generators (Steiner systems,
  sunflowers), arc-value editing, composite layout — toward a usable design tool.

---

## Insertion notes (pending `python-pptx`)
- Tensor-view slide is **slide 12** ("Star vs clique: the efficiency argument").
  Background images: `figures/star_expansion.png` (sparse signed incidence) and
  `figures/clique_expansion.png` (dense co-membership) — place behind the text at
  reduced opacity (≈25–35 %), star left / clique right.
- Add **Slide A** after slide 33 (Contributions) and **Slide B** before/with
  slide 34 (Conclusion & outlook), or fold B into the outlook.
- Editing the `.pptx` in place needs `python-pptx` (not installed → a §1
  dependency add, awaiting approval). Heatmaps + this content are dependency-free.

---

## Further resources (collected 2026-06-16)

Additional related work to position HyMeKo against, organised by the deck's threads.
These are candidates for a backup/appendix references slide or for the related-work
section of the journal extensions — not all need to appear in the talk.

**Hypergraph neural networks (beyond HGNN)**
- Yadati et al., *HyperGCN: A New Method for Training GNNs on Hypergraphs*, NeurIPS 2019
- Dong, Sawin & Bengio, *HNHN: Hypergraph Networks with Hyperedge Neurons*, 2020
- Huang & Yang, *UniGNN: a Unified Framework for Graph and Hypergraph NNs*, IJCAI 2021
- Chien et al., *You are AllSet: a Multiset Function Framework for Hypergraph NNs*, ICLR 2022
- Antelmi et al., *A Survey on Hypergraph Representation Learning*, ACM CSUR 2023

**Signed / directed graph learning (context for HSiKAN)**
- Tang et al., *A Survey of Signed Network Mining in Social Media*, ACM CSUR 2016
- Li et al., *Learning Signed Network Embedding via Graph Attention (SNEA)*, AAAI 2020
- Shu et al., *SGCL: Signed Graph Contrastive Learning*, 2021
- Ko et al., *Signed graph learning surveys*, 2023–2024 (for the strict/transductive convention debate)

**Kolmogorov–Arnold networks (the fast-moving 2024–25 line)**
- Liu et al., *KAN 2.0: Kolmogorov–Arnold Networks meet Science*, 2024
- *FastKAN* (RBF approximation of the spline basis), 2024
- *Wav-KAN* (wavelet bases), 2024; *Chebyshev-KAN*, 2024 (basis-choice alternatives to Catmull–Rom)
- Bozorgasl & Chen, and the broader "KAN-for-graphs" preprints, 2024–25

**Curvature, over-squashing & geometric deep learning**
- Di Giovanni et al., *On Over-Squashing in Message Passing NNs*, ICML 2023
- Nguyen et al., *Revisiting Over-smoothing and Over-squashing using Ollivier-Ricci Curvature*, ICML 2023
- Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, Gauges*, 2021
- Brandstetter et al., *Clifford Neural Layers for PDE Modeling*, ICLR 2023
- Ruhe et al., *Clifford Group Equivariant Neural Networks*, NeurIPS 2023

**Systems / MBSE / dataflow (the framework side)**
- Steinberg et al. and the OMG *SysML v2 / KerML* specifications (2023–) — the metamodel HyMeKo targets
- Lattner & Adve, *LLVM*; the *MLIR* dataflow-IR line (context for the `.hymeko` IR → codegen path)
- *Apache Arrow* + *DLPack* specifications (the zero-copy interchange HyMeKo uses)

> Provenance: titles/venues are from widely-cited works; confirm exact years/authors
> against the project `.bib` files before they go into a submitted manuscript.
