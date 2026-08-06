# Kato Seminar Prep

This is the rehearsal-facing version of `kato_seminar_review`: keep the deck
large, but keep the spoken frame narrow and controlled.

## One Sentence Frame

HyMeKo has two coupled contributions: first, a canonical hypergraph
infrastructure for engineering structured systems; second, a structural-prior
learning family where cycles and walks become inductive features.

Use this sentence early, then return to it during transitions:

> This is a unified representation-and-learning substrate, not a bag of
> unrelated techniques.

## Four-Block Talk Arc

1. **Problem** - real systems are n-ary; pairwise projection loses relation
   identity.
2. **Framework** - HyMeKo DSL, canonical IR, star expansion, and code generation.
3. **Learning** - cycles and walks as structural priors; SignedKAN, HSiKAN, and
   Gomb.
4. **Evidence** - honest protocol, leakage audit, results, demo, and future
   collaboration.

Avoid presenting the talk as 43 separate slides. Treat each slide as evidence
inside one of these four blocks.

## Strongest Anchors

- **Star vs clique efficiency:** 1,498 vs 10,991 NNZ; 79.7 ms vs 496 ms. This is
  the cleanest engineering claim.
- **One graph, many targets:** URDF, SDF, MJCF, SysML, and PyTorch from one IR.
  This is the robotics-lab relevance hook.
- **Leakage / honesty protocol:** the previous sigma-leakage issue is not hidden;
  it becomes methodological evidence.
- **Gomb-strict results:** present as honest, 5-seed, paired evidence, not
  "magic SOTA everywhere."

## Tame The Big Side Branches

Use one-sentence boundaries for the topics that can make the talk feel too wide.

**Clifford-FIR / quaternion / visual cortex analogy:**

> This is the geometric machinery under the cascade; I will not claim
> neuroscience equivalence, only a useful layered analogy.

**HyMeYOLO / Ricci / Hodge:**

> This is not the main claim today; it is a stress test showing that the same
> structural-prior idea transfers outside signed graphs.

**Gomb name:**

> "Gomb" means sphere in Hungarian; here it names the strict cascade
> configuration.

## Likely Kato Questions

**What is the minimum working demo?**

`.hymeko` source to IR, star expansion, PyTorch / URDF output, plus the HSiKAN
graph-only kinematics demo.

**What is novel compared with existing hypergraph neural networks?**

Not just an HGNN layer: canonical hypergraph infrastructure, signed cycle/walk
tuple priors, and a strict leakage-audited protocol.

**What is the relation between the engineering framework and the learning model?**

The same IR provides both the engineering transform and the learning substrate;
they are one representation principle, not two projects.

**Is the 0.996 result leakage-free?**

No. That is the transductive-convention number. The strict baseline is the honest
claim.

**What can we collaborate on?**

Robot-structure learning: encode robot morphology and task structure in HyMeKo,
then test whether structural priors improve sample efficiency or generalization
in control.

## Rehearsal Discipline

- Start with the two contributions.
- Name what is proven, what is prototype, and what is research program.
- Keep the baseline story objective: competitive-to-leading, especially strong
  on Epinions, not universal SOTA.
- Do not add more claims during Q&A. Translate questions back into the four-block
  frame.
- End by making the collaboration concrete: morphology + task structure in
  HyMeKo, structural priors for control.

