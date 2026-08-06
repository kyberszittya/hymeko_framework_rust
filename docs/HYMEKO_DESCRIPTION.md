# HyMeKo — description

**One line.** HyMeKo (Hypergraph Model Cognition) is a Rust framework and
domain-specific language for declaratively defining, compiling, transforming, and
learning from hypergraphs — one substrate that both represents structured systems
and learns from their structure.

**Short.** Many systems we model — robot mechanisms, signed trust and affect
networks, chemical processes, neural dataflow — are inherently *n-ary*: their
relations bind several parts at once. Pairwise graphs cannot hold those relations
without losing each one's identity and inflating the representation. HyMeKo takes
the hypergraph as a first-class object: a small declarative language describes a
system's parts and the signed, hierarchical, many-to-many relations among them,
and a high-performance engine compiles that description into a single canonical
intermediate representation with a content-hash identity. From that one
representation the framework renders many target formats with guaranteed
cross-view consistency, streams the structure into PyTorch with near-zero
overhead, and exposes the same structure to a family of learning models that use
the graph's own cycles as an inductive prior.

**What it includes.**

- A declarative DSL (`.hymeko`) for hypergraphs, parsed by a SIMD-accelerated
  lexer and an LR(1) (LALRPOP) grammar.
- A canonical hypergraph intermediate representation with Blake3 content-hash
  identity, so isomorphic descriptions share one structural fingerprint.
- A query-and-template transform engine that emits URDF, SDF, MJCF, Graphviz DOT,
  ROS 2 launch, SysML v2, Mermaid and a PyTorch `torch_dataflow` module from the
  same IR — no recompile per target.
- A zero-copy runtime that materialises the O(|E|·d) **star expansion** and
  streams it into PyTorch via Iceoryx2 shared memory, Apache Arrow and DLPack,
  with a topology-hash gate separating structural updates from weight streams.
- A structural-prior learning stack — **SignedKAN → HSiKAN → Gömb** — that turns
  enumerated signed cycles and walks into Kolmogorov–Arnold spline activations for
  signed-graph link prediction, with a vision variant (HyMeYOLO) built on
  Forman–Ricci curvature and Hodge decomposition.

**The idea in one sentence.** The hypergraph the framework compiles is the same
hypergraph the model reasons over — structure is the shared substrate for both
representation and learning.

*Maintainer: Csaba Hajdu. See `README.md` for build/usage and `docs/seminar/` for
the seminar deck and abstract.*
