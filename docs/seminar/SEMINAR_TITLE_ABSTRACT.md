# Seminar — title & summary (for Prof. Shohei Kato)

## Recommended title
**HyMeKo: One Hypergraph Substrate for Representing and Learning from Structured Systems**
*From declarative hypergraph structure to structural-prior learning.*

### Alternative titles
- Structure as a Prior: A Hypergraph Framework Bridging Systems Modelling and Signed-Graph Learning
- HyMeKo: When the Graph You Compile Is the Graph the Model Learns From

---

## Abstract (~200 words)
Many systems we care about — robot mechanisms, trust and affect networks,
chemical processes, neural dataflow — are inherently *n-ary*: their relations
bind several parts at once. Forcing them into pairwise graphs both loses the
identity of each relation and inflates the representation to O(|E|·d²). HyMeKo
takes the hypergraph seriously as a first-class object.

This talk presents HyMeKo as one hypergraph substrate with two payoffs. As a
**framework**, a declarative language compiles to a canonical hypergraph
intermediate representation with content-hash identity; from that single source
a query-and-template engine emits URDF, SDF, MJCF, SysML v2 and PyTorch modules
with guaranteed cross-view consistency, and an O(|E|·d) star expansion streams
zero-copy into PyTorch. As a basis for **learning**, the same hypergraph's
cycles become an inductive prior: the SignedKAN → HSiKAN → Gömb family replaces
fixed activations with learnable Kolmogorov–Arnold splines over cycle and walk
tuples, reaching competitive-to-leading signed-graph link prediction at a
fraction of the parameters, under leakage-audited honest protocols. The same
primitive transfers to vision and to recovering kinematics from topology alone.

The unifying idea — and the closing live demos — show that the structure the
framework compiles is the structure the model reasons over.

---

## One-line summary (for a programme listing)
A hypergraph framework and intermediate representation that both *describes*
structured systems (robotics, SysML, signed graphs) and *learns* from them via
cycle-based Kolmogorov–Arnold priors — one substrate, two payoffs.

## Speaker
Csaba Hajdu — hypergraph systems and signed-graph learning.

## Keywords
hypergraphs · intermediate representation · canonical hashing · model-based
systems engineering · signed-graph link prediction · Kolmogorov–Arnold networks
· structural priors
