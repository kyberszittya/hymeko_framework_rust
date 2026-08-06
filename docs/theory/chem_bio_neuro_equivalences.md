# Chemical, biological, and neurological equivalences of the signed-hypergraph / walk-holonomy controller

**Drafted:** 2026-06-28 (overnight) · for Dr. Cs. Hajdu · **Status:** cross-disciplinary synthesis; connections
graded *deep* (established, structural) vs *loose* (suggestive analogy). Honesty matters here — the value is in the
*deep* ones being testable, not in the count.

## The shared object

Everything we generated is the same mathematical animal: a **signed (or rotor-) hypergraph carrying a connection**,
whose **holonomy** (sign/phase product around a cycle) is the load-bearing invariant, and whose best basis is a
**balanced combinatorial design**. That object is *not* unique to control — it is exactly the structure of several
physical, chemical, and neural systems. Where the SAME structure recurs, the math transfers.

---

## Chemistry

**(DEEP) Aromaticity = cycle holonomy — signed graphs *are* Hückel theory.** The Hückel Hamiltonian of a
π-conjugated molecule is the (signed) adjacency matrix of its atomic graph; orbital phases are the ±. A ring's
electronic stability is **Hückel-aromatic (4n+2)** when the phase closes trivially and **Möbius-aromatic (4n)**
when it closes with a sign flip — i.e. **aromaticity is the Z₂ holonomy of the π-system around the cycle**, and
Möbius aromaticity (Heilbronner) is a *frustrated* (unbalanced) ring. Our "balance = trivial holonomy" is
literally the aromaticity rule. *This is the single deepest equivalence: our cycle-holonomy and chemists'
aromaticity are the same theorem.*

**(DEEP) Sunflower = coordination complex.** A central metal + ligands is a shared core with disjoint petals —
the sunflower hypergraph exactly. **(MODERATE) k-uniform = multi-centre bonds** — a 3-centre-2-electron bond
(B–H–B bridges in boranes) is a genuine 3-uniform hyperedge, not two 2-edges. **(MODERATE) Reaction networks**
are hypergraphs (a reaction with k reactants = a k-edge); chemical reaction network theory and flux balance live
on them, and the holonomy is the thermodynamic loop law (Kirchhoff/Wegscheider).

---

## Biology

**(DEEP) Gene-regulatory & signalling networks = signed graphs; loop sign = holonomy = stability.** Activation
(+) / inhibition (−) are signs; a feedback loop's **net sign is the product around the cycle** — *positive*
feedback (even # of −, trivial holonomy) gives bistable switches/memory, *negative* feedback gives homeostasis or
oscillation. "Balanced vs frustrated cycle" is the biologist's "positive vs negative feedback loop." (MODERATE)
Metabolic networks are hypergraphs; **(LOOSE)** developmental/morphogen patterning as a discrete connection.

---

## Neuroscience (the emphasis — and where it is strongest)

**(DEEP) E/I balance = signed-graph balance.** Excitatory/inhibitory synapses are the ±; cortical dynamics are
organised around E/I *balance*, and a recurrent loop's product-of-signs sets runaway (epileptiform, positive) vs
regulated (negative) behaviour. Our signed adjacency *is* the E/I connectivity matrix; frustration = the seizure/
stability axis.

**(DEEP — the crown jewel) Grid cells = path integration = holonomy over an affine/hexagonal geometry.**
Entorhinal **grid cells** fire on a triangular (hexagonal) lattice — a finite-affine-plane-like, tight-frame-like
tiling — and are believed to perform **path integration**: integrating velocity to track position. *Path
integration is parallel transport; the accumulated position is the **holonomy** of a connection over the lattice.*
Recent work (Gardner et al., 2022) shows grid-cell population activity lives on a **torus** — a flat compact
manifold with a connection. So the brain's spatial system is, structurally, **a walk-holonomy computer over a
designed lattice** — which is exactly the StructuralActor. *This is the most important equivalence: our
architecture is a normative model of grid-cell computation.*

**(DEEP) Ring/torus attractors = rotors.** Head-direction cells form an **S¹ ring attractor** — our SO(2) rotor;
the grid torus is the SO(2)×SO(2) case. The rotor toy is the head-direction system; transport around the ring is
the holonomy that updates heading.

**(DEEP) Spike-rotor = phase coding.** Your spike-rotor (event timing + phase) is **theta–gamma coupling** and
**phase precession**: a place/grid cell's spike *timing relative to the oscillation phase* carries position —
spike (event) × rotor (phase). "Spike-rotor for jumps / message-passing channels" = bursts and phase-multiplexed
communication channels (Communication-through-Coherence, Fries). The rotor is the carrier; the spike selects the
slot.

**(MODERATE) Dendrites = KAN nonlinearity.** Dendritic branches integrate inputs nonlinearly (each branch ≈ a
learned univariate nonlinearity before summation) — the KAN's per-edge function, lifted in our block-wise
Steiner-KAN to per-hyperedge dendritic subunits. **(MODERATE) Higher-order connectome = hypergraphs / simplicial
complexes** — cortical microcircuits contain high-dimensional cliques and topological cavities (Reimann et al.,
2017); neural computation is higher-order, not pairwise.

---

## The synthesis, and the honest line

Read top to bottom, the *deep* equivalences are one statement: **a signed-hypergraph connection whose holonomy is
the invariant, best-conditioned on a balanced design, is the common structure of aromatic stability, regulatory
feedback, E/I balance, and — most strikingly — grid-cell path integration.** The StructuralActor is therefore not
just an engineering shortcut; it is the *same computation* the entorhinal system appears to run: transport over a
designed geometry, read out by holonomy.

That makes it a **normative model**, and a normative model earns a falsifiable prediction:

1. **Grid-cell test:** a StructuralActor whose topology is a hexagonal/affine lattice, trained to path-integrate
   (velocity → position), should develop grid-like / toroidal population structure — the way RNNs and LSTMs do
   (Cueva–Wei, Banino et al.). If it does, "control in a bounded affine geometry via holonomy" is the grid-cell
   computation, stated in control terms.
2. **Aromaticity test (chemistry, free):** the actor's cycle-holonomy classifier should reproduce Hückel vs Möbius
   (4n+2 vs 4n) labels exactly — a zero-cost validation that our holonomy *is* the chemists' invariant.
3. **E/I test:** frustration (unbalanced cycles) should predict instability/oscillation in the closed loop, as it
   predicts seizure-like runaway in E/I circuits.

The loose analogies (morphogenesis, generic reaction networks) are worth noting but not worth leaning on. The
three deep ones — **aromaticity, E/I balance, grid-cell holonomy** — are where a real bridge exists, and the
grid-cell one is the prize: if the architecture spontaneously produces grid codes, the geometry-as-mechanism claim
stops being a metaphor and becomes a result in computational neuroscience.
