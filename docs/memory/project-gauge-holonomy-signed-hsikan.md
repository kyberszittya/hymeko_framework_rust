---
name: project-gauge-holonomy-signed-hsikan
description: "BIG synthesis (Hajdu/Kato 2026-06-25): signed-graph computation is gauge-theoretic — balance = Z2 holonomy (THEOREM); HSiKAN=fiber, rotor=connection, spikes=timing; the spiral collects parallel-transported fibers at spike-walk ends = a learned continuous holonomy generalizing balance parity. Likely the paper. Long report on disk."
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

**The thesis (technical report `docs/theory/gauge_holonomy_signed_hsikan.{tex,pdf}`, multi-page, compiles).**
Signed-hypergraph computation is a **semiring convolution** B⊛x; the predictive SIGNAL is **holonomy**, not node
features. Established (THEOREM): a signed graph is balanced iff its **Z2 holonomy** (product of signs around every
cycle, Cartwright–Harary) is trivial — signs ARE a Z2 connection, balance IS its trivial holonomy; frustration =
non-trivial holonomy = where the AUROC signal hides. The **Spin lift**: replace Z2 by Spin(d), each arc a learned
**rotor** R_uv (a connection); π:Spin→Z2 recovers the sign, so the rotor holonomy Hol_Spin(C) is a continuous
invariant whose Z2 projection IS the balance parity (PROVEN refinement, Prop.).

**Architecture (the key distinction from the SETTLED negative).** Prior: rotor/quat AS READOUT on raw cycle
features did NOT beat real bilinear (readout algebra not the bottleneck — [[project-hsikan-geometric-attention-berge]]).
NEW: HSiKAN = the **fiber** (learned per-vertex section via node_activations); the rotor = the **connection** that
parallel-TRANSPORTS those learned fibers along **spike-walks** (tropical min,+ timing = shortest signed walk first);
the **spiral collects the transported fibers at the walk ENDS** → ŷ_uv = bilinear(⊕_W α_W Hol_Spin(W)·h), readout
FIXED at real-bilinear. So the new signal is in the TRANSPORT, not the readout — does NOT re-run the settled
experiment. Three-part code: sign (Z2) · timing (tropical) · ordered phase (Spin holonomy). Ordered ⊗ on walks may
be non-commutative (rotor) — fine; only ⊕ over unordered hyperedge members must be commutative.

**Claims (tagged):** T1 balance=holonomy + Spin refinement (THEOREM, done). C1 AUROC bet: holonomy-transport vs
flat readout on OTC, readout fixed (CONJECTURE, untested, the principled lift — estimates frustration the flat code
discards). C2 spike sparsity/event-rate (near-certain efficiency → neuromorphic = the rate-asymmetric fast reflex).
C3 HSiKAN>MLP ∝ topological complexity (quadruped/k-agent, untested). E1 hsikan_diagnose debuggability (BUILT +
**DEMONSTRATED 2026-06-26**): wired into the off-policy loop (`offpolicy_eval` diagnoses actor+critics each eval,
aborts+localizes), it collapsed a multi-hour slow-vs-diverging ambiguity into one signal (`diverged=False` →
quadruped is COMPUTE-bound not diverging). **Value prop = HSiKAN spares trial-and-error as a DEBUGGING INSTRUMENT**
(read the named failure — "up-chain signed-agg W− L0" — instead of bisect-searching LR/clip/seed). Lead with THIS
at the venue. Keep SEPARATE from the structural-prior claim (less architecture/data search = C3, still open) — the
debuggability win must not be used to sell the prior win. NB the actual divergence FIX (grad-clip + reward-norm in
ddpg/sac) was a STANDARD RL fix, not HSiKAN-specific; HSiKAN's contribution was the *localization*, not the cure.

**HOLONOMY TOY — PRINCIPLE DEMONSTRATED + ANALYTICALLY VERIFIED (2026-06-26, `hymeko_rl/rotor_probe.py`,
report `2026-06-26-rotor-holonomy-toy`, plan `2026-06-26-rotor-spikes-ablation`):** the cleanest isolation of
T1's core — transport a source vector x∈R² around a cycle, target y=R(Φ)x (Φ=holonomy). Rotor (SO(2), learned
angle) MSE≈0 at EVERY Φ; signed (Z2 scalar, learned gain) MSE=**sin²Φ** — zero ONLY at Φ∈{0,π} (the Z2 points
where holonomy IS a sign), blind to the rotation everywhere else. signed/rotor ratio @π/2 ≈1e9, and the measured
signed curve matches the CLOSED-FORM sin²Φ to ~1% (c*=cosΦ is the cosine-shadow projection). So "signed sees only
the Z2 quotient; rotor sees the full continuous holonomy" is now a verified picture, not just a theorem — a clean
Kato-facing figure (`reports/rotor_probe/rotor_probe.png`). This is the PRINCIPLE behind C1 (still must run the
OTC AUROC test, readout-fixed, to show it pays on real data). Production rotor already exists (quaternion/SO(3)
`RotorInjector`/`CayleyRotorEmbedding`/`SignedRotorPropagation` in signedkan_wip run_hsikan_rotor.py) — toy is a
minimal SO(2) isolation, NOT a rebuild. SPIKES toy (designed, not built): needs non-abelian SO(3) where walk
holonomy is order-dependent → spike timing selects the time-ordered walk (diamond graph, two non-commuting paths).

**SPIRAL TOY — HIGHWAY=STRUCTURE-FREE-SPIRAL CONJECTURE CONFIRMED (2026-06-27, `hymeko_rl/spiral_probe.py`,
report `2026-06-27-chain-and-spiral-toys`, user insight):** the HSiKAN highway gate `out=T·H+(1−T)·carry` is the
structure-FREE skeleton of the spiral `ŷ=⊕_W α_W Hol(W)·h` — the highway's CARRY transports with IDENTITY (no
rotation/holonomy); the spiral replaces it with rotor parallel-transport along walks + α-collect. Toy: θ-graph,
K parallel walks, vary connection θ, target y=mean_k R(Σ_{e∈W_k}θ_e)·x. **spiral MSE≈0 at all K** (rotor-transport
+ α-collect, handful of params); **plain highway_mlp FAILS (0.3–0.6), = the flat MLP** → the gate carries NO
structural signal. This EXPLAINS the galambos `skip=highway` null (2026-06-26-pernode-actor: pernode_hw 0.20 =
pernode 0.20, +5.3k params, 0 gain): plain highway adds capacity not signal. So the fix is the SPIRAL (carry =
rotor-transport-along-walks), not the plain highway — `highway = spiral with identity connection + walk collapsed
to depth`. Spiral unifies per-node-readout(✓ fires) + rotor-connection + spike-walk-timing into one collector,
highway as skeleton = C1 realized AS the skip layer. **NEXT (real prize):** replace HSiKAN's highway carry with
rotor-transport-along-walks (CayleyRotor + hymeko_graph topk_walks + α-collect), test on a structural robot
reward. Also CHAIN toy (`2026-06-27-chain-and-spiral-toys`): HSiKAN's sparse signed reasoning beats MLP on a
CHAIN (the audit's "structure-poor" worst case) — 4.5× at len-4 → **41× at len-16**, HSiKAN error FLAT in length
while MLP explodes = sparse-reasoning length-invariance. Even a chain's kinematic structure is exploitable.

**SPIKES TOY — TRILOGY COMPLETE (2026-06-27, `hymeko_rl/spike_probe.py`, report `2026-06-27-spike-toy`):** the
non-abelian piece. Two non-commuting SO(3) rotations R_a=R_x(θ),R_b=R_y(θ); target y=(spike? R_aR_b:R_bR_a)·v;
sweep θ (=non-commutativity). At θ=0 (commuting) the ORDER-BLIND model fits (order irrelevant); as θ grows it
fails ∝ commutator (0→0.45 MSE) — can't select order without the spike; SPIKE_GATED ≈0 everywhere (uses the
spike to select order). Oracle test (true angles reproduce target to 1e-5) confirms order convention. mlp also
fits (sees s) → load-bearing point: the SPIKE INFO is necessary, order_blind (no s) can't represent non-abelian
holonomy. **STRONGER VERSION (`--walks`/run_spike_walks): a GENERALIZATION win.** Varying connection + 2 walks ×
length-L SO(3) chains (cycling axes, non-commuting), spike selects walk, target=selected-walk holonomy·v.
spike_gated (composes along the spike-selected walk) MSE≈0 at all L; flat MLP degrades with L (0.01→0.30 at L=4)
— can't learn the L-fold conditional non-commuting product. So spike_gated BEATS the flat net and the gap GROWS
with composition depth (chain-probe signature) — turns "spike necessary" into "structured spike-gated WINS".
Oracle test (unit gain reproduces target to 1e-5, L=1..4) confirms the batched composition is correct. **ROTOR/SPIKES TRILOGY now all built+tested+figured:** rotor(SO2: connection, signed sees only sin²Φ
shadow) → spiral(SO2: collects rotor-transport along walks, plain highway carries none) → spike(SO3: timing
selects walk order when non-abelian). That's the full gauge stack as clean supervised demonstrations. **NEXT
(the real prize, item #2 on the roadmap):** rotor IN HSiKAN — production rotor-connection carry (reuse
CayleyRotor + hymeko_graph topk_walks + α-collect), spike-ordered walks for non-abelian, tested on a structural
robot reward.

**Risks:** Hol_Spin may overfit frustration noise (regularize toward Hol≈1); the head must be GAUGE-INVARIANT
(use ‖log Hol_Spin‖ / class traces, else the holonomy washes out); fuzzy non-distributive → single-layer.
**Paper framing:** gauge-theoretic signed-graph learning (balance=Z2 holonomy, HSiKAN-fiber+rotor-connection=Spin
lift, spikes=event timing) — deep+novel, not "another KAN variant"; opens neuromorphic venues. Unifies
[[project-hymeko-aggregation-semantics]] (the semiring substrate) + [[project-cayley-rotor-idea]] (rotor re-roled
metric→connection) + walk-spikes + [[project-actor-critic-shared-reasoning]] (structural critic = holonomy aggregate).
**How to apply:** implement = rotor-connection layer over the HSiKAN fiber + holonomy readout (walks already in
hymeko_graph) + the aggregate semiring extension; parity guard π(Hol_Spin)=BalanceScorer; test C1 readout-fixed.
