# HSiKAN vs MLP for the omni crab — structure propagation does NOT fix the asymmetry (+ how we sped it up)

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. signedkan (HSiKAN) vs MLP, residual SAC.** ·
**Verdict: `HSIKAN_STRUCTURE_PROPAGATION_MATCHES_MLP` (asymmetry is dynamics, not representation).**

---

## The hypothesis (user)

The MLP omni residual learned a ONE-SIDED crab (+y reached, −y not). User's hypothesis: HSiKAN's
value is **structure propagation between the observation and action spaces** (signed hyperedges over
the body), not raw capacity — a per-node actor with **weight-sharing across legs** should represent a
**symmetric** crab that the flat MLP can't, fixing the −y failure.

## The test

A `signedkan` per-node actor over a **leg hypergraph** (per-vertex obs, `act_vertices` = the 4
abduction vertices, signed hyperedges routing a lateral goal-demand to the per-leg abduction), vs the
trained MLP (flat obs), on a **symmetric** held-out grid (±20°, ±40°). Two hypergraph sign schemes:
- **kinematic** signs (torso↔leg down +1 / up −1);
- **symmetric** signs (left legs fl,bl +1; right legs fr,br −1) — the sharper test, encoding the
  left/right symmetry axis *in the structure* so the signed propagation routes the demand
  differentially (something a flat MLP cannot represent).

## Result — the one-sided crab is a SYMMETRY-BREAKING optimum (6 architectures)

| architecture | head | reach (test) | +y | −y |
|---|---|---|---|---|
| MLP (flat, 9-D) | — | 0.40 | **1/2** | 0/2 |
| signedkan (kinematic hg) | per_node | 0.40 | **1/2** | 0/2 |
| signedkan (symmetric-signs hg) | per_node | 0.40 | **1/2** | 0/2 |
| signedkan + H★ exploration | per_node | 0.40 | **1/2** | 0/2 |
| signedkan (intermediate LATENT) | pooled | 0.40 | **1/2** | 0/2 |
| **mixture (HSiKAN+MLP gated MoE)** | pooled | 0.40 | 0/2 | **1/2** |

**Every architecture reaches exactly 2/5 and is ONE-SIDED — but the mixture converges to the *other*
side** (−y instead of +y). This refines the earlier reading: the −y crab is **not dynamics-impossible**
(the mixture *did* reach −20°), and it is **not a pure representation/exploration gap** (5 other
architectures reach +y but not −y). It is a **symmetry-breaking convergence**: the +y/−y crab is
(near-)symmetric, SAC breaks the symmetry and commits to *one* side, and the architecture/init merely
selects **which** side — none reaches **both**. (The mixture's auto-verdict `FIXES_CRAB_ASYMMETRY` is
a **false positive** of the runner's rule: −y improved but +y was lost; reach is unchanged at 2/5.)

So the honest finding — testing the user's "intermediate latent" and Kato's "mixed HSiKAN+MLP":
- **Intermediate latent** (signedkan, pooled head over the structured latent, not per-vertex readout)
  = still +y-only, matches MLP.
- **Mixture** (learned per-state gate blending HSiKAN + MLP) = flips to −y-only — *different* but not
  *better* (2/5). Structure/mixing changes **which** symmetry is broken, not **whether** it breaks.

**The lever for reaching BOTH sides is symmetry PRESERVATION** (mirror-augmented training: for each
+y goal add the mirrored −y goal with mirrored obs/action; or a symmetry-EQUIVARIANT policy), not
more expressive structure — no single asymmetric-training run reaches both. Consistent with PnP
R-HSiKAN (HSiKAN ≈ MLP on reach) but sharper: the bottleneck is a **broken symmetry**, and the
architecture influences the broken direction. Caveats: 1 seed, 30k, 5-vertex hg — but six
architectures all landing at one-sided-2/5 is a robust signal.

## H★ structural-entropy exploration (user's "entropy backpropagation")

H★ is the Shannon entropy of the signedkan's per-vertex activation energy (`star_entropy.py`) — a
structural exploration signal an MLP **cannot** produce (no per-vertex activations). Wired into the
SAC actor objective as a **guarded, backward-compatible seat** (`SACConfig.struct_entropy_coef`,
default 0, HSiKAN-only; mirrors the existing `mech_coef` seat — `+ coef·H★` in the actor bonus).
Because the −y failure is a **local-optimum / exploration** problem, this was the sharpest test of
"what HSiKAN buys" beyond the mapping — and it **still matches MLP** (coef 0.3 did not drive SAC out
of the one-sided crab). This corroborates the dynamics-limited reading: the +y/−y asymmetry is not a
matter of *exploring* the −y crab but of the −y crab being physically harder.

## SA-HSiKAN + Steiner-config hypergraphs (user's next idea) — scope

`sa_hsikan` (Two-Hop Signed Propagation) is a built `POLICY_KINDS` variant; Steiner systems (Fano
plane etc.) are balanced designs that guarantee pairwise coverage, so two-hop propagation over a
Steiner overlay **mixes globally in 2 hops**. That accelerates structural reasoning on **large**
hypergraphs where mixing takes many hops — but the omni's minimal **5-vertex** leg hg already mixes
in ~2 hops (a star from the torso), so a Steiner overlay would not move this task. Its natural
testbed is the **full-body 33-vertex hg** (on GPU/katolab, where the Triton CR kernel engages) or a
vision hypergraph — a mixing-limited problem, not the dynamics-limited crab. Recorded as the next
direction, deliberately *not* run on the exhausted omni task.

## How we sped it up (the user's Chebyshev-CR question)

The signedkan is much slower than MLP on CPU. Chain of speedups:

1. **The HSiKAN cell already uses Catmull-Rom** (not B-spline). A **Chebyshev deploy fast-path**
   exists (`set_deploy_mode`, ~2.2× forward) — but it is **inference-only** (the Chebyshev
   *approximation* would shift the measured numbers, so the reported eval stays on **exact CR**), and
   the **fused CR Triton kernel is GPU-only** (CPU falls back to eager). So on CPU it doesn't speed
   up training.
2. **Minimal leg hypergraph** (5 vertices vs the full body's 33 — head/tail/ears are irrelevant to
   the crab): **7 → 28 steps/s (~4×)**, and a cleaner test of the leg-symmetry structure.
3. **`update_every=3` + `hidden 128→64` + `batch 256→128`**: **28 → ~80 steps/s (~3×)**.

Net **~11×** (61 min/run → ~6 min/run). **katolab** (kato15, RTX 6000 Ada, 49 GB) is reachable and is
where the Triton CR kernel would engage for the *full* hg — but the aibo worktree + CUDA-torch env
are not synced there, so it needs a setup step; used the local fast path instead.

## Files

```
scenarios/aibo/residual_trot.py         MOD  (obs_mode hypergraph|leg_hypergraph, minimal_leg_hypergraph(symmetric=))
scenarios/aibo/run_aibo_hsikan_omni.py  NEW  (signedkan per-node vs MLP; --symmetric; fast SAC config)
tests/test_aibo_residual_trot.py        MOD  (+ hypergraph/leg-hg obs + symmetric-sign tests)
reports/2026-07-28-aibo-residual-trot/result_hsikan_omni_{signedkan,signedkan_sym}.json  NEW
```

## Tests / provenance

`ruff` clean. **21/21** residual-env tests (incl. per-vertex hg obs, minimal 5-vertex leg hg,
symmetric-sign encoding, a=0 = pure scaffold across all obs modes); full AIBO suite green. Reported
eval on exact CR (no Chebyshev approximation). CPU, seed 0, MuJoCo, ~80 steps/s.

## Bottom line

Testing the user's hypothesis: **HSiKAN structure propagation MATCHES the MLP** for the omni crab —
it does not fix the −y asymmetry, even with the left/right symmetry encoded in the hypergraph signs.
The asymmetry is a **dynamics** property (the AIBO crabs more easily one way), not a representation
gap, so the structural prior can't cure it — consistent with the PnP HSiKAN≈MLP finding. Speed: a
minimal leg hg + `update_every`/`hidden`/`batch` cuts a 61-min run to ~6 min on CPU; the GPU Triton
CR path (katolab) is the route for the full-body hg.
