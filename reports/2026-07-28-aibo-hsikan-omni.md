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
| **signedkan + HIGHWAY skip** (the 'H') | per_node | 0.40 | **2/2** | 0/2 |
| **sa_hsikan (cr_cheby Chebyshev cells)** | per_node | 0.40 | **1/2** | 0/2 |

**Every architecture reaches exactly 2/5 and is ONE-SIDED — but the mixture converges to the *other*
side** (−y instead of +y). (The bottom two rows — **highway skip** and **Chebyshev `cr_cheby`** — are
the two *namesake* HSiKAN features that were **default-OFF** in all earlier runs; §"Two namesake
features" below. Turning them on leaves reach at 0.40 and **−y still 0/2** — the decisive
confirmation that the −y failure is scaffold dynamics, not representation. Highway *did* improve the
**+y** side to 2/2 by hitting both off-axis +y goals, i.e. it re-selected *which* +y goals within the
already-broken side, never crossing to −y.) This refines the earlier reading: the −y crab is **not dynamics-impossible**
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

## Two namesake features that were default-OFF — highway + Chebyshev (user's follow-up)

The user asked whether the trained HSiKAN actually used **Chebyshev-CR** and **highway connections**.
Reading the code (not memory): **neither**. The signedkan/mixture backbones default to `activation="cr"`
(exact Catmull-Rom, piecewise cubic — *not* Chebyshev) and `skip="none"` (plain signed-conv). The two
features exist but were off:
- **highway** (`skip="highway"`, the Schmidhuber gate — literally the *H* in **H**SiKAN) — available,
  never enabled;
- **Chebyshev** (`cr_cheby` cells + the `deploy_policy`/`set_deploy_mode` fast path, ~2.2× forward) —
  **inference-only** and **validated** (kept off in eval so the approximation can't shift the numbers),
  and only the **default for `sa_hsikan`**, which we had not run on the omni. The fused CR Triton kernel
  is GPU-only anyway (CPU falls back to eager).

So we ran the fair "full HSiKAN" A/B (30k, same fast config): `signedkan --skip highway` and
`sa_hsikan` (cr_cheby). **Both stay at reach 0.40 and −y 0/2** (the last two table rows). Highway
raised **+y to 2/2** (both off-axis +y goals) but never crossed to −y and *lost* the bearing-0 goal —
i.e. it re-shuffled *which* goals inside the already-broken +y side, exactly the "architecture selects
the broken direction, not whether it breaks" reading. cr_cheby matched the MLP one-sided. **This is the
sharpest confirmation of the scaffold-asymmetry conclusion** (`reports/2026-07-28-aibo-crab-symmetry-resolved.md`):
enabling the two namesake HSiKAN features does not touch the −y failure, because the failure is a
property of the **running diagonal-trot scaffold**, not of the policy representation.

Wall/RSS: highway ~70 steps/s (the per-layer gate adds ops), sa_hsikan ~131–166 steps/s (the Bᴸ
collapse is the cheapest backbone); peak RSS 379 MB / 354 MB — far under the 16 GB cap.

## SA-HSiKAN + Steiner-config hypergraphs (user's earlier idea) — scope

`sa_hsikan` (Two-Hop Signed Propagation) is a built `POLICY_KINDS` variant; Steiner systems (Fano
plane etc.) are balanced designs that guarantee pairwise coverage, so two-hop propagation over a
Steiner overlay **mixes globally in 2 hops**. That accelerates structural reasoning on **large**
hypergraphs where mixing takes many hops — but the omni's minimal **5-vertex** leg hg already mixes
in ~2 hops (a star from the torso), so a Steiner overlay would not move this task (confirmed above:
sa_hsikan is one-sided-2/5 like the rest). Its natural testbed is the **full-body 33-vertex hg** (on
GPU/katolab, where the Triton CR kernel engages) or a vision hypergraph — a mixing-limited problem,
not the dynamics-limited crab.

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
scenarios/aibo/run_aibo_hsikan_omni.py  MOD  (signedkan/mixture/sa_hsikan per-node vs MLP; --symmetric; --skip {none,residual,highway}; fast SAC config)
tests/test_aibo_residual_trot.py        MOD  (+ hypergraph/leg-hg obs + symmetric-sign tests)
tests/test_aibo_hsikan_skip_wiring.py   NEW  (5 tests: highway skip adds gate params / none has none / sa_hsikan builds+ignores skip)
reports/2026-07-28-aibo-residual-trot/result_hsikan_omni_{signedkan,signedkan_sym,signedkan_per_node_highway,sa_hsikan_per_node}.json  NEW
```

## Tests / provenance

`ruff` clean. **102/102** AIBO tests green — incl. the residual-env suite (per-vertex hg obs, minimal
5-vertex leg hg, symmetric-sign encoding, a=0 = pure scaffold across all obs modes) and the new
**5 skip-wiring tests** locking that `--skip highway` genuinely adds the per-layer Schmidhuber gate
(not a silent no-op) and `sa_hsikan` builds while ignoring skip. The reported signedkan/mixture eval is
on exact CR; `sa_hsikan` uses `cr_cheby` cells in exact (non-deploy) mode — no Chebyshev *fast-path
approximation* engaged, so the numbers are un-shifted. CPU, seed 0, MuJoCo; ~70 steps/s (highway),
~131–166 (sa_hsikan); peak RSS 379/354 MB.

## Bottom line

Testing the user's hypothesis: **HSiKAN structure propagation MATCHES the MLP** for the omni crab —
it does not fix the −y asymmetry, even with the left/right symmetry encoded in the hypergraph signs,
**and even with the two namesake features (highway skip + Chebyshev `cr_cheby`) turned on** — both
stay at reach 0.40, −y 0/2. The asymmetry is a **dynamics** property (the AIBO crabs more easily one
way, rooted in the running diagonal-trot scaffold — `reports/2026-07-28-aibo-crab-symmetry-resolved.md`),
not a representation gap, so no structural prior, gate, or spline basis cures it — consistent with the
PnP HSiKAN≈MLP finding. Speed: a minimal leg hg + `update_every`/`hidden`/`batch` cuts a 61-min run to
~6 min on CPU; the GPU Triton CR path (katolab) is the route for the full-body hg.
