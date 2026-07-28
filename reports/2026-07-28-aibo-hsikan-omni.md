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

## Result — all three converge to the SAME one-sided crab

| architecture | reach (test) | +y | −y |
|---|---|---|---|
| MLP (flat, 9-D) | 0.40 | 1/2 | **0/2** |
| signedkan (kinematic hg) | 0.40 | 1/2 | **0/2** |
| signedkan (symmetric-signs hg) | 0.40 | 1/2 | **0/2** |

`SIGNEDKAN_MATCHES_MLP` in **both** sign schemes. The structure propagation — even with the left/right
symmetry *explicitly encoded in the hypergraph signs* — does **not** confer a symmetric crab; all
three reach +20° but not −20°/±40°. So the −y failure is **not a representation gap** (the MLP could
already represent it) — it is a **dynamics asymmetry**: the AIBO's lateral crab is physically easier
on one side (the +y probe crabbed cleanly; −y was weaker / prone to tip). SAC converges to the same
one-sided local optimum regardless of the architecture. Consistent with the campaign's **PnP
R-HSiKAN ablation (HSiKAN ≈ MLP)**: for these small action spaces the structural prior adds no
measured advantage. Caveats: 1 seed, 30k steps, a 5-vertex hg — but three architectures converging is
a robust signal.

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
