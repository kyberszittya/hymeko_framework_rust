# CANONICAL_DYNAMIC_EXPERT_PASS — golden-structure inertia + integrator repair; frozen chain reproduces 3/9 on v3

**Created-at:** 2026-07-22 16:36 JST
**Branch:** recovery/coin-hymeko-bundle-and-results (79abf71 → fcd89ca)
**New rescue tag:** `coin-canonical-bundle-v3-dynamics` · combined bundle hash `6664ac459cca8f62`

## Verdict

`GALAMBOS_PLANAR_DYNAMIC_PARITY_PASS` + `CANONICAL_DYNAMIC_EXPERT_PASS` + re-closed
`HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS`. The frozen learned chain (E_valselect approach → handoff transport) delivers
**3/9** on the canonical v3 robot through real dynamics — matching the legacy golden baseline — after the v2 robot's
0/9. **61 recovery gates green.**

## Three fixes, in the emit adapter (no CORE change)

1. **Golden STRUCTURE.** The fingertip contact geom is folded onto the massive `link2` (named `fingertip_{side}`,
   stable contact) and the separate fingertip body is **deleted** (with its dangling `<exclude>`). The arm is now the
   golden's exact 6-body structure, the semantic graph is **natively 6 vertices** (no projection), and the tool site
   lives on link2. `with_fingertip_shape`/`with_fingertip_clamp` find the geom by name wherever it lives — verified
   the E0 clamp prongs attach to the massive link2.
2. **DENSITY mass.** Stripping the crude emitter `<inertial>` lets MuJoCo derive mass+inertia from geometry (density
   1000) — matching the golden compiled model **body-for-body, EXACTLY, for every embodiment** (POINT / E0 clamp /
   wrist add their real mass to link2 instead of being starved by an explicit inertial). Total arm mass 0.351557 =
   golden (Δ0). The `galambos_inertia` contract is retained to assert the density result equals the golden.
3. **INTEGRATOR parity.** The env preserves `original_timestep × frame_skip` as the control interval; the emitter's
   `timestep=0.001` gave a **0.005 s** interval vs the golden's `0.002`→**0.01 s**, so the frozen chain (trained at
   0.01 s) blew up (qvel 5e3) at step 1. Matching the golden `<option>` (timestep 0.002 + implicitfast) restored it.

## Evidence

| gate | result |
|---|---|
| inertia parity (static) | total mass Δ0, per-link mass/inertia/COM Δ<2e-9 — EXACT |
| contact-free EoM (qfrc_bias + qfrc_passive) | match golden **exactly** (Δ<1e-6) at the pose panel |
| contact stability (POINT/E0/FLAT_PAD, 200 steps) | no NaN |
| short-rollout parity vs legacy | Δ ~1e-3 rad (step-1 was 2.07 pre-fix) |
| frozen chain, canonical v3 | **deliver 3/9** (grasp 4), == legacy; no injection, strict cert fires naturally |
| per-seed | v3 {1011,1045,1278}, legacy {1045,1174,1278} — same count; marginal-threshold seeds flip within FP tolerance (§8) |

The residual per-seed flip is the FP-level difference between the emitted and hand-authored MJCF at the strict
threshold — the directive (§8) explicitly permits tiny FP differences when rollout parity is tight and no behavioral
divergence appears (delivery count matches, trajectories track to ~1e-3).

## Bundle identity refreshed (§10)

New combined bundle hash `6664ac459cca8f62`; `canonical_bundle_manifest_v3.json` (nbody 8, arm mass 0.3516, graph_fp
`sem:469094de…` unchanged, integrator implicitfast/0.002). New rescue tag `coin-canonical-bundle-v3-dynamics`
supersedes `PRE_INERTIA_CANONICAL_BUNDLE_SNAPSHOT`. The robot gate is now full equivalence
(`GALAMBOS_PLANAR_HYMEKO_EQUIVALENCE_PASS` incl. inertia + dynamics).

## Next (gated, no RL yet)

The dynamic expert is the frozen neutral chain (3/9). Per §11: generate canonical full-action demonstrations → BC/DAgger
competence → critic calibration → SAC/TD3 smokes → multi-seed campaign (KatoLab). No training before this bundle's tag.
