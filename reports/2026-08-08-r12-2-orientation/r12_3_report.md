# R12.3 — orientation representation: encoding sanity (A) → relative-frame geometry (B)

**Date:** 2026-08-11 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
On the powered R12.2-B dataset (76 handoffs, 4940 pairs; the ranker is learnable, AUROC ~0.67). R12.2-B established the
baseline: an *absolute* orientation feature adds ~0 and structure ≈ flat. R12.3 asks whether a richer *geometric*
representation of orientation helps — carefully, so a coordinate reparameterization is not mistaken for a rotor win.

## R12.3A — encoding sanity (a control)

For planar yaw θ, `sin-cos = (sinθ,cosθ)`, `z-quaternion = (cos θ/2, sin θ/2)` and `2D-rotor coefficients` all encode
the SAME one d.o.f. So the honest hypothesis is ≈0 difference; the harness fixes dataset/split/seeds/optimizer AND
**parameter budget** (every encoding zero-padded to the same 4 extra dims → identical MLP params 29591) and varies only
the encoding. `b_encoding_sanity.json`.

| encoding | AUROC (12 seeds) | Δ vs none |
|---|---|---|
| none | 0.681 ± 0.037 | — |
| raw yaw | 0.677 ± 0.037 | −0.004 |
| sin-cos | 0.682 ± 0.037 | +0.001 |
| sin2θ/cos2θ (symmetry-aware) | 0.684 ± 0.041 | +0.003 |
| quaternion | 0.664 ± 0.036 | −0.017 |
| rotor | 0.664 ± 0.036 | −0.017 |

**Reparameterization-neutral, as expected.** (i) `quat == rotor` exactly — the same object for planar yaw. (ii) The
`quat − sincos = −0.018±0.014` dip is a flagged **half-angle scaling artifact** (cos θ/2 ∈ [0.71,1] over our [0,90°]
band is worse-conditioned than full-angle sin-cos) — not "rotor is worse". (iii) The symmetry-aware `sin2θ/cos2θ ≈
sin-cos` (+0.002±0.029): no π-symmetry advantage *as a flat feature*. No encoding beats `none`. ⇒ on planar 1-DOF
orientation, the *coordinate* does not matter; any geometric advantage must come from **relative frames** (B).

## R12.3B — relative-frame geometry (the real test) — PLANAR CAVEAT

**Key honesty point:** for a planar object the relative rotor `R_ot = R_t⁻¹R_o` reduces to an angle *subtraction*
(SO(2) is abelian, 1-D). So B4/B5 (quaternion/rotor of a relative rotation) are reparameterizations of B3 (the relative
*angle*) — A already showed reparams don't matter. The rotor's genuine power — non-commutative composition — is an
SO(3) phenomenon and needs a 3-D-tumbling object, which this planar prism does not have. Therefore on THIS substrate
R12.3B tests the substantive geometric hypothesis that survives in 2-D: **do RELATIVE orientations (object vs transport
direction, object vs contact frames) carry ranking signal that ABSOLUTE yaw does not?** The full rotor-vs-scalar test
is deferred to a 3-D orientation substrate (a later rung).

Ablation (same dataset/split/seeds/budget): B0 base · B1 +absolute sin/cos yaw · B2 +symmetry-aware · B3 +relative
angle(s) object↔transport (and object↔contact where extractable) · [B4/B5 rotor-of-relative = reparam of B3 in 2-D,
run only as a control].
