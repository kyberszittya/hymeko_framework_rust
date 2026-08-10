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

Ablation (flat MLP, same dataset/split/seeds/budget, `b_relative_frames.json`; transport frame =
`atan2(req_transport)` = direction of target−coin, verified, and it varies across handoffs so `rel = yaw − transport`
is genuinely distinct from absolute yaw). **PAIRED Δ vs none across 24 seeds** (paired to cancel seed variance — an
unpaired test like R12.2-B's has a ~±0.037 noise floor that buries a ~0.01 effect):

| encoding | AUROC | Δ vs none (paired) |
|---|---|---|
| none | 0.682 ± 0.033 | — |
| abs (sin/cos yaw) | 0.687 ± 0.032 | +0.005 ± 0.007 (CI-excl-0) |
| rel (sin/cos(yaw−transport)) | 0.689 ± 0.034 | +0.007 ± 0.014 |
| **rel_sym (sin/cos(2·rel))** | 0.700 ± 0.030 | **+0.017 ± 0.016 (CI-excl-0)** |
| abs_rel | 0.695 ± 0.030 | +0.012 ± 0.014 |

`rel_sym − abs = +0.012 ± 0.015` (includes 0).

**Verdict (disciplined, no over-claim):**
1. **Orientation is NOT fully redundant.** With a paired test, even absolute yaw carries a tiny real gain (+0.005*),
   and rel_sym +0.017*. R12.2-B's "adds ~0" was an *unpaired* power artifact, not zero signal.
2. **rel_sym (symmetry-aware relative) is the largest and beats none** (+0.017, CI just clears 0) — the physically-right
   feature (object axis relative to transport, folded by the rectangle's 180° symmetry). NOT a rotor/encoding win (A
   showed encodings are neutral) — a symmetry-group + relative-frame effect.
3. **But it does NOT conclusively beat absolute yaw** (rel_sym − abs = +0.012±0.015, includes 0), and the effect
   *shrank* 0.024→0.017 from 12→24 seeds (honest regression to the mean).
⇒ a **small (~1–2% AUROC) geometric signal exists**; the symmetry-aware relative encoding is its best candidate, but
the STATIC PLANAR ranker extracts only a little of the (real) orientation physics — not a clean structural win. No
artifact (rel_sym is a well-conditioned sin/cos feature). The larger geometric signal, if any, is where the planar
static ranker can't reach: **contact-relative frames**, **3-D rotor composition** (SO(3), a tumbling object), or the
**dynamic Rotor-Spike** (R12.4).

## R12.3B-contacts — object↔contact relative frames (last planar control)

The one relation not yet tested: the object's orientation relative to the CONTACT frames. Left/right contact bearings
computed from the arm joints `q[0:4]` via analytic 2R FK (`PlanarArm2R.link_points`, verified: tips ~30 mm from the coin
at bearings ~122°/−120°, matching the straddle), π-symmetry-folded where they involve object yaw. Same MLP / dataset /
split / seeds / budget. `b_contacts.json`.

| encoding | AUROC (24 seeds) | Δ vs none (paired) |
|---|---|---|
| none | 0.685 ± 0.033 | — |
| rel_sym (transport-relative) | 0.711 ± 0.031 | +0.027 ± 0.017 (CI-excl-0) |
| contacts (object↔contact + contact↔transport) | 0.696 ± 0.033 | +0.011 ± 0.020 (incl 0) |
| rel_sym + contacts | 0.713 ± 0.033 | +0.028 ± 0.023 |

`contacts − rel_sym = −0.015 ± 0.018` (incl 0). Contact-relative frames add **no more** than the transport-relative
whisper: `contacts` alone is weaker (+0.011, not significant), and stacking them onto rel_sym gives no lift
(+0.028 ≈ +0.027). No contact jump ⇒ no structure test warranted (closure rule).

## CLOSURE — `R12_3_PLANAR_RELATIVE_GEOMETRY_SMALL_SIGNAL_ONLY`

The **planar static representation rung is closed**. Summary across A + B + B-contacts:
- **Encoding is neutral** (A): raw/sin-cos/quaternion/rotor all ≈; quat==rotor; no encoding beats none.
- **A small geometric signal exists** (B): the symmetry-aware **transport-relative** orientation (object axis vs
  transport, mod the rectangle's 180°) is the only feature that clears none — **~+0.02–0.03 AUROC**, real but small.
- **Contact-relative frames add nothing more** (B-contacts): +0.011 (incl 0), no lift over rel_sym.
- **No clean structural win** anywhere; no planar HSiKAN-C warranted.

⇒ On this planar static substrate, relative geometry is a *real but small* signal — too small to carry the big
structural claim. The genuine rotor program moves to **3-D / SO(3)** (R12.4), where the honest preconditions hold:
a non-circular **tumbling / edge-contact** object where rotations do NOT commute, orientation genuinely reshapes the
contact, there are real **face→edge→corner** mode transitions, and ω / angular momentum matter — the setting where a
rotor is not merely another encoding of an angle, and where the **Rotor-Spike** (large relative-rotor change,
contact-frame transition, angular-velocity reversal, hybrid-mode switch) finally has genuine events to fire on.

**Next program (3-D):** R12.4A SO(3) orientation benchmark · R12.4B relative rotor geometry · R12.4C Rotor-Spike ·
R12.5 dynamic HyMeKo incidence · R12.6 k-actor × n-critic tensor.
