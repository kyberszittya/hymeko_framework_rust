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

## Next (decision, not assumed)
- **R12.3B-contacts** — add object↔left/right-contact relative angles (needs FK from the arm joints in the descriptor)
  to see if the contact-relative frames carry more than the transport-relative one. Cheap on the same dataset.
- **R12.3C — rotor-aware structured critic** — wire rel_sym (+ contact-relative) into the HSiKAN's object node/edges and
  test if STRUCTURE exploits the small relative signal better than flat. (Given the signal is ~0.017, likely also small.)
- **R12.4 — Rotor-Spike / 3-D** — the genuine rotor-composition advantage needs SO(3); the static planar substrate has
  now been characterized (small relative-geometry signal, no clean structural win).
