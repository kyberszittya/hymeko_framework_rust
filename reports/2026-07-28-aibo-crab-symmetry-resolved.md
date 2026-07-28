# The AIBO crab one-sidedness RESOLVED — a scaffold-induced dynamics asymmetry

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Diagnostic (no training).** · **Verdict: `CRAB_ASYMMETRY_IS_SCAFFOLD_INDUCED_NOT_SPONTANEOUS`.**

---

## The question

Across the whole HSiKAN/symmetry thread, six trained architectures (MLP, signedkan kinematic /
symmetric-signs / H★ / pooled, mixture) all reached a **one-sided** crab (2/5, +y — the mixture flips
to −y), and mirror-augmentation *degraded* rather than fixed it. Is the +y/−y crab **symmetric**
(spontaneous symmetry-breaking — SAC picks a side) or **dynamically asymmetric** (one side genuinely
easier)? The mirror-augmentation failure suggested the latter; this measures it directly.

## Measurement — the crab is NOT left-right symmetric

Constant abduction pattern `r` over the running scaffold, torso lateral displacement `dy` + minimum
uprightness:

| abduction pattern | dy | upright | | left-right SWAP | dy | upright |
|---|---|---|---|---|---|---|
| left legs `[1,0,1,0]` | **+0.209** | 0.86 | | right legs `[0,1,0,1]` | **+0.251** | 0.89 |

**Left-abduction and right-abduction BOTH crab +y** — not opposite. A left-right-symmetric crab would
have the swap flip `dy`'s sign; instead both push +y. Likewise `r` vs `−r` do not give opposite `dy`.
So the crab is a **genuine dynamics asymmetry**, not spontaneous symmetry-breaking.

## Root cause — the running diagonal-trot scaffold

The AIBO model is left-right symmetric *by construction* (fl/fr, bl/br mirror), so the asymmetry is
not in the morphology. It is induced by the **scaffold**: the omni action is a residual over the
**forward trot**, whose diagonal phase `DIAG_PHASE=(0,π,π,0)` puts the **left and right sides in
different gait phases at every instant** (fl at phase 0 while fr is at π). Abduction added over this
temporally left-right-asymmetric scaffold therefore interacts differently with the two sides — even a
*static* abduction is asymmetric (left `[1,0,1,0]` → dy −0.05 **upright**; right `[0,1,0,1]` → dy −0.32
**tipped**). The crab inherits the trot's instantaneous left-right asymmetry.

## Why every earlier intervention failed — now explained

- **6 architectures one-sided** (MLP/signedkan×4/mixture): the *dynamics* over the trot isn't
  mirror-symmetric, so no policy class confers a symmetric crab; the architecture only selects **which**
  emergent side it exploits (mixture found the −y one).
- **H★ structural-entropy exploration** didn't help: it's not an exploration gap; the −y crab over the
  running trot is genuinely a different, harder emergent mode.
- **Mirror-augmentation degraded** (3/5→2/5): mirroring the residual data teaches a symmetry the
  scaffold **does not have** — the trot's phase would also have to be mirrored (a half-period shift),
  which residual-only augmentation cannot express.

## The real lever (not delivered — a scaffold change)

A symmetric crab needs a **temporally symmetric substrate**: abduction on a **static / symmetric
stance** (a stance-crab with no trot underneath — loses forward motion), or a crab **phase-locked to
each leg's own stance so the stride-average is symmetric** (a co-designed gait+crab), or the full
system (scaffold phase + residual) mirrored together. All are **scaffold/gait redesigns**, consistent
with the campaign's recurring finding that the AIBO scaffold is the binding constraint.

## Bottom line

The crab one-sidedness is **RESOLVED**: it is a **scaffold-induced dynamics asymmetry** — the abduction
residual sits over a forward trot whose diagonal phase makes the two body sides temporally asymmetric,
so left and right abduction both crab +y and no mirror symmetry exists to exploit. This cleanly
explains the six one-sided architectures, the null H★, and the failed mirror-augmentation: the fix is
**not** in the policy (architecture / exploration / data symmetry) but in the **substrate** (a
temporally-symmetric gait). The HSiKAN thread's honest verdict stands — HSiKAN ≈ MLP here — and the
symmetry-breaking is now fully characterised as a dynamics property of the scaffold.
