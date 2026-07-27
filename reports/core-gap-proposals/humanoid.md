# Core-gap proposals from CIP-HUM-01 (humanoid)

Proposals only — no promotion from this scenario branch. Promotion happens later
on `integration/hymeko-control-core-v1`, gated on genuine scenario-independence
and a second implementer.

## Candidate 1 — generic support/stability certificate interface (STRONG, but needs a genuine second implementer)

Both the humanoid ("support margin maintained, no fall") and AIBO ("no fall,
stable stop") need a **support/stability safety certificate**: a predicate over
(base pose, contact set, COM) that a floating-base embodiment must hold. This
scenario declared it (`support_margin_maintained`) but could only implement it
**vacuously** (fixed base). Promote a generic `support_stability` certificate
interface ONLY once a floating-base embodiment implements it non-vacuously — else
the core would ship an untested, vacuous safety type. Recommendation: **defer**
until a floating-base humanoid or the AIBO provides a genuine stability signal.

## Candidate 2 — computed-torque / gravity-feedforward PD as a reusable realizer (MEDIUM)

The `HumanoidSim.pd_step` (PD error + `qfrc_bias` feedforward) is a generic
model-based joint controller reusable by any MuJoCo-backed adapter (the arm, a
future floating humanoid). It is embodiment-agnostic (reads only model/data). But
it depends on MuJoCo, so it cannot live in the stdlib-only core `hymeko_control`;
it belongs in a shared *scenario-side* helper (e.g. a `scenarios/_sim_util.py`),
not in the core. Recommendation: **promote to a shared scenario helper, not to core.**

## Candidate 3 — "reset-on-transition" reuse (NO ACTION)

The contract uses `Mode.reset` (`TOUCH: {touch_dwell:0}`, `RECOVER: {recover_dwell:0}`)
— already a core feature. No promotion needed.

## Recommendation

Nothing from the humanoid is ready for core promotion yet. Candidate 1 (support/
stability certificate) is the important one but must wait for a genuine
(non-vacuous) implementer — likely the AIBO or a floating-base humanoid. Candidate 2
is a scenario-side helper, not core. Reconcile with the pick-place proposal's
"contact-retention" / "handoff" candidates at the final review.
