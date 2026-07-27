# Core-gap proposals from CIP-PNP-01 (pick-and-place)

These are candidate promotions into the shared core `hymeko_control`. They are
**proposals only** — nothing here is promoted from the scenario branch (per
campaign rule: no silent shared-core changes from a scenario branch). Promotion
happens later, on `integration/hymeko-control-core-v1`, only if the capability is
genuinely scenario-independent and another embodiment could implement it.

Scenario-specific code (the PickPlaceEnv wiring, expert v3, FANUC config, the
`both_contact/lifted/...` signal names) **stays in the scenario**.

## Candidate 1 — generic contact-retention intent + certificate (STRONG)

The pattern "a payload must remain in controlled contact while the embodiment
moves it" recurs across embodiments: a gripper retaining a box in CARRY, a
humanoid retaining a foot/hand contact during a support shift, an AIBO retaining
ground contact of the stance legs. In this scenario it appears as
`transport_authority` (drop margin) + the `object_not_dropped` /
`contact_retained_in_carry` safety certificates.

Promotable form: a generic `contact_retention` safety-certificate factory in
`hymeko_control.cip.certificate` that takes a per-step boolean "contact held"
signal and a mode predicate, and a matching authority channel convention. No
pick-place constant would appear in it.

## Candidate 2 — generic handoff certificate (MEDIUM)

The semi-MDP `OptionEnd.HANDOFF` between two modes should carry a certificate
that the invariant surviving the mode switch (here: object still grasped/lifted)
holds at the boundary. Currently the adapter checks this ad-hoc in `execute`.
A generic `HandoffCertificate` value type (state/trace boundary predicate) would
be reusable by the humanoid (support-margin preserved across a stance switch) and
AIBO (stability preserved across a gait-mode switch).

## Candidate 3 — generic stored-energy-safe release certificate (MEDIUM)

RELEASE→SETTLE here requires the object to be set down with bounded residual
velocity before the gripper opens (no ballistic drop). Generalised: "release only
when the stored kinetic/potential energy about to be freed is below a bound." A
humanoid retract and an AIBO stop share this shape (don't let go / don't stop with
unsafe residual energy). Promotable as a `bounded_release` certificate over a
velocity/energy signal.

## Candidate 4 — reset-on-transition already in core (NO ACTION)

The `Mode.reset` map (e.g. `LIFT: {grip_settle_counter: 0}`, `SETTLE:
{settle_dwell_counter: 0}`) is already a first-class part of the core language.
No promotion needed; noted so the humanoid/AIBO reuse it rather than re-inventing.

## Recommendation

Carry Candidates 1–3 to the final core-promotion review. Candidate 1 is the
strongest (clear second implementer in both remaining scenarios). Do not promote
anything until at least two adapters demonstrate the interface.
