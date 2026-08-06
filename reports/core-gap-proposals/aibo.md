# Core-gap proposals from CIP-AIBO-01 (AIBO / quadruped)

Proposals only — no promotion from this scenario branch. Reconcile at the final
core-promotion review.

## Candidate 1 — support/stability certificate interface — SECOND implementer found (STRONG)

The humanoid proposed a generic support/stability certificate but could only
implement it **vacuously** (fixed base). The AIBO implements a **genuine**
`no_fall` (torso uprightness > threshold, no divergence) on a free-base body.
That gives the interface a real second/first genuine implementer. **Recommend
promoting** a generic `stability`/`no_fall` safety-certificate factory to
`hymeko_control.cip.certificate` that takes an uprightness (or support-margin)
signal + threshold — usable by AIBO now and a future floating-base humanoid.

## Candidate 2 — heading/response provenance in the interface audit (MEDIUM)

The AIBO interface-response audit (commanded → measured forward velocity + yaw
rate + latency + stopping) is a generic **command-response characterization** any
mobile embodiment needs before control. Promotable as a small
`response_audit(command_fn, measure_fn, steps)` helper. But it depends on a live
env, so it belongs in a **scenario-side** shared helper, not the stdlib core.

## Candidate 3 — bounded-velocity "stop" certificate (MEDIUM, overlaps pick-place)

`speed_bounded_at_stop` (planar body speed below threshold at STOP/HOLD) is the
locomotion analog of the pick-place "bounded release" candidate (both certify a
mode entered with bounded residual energy/velocity). Merge with the pick-place
`bounded_release` proposal into one generic `bounded_terminal_state` certificate.

## Candidate 4 — response-history contract (MEDIUM)

The AIBO STOP mode reads whether the body has halted (a short response history).
Combined with the pick-place contact-retention-over-CARRY need, this suggests a
generic **response-history window** contract in the core (a rolling buffer of the
last N `ResponseTrace` signals). Consider for core promotion if two adapters need it.

## Recommendation

**Candidate 1 (stability / no_fall certificate)** is the strongest cross-scenario
promotion: it is scenario-independent, has a genuine AIBO implementer, and the
humanoid + a future floating-base humanoid are second implementers. Promote it at
the final review. Candidates 2–4 are scenario-side helpers or need a second
concrete implementer first.
