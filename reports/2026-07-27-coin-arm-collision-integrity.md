# Coin↔arm collision integrity — investigation + regression lock-in

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-teacher-to-rl` · Model: coin-teacher scene (ball-tip BALLTIP
arm + cylinder coin), built via `CoinRL4Dof(geom="POINT", arm_mjcf_transform=_ball_tf, coin_shape="cylinder",
disk_radius_override=0.020)`.

## Trigger

Reported concern: "the coin does not collide with the robotic arms."

## What was actually found

**No coin↔arm collision bug.** The coin collides with every arm geom, cannot tunnel through the links, and contacts the
arm links during real delivery.

### The wrong turn (recorded as a methodology lesson)

An initial characterisation placed the coin and each arm geom at **coincident centres** and read `data.ncon`. It reported
`cylinder-coin vs {capsule, cylinder, box} = 0 contacts` (only sphere = 1) and suggested a MuJoCo cylinder-vs-capsule
narrowphase gap. This was a **degenerate convex-collision configuration**: GJK/MPR has no well-defined separating axis at
coincident centres, so it returns no contact for convex↔convex pairs regardless of true overlap. The tell was that a
convex-mesh cylinder returned 0 against **everything** (including sphere) — a harness degeneracy, not a shape limitation.
A geom-type replacement plan (sphere-decomposing the links / changing the coin shape) was drafted on this false premise
and then **discarded**.

### The correct, discriminating tests

1. **Shallow-penetration (1 mm, non-coincident, along each geom's outward normal).** Every arm geom — the 4 capsule
   links (link1/link2, both arms), the 2 base cylinders, and both fingertip spheres — produces the exact `(disk, geom)`
   pair in `data.contact` with `dist = -0.001` and a nonzero separating normal force.
2. **Swept pass-through (0.25 mm increments across each capsule).** A `(disk, link)` contact appears before the coin
   reaches the far side; the coin **cannot tunnel**. (Sim `dt = 5e-4`, coin speed < 1.5 m/s ⇒ ≤ 0.75 mm/step ≪ the
   24 mm link diameter, so tunneling is physically impossible.)
3. **Real s1 delivery.** Observed coin↔arm contacts: `{fingertip_left, fingertip_right, link2_left(g3),
   link2_right(g7)}` — the coin touches both fingertips **and** the link2 capsules — over 26 of 60 steps. No spurious
   same-arm or cross-arm self-contacts occurred.

## Collision-contract status (current model)

- **Coin ↔ every arm geom: enabled and functional** (arm `contype=1/conaffinity=3`, coin `2/2` ⇒ mask-collidable;
  contacts confirmed).
- **Left ↔ right arm: collidable** (mask-collidable, no cross-arm exclude) — does not trigger in the delivery
  trajectories, but the contact is enabled if the arms meet.
- **Same-arm: adjacent pairs excluded** via `<exclude>` (base–link1, link1–link2, per arm); non-adjacent same-arm pairs
  remain mask-collidable but never overlap in practice (no same-arm contact observed at any pose in delivery).

Per the "don't change geom types unless a non-degenerate shallow + swept test proves a supported pair genuinely fails"
rule, and since none failed, **no physics/geom/mask change was made** — the model's collision behaviour is correct.

## Lock-in (this commit)

`hymeko_rl/tests/test_coin_collision_contract.py` (5 tests, ~1.3 s, no acquisition) encodes the discriminating
methodology so the verified behaviour is a regression:

- `test_arm_geom_inventory_is_complete_and_expected` — the full set of physical arm collision geoms is fixed; a new or
  removed link trips the test (so a link cannot silently become non-collidable with the coin).
- `test_coin_collides_with_every_arm_geom_shallow_penetration` — parametric shallow-penetration; forces overlap and
  inspects `data.contact` by geom ID + separating force.
- `test_coin_cannot_tunnel_through_capsule_links_swept` — swept crossing, asserts contact before the far side.
- `test_cross_arm_pairs_are_collidable` / `test_same_arm_adjacent_pairs_are_excluded_and_do_not_contact` — document the
  cross-arm-collide / adjacent-same-arm-exclude structure.

## Impact on prior results (of the verified-collision lock-in)

None from the test-only lock-in itself. The coin was colliding with the arm all along; the earlier "pass-through"
reading was a test artifact.

---

## Follow-up — per-side collision contract + task-legality separation (supersedes "test-only")

The user then directed the explicit per-side collision contract (the "test-only" choice above was overridden), and a
deeper architectural correction: **physical collision** and **task legality** are separate concerns. The old model
conflated them — it disabled arm↔coin collision (fake physics, interpenetration) to enforce a *behavioural* preference
(no arm-body "knock"). That is unsuitable for real-robot transfer.

**Physical collision contract (`hymeko_rl/env/collision_contract.py`, applied in `PlanarGraspEnv` for every planar coin
scene):** per-side category masks — left arm `1/14`, right arm `2/13`, coin `4/11`, floor/world `8/7`. Semantics: every
arm geom collides with the coin; left↔right collide; same-arm pairs are mask-isolated (not just the adjacent excludes).
The legacy Galambos fingertip-only-via-noncollision model is retired.

**Task certificate (`hymeko_rl/coin_delivery/theta_option/insertion_certificate.py`), separate from masks:**
`CONTROLLED_INSERTION = target-directed displacement ∧ bounded coin speed ∧ active braking (low terminal speed) ∧
terminal K6 dwell ∧ ¬ballistic-knock`. Link contact is **allowed** (morphology-assisted guiding); the forbidden shortcut
is a ballistic knock. Delivery levels: **E0** whole-arm assisted (link contact allowed), **E1** fingertip-dominant
(fingertip impulse share ≥ 0.5), **E2** fingertip-only. The fingertip vs arm-body impulse split reuses
`contact_legality.classify_contacts` (no re-implementation).

**Impulse audit of the frozen teacher (`contact_quality_audit.json`):** overall fingertip impulse share ≈ **0.06**
(s1 0.107, s3 0.200, s4 0.028, s7 0.091; 126/240 arm-body-contact frames). None is a ballistic knock;
**3/4 pass CONTROLLED_INSERTION** (s1, s3, s7) — **s4 is K6-valid but drifts after dwelling** (terminal coin speed ≥
SETTLE_VEL), so it fails the "ends settled" clause (the known "s4 least clean"). **Teacher label:
`WHOLE_ARM_ASSISTED_INSERTION` (E0).** The teacher guides the coin primarily with the arm links; the fingertips are
supplementary. It is **not** fingertip-dominant — no fingertip-grasp claim is made, and the strict CONTROLLED_INSERTION
certificate (stricter than K6, it also requires terminal rest) already flags s4's drift.

**Validation of the mask change (physics-neutrality + coverage re-freeze):** the frozen teacher replays **4/4 unchanged**
(snapshot hashes + K6 + dwell match), and all previously-usable dev cradles (s1,s3,16500,17750,19500) match. Re-scouting
under the corrected masks changed the certified SET in both directions: **seed 23000 no longer certifies** (its old
certification was an artifact of the incorrect same-arm collision — a false positive dropped) while **seed 24000 now
certifies AND delivers K6** (a false negative recovered — a spurious same-arm collision had been blocking it). Net: the
usable dev pool **grew 5 → 6** (s1, s3, 16500, 17750, 19500, 24000; delivery yield 0.667), so the coverage N-curve is now
**N = 2, 4, 6**. Held-out s4/s7 unchanged.

**SAC/TD3 authorisation is unaffected** — the E0 teacher is a valid whole-arm-assisted delivery; the next gate remains
update-0 coverage at N = 2, 4, 5. E1/E2 (fingertip-dominant / fingertip-only) are later hardenings.

### Classification of the 5 failing regression tests (migrated, not deleted)

- **A — obsolete fingertip-only-via-noncollision assumption** (migrated to the physical contract, historical intent
  recorded): `test_only_fingertip_can_touch_the_coin` → `test_arm_and_fingertip_physically_collide_coin_same_arm_isolated`;
  `test_compiled_model_coin_collides_with_fingertip_not_arm_capsule` →
  `test_compiled_model_coin_physically_collides_with_arm_and_fingertip`; the degenerate coincident-centre
  `test_coin_passes_through_arm_capsule_but_contacts_fingertip` → `test_coin_does_not_pass_through_arm_capsule`
  (non-coincident shallow penetration); `test_arm_links_collide_with_coin_bitmask` (POINT + CONCAVE_CLAMP) — value
  migration only (intent aligned: coin↔arm collides; masks updated to per-side).
- **B — genuine regression:** none.
- **Pre-existing (collision-independent):** `test_env_shapes_and_coin_placed_in_reach` fails **without** the collision
  change too (the coin-spawn `_clear_of_arms` is a purely geometric point-to-segment check, unaffected by masks). Left
  as-is; not in scope for this change.

New task-semantics tests: `hymeko_rl/tests/test_coin_insertion_certificate.py` (controlled-insertion accepted,
ballistic-knock rejected, teacher grades E0).
