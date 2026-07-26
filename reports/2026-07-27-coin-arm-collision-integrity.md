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

## Impact on prior results

None. No physics changed, so the teacher bank, the update-0 result, and the cradle coverage inventory are unaffected.
The coin was colliding with the arm all along; the earlier "pass-through" reading was a test artifact.
