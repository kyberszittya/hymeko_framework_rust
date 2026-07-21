---
campaign: COIN wristed independent-pad embodiment (wrist-yaw + independent closure DoF) + 2×2 physical oracle
title: The wrist + independent-closure embodiment is built and verified, but the delivery-level physical oracle is BLOCKED on a named integration boundary
date: 2026-07-21
branch: exp/coin-independent-concave-clamp
source_commit: bdcb8fb
classification: BLOCKED (embodiment built + verified; the delivery-env actuator-driver integration for the 2×2 oracle is the exact remaining piece — no oracle result fabricated)
---

# Wristed independent-pad embodiment

**Created-at:** 2026-07-21 12:55 JST. The corrected-ring result pinned the residual limitation to the embodiment: the
2-link arms can position the fingertips but cannot **orient** the contact faces or **independently regulate closure
force**, and the coin is a flat-faced box. This iteration adds those two DoF and tries to verify force closure
physically before RL.

## §1 embodiment — built and verified
Two per-arm DoF, reusing the canonical `arm_mjcf_transform` hook:
- **Wrist-yaw** — reused the existing K1 `with_distal_pad_orientation`: a `pad_hinge_{side}` in-plane hinge + position
  actuator `a_pad_{side}` per pad (the pad orients about vertical to face a box surface).
- **Independent closure** — NEW `with_pad_closure`: a `pad_slide_{side}` slide joint along the pad's local X (the
  *measured* contact normal) + position actuator `a_close_{side}`, so each pad presses toward the coin **independently
  of the arm's coupled squeeze** — a regulated normal-force clamp.
Simple flat pad (`fingertip_shape=box`), no prong rings/suction/welds/magnets/hidden constraints (as instructed).

## §2 the 2×2 embodiment — compiles with correct typed actuator schemas
| embodiment | DoF | nu | nq | model hash | actuators |
|---|---|---|---|---|---|
| **E0** passive ring | — | 4 | 7 | `935fa3c0…` | 4 arm |
| **E1** wrist | wrist | 6 | 9 | `efe80ed1…` | 4 arm + `a_pad_{L,R}` |
| **E2** closure | closure | 6 | 9 | `8f7c6b33…` | 4 arm + `a_close_{L,R}` |
| **E3** wrist+closure | both | 8 | 11 | `3112288…` | 4 arm + `a_pad_{L,R}` + `a_close_{L,R}` |

Coin/target/arm-lengths/base-positions/friction/reward/strict-predicate/clearance-bands unchanged; fresh
model-compatible state layout (K1-style qpos padding). Schema recorded:
`experiments/2026_07_21_coin_wristed_pad/embodiment_schema.json`.

## §3–4 physical oracle — **BLOCKED** on a concrete, named integration boundary (no result fabricated)
The delivery env is `CoinDeliveryTrainEnv → ContactFormationEnv → PlanarGraspEnv`; its action maps a 6-DoF cooperative
command to the **4 arm** actuators only. To run the 2×2 delivery oracle (approach → wrist-align → close → hold →
transport → release → settle → strict), the delivery env must **drive the new `a_pad`/`a_close` actuators** with a
wrist-alignment + bounded-normal-force controller. The existing K1 `PadAwareContactFormationEnv._physics_motor` does
exactly this **but only for `a_pad` and only for the CONTACT-FORMATION task** (which terminates on grip-ready, not
delivery). The remaining integration is specific:
1. a delivery-task physics-motor override that appends **both** `a_pad` (align) and `a_close` (force) targets to the
   4-DoF arm motor — extending the `PadAwareContactFormationEnv` pattern to closure **and** to `CoinDeliveryTrainEnv`;
2. the snapshot-padding for the 8-actuator/11-qpos E3 layout (the K1 code pads 9-qpos; E3 needs 11);
3. a bounded-force closure controller (regulate `a_close` to a normal-force target, not a fixed position).
**A raw-physics bypass** (driving the raw `PlanarGraspEnv` ctrl vector with a hand-rolled arm P-controller) was
attempted and **failed to produce a valid grasp** (0 coin contacts, 0 displacement, for both E0 and E3) — confirming the
proper cooperative arm controller (which lives in the delivery mapping) is required, not replaceable by a crude bypass.
So the physical-feasibility signal **cannot be obtained** without that integration.

## Classification: **BLOCKED** — honestly, not a fabricated oracle
Per §4 the oracle must demonstrate force-closure transport before RL; I could not obtain that measurement because the
delivery-env driver for the new actuators is not built, and the raw bypass does not grasp. Rather than fabricate an
oracle result or an RL launch on an unverified embodiment, I report BLOCKED with the exact boundary. **No RL was
launched. No friction or hidden actuator was added. Coin Delivery remains open.**

## Which DoF is load-bearing — not yet determinable
The 2×2 was designed to isolate wrist (E1) vs closure (E2) vs both (E3), but the oracle that would answer it is blocked
above. The arc's prior K1 finding (wrist-only, contact task) was **INCONCLUSIVE** (neutral distal bodies collapsed the
L/R balance); the closure DoF is newly available and untested. The *physical* hypothesis remains sound — two flat pads
that align to a box face and press with regulated force are a classic parallel-jaw grasp — but it is **unproven here**.

## §6 transfer relevance
`with_pad_closure` + `with_distal_pad_orientation` live in shared env infrastructure
(`env.planar_grasp_env`, `coin_delivery.scenarios.kinematic_variant`), reusable by PickPlaceEnv / a future Beni or AIBO
end-effector — the wrist+closure abstraction is embodiment-general, not Coin-only. Not claimed as working manipulation
until the oracle passes.

## Files / provenance
- `hymeko_rl/coin_delivery/scenarios/kinematic_variant.py` — `with_pad_closure` (NEW); reused `with_distal_pad_orientation` (wrist).
- `experiments/2026_07_21_coin_wristed_pad/embodiment_schema.json` — the 4 typed schemas + hashes.
- 9 golden tests pass; POINT byte-identical; no CORE.YAML items; no deps. Preserved: transport `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`. Host Apple M5 Pro.
- **Next concrete step:** a `WristedPadDeliveryEnv(_physics_motor)` extending the `PadAwareContactFormationEnv` pattern to closure + the delivery task, then the align/force controllers, then the 2×2 oracle. That is a bounded, well-scoped integration — not a research unknown.
