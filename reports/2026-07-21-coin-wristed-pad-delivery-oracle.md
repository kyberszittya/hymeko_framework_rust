---
campaign: COIN wristed independent-pad delivery integration + 2×2 physical oracle
title: The delivery integration is complete and verified; the correctly-integrated 2×2 oracle achieves 0 strict — NO_FORCE_CLOSURE (control-limited, coin reaches center but does not settle)
date: 2026-07-21
branch: exp/coin-wristed-pad-delivery-integration
source_commit: 4b22b42
classification: NO_FORCE_CLOSURE (correctly-integrated E0/E1/E2/E3 oracle; no DoF is load-bearing; RL not launched)
---

# Wristed independent-pad delivery integration + 2×2 physical oracle

**Created-at:** 2026-07-21 13:20 JST. The prior BLOCKED report named the exact missing piece: the delivery motor path
drove only the 4 arm actuators. This iteration builds that integration and runs the 2×2 oracle.

## §2–5 integration — complete and verified (7 integration tests pass)
- **Schema-aware motor path** (`hymeko_rl/env/pad_actuation.py`, shared infra §13): `actuator_groups(model)` discovers
  ARM / WRIST_YAW / PAD_CLOSURE by **named** actuators (never fixed indices); `WristCloseController.motor_override`
  overwrites the wrist/closure indices on the nu-sized arm motor, leaving ARM untouched — **E0 (nu=4) byte-identical**.
- **Wrist alignment** — world-frame: pad contact-normal (measured local basis) → coin direction; bounded rate,
  anti-windup, continuous angle wrapping; logs target/actual/error/saturation.
- **Bounded closure force** — regulates the `a_close` slide toward a **normal-force target (2.5 N)** read from **real
  MuJoCo contacts** (`mj_contactForce`); bounded approach before contact, force regulation during hold/transport,
  smooth ramp-to-zero + open on release; position/velocity/force limits + anti-windup. Force target justified: well
  within the slide servo (kp 30, range 0.02 m).
- **Generalized restore** — `build_wristed_contact_env` pads the canonical 7-qpos snapshot into the model's **typed**
  layout (validates nq; fails loudly on schema mismatch; never silently truncates); added joints init at neutral.
- **Canonical builder** — `make_env(fingertip_geometry=E1_WRIST/E2_CLOSURE/E3_WRIST_CLOSURE)`; no separate simulator.
- **§7 tests (7, all pass):** typed actuator groups; E0 motor byte-identical; E1 drives only wrist; E3 drives all 8
  finite; qpos-addr layout; 11-qpos restore round-trip; closure force bounded + release ramps down. POINT golden intact.

## §6 explicit oracle state machine
APPROACH → WRIST_ALIGN → PAD_CLOSE → FORCE_HOLD → TRANSPORT → BRAKE → RELEASE → WITHDRAW → SETTLE, with public named
transition measurements and a **phase-aware arm base** (grasp_carry HOLD during grip acquisition, CARRY during
transport, OPEN during release — so closure can form a grip without the arm shoving the coin). Per-phase failure
taxonomy (APPROACH/WRIST_ALIGNMENT/PAD_CONTACT/FORCE_HOLD/…_FAILURE).

## §8 2×2 physical oracle — 48 matched seeds, 4 clearance bands
| embodiment | strict ≥+0.030 | min_dtz reached (near-goal band) |
|---|---|---|
| E0 passive ring | **0** | 0.020 |
| E1 wrist | **0** | 0.020 |
| E2 closure | **0** | 0.088 |
| E3 wrist + closure | **0** | 0.021 |

**Every embodiment: 0 strict, all bands.** The coin reaches near-center (min_dtz ~0.020 = the `center_tol`) transiently
under E0/E1/E3, but never satisfies the strict **centered-AND-settled** condition; failures are distributed across
wrist-alignment, bilateral pad-contact and settle. **No DoF is load-bearing.**

## Classification: **NO_FORCE_CLOSURE** — earned on a *correctly-integrated* oracle
Per §9, NO_FORCE_CLOSURE is returned only when the correctly integrated E3 oracle cannot transport and settle the coin —
which the 7 passing integration tests establish is the case here (the motor path, controllers, and restore are verified
correct, and E3 drives all 8 actuators). Per §10, the oracle is **not positive → no RL was launched.**

## Honest nuance (not an absolute impossibility)
The coin **does reach center** (min_dtz ~0.020) — transport is partially achieved; the gap is a **reliable bilateral
grasp + low-velocity settle**, and the failures are spread, not a single wall. The bounded controllers are **minimally
tuned** (one force target, one set of gains, a hand-written phase machine) — this is a **control-limited null on a
verified integration**, not proof that wrist+closure cannot grasp the box. A tuned symmetric-grasp controller (or the
learned option chain the integration now enables) could still succeed; the physical hypothesis (parallel-jaw grasp of a
box) remains open. But per the discipline, RL is gated on a positive oracle, which this is not.

## Files / provenance
- `hymeko_rl/env/pad_actuation.py` (NEW, shared) — actuator groups, WristCloseController, build_wristed_contact_env.
- `hymeko_rl/coin_delivery/scenarios/kinematic_variant.py` — `with_pad_closure` (prior commit).
- `hymeko_rl/experiments/coin_wristed_delivery.py` (NEW) — make_wristed_delivery_env, oracle state machine, 2×2 runner.
- `hymeko_rl/experiments/exp_galambos_coord_ab.py` — E1/E2/E3 through the canonical builder.
- `hymeko_rl/tests/test_pad_actuation.py` (NEW, 7 tests). 9 golden/coin tests pass; POINT byte-identical; no CORE.YAML; no deps.
- **Preserved:** transport `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`.
- Data: `experiments/2026_07_21_coin_wristed_pad/oracle/wristed_oracle.json` sha `64f42b3a`; E0/E1/E2/E3 hashes `935fa3c0`/`efe80ed1`/`8f7c6b33`/`3112288`. Final HEAD `5a36737`. Host Apple M5 Pro; oracle wall ≈ 6 min, RSS ~0.45 GB.
- **Next concrete step:** a symmetric-grasp controller tune (or the learned GRASP/FORCE_HOLD → RELEASE_SETTLE → TRANSPORT option chain the integration now enables) — the delivery motor path + controllers + oracle are in place; only the grip/settle control is unsolved.
