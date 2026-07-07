# PlanarGraspEnv v2 — declarative graded contact-quality contract

**Date:** 2026-07-06 20:13 JST · **Base SHA:** 4320202 (working tree dirty — see Files) · **Author:** Aiko (for Dr. Cs. Hajdu)
**Plan:** `docs/plans/2026-07-06-planar-grasp-contact-legality/` (tex/pdf/tikz/mmd, compiles)

## Summary

Made arm-body↔coin collision **physically real** in the planar-grasp env (v1 let the coin pass through the arm
bodies — an abstract fingertip-only prototype), and expressed contact quality as a **declarative, graded**
contract rather than a binary forbidden rule. Every object contact is classified by **geom role** (fingertip vs
arm-body) against a `ContactLegalitySpec` built once at env init; an arm-body↔coin contact is physically allowed
but non-preferred — tracked (count/duration/impulse), penalised by a reward term (body-only pushes, not grasps),
and used to grade each delivery **raw / clean / assisted / exploit** by progress attribution. A `contact_mode`
switch selects the consequence: **graded** (default; never voids the episode) or **strict** (voids + terminates,
for clean-paper validation). The whole v2 path is gated and declared in HyMeKo/EnvSpec; v1 is untouched and
bit-reproducible.

This design is the result of **two user corrections** during implementation: (1) make it a declarative
`ContactLegalitySpec`, not flags threaded through `compute_planar_metrics`; (2) make it **graded**, not binary
terminate/invalidate. Both were adopted; the report records the diagnosis that drove the graded model.

## Design evolution (measured, on record)

| stage | rule | scripted controller (N=12) |
|---|---|---|
| v1 prototype | coin passes through arm bodies | delivery **0.833**, arm-body rate 0.0 |
| binary v2 (rejected) | any arm-body↔coin contact → invalidate/terminate | delivery **0.0**, arm-body rate **1.0** |
| **graded v2 (adopted)** | arm-body contact tracked + reward-penalised + grades the delivery | raw **0.75**, fingertip-dominant **0.75**, zero-body-contact **0.083**, assisted/exploit 0 |

**Terminology (paper-facing, corrected 2026-07-06):** the graded 0.75 is **fingertip-dominant** delivery
(the fingertips do the directed work; an incidental distal-link graze is allowed), **not** "clean" and **not**
zero-contact. Zero-arm-body-contact delivery is a *separate, stricter* number (0.083). "Clean" is deliberately
retired because with `arm_body_rate = 0.917` it reads as "no contact", a stronger claim than the metric supports.

**Diagnosis that killed the binary rule (discriminating measurement):** of the arm-body contacts the *correct*
two-fingertip controller commits, **100% are on the distal fingertip-bearing link (`link2`), 0% on upper links,
and all during the push** (never during carry). The fingertip sphere does not protrude past its own distal
capsule, so a legitimate fingertip grasp *inevitably* grazes the hand link against the coin. A binary rule
therefore voids every legitimate delivery. The graded model resolves this: the incidental hand graze contributes
~0 body-only progress, so the delivery grades **clean**; only a body that actually *moves* the coin (a shove)
grades assisted/exploit. The existing reward's `arm_body_collision` term already encoded this intuition
("LIGHT upper-arm collision"; the distal link was never penalised).

## The declarative contract

- **`hymeko_rl/env/contact_legality.py`** (new): `GeomRole {OBJECT, FINGERTIP, ARM_BODY}`, `ContactMode
  {GRADED, STRICT}`, `ContactLegalitySpec` (geom-id sets by role + mode; `from_model` assigns roles **once** by
  the fingertip name-prefix convention — the single home of that convention), `ContactLegalityState`
  (left/right fingertip + arm-body count/impulse; `fingertip_contact`/`both_fingertip_contact` derived),
  `classify_contacts(model, data, spec) → state` (reads **roles**, never names), `contact_force_magnitude`.
- **Collision mask (v2):** arm-link geoms get `Collision.ARM_LEGALITY (1,3)` so
  `collide(ARM, COIN) = (1&2)|(2&3) = 2 > 0`. The mask only *enables* the contact; legality is a software
  decision by geom role.
- **Graded tiers (progress attribution):** per step the coin's toward-zone progress is credited to
  `fingertip_progress` (a fingertip is in contact) or `body_progress` (an arm link is, with no fingertip). A held
  delivery is **clean** if `body_progress ≤ ε` (`_CLEAN_BODY_EPS = 0.005 m`), **exploit** if
  `body_progress > fingertip_progress`, else **assisted**.
- **Reward alignment:** `arm_body_coin_contact` penalises the *same* body-only steps (arm-body contact, **no**
  fingertip), so reward and metric agree; a hand touch *during* a grasp costs nothing — the penalty never fights
  the only feasible grasp (cf. whole-arm `arm_collision` 2.0 that killed grasping, 2026-06-28).
- **Declarative in HyMeKo:** `meta_env.hymeko` gains a `@contact` param; `galambos_env_v2.hymeko` declares
  `@contact { legality 1.0; strict 0.0; }`; `EnvSpec` reads it into `contact_legality`/`contact_mode`.
  `galambos_task_deliver_v2.hymeko` adds the graded penalty term (weight 2.0, an **oracle-pending seed**).

## Files touched

New:
- `hymeko_rl/env/contact_legality.py` (+188)
- `hymeko_rl/tests/test_contact_legality.py` (+111)
- `data/robotics/galambos_env_v2.hymeko` (+44), `data/robotics/galambos_task_deliver_v2.hymeko` (+62)
- `docs/plans/2026-07-06-planar-grasp-contact-legality/{plan.tex,plan.pdf,plan.tikz,plan.mmd}`
- `scratchpad/validate_v2_contact.py` (validation harness)

Modified:
- `hymeko_rl/env/constants.py` — `Collision.ARM_LEGALITY`
- `hymeko_rl/env/planar_grasp_env.py` — `with_arm_coin_collision`; `PlanarGraspMetrics.legality`;
  `compute_planar_metrics(contact_spec)`; env builds the spec once, accumulates arm-body duration/impulse,
  applies graded/strict consequences; graded info keys
- `hymeko_rl/env/reward.py` — `arm_body_coin_contact` term (+ registration)
- `hymeko_rl/env/env_spec.py` — `contact_legality` / `contact_mode` / `valid_contact_prefix` (+ `contact` term)
- `hymeko_rl/experiments/exp_galambos_coord_ab.py` — graded delivery tiers + arm-body rate/duration/impulse
- `data/robotics/meta_env.hymeko` — `@contact` param type

**CORE.YAML items touched:** none (verified: env files, `meta_env.hymeko`, `env_spec.py` not in CORE.YAML).

## Test results

| suite | count | result |
|---|---|---|
| `test_contact_legality.py` (new) | 6 | pass |
| `test_planar_grasp_env.py` (+6 v2 cases) | — | pass |
| env/reward regression (`test_htl_reward`, `test_reward`) | 66 | pass (v1 byte-identical) |
| broader sweep (`test_galambos_demo`, `test_bc`, `test_offpolicy_framework`, CTDE, reward-oracle) | 112 | pass |

- **ruff:** clean on all changed modules and the two new test files.
- **mypy --strict:** only the pre-existing project-wide `mujoco` missing-stub note; no new type errors.
- New/modified functions each exercised by a test in the same change (§3 coverage): `from_model` (roles +
  fail-loud), `classify_contacts`, `contact_force_magnitude`, graded/strict env gates, duration/impulse
  accumulation, reward term (body-only vs grasp).

## Validation (scripted two-fingertip controller, N=12, deliver reward)

The scripted controller ignores the reward, so it is a pure affordance probe. `scratchpad/validate_v2_contact.py`.
The delivery tiers are reported **separately** with precise names (no "clean"); v1 and v2 are kept in separate rows.

**v1 (abstract prototype — coin passes through arm bodies):**

| raw | fingertip-dominant | zero-body-contact | body-assisted | body-driven-exploit | arm_body_rate |
|---|---|---|---|---|---|
| 0.833 | 0.833 | 0.833 | 0 | 0 | 0.0 |

**v2 (physically-real collision):**

| mode | raw | fingertip-dominant | zero-body-contact | body-assisted | body-driven-exploit | arm_body_rate | arm_body_steps |
|---|---|---|---|---|---|---|---|
| **graded** (default) | 0.75 | **0.75** | **0.083** | 0 | 0 | 0.917 | 52 |
| strict | 0.083 | 0.083 | 0.083 | 0 | 0 | 0.917 | 0.9 |

- **Graded is viable and honestly labelled:** raw 0.75 ≈ v1 0.833 (physical collision only mildly perturbs the
  dynamics — the coin now bumps off the arm during approach). The delivery is **fingertip-dominant (0.75)** —
  the fingertips do the directed work (body-only progress ≈ 0) — while the hand grazes in 92% of episodes, so
  **zero-body-contact is only 0.083**. These are reported as distinct tiers; the 0.75 is *not* zero-contact.
  **Body-assisted = body-driven-exploit = 0** for the scripted controller (correct — it is not an exploiter).
- **Cross-check:** `zero_body_contact_delivery` (graded, 0.083) equals `raw` under **strict** mode (0.083) —
  both count "held with no arm-body contact at all", from opposite directions.
- **Strict works as designed:** the ultra-strict paper-validation mode invalidates on the incidental hand graze;
  low `arm_body_steps` (0.9) reflects strict terminating on the first contact.

## Geometry finding (explicit — a modeling issue, not an algorithmic one)

Strict zero-arm-body-contact success is low (0.083) **because the fingertip geom does not protrude beyond its
own distal-link capsule.** A legitimate two-fingertip grasp therefore *inevitably* grazes the distal (hand) link
against the coin (diagnosis: 100% distal `link2`, 0% upper links, all during the push). This is a
**geometry/modeling** property of the current manipulator, **not** an algorithmic or controller failure — the
graded model already scores such a delivery correctly (fingertip-dominant). If a higher zero-body-contact number
is wanted for a clean-paper strict result, a **future gated geometry step** may adjust the fingertip protrusion
or the distal-link collision geometry (longer/forward fingertip sphere, recessed or thinner distal capsule, or a
smaller coin). Deferred deliberately — not improvised.

## Follow-ups closed after the initial build (2026-07-06 20:46)

1. **Oracle-certified `deliver_v2` — DONE.** `reward_oracle.certify` returns `delivers=True` (optimal_return
   25.40, identical to `deliver`): the graded arm-body penalty leaves the delivering optimum intact in the
   abstract MDP. Caveat: the oracle does not model the contact penalty's effect on the real grasp, so its
   *weight* (2.0) is still a seed — the A/B on real RL is the tuning gate, but the `delivers=True` prerequisite
   is met.
2. **`exploit`/`assisted` tiers now directly tested — DONE.** The tier logic was extracted into a pure
   `grade_delivery(held, body_progress, fingertip_progress)` (`exp_galambos_coord_ab.py`) and unit-tested across
   all four outcomes (`test_grade_delivery_tiers`), so the exploit/assisted branches an all-clean scripted
   rollout never reaches are covered. `_coordination_metrics` and the validation script both call the shared
   function (no duplicated logic).
3. **Visual evidence — DONE.** `reports/gifs/coin_v1_passthrough_vs_v2_collision.gif` (side-by-side, same seed):
   left = v1 (coin passes through the arm links), right = v2 (arm bodies physically collide with the coin).

## Before any RL/learning on v2 (gate — do not skip)

1. **Certify `deliver_v2` AND its `arm_body_coin_contact` penalty weight.** Oracle `delivers=True` is met (above);
   the penalty *weight* (2.0) is still a seed — the oracle cannot score its effect on the grasp, so the A/B on
   real RL (or the reward-shape optimisation) is the weight gate. Do not accept the weight as final on the seed alone.
2. **Run the v2 validation gates** (`validate_v2_contact.py`: v1 / graded / strict) and confirm the affordance
   (fingertip-dominant ≈ v1, exploit 0 for the scripted controller) before trusting any learned number.
3. **Report all five tiers separately** — raw / fingertip-dominant / zero-body-contact / body-assisted /
   body-driven-exploit — never a single "delivery" scalar, and never "clean".
4. **Keep v1 and v2 tables clearly separated** (as above); they are different physics.

## Open issues

1. **v2 is a separate declared scene; v1 remains the default** (§6.5 #19). Flipping the default to graded-v2 for
   future work is a deliberate, evidence-gated step (raw 0.75 vs 0.833 is a real dynamics change, not a no-op).
2. **Zero-body-contact strict number is geometry-bound** (see Geometry finding) — a future gated geometry step,
   not an algorithmic fix.
3. **RL on `env_v2` is the next natural step** (not run here, and gated on #1 above): train a collaborative CTDE
   policy on the graded v2 physics and confirm the reward penalty + exploit tier discourage a body-shove policy
   in practice.

## Provenance

- Working tree dirty (pre-existing session changes + this task's files, listed above).
- Validation seeds: eval seed0 9000, N=12; scripted `PushDemonstrator`, `robot=None`, difficulty 0.3,
  `galambos_task_deliver.hymeko`, `max_steps=300`. Host: this Windows box (single env, no GPU training).
- No persistent-state mutation; no checkpoints written.
