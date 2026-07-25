# Coin V4 intermittent-contact freeze + mass×friction + C2 intermittent ablation

**Date:** 2026-07-25
**Branch:** `feat/architectural-assimilation-v1`
**Directive:** freeze V4 as an intermittent-contact contract; defer sustained/grasp to a separate benchmark; characterize
mass×friction; then run C2 with intermittent-physics options.
**One-line outcome:** V4 froze as an **intermittent-contact** contract; the coin transport wall is the **tangential
contact coupling** (friction-limited), not the controller's option language — both continuous (C1) and intermittent (C2)
option languages hit the same wall, and mass is provably not the lever.

---

## 1. V4 frozen — `COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT`

The coin's real contact class is **intermittent** (push-and-coast), not sustained-pressure grasp: a single tip on a
low-mass/low-friction disk cannot hold continuous contact. The gate certifies the regime the task actually uses; the
sustained/grasp criterion is **deferred to a separate future benchmark** (two-sided clamp / concave tip / graspable
object). All three swept configs **PASS**; frozen at least-intervention `over_hard_brake = 1.5`.

| criterion (pre-declared) | result |
|---|---|
| genuine contact (majority of states) | 3/4 ✓ |
| motion-legal every episode (peak-in-contact ≤ 3.45, integ < 0.4) | peak 1.92, integ 0 ✓ |
| object not launched (peak coin speed ≤ 1.0 m/s) | 0.22 m/s ✓ |
| arm recovers below safe band | ~20 frames ✓ |
| re-contact demonstrated (≥2 episodes) | 7–8 episodes ✓ |
| terminal certificate (arm + coin near-rest) | coin 0.002–0.046 m/s ✓ |
| no persistent torque saturation (≤0.5) | 0.5 ✓ (at the bound — noted) |

**Force decomposition (the flagged caveat, confirmed):** normal force **Fn 17.2 N** but tangential **Ft only 3.57 N**
(Ft/Fn ≈ **0.21**). The 17 N is mostly a normal press spike; the *useful* tangential shear that drags the coin is small.
Artifact: `dynamics_contract_v4.json`.

## 2. Mass × friction characterization (delivery-independent)

Swept object mass {0.5, 1, 2}× × friction {0.5, 1, 2}× under the same oscillating-press driver, measuring the contact
regime only (K6/zone never read).

- **MASS-INVARIANT** — byte-identical metrics across the 4× mass range. This is *physically correct*, not a bug (verified
  the scaling changes body_mass 0.0025→1.005 kg): under **Coulomb friction the coast deceleration a = μg is
  mass-independent**, and a **position-controlled tip makes the light coin a kinematic follower** during contact. ⇒ **mass
  is NOT the transport lever.**
- **FRICTION is the lever** — at 2× friction: Ft 1.26→**6.25 N**, Ft/Fn 0.22→**0.58**, contact 8.7→**10.3** frames. Higher
  friction ⇒ more tangential drag ⇒ longer, more effective contact.
- **Caveat (measured limitation):** sub-unity friction cells are degenerate (`0.5× == 1.0×`) because MuJoCo combines two
  geoms' friction by the elementwise **maximum**, and the tip/floor friction (~1.0) dominates the push; only scaling
  *above* baseline (2×) changes the contact. The sub-unity axis is not interpretable; the ≥1× signal is.

Artifacts: `coin_mass_friction.json`, `coin_mass_friction.png`.

## 3. C2 — intermittent-option ablation on frozen V4 (16 states)

The option language that fits push-and-coast: FSM **impulse → coast (observe) → re-contact → brake → settle**, all through
the shared V4 GovernedArm stack (no raw-torque bypass). Progressive ablation:

| arm | acquire | transport-dist | zone-entry | K6 |
|---|---|---|---|---|
| A legacy (searched open-loop macro) | 0.938 | **0.036** | **0.312** | **0.312** |
| B impulse+coast | 0.812 | 0.021 | 0.062 | 0.062 |
| C +re-contact | 0.812 | 0.021 | 0.062 | 0.062 |
| D +brake | 0.812 | 0.021 | 0.062 | 0.062 |
| E +settle (full) | 0.812 | 0.021 | 0.062 | 0.062 |

```
VERDICT: INTERMITTENT_OPTION_LANGUAGE_INSUFFICIENT_UNDER_REALISTIC_CONTACT_DYNAMICS
```

- **B = C = D = E, identically.** The downstream options (re-contact / brake / settle) **never bear differential
  effect** because the coin never gets close enough to the zone for them to fire. The wall is the **transport primitive**
  (impulse-coast), *upstream* of the option timing.
- All intermittent arms **under-transport** the searched legacy macro (0.021 vs 0.036; K6 0.062 vs 0.312). Legacy K6
  0.312 matches the earlier realistic-dynamics expert measurement (0.312) — a consistency check.

Artifact: `c2_intermittent_ablation.json`.

## 4. Synthesis — where the coin transport wall actually is

Three independent results converge:
1. **Force decomposition (V4):** useful tangential force ≈ 21 % of the normal peak.
2. **Mass×friction:** transport is mass-invariant (Coulomb + position control) and friction-limited; the effective
   contact friction is capped by the max-combination with the tip/floor.
3. **C2:** neither the continuous (C1, prior) nor the intermittent (C2) option language restores transport; the options
   downstream of the push never fire.

⇒ **The coin transport wall is the tangential contact coupling — friction-limited shear on a low-friction disk driven by
a single tip — not the controller's option language.** Both option languages hit the same wall because the wall is
upstream of them, in the contact mechanics.

## 5. Claims / non-claims

**Claimed (measured):**
- V4 froze as an intermittent-contact contract; the stack is motion-legal + recovers + re-contacts under intermittent
  loading, object not launched, terminal-safe.
- Coin transport is mass-invariant (with the Coulomb + position-control mechanism) and friction-dependent.
- The intermittent option language (as implemented) does not beat the searched legacy macro; its downstream options never
  fire because transport is the wall.

**NOT claimed / provisional:**
- The intermittent controller is **un-searched/un-tuned** vs the 128-shot searched legacy — the *absolute* K6 gap is an
  unfair comparison; a searched intermittent controller might transport more. The **structural** finding (downstream
  options never fire; transport is the wall) is robust across 16 states, but the absolute intermittent K6 is provisional.
- "Friction-limited tangential coupling is the *sole* cause" is **inferred** (from 3 converging results), not proven by a
  single discriminating experiment.
- Sub-unity friction cells are not interpretable (max-combination).

## 6. Exact next gate (options for the user)

The wall is now localized to the contact mechanics. Concrete next steps, in increasing scenario change:
- **(a)** Raise the *effective* tip↔coin friction (it currently caps the tangential drag) and/or search the intermittent
  impulse parameters — a fair, tuned intermittent controller vs the searched legacy, to see if transport recovers within
  the current geometry.
- **(b)** Change the end-effector geometry toward the deferred **SUSTAINED/GRASP-TRANSPORT** benchmark (two-sided clamp /
  concave tip) so tangential coupling is not friction-limited — this is the separate benchmark, not the coin task.
- **(c)** Accept the coin as a genuinely hard, friction-limited push task and rest the O3/coin claims on the intermittent
  *contact-safety* certificate (V4) rather than delivery.

I did not choose — (a) vs (b) vs (c) changes the scenario's scope.

---

### Commits
- `b2ea05e7` — freeze `V4_INTERMITTENT_CONTACT` + force decomposition.
- `d693e0f3` — intermittent controller + C2 ablation + mass×friction characterization.
- this report — final.

### Preserved unchanged
V2 / V3_AGILE frozen contracts, `COIN_LEGACY_FAST_V1`, the continuous-transport C2 (`coin_c2_ablation.py`, superseded by
the intermittent one), all prior results.
