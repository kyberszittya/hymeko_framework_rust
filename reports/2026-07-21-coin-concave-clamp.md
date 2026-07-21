---
campaign: COIN independent concave-clamp embodiment + force-closure physical oracle
title: A fingertip concave cradle does not create force closure on the canonical two-arm rig — a genuine clamp needs an independent-pad rig, so no RL is launched
date: 2026-07-21
branch: exp/coin-independent-concave-clamp
source_commit: f30ef8e
classification: NO_FORCE_CLOSURE (fingertip-cradle geometry; the sufficient lever is an independent-pad rig — a larger embodiment change, not attempted, not an impossibility)
---

# Independent concave-clamp — build it, verify force closure physically first

**Created-at:** 2026-07-21 12:05 JST. The flat-pad result (`f30ef8e`) showed a larger contact patch improves
persistence but not force closure. The next lever was a genuine opposing-clamp geometry. Per the instruction I built
the clamp and ran the physical oracle **before** any RL — and the oracle gates it out.

## §1–2 embodiment (canonical builder, POINT/FLAT_PAD preserved)
Added `CONCAVE_CLAMP` to `make_env(fingertip_geometry=…)` via the existing `arm_mjcf_transform` hook →
`with_fingertip_clamp`: each single `fingertip_{side}` geom is replaced by **two prong geoms** straddling the coin in
the fingertip's local X (perpendicular to the +Y approach), forming a V-cradle whose two contact normals pass on both
sides of the coin centre (the flat pad had one central normal → the cylinder rolled). No new actuator — the existing
aperture/squeeze closes the opposing cradles. POINT model **unchanged** (golden), FLAT_PAD and CONCAVE_CLAMP are
**separate** compiled models (distinct geom-hashes); fresh per-geometry states (no cross-model snapshot restore).
Two cradle geometries were tried: **wide** (prong sep 0.048, r 0.007) and **snug** (prong sep 0.034 gripping the coin
flanks, r 0.011). Coin radius 0.02.

## §3–4 physical force-closure oracle (matched seeds, both push AND grasp_carry controllers)
Canonical scripted actors — the generic push/plow `A0–A4` **and** the proper `p_grasp_carry` clamp controller
(direction-to-zone + squeeze 0.8 = grasp-and-carry) — actuator-limited, no teleport / coin-manipulation / reward
override, best-of-actor, 4 clearance bands, 16 seeds:

| geometry | strict ≥+0.030 | max strict clearance | both_frac (contact persistence) |
|---|---|---|---|
| POINT | 1 | +0.037 | 0.05–0.13 |
| FLAT_PAD | 1 | +0.039 | 0.10–0.17 |
| **CONCAVE_CLAMP** | **0** | **—** | **0.0–0.13 (LOWER than flat pad)** |

**Verdict: NO_CLAMP_ADVANTAGE** — consistent across **both** cradle geometries and **both** controller families.

## Classification: **NO_FORCE_CLOSURE** — with the correct lever identified
- The fingertip cradle **does not restrain the cylinder**: it achieves **0 strict** ≥+0.030 and, tellingly, makes
  *less* bilateral contact than a flat pad (both_frac 0.0–0.13 vs 0.10–0.17). Two discrete prongs on a *single*
  fingertip make sparse point contact and, because both prongs sit on the *same* side of the coin (the two arms already
  close from opposite sides), they do **not wrap/cage** the cylinder — they poke it and it slips/rolls out.
- Not CLAMP_POSITIVE / CLAMP_ORACLE_POSITIVE / RELEASE_SETTLE_POSITIVE (no transport, no oracle strict). Not merely
  NO_EFFECT — the diagnostic is specific: **no force closure**.

## Why fingertip geometry alone is insufficient (measured → inferred)
**Measured:** changing only the fingertip geom (patch → cradle prongs) on the canonical two-arm rig does not create
force closure — contact drops and strict stays 0. **Inferred:** force closure on a cylinder needs contacts that *wrap*
around it (a concave face MuJoCo can't do with convex primitives) **or** two *independent* opposable pads with their own
closure degree of freedom. This matches the arc ledger's **CLAMP-ORACLE** finding exactly: the canonical rig is
**kinematically** limited; an *isolated 2-independent-pad rig* (O1 ρ0.86) **does** position-transport the cylinder. §2
anticipated this — "when an extra clamp-closure actuator is physically necessary, add it explicitly" — and that is the
**sufficient** lever: a larger embodiment change (new closure actuator + opposable-pad topology), **not** a fingertip
geometry tweak.

## §12 honest scope — not a geometric impossibility, and no RL launched
Per §4 the physical oracle does not demonstrate force-closure transport, so **the RL campaign (§§5–11) is not
launched** (launching it would train against an embodiment the physics shows cannot cage the coin). Per the explicit
instruction, this is **not** an absolute geometric impossibility: the independent-pad clamp rig the ledger already
validated remains the sufficient (larger) embodiment change; a MuJoCo concave mesh / multi-geom wrap face is also
untested. But neither is a *fingertip* intervention, and the fingertip lever — the minimal one this task scoped — is
**exhausted**. **Coin Delivery remains open.** No demo produced.

## §14 transfer note
The finite-area / force-closure abstraction still belongs in shared manipulation infrastructure for PickPlaceEnv gripper
pads and a future Beni end-effector — but the honest lesson is sharper: a manipulation end-effector needs an
**independent opposable-pad clamp with its own closure DoF**, not a fingertip geometry bolted onto a push rig. Recorded
for the next real manipulation adapter; **not** claimed as Beni manipulation (Beni is still locomotion-only).

## Files / tests / provenance
- `hymeko_rl/env/planar_grasp_env.py` — `with_fingertip_clamp` (NEW: V-cradle from prong geoms).
- `hymeko_rl/experiments/exp_galambos_coord_ab.py` — `make_env(fingertip_geometry=CONCAVE_CLAMP)`.
- `hymeko_rl/experiments/coin_pad_oracle.py` — 3-way oracle + `grasp_carry` clamp controller + release-settle diagnosis.
- 9 golden/coin tests pass; POINT byte-identical; no CORE.YAML items; no deps.
- **Preserved:** frozen transport `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`; no canonicalization repeated.
- Data: `experiments/2026_07_21_coin_concave_clamp/oracle/clamp_oracle.json` sha `4df1dc3a`. Source `f30ef8e`; host Apple M5 Pro. Oracle wall ≈ 4 min, RSS ~0.45 GB.
