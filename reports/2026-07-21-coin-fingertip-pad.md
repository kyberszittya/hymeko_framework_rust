---
campaign: COIN fingertip-pad contact-geometry intervention + physical feasibility oracle
title: A finite-area fingertip pad improves contact persistence but not strict transport — the physical oracle shows no pad advantage, so no RL campaign is launched
date: 2026-07-21
branch: exp/coin-fingertip-pad
source_commit: 8835db5
classification: NO_EFFECT (on the decisive strict-transport / frontier endpoint; the pad improves contact persistence but not strict coverage; §4 gate not passed → RL campaign not launched)
---

# Fingertip-pad contact geometry — the minimal embodiment lever

**Created-at:** 2026-07-21 11:45 JST. The live-frontier result (`8835db5`) closed the point-contact relay line as
NO_EARLY_FRONTIER — the transport-solvable basin is narrowly near-goal because sphere-on-cylinder is a rolling *point*
contact. The evidenced next lever was contact **geometry**. This iteration parameterizes it and gates on a physical
oracle *before* any RL, as required.

## §2 canonical parameterization (POINT vs FLAT_PAD)
Added `fingertip_geometry ∈ {POINT, FLAT_PAD}` threaded through the **existing** builder chain
`make_env → _roll_env → _env → make_delivery_rl_env → direct_env(fingertip_geometry=…)` (no second env builder).
POINT = the existing sphere fingertip (a no-op — golden). FLAT_PAD retypes the collision `fingertip_{side}` geoms to a
**box** via the already-present `with_fingertip_shape` hook, half-extents **0.004 × 0.016 × 0.02** (thin in the
contact-normal x = a flat face, wide in y/z), symmetric L/R, **existing friction** (no silent increase), no
suction / weld / magnet / coin constraint. Joint / actuator / arm / coin / target / reward / strict predicate /
body-shove rule all unchanged.

## §3 POINT backward compatibility — verified
POINT model geom-hash **unchanged**; FLAT_PAD is a **separate** compiled model (distinct geom-hash `cd51f238…` vs POINT
`f7df238b…`). Frozen transport on state `04870b0e`: **10/10 strict** under POINT (golden preserved). Fresh per-geometry
states only — no POINT snapshot restored into the FLAT_PAD model.

## §4 physical feasibility oracle (canonical scripted A0–A4 push, matched seeds, 16/band)
Actuator-limited scripted push/plow (`A0_sym_push … A4_recovery`) — no teleport, no coin manipulation, no reward
override. Best-of-actor per state, 4 clearance bands:

| band | POINT strict | POINT both_frac | FLAT_PAD strict | FLAT_PAD both_frac | FLAT_PAD loose |
|---|---|---|---|---|---|
| +0.018–0.030 | 1 | 0.05 | **2** | **0.16** | 4 |
| +0.030–0.045 | 1 | 0.10 | 1 | 0.11 | 2 |
| +0.045–0.060 | 0 | 0.13 | 0 | **0.17** | 1 |
| +0.060–0.080 | 0 | 0.13 | 0 | **0.14** | 0 |

**Strict ≥+0.030:** POINT **1**, FLAT_PAD **1** (max strict clearance +0.037 vs +0.039). **Verdict: NO_PAD_ADVANTAGE.**

## Classification: **NO_EFFECT** (on the decisive endpoint) — with the honest mechanistic nuance
- **The pad does what it was supposed to at the contact level:** bilateral contact persistence roughly **doubles**
  (both_frac 0.05–0.13 → 0.10–0.17) and loose zone-entry coverage rises — the finite-area patch holds the cylinder
  better than a rolling point contact, exactly the hypothesized mechanism.
- **But it does not convert into strict transport.** POINT and FLAT_PAD each certify exactly **1** state ≥+0.030 with
  the scripted oracle, at essentially the same max clearance (+0.037 / +0.039). The pad does **not** move the
  transport-solvable frontier outward. Not PAD_POSITIVE, not PAD_FRONTIER_POSITIVE, not PAD_ORACLE_ONLY (the oracle
  shows no pad *advantage* for a policy to then exploit).
- **Per §4, the pad oracle does not demonstrate a transport advantage over POINT, so the large RL campaign
  (§§5–11) is NOT launched** — launching it would chase a lever the physical measurement shows is absent, against the
  "gate expensive work on evidence" discipline.

## Why the pad helps contact but not strict transport (inferred)
Better persistence increases *loose* zone entries but the strict predicate additionally requires low settle-velocity
dwell in the zone; the extra contact the pad buys is spent holding a *rolling/slipping* cylinder rather than achieving a
stable *grasp*. This matches the arc ledger's terminal reading (`F-COIN-GRIPPER-GEOMETRY`, `COMPLIANT-PAD` STRICT 0
slip; `CLAMP-ORACLE` — a stable clamp needs an *independent-pad clamp rig*, a larger embodiment change than a
flat-face fingertip). A thin flat box is not enough curvature/geometry to convert push into grasp.

## §12 honest scope (not an impossibility)
This is **one** pad dimension (a thin flat box at the existing friction). Per the instruction, this is **not** an
absolute geometric impossibility — a different pad face (rounded / concave to cradle the cylinder), an explicitly
isolated friction change, or a learned policy exploiting the measured persistence gain remain untested. But none of
those is warranted by *this* oracle, which shows no strict-transport advantage; pursuing them would be the uncontrolled
geometry sweep the instruction forbids. **Coin Delivery remains open.** No demo produced (§4 gate not passed).

## Files / tests / provenance
- Threaded `fingertip_geometry` (POINT default, golden) through `exp_galambos_coord_ab.make_env`, `exp_v3_handoff_gate._roll_env`, `pedc_selection._env`, `coin_delivery_rl.make_delivery_rl_env`, `coin_two_arm_sac.direct_env`.
- `hymeko_rl/experiments/coin_pad_oracle.py` (NEW) — the §4 physical oracle (ruff-clean; the 65 pre-existing ruff errors in `pedc_selection.py` are baseline debt, not introduced here).
- 17 golden/coin tests pass; POINT path byte-identical (geom-hash + 10/10 unchanged). No CORE.YAML items; no deps.
- **Preserved:** frozen transport `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`; no canonicalization repeated.
- Data: `experiments/2026_07_21_coin_fingertip_pad/oracle/pad_oracle.json` sha `62f3766a`. Source `8835db5`; host Apple M5 Pro. Oracle wall ≈ 3 min, RSS ~0.45 GB.
