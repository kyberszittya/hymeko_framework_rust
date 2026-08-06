---
campaign: COIN concave-clamp — tandem-prong geometry correction + re-test
title: The reviewer was right — the prongs were tandem (frame bug); corrected to an orientation-invariant ring, contact improves but force closure still does not certify
date: 2026-07-21
branch: exp/coin-independent-concave-clamp
supersedes: the tandem-geometry NO_FORCE_CLOSURE conclusion in 2026-07-21-coin-concave-clamp.md
classification: NO_FORCE_CLOSURE (now on a CORRECTED, orientation-invariant multi-geom cradle — not just the tandem placement)
---

# Concave-clamp geometry correction

**Created-at:** 2026-07-21 12:30 JST. The review flagged a likely local-frame error: the FLAT_PAD box is thin in its
first (X) half-extent, implying **local X is the contact normal**, yet the clamp prongs were separated **along local
X** — i.e. one behind the other along the closing direction (the front prong shadowing the rear), which would explain
the low bilateral contact and slip. This was correct.

## 1. Measured fingertip local basis (empirical, at a grasp contact)
| local axis | world direction | role |
|---|---|---|
| X | ≈ −world Y | **contact normal** (`|dot(localX, fingertip→coin)| = 0.86`) |
| Y | ≈ +world X | in-plane tangent |
| Z | = world Z | vertical (cylinder/box axis) |

The original prong separation was along local X → **`|dot(prong-sep, contact-normal)| = 0.86` = tandem.** Confirmed.

## 2. The deeper reason a *fixed directional* cradle can't work here
Expressing the fingertip→coin direction **in the local frame** across grasp configs (multiple seeds) gives
`+Y`, `−Y`, `+X` — it **varies with the grasp**. The 2-link arm (j1/j2, both Z-hinges) has **no wrist DoF to orient the
cradle toward the coin**, so *any* fixed 2-prong offset is side-by-side for some grasps and tandem for others. A single
directional cradle is therefore not robust — this is the kinematic limitation the ledger's CLAMP-ORACLE named, now
pinned to its cause.

## 3. Correction — an orientation-invariant ring cradle
`with_fingertip_clamp` now builds a **horizontal RING of 6 prongs** (local X-Y plane; local Z is the vertical coin
axis). Whatever horizontal direction the coin contacts from, the two nearest prongs on the ring straddle it. This
removes the orientation dependence. (Also discovered: the coin is a flat-faced **box**, half-extent 0.02 — not a
cylinder — so a ring contacts a flat face at ≈1 point, an independent limitation.) Top-down debug render (fingertip
prongs, coin box, MuJoCo contact points):
[reports/figures/2026-07-21-concave-clamp/clamp_topdown_debug.png](figures/2026-07-21-concave-clamp/clamp_topdown_debug.png)
— shows 2 coin contacts at the illustrated grasp.

## 4. Contact-angle oracle (contacts per fingertip side at grasp, over seeds)
The corrected ring reaches ~0.8 contacts/side on average (up to 2 at favourable configs) — **better than the tandem
placement** (which shadowed to ~0.5) but short of a reliable ≥2/side cradle, because the flat box face is contacted at
one point and the arm cannot press/orient the ring firmly.

## 5. Re-ran the physical oracle — corrected ring vs POINT / FLAT_PAD (16 seeds, push + grasp_carry)
| geometry | strict ≥+0.030 | max strict clearance | both_frac (persistence) |
|---|---|---|---|
| POINT | 1 | +0.037 | 0.05–0.13 |
| FLAT_PAD | 1 | +0.039 | 0.10–0.17 |
| **CONCAVE_CLAMP (corrected ring)** | **0** | **—** | **0.07–0.19 (now ≥ flat pad)** |

The corrected geometry **fixes the low-contact symptom** (both_frac now up to 0.194, comparable-to-above the flat pad)
but **still achieves 0 strict** transport ≥+0.030. **Verdict: NO_CLAMP_ADVANTAGE.**

## Classification: **NO_FORCE_CLOSURE** — now earned on a corrected multi-geom cradle
The tandem-based conclusion is **superseded**: with the geometry corrected (orientation-invariant ring, better contact
than the flat pad), the clamp *still* does not force-close a certifiable delivery. The residual blockers are now
precisely: (a) the 2-link arm cannot **orient or firmly press** a fingertip cradle (no wrist DoF), and (b) the coin is a
**flat-faced box**, so a fingertip cradle contacts one face and the two arms already grip opposite faces — adding
prongs does not add closure. Per §4, **no RL was launched** (no force-closure demonstrated). No friction or actuator
was added, as instructed.

## Honest scope — the reviewer's caveat, now resolved
The prior NO_FORCE_CLOSURE was, as the reviewer noted, valid only for the tandem placement. This report tests a
**correctly oriented multi-geom concave face** and the conclusion holds — with the added, more informative diagnosis
(no-wrist-DoF + box-coin). It is still **not** an absolute impossibility: an **independent-pad rig with its own closure
+ wrist DoF** (the ledger's validated CLAMP-ORACLE) remains the sufficient lever — a larger embodiment change, not a
fingertip cradle. **Coin Delivery remains open.**

## Files / provenance
- `hymeko_rl/env/planar_grasp_env.py` — `with_fingertip_clamp` corrected (tandem → orientation-invariant ring).
- `experiments/2026_07_21_coin_concave_clamp/oracle_corrected/clamp_oracle.json`; top-down render under `reports/figures/2026-07-21-concave-clamp/`.
- 9 golden tests pass; POINT byte-identical; no CORE.YAML items; no deps. Preserved: transport `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`. Commits `b3c20a6` (embodiment+oracle) · `1f41d34` (initial report) · (this) correction. Host Apple M5 Pro.
