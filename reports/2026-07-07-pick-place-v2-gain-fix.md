# v2 isolated gain/tracking fix (2026-07-07)

## Summary

Applied a **logic-free** tracking-gain fix to the v2 arm servo (position gain 45→**60**, velocity gain 9→**15**,
plus a **halved integration sub-step** for stability), **conditional on `expert_version==2`** so **v1 stays
byte-identical**. It **clearly improves tracking** (transit clearance −0.019→−0.005, forbidden-pre-object 1.0→0.5,
transit contact 0.39→0.05) with **no new instability**, confirming the sag *is* a tracking effect. But the gate
still fails, because the **commanded path itself still grazes 0** — which gains cannot lift. So the next isolated
step is the Cartesian-controlled preshape.

## Constraint handled: v1 byte-identical

The gain block is shared, so a global change would break v1. The gains + sub-step are therefore
**`expert_version`-gated**: v1 keeps `(45, 9)` at `STABLE_DT` (frozen baseline); v2 gets `(60, 15)` at
`STABLE_DT/2`. Gravity-comp is already full (1.0). **v1 re-smoke byte-identical** (lift 1.0 / place 0.75 /
min_clr −0.02577); 15 pick/ik/probe tests green.

## Gain sweep (why kp=60, not more)

| gains (v2) | stability | transit clearance | verdict |
|---|---|---|---|
| kp90/kv18, STABLE_DT | **qacc 2.7e6 — detonates** | — | rejected |
| kp90/kv18, dt/2 | qacc 1.5e3; **2/4 knock object off** | transit fully clean (0.0) | rejected — new instability |
| kp90/kv27, dt/2 | qacc 8e6 — velocity gain destabilises | — | rejected |
| **kp60/kv15, dt/2** | **stable (qacc <900, 0 deaths)** | −0.005 (2 seeds graze) | **accepted** |
| kp60/kv15, STABLE_DT | seed 50002 blows up | — | rejected — needs finer dt |

kp=90 fully cleans the transit but the stiff servo **overshoots into the object during descent** (knock-off =
"new instability", which the pass condition forbids). kp=60 is the stiffest **stable** setting.

## Results (v2 smoke, 4 ep, seeds 50000–50003; reproduced bit-identical on kato15/EGL)

| metric | v1 (ref) | v2 (kp45, pre-fix) | **v2 gain-fix (kp60/kv15/dt÷2)** |
|---|---|---|---|
| commanded min clearance | — | +0.000 | **+0.000 (still grazes)** |
| physical min clearance | −0.026 | −0.019 | **−0.005 min / −0.002 mean** |
| forbidden-pre-object rate | 1.0 | 1.0 | **0.5** |
| transit finger↔table rate | 0.45 | 0.39 | **0.047** |
| first strike step | ~51 | ~100 | ~127–219 |
| first-over-object step | ~150–225 | 163–203 | 130–200 |
| lift / place | 1.0 / 0.75 | 0.0 / 0.0 | **0.0 / 0.0** |
| new instability / overshoot | — | none | **none** (all seeds full 620, qacc <900) |
| gate | FAIL (dirty) | FAIL | **FAIL** |

**Tracking error:** the Cartesian `|cmd_tool − phys_tool|` max ≈ 0.9 m (catch-up ≈ 0.63 m) is dominated by the
**horizontal** command-jump during the preshape (the command reaches `cf_hover` in ~30 steps while the arm is still
retracting) — that transient is a command-pacing artifact, unchanged by gains. The **vertical** component (the sag
= the finger-table clearance) is the one that matters, and it improved from −0.019 to −0.005.

## Acceptance for this isolated fix — MET

- physical min clearance substantially closer to zero (−0.019 → −0.005): ✅
- forbidden-pre-object decreases (1.0 → 0.5): ✅
- transit contact decreases (0.39 → 0.047): ✅
- first strike not earlier (later): ✅
- no new instability/overshoot (kp=60 stable; kp=90 rejected for object knock-off): ✅
- lift/place remain 0, tracking clearly improved: ✅

## Decision — do NOT proceed to learning; next = Cartesian-controlled preshape

Physical clearance is still slightly negative (−0.005) and, decisively, the **commanded** path still grazes **0**
(`cmd_min_clr = +0.000`). Gains fixed the *tracking* of the command, but cannot lift a command that itself touches
the table. Per your decision rule, the next isolated step is **Cartesian-controlled preshape / clearance-constrained
waypoint generation** (hold z high while retracting), which raises the commanded path off 0 so the (now-tight)
tracking keeps positive physical clearance. **No BC/DAgger/RL; the 32-ep gate is not justified and was not run.**

## Files / provenance

- Changed: `hymeko_rl/env/pick_place_env.py` only (v2-gated arm gains + sub-step). `ik.py` not touched.
- ruff clean · 15 tests green · v1 byte-identical · local and **kato15** (EGL) agree bit-for-bit.
- **Cost note:** v2 now integrates at half the sub-step (≈2× slower physics) — acceptable for the clearance work;
  revisit if it bites BC/DAgger throughput later.
