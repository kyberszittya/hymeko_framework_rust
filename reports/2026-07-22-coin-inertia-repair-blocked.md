# GALAMBOS_PLANAR_INERTIA_REPAIR_BLOCKED — static parity achieved, dynamic parity blocked by fingertip-contact-body structure

**Created-at:** 2026-07-22 16:00 JST
**Branch:** recovery/coin-hymeko-bundle-and-results
**Canonical robot:** kept on **v2** (tree green, 42 gates) — v3 not promoted until the structural fix lands.

## Progress: GALAMBOS_PLANAR_INERTIA_PARITY_PASS (static)

The golden `make_planar_arms_mjcf` compiled model was extracted as the source of truth and encoded in a new
`galambos_inertia { base{…} link1{…} link2{…} }` contract in `galambos_planar_v3.hymeko` (plain decimals — the
`.hymeko` scalar grammar rejects scientific notation; the adapter re-emits MJCF exponent form). The adapter replaces
each structural link's `<inertial>` with the golden values. Result (`test_galambos_inertia_parity`, 4):

| quantity | golden | v3 | delta |
|---|---|---|---|
| total arm mass | 0.351557 | 0.353557 | +0.002 (fingertip floor) |
| worst structural-link mass | — | — | 4.2e-9 |
| worst structural-link inertia | — | — | 2.0e-15 |
| worst structural-link COM | — | — | 2.4e-8 |

Base/link1/link2 mass + inertia + COM match the golden **body-for-body**.

## Blocker: dynamic parity — the fingertip contact-body cannot be made massless

The golden folds the fingertip sphere into **link2** (COM at y=0.08348) — one body, mass+contact unified. The v3
design keeps the fingertip as a **separate welded child body** (required for the six-vertex graph projection +
POINT/RING/wrist embodiment transforms). To match golden mass exactly, that body must be massless — but a **welded
contact body with ~0 mass produces a NaN contact impulse** (`QACC NaN`, DOF 1–2, t≈0.01) the instant the coin
touches it. Confirmed across mass values:

- fingertip 1e-9 (exact parity): unstable;
- fingertip 0.001 (POINT): stable in isolation, but the neutral chain uses **E0 CONCAVE_CLAMP** — `with_fingertip_clamp`
  adds 12 clamp geoms to the fingertip body, and the explicit inertial override starves them → unstable;
- the natural physical split (link2 capsule-only 0.048 + fingertip sphere 0.011 = golden 0.0597) is **also** unstable
  (light welded link2 + separate tip body);
- only an EXCESS-mass fingertip (link2 = golden + non-degenerate fingertip) is stable — but then mass ≠ golden.

So the fingertip-as-separate-contact-body structure cannot simultaneously (a) match golden mass and (b) be
contact-stable. The naive override also perturbs the POINT strict-reference dynamics enough to break the discounted-
alignment strict reference.

## Robust fix (identified, not yet implemented) — the golden STRUCTURE

Emit the fingertip **contact geom on link2** (unified mass + contact, exactly like the golden, so contact is stable
against the massive link2), and keep the fingertip **body as a massless geomless FRAME** carrying only the graph
marker (`graph_node 0`) + the tool `site`. For E0/wrist, the clamp/pad geoms attach to the same massive body. This
reproduces the golden dynamics body-for-body while preserving the six-vertex graph projection and the tool point. It
is a bounded emit restructuring (touches `emit_galambos_v2_mjcf`, `with_fingertip_shape`, `with_fingertip_clamp`, the
contact-legality geom-name lookup) — designed carefully, not rushed.

## Why compute cannot fix it

Robot-specification structural defect. No search/training changes it.

## Reproduction

```
# static parity passes:
python -m pytest hymeko_rl/tests/test_galambos_inertia_parity.py -q
# dynamic instability (E0 chain, canonical=v3):  QACC NaN, deliver 0/9
```

## Status ledger (per directive §1)

- Tag `PRE_INERTIA_CANONICAL_BUNDLE_SNAPSHOT` @1039005 stands.
- Robot gate downgraded: `GALAMBOS_PLANAR_GEOMETRY_KINEMATICS_CONTACT_PASS` (full equivalence NOT claimed — inertia/
  dynamic parity pending).
- `HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS` reopened (the physical fingerprint represented incorrect dynamics); it will
  be re-closed after `GALAMBOS_PLANAR_DYNAMIC_PARITY_PASS` + `CANONICAL_DYNAMIC_EXPERT_PASS`.
- Static parity (`GALAMBOS_PLANAR_INERTIA_PARITY_PASS`) committed as a regression guard.
