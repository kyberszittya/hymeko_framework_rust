# R2 relational representation contract (FROZEN before training)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-decision-representation` · **Base:** `c70c844a`
(`FLAT_R1_LEARNED_AMORTISATION_FAILS`). This document is the **frozen experimental contract** for R2. It exists so no new
feature, capacity, or search change can enter unnoticed. R2 is authorised **only** as the isolated *relational-organisation*
axis.

## Invariant — the single thing that changes

> R2 receives **exactly** the verified R1 v3 canonical physical information, organised as explicit HyMeKo relations instead
> of a 43-D flat vector.

**Frozen (identical to the R1 learned update-0, `c70c844a`):** dev teacher sets, dev acceptable sets, canonical θ labels,
K-head acceptable-set output + loss, dev-only K-selection procedure, optimiser/epochs/init-seed policy, **total budget-8**
search + centre-inclusion + per-mode split, physical PUSH/BRAKE/RELEASE option, K6 monitor, frozen 4-state panel, held-out
discipline. **Forbidden:** any new state variable, any held-out-derived feature, changed action semantics, larger search
budget, or changed teacher sets.

## Graph schema

**Typed nodes** (the two sides share a node type AND tied encoder weights — no separate left/right embedding, so the net
cannot re-learn the arbitrary labelling):

| node | type | attributes (all from the R1 canonical extractor) |
|---|---|---|
| `coin` | COIN | target-frame position (dtz along/perp), target-frame velocity (coin_vel_along/perp), phase/horizon (const at t=0) |
| `target` | TARGET | zone tolerance (CENTER_TOL), settle limit (SETTLE_VEL), dwell (HELD_DWELL) — constant in the canonical frame |
| `tip_{L,R}_canonical` | TIP | coin-relative along/perp (tip_coin_along/perp) |
| `contact_{L,R}_canonical` | CONTACT | normal along/perp (normal_along/perp), v_n (vrel_normal), v_t (vrel_tangent), F_n (fn), friction utilisation (bounded slip) |
| `actuation_port_{L,R}` | PORT | prev post-governor torque (prev_tau_arm), ±slew headroom (slew_head_up/dn), per-side authority magnitude (btau_side_auth) |

**Typed directed relations / hyperedges** (attributes are R1 groups; nothing new):

1. **Goal** `coin → target`: distance along/perp, velocity along/perp, zone-entry margin, settling margin.
2. **Contact geometry** `tip_i → contact_i → coin`: relative position, normal/tangent frame, v_n/v_t, F_n/F_t proxy,
   friction utilisation.
3. **Actuation→contact authority** `port_i → contact_i → coin`: forward-push reach, forward-reverse reach, lateral ±
   reach (side-consistent), slew-admissible margin, **governor attenuation** (bcoin_min_attenuation).
4. **Bimanual hyperedge** `{contact_L, contact_R, coin, target}` — the load-bearing HyMeKo element (what flat R1 had to
   implicitly cross-multiply): straddle geometry, contact separation, normal alignment, **squeeze/internal-force reach**
   (normal_force_reach_pair), **L/R balance reach** (balance_reach_signed), combined forward authority, combined brake
   authority, combined lateral authority.
5. **Option-mode context** `{coin, target, both contacts, phase}`: a global/hyperedge context (no separate phase node —
   all decisions start in the same phase), schema-ready for PUSH/BRAKE/RELEASE.

## Information-parity map (EVERY R1 v3 group has exactly one home)

| R1 v3 group (43-D) | R2 location |
|---|---|
| dtz | coin node |
| coin_vel_along / coin_vel_perp | coin node |
| straddle | bimanual hyperedge |
| tip_coin_along / tip_coin_perp | tip node |
| normal_along / normal_perp | contact node |
| fn | contact node |
| vrel_normal / vrel_tangent | contact node (+ contact-geometry edge) |
| friction_util | contact node |
| slew_head_up / slew_head_dn | port node |
| prev_tau_arm | port node |
| btau_svals / btau_summary | actuation→contact authority edge (global-ish) |
| btau_side_auth | port node |
| forward_push_reach / forward_reverse_reach | actuation→contact authority edge + bimanual (combined) |
| lateral_reach_pair | actuation→contact authority edge (side pair) |
| brake_opposed_reach | actuation→contact authority edge + bimanual (combined) |
| bcoin_min_attenuation | actuation→contact authority edge |
| normal_force_reach_pair | bimanual hyperedge (squeeze) |
| balance_reach_signed | bimanual hyperedge (balance) |

Because the R2 adapter is built **from the verified R1 canonical extractor** (not the env's raw `node_features()`), this
map is exact and testable both ways (no new information, no lost information).

## Encoder (minimal first — capacity-matched, no tricks)

`typed node encoders → 2 rounds typed message passing → coin + target + bimanual/global pooling → the SAME frozen K-head
acceptable-set head`. Rules: **tied weights** for the two sides; one MLP per relation type; sum or mean aggregation; **2**
message rounds; **no attention** in v1; hidden width bounded so the **parameter count ≈ the R1 K-head model** (or a
declared capacity-matched variant); the same canonical θ-label and K-head set-loss. Output unchanged:
`canonical graph → K canonical θ centres → inverse T_θ → total budget-8 search → 6-D option → frozen K6`.

## `planar_grasp_env` HyMeKo topology — audit, don't reuse blindly

`self.hg` topology may be a useful starting point, but the existing `node_features()` may contain raw world coordinates,
raw joint angles, the original L/R label, non-canonical ordering, or R1-excluded/mis-scaled features. **Safe path:**
`verified R1 canonical extractor → R2 graph adapter → existing HyMeKo runtime/topology where compatible`. Do **not**
recover invariance from the env's old node features.

## Mandatory tests (beyond the 6 canonical-frame tests)

1. **Graph mirror-equivalence** — `G(Mx) ≅ G(x)` after canonicalisation (identical node/edge attributes).
2. **Node-permutation invariance** — swapping the physical L/R node order leaves the graph embedding unchanged (tied weights).
3. **θ-output equivariance** — decoding the canonical output flips the balance sign correctly.
4. **R1 information parity** — every R1 v3 feature group is recoverable from the R2 graph; nothing new, nothing lost.
5. **Parameter-budget audit** — R2 param count ≈ the R1 K-head, or a declared capacity-matched variant.
6. **Search/provenance regression** — same K, centre-inclusion, `budget/K` split, θ₀/θ_exec provenance, K6 monitor.

## Decision tree

| result | verdict | RL |
|---|---|---|
| **4/4, held-out 2/2** | `HYMEKO_STRUCTURAL_REPRESENTATION_LOAD_BEARING` → UPDATE_ZERO_NO_REGRESSION_PASS | **SAC/TD3 authorised** |
| 3/4 or held-out 1/2 | `RELATIONAL_ORGANISATION_IMPROVES_GENERALISATION` (update-0 still FAIL) | blocked |
| 2/4, held-out 0/2 | `RELATIONAL_ORGANISATION_ALONE_INSUFFICIENT` → next axis: action intent / deterministic decoder | blocked |
| dev regression | implementation / capacity / training audit — **not yet a scientific negative** | blocked |

Any honest failure is reported **without** changing the physics, teacher sets, search budget, or held-out discipline.

## Build order (after this frozen contract commit)

1. `theta_option/relational_graph.py` — the R2 graph adapter (R1 canonical extractor → typed nodes/edges/hyperedges) +
   the parity/mirror/permutation tests (no training).
2. `theta_option/relational_encoder.py` — typed 2-round message passing → pooling → the existing K-head; capacity-matched.
3. `coin_theta_rl_benchmark --r2-update0` — same procedure as `--r1-update0`, graph representation only; one frozen-panel
   evaluation; the decision tree above.
