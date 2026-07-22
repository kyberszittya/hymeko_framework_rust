# Option B — one canonical robot graph: GALAMBOS_PLANAR_GRAPH_CONTRACT_EQUIVALENCE_PASS

**Created-at:** 2026-07-22 14:50 JST
**Branch:** recovery/coin-hymeko-bundle-and-results (commits 2be779f → c6aa11f)
**Decision:** keep the HyMeKo-emitted v2 physical robot canonical; project its semantic policy graph to the
legacy 6-vertex / 48-dim contract so the frozen deploy stack loads unchanged. No retrain, no reshape.

## Result

The canonical v2 Coin robot now has:
- **physical MuJoCo model:** v2, 10 bodies (2 fingertip helper bodies included, contact-active);
- **semantic policy graph:** legacy-compatible **6 vertices** (`base/link1/link2 × left/right`), **48**-dim actor input;
- **frozen checkpoints:** load unchanged; graph-state actor step-zero action legacy-vs-v2 **Δ = 0.0**;
- **physical/kinematic/contact parity:** unchanged (`GALAMBOS_PLANAR_HYMEKO_EQUIVALENCE_PASS` preserved).

## Mechanism (spec-driven, explicit — not a name heuristic)

1. **Semantic metadata (2be779f).** `galambos_planar_v2.hymeko`: `graph_node 0` on `fingertip_{left,right}`
   (marks them non-semantic geometry/emission helpers), and joint names `j{1,2}_{left,right}` (legacy-identical,
   left-arm-first) so the emitted actuators carry the `_left`/`_right` suffix the arm-action partition needs.
2. **Extractor projection (e51efd7).** `emit_galambos_v2_mjcf` reads the `graph_node 0` links
   (`semantic_helper_bodies`) and injects a MuJoCo-legal `<custom><text name="hymeko_non_semantic_bodies">`
   marker. `HypergraphState.from_mjcf` parses it and drops the flagged **leaf** helpers from the vertex set,
   re-indexing contiguously (a flagged non-leaf is rejected loudly). Default-safe: no marker → byte-identical
   legacy behaviour for every other robot. `PlanarGraspEnv._vtx2body` maps each semantic vertex to its correct
   physical body **by name**, so `node_features` reads the right body despite interspersed helpers
   (v2 bodies `[1,2,3,5,6,7]` vs legacy `[1..6]`). Added `HypergraphState.semantic_fingerprint()`.

## Gates

| Gate | Status | Evidence |
|---|---|---|
| `GALAMBOS_PLANAR_GRAPH_CONTRACT_EQUIVALENCE_PASS` | **PASS** | `test_galambos_graph_contract_equivalence.py` (6): identical vertices/edges/signs/fingerprint, raw node-feature + actor-input Δ<1e-6 over a 6-pose panel, identical actuator ordering, RING→6 |
| Frozen-checkpoint v2 compatibility (§5) | **all 3 COMPATIBLE** | `test_frozen_checkpoint_v2_compat.py` + `logs/checkpoint_compat_v2.json`: E_valselect (7dbbf1a7, Δ0.0), handoff (8955e8db), frozen_transport (8bd73d8c) |
| Physical parity (§6) | **unchanged** | `test_galambos_planar_v2_parity.py` (5) still green after projection |
| Control + scene authority | **PASS** | 6 + 8 sentinels (230e44b, 71fe99f) |
| Canonical bundle identity (§7) | **manifest** | `logs/canonical_bundle_manifest_v2.json`: physical fp (nbody 10) + semantic fp (sem:469094de…, 48-dim) |

Pre-existing, unrelated failures (verified identical on the pre-graph baseline; NOT introduced here):
`test_topology_zoo[petersen/kneser/grotzsch/expander]`, `test_planar_grasp_env::{env_shapes_and_coin_placed_in_reach,
compiled_model_coin_collides_with_fingertip_not_arm_capsule}`, `test_coin_delivery_rl::scripted_baseline_in_delivery_env0_band`.

## Remaining (ordered, per directive §10)

4. **§9 discounted-alignment gate** (`COIN_DISCOUNTED_REWARD_ALIGNMENT_PASS`) — relaxed load-bearing form:
   discounted(strict K=6 delivery) > every non-success class (γ=0.99); internal failure ordering diagnostic; no
   farming. Needs the K=6 wrapper + a faithful behavior harness (not rushed — an untrustworthy alignment verdict is
   worse than a deferred one).
5. **§5 bundle gate** (`HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS`) — combine control + scene + graph + reward + checkpoint.
6–11. Historical repro (legacy column) → canonical v2 repro → full-action BC competence → critic → SAC/TD3 smokes →
   multi-seed campaign.
