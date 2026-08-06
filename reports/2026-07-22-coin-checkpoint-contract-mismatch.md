# §7 Checkpoint-size-mismatch diagnosis — CHECKPOINT_CONTRACT_MISMATCH

**Created-at:** 2026-07-22 14:10 JST
**Branch:** recovery/coin-hymeko-bundle-and-results
**Scope:** diagnose the `test_certifier_matches_raw_oracle_on_rollout` load failure (`arm_actors.0.1.bias shape [2] vs [4]`). Diagnosis only — no retrain, no reshape (per §7 of the directive).

## Verdict: `CHECKPOINT_CONTRACT_MISMATCH`

The frozen E-approach deploy checkpoint `experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt`
(sha256 `7dbbf1a7782f`, the b822a660 deploy actor) matches its **own** declared architecture contract
— the **legacy** robot graph — but not the architecture the loader now declares against the **v2**
canonical robot. It is neither corrupt nor the wrong file.

## Evidence (discriminating tests)

| Check | Result |
|---|---|
| `E_valselect_v2.pt` sha256 prefix vs pinned `7dbbf1a7782f` | **match** → artifact intact, not corrupt |
| load into a **legacy-robot** actor (`build_collaborative_offpolicy`, `robot_source="legacy_python"`) | **CLEAN** (0 missing / 0 unexpected) |
| load into the **v2-robot** actor (canonical `make_coin_env`, `robot_source="hymeko_spec"`) | **FAILS** — shape mismatch |

Mismatched tensors (checkpoint → current v2 model):
- `arm_actors.0.0.1.weight` `[64, 48]` → `[64, 64]`
- `arm_actors.0.1.weight` `[2, 64]` → `[4, 64]`
- `arm_actors.0.1.bias` `[2]` → `[4]`

## Root cause

The per-arm actor's input width is the **flattened `node_features`** of the Coin env, and
`node_features` has one row per hypergraph vertex, i.e. one per MJCF body:

| robot source | bodies | graph vertices | `node_features` | per-arm actor input |
|---|---|---|---|---|
| `legacy_python` (make_planar_arms_mjcf) | 8 | 6 | (6, 8) | **48** |
| `hymeko_spec` (galambos_planar_v2) | 10 | 8 | (8, 8) | **64** |

`galambos_planar_v2.hymeko` realises each fingertip as a **separate massless child body** (the
kinematics IR carries one geom per link), so the v2 robot has **+2 bodies → +2 graph vertices** than
the golden's in-link fingertip geoms. This was documented as the one explained structural delta in the
robot-parity proof (`test_fingertip_body_difference_is_the_only_structural_delta`) and is
**kinematically equivalent** (0.00 mm fingertip parity) — but it is **not graph-identical**. The v2
change therefore altered the `node_features` contract that every graph-state policy consumes.

Wiring `make_coin_env` to `robot_source="hymeko_spec"` (commit 9e0c26f) propagated this 6→8 node
change into the canonical env. So `load_e_approach_policy` now builds a 64-dim / 4-DoF-per-arm actor
that the frozen 48-dim / 2-DoF-per-arm checkpoint cannot fill.

## Reproduction

```
.venv/bin/python -m pytest hymeko_rl/tests/test_strict_monitor_contract.py::test_certifier_matches_raw_oracle_on_rollout -p no:randomly -q
```
(fails at line 33 `build_actors("P4_E_APPROACH_HANDOFF")` → `load_e_approach_policy()` → v2-robot actor
build + frozen-checkpoint load.)

## Implication for §10 reproduction (blocker)

Every frozen deploy checkpoint in the reproduction set (E_valselect, handoff transport, relay/neutral
bridge, POINT zero-shot, residual, corrected bridge, full-action BC) was trained on the **legacy
6-node** graph. None can load against the v2-robot canonical env. §10 reproduction is therefore
**blocked** on a canonical-robot-lineage decision (see the handoff message). No retrain/reshape was
performed, per directive.
