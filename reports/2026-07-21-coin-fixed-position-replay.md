---
campaign: Fixed-initial-position Coin Delivery replay on the released source tag
title: FIXED_POSITION_REPRODUCED — seed-1011 exact position reproduces 10/10 strict through both replay modes (bit-identical), and a new user-selected position also succeeds (FIXED_POSITION_POSITIVE)
date: 2026-07-21
source_tag: coin-delivery-neutral-v1-source
tested_commit: 5026152ef80189d92aaad20f5bf0541fbc207c19
branch: exp/coin-fixed-position-replay
verdict: FIXED_POSITION_REPRODUCED
---

# Fixed-position Coin Delivery replay — result

**Created-at:** 2026-07-21 21:27 JST. Built on the released source tag `coin-delivery-neutral-v1-source`
(`5026152`); the tag is **not** modified. Focused branch `exp/coin-fixed-position-replay`.

## What was added (production, experiment-web-free)
| module | role |
|---|---|
| `hymeko_rl/coin_delivery/fixed_position.py` | `CoinInitialState` schema + validation (fail-loud), exact `apply`/`extract`, `problem_hash`, checkpoint-hash manifest |
| `hymeko_rl/coin_delivery/fixed_position_replay.py` | reachability analysis, the fail-loud replay gates (§2), the traced two-phase rollout (mirrors `eval_composed`) |
| `hymeko_rl/coin_delivery/fixed_position_campaign.py` | the two replay modes + P0/P1/P4 causal controls orchestration |
| `hymeko_rl/coin_delivery/fixed_position_video.py` | 100 FPS real-time mp4 export with overlays |
| `hymeko_rl/campaign/{__main__,adapters}.py` | CLI flag form `run --domain coin --seed … / --initial-state …` + `CoinFixedPositionAdapter` |
| `configs/problems/{coin_point_1011_exact,coin_fixed_position}.json` | the exported seed-1011 exact state + the user-selected position |

Coordinate conventions (the planar Coin env, `nq=7`): `qpos=[j1_left,j2_left,j1_right,j2_right, disk_x,disk_y,disk_rz]`;
the coin is **planar** (x, y, yaw), the target is the zone site, `control_dt = 20×5e-4 = 0.01 s`. A non-planar
`coin_quaternion`, an off-table `z`, a non-finite value, a `control_dt` mismatch, an in-contact neutral start, a
target-overlapping coin, an embodiment or obs/action-schema mismatch, or a checkpoint-hash mismatch each **fails loud**
— no fallback to a generated seed or contact-prepared bank (§2, tested).

## §1A/§4 Deterministic problem replay — seed 1011 reproduces the released headline
```
python -m hymeko_rl.campaign run --domain coin --seed 1011 --embodiment POINT --policy-chain causal --repetitions 10
problem_hash = f175a7c5b269e3f4   geom_fp = 498e4e575065   initial signed clearance = +0.079284
arm_qpos = [0,0,0,0]   coin = (-0.158877, 0.121579, yaw 0)   target = (-0.021588, 0.145066)   initial contact = none
```
| policy | strict | first | bilateral | zone | trajectory hash |
|---|---|---|---|---|---|
| **P4** E-approach + handoff-matched transport | **10/10** | 10 | 10 | 10 | `a3823a8455a6151e` |
| P1 frozen transport (`8bd73d8c`) | 0/10 | 10 | 10 | 0 | `0ea5650fffeaeb89` |
| P0 zero action | 0/10 | 0 | 0 | 0 | `219ff527248cba98` |

Matches the released report (P4 10/10, P1 0/10, P0 0/10; geom_fp `498e4e57`; clearance ≈ +0.079). **No regression.**

## §5 Exact-state replay — bit-identical to the seed run
The seed-1011 state was exported to `configs/problems/coin_point_1011_exact.json` and replayed through the exact-state
path:
```
python -m hymeko_rl.campaign run --domain coin --initial-state configs/problems/coin_point_1011_exact.json --policy-chain causal --repetitions 10
```
→ same `problem_hash f175a7c5b269e3f4`, and **identical trajectory hashes** for all three policies
(`a3823a84` / `0ea5650f` / `219ff527`), identical strict results, identical initial `node_features` (max|Δ|=0.00e+00)
and identical initial physics (`apply(extract(seed 1011))`: max|Δqpos|=max|Δqvel|=Δzone=0). Exact-state replay is a
pure, seed-hint-independent function of the state (proven).

## §6 User-selected position — FIXED_POSITION_POSITIVE
A **new** coin position `(-0.115, 0.115)` (neutral arm, same target/embodiment/checkpoints, `configs/problems/coin_fixed_position.json`):
```
signed clearance +0.03813   coin_reachable True   collision_free True   initial contact none
```
| policy | strict | trajectory hash |
|---|---|---|
| **P4** E-approach + handoff | **10/10** | `895b479a42a683ad` |
| P1 frozen transport | 0/10 | `67c0bef4e7b96216` |
| P0 zero action | 0/10 | `89d1cc5d047feb65` |

The learned chain **succeeds at a new position where the frozen-transport control fails** — a genuine generalization,
not a memorized seed. (Probed workspace: positions with larger clearance need the learned transport; a
target-overlapping coin is rejected as `INVALID_INITIAL_STATE`.)

## §7 Video (100 FPS real time, 960×720, overlaid)
- `reports/video/coin_fixed_position/coin_delivery_fixed_position_real_time.mp4` sha256 `ffe9dac97f0c553a` — P4 delivery.
- `reports/video/coin_fixed_position/coin_delivery_fixed_position_controls.mp4` sha256 `484486b6bd8b579d` — P4/P1/P0.
Overlays: EXACT FIXED START, initial coin coordinates, initial clearance, policy, phase, elapsed physical time, live
strict result. (mp4s are external artifacts, gitignored; hashes recorded in `video_manifest.json`.)

## Tests
`hymeko_rl/tests/test_fixed_position_replay.py` (13) — schema validation + fail-loud, extract/apply bit-identity,
problem-hash determinism, reachability, target-overlap/embodiment rejection, seed-vs-exact rollout bit-identity. Plus
the architecture-guard ratchet updated 40→43 (the new modules import ONLY the canonical coin_* chain, **no arc** —
`test_coin_delivery_import_closure_is_experiment_free` still passes). All green on the tag (`5026152`).

## Checkpoint hashes (external artifacts, provisioned by manifest)
E_valselect `7dbbf1a7782f` · handoff `8955e8db8ac1` · frozen `8bd73d8cbea0`.

## Verdict: **FIXED_POSITION_REPRODUCED**
The released seed-1011 exact position reproduces **10/10 strict** through both the deterministic-problem and the
exact-state replay paths, bit-identically; the causal controls fail as expected (P1 0/10, P0 0/10); the two replay
modes are provably equivalent. **Additionally FIXED_POSITION_POSITIVE**: a new user-selected position `(-0.115, 0.115)`
succeeds (P4 10/10) where the frozen control fails. No regression to the released result.
