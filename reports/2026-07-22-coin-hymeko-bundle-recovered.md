# COIN_HYMEKO_BUNDLE_RECOVERED — the executable Coin spec bundle is load-bearing, gated, and reproduces

**Created-at:** 2026-07-22 15:12 JST
**Branch:** recovery/coin-hymeko-bundle-and-results (9e0c26f → 111a5c9)
**Combined bundle hash:** `388dd238c2546354`

## Verdict

The entire executable HyMeKo Coin task specification bundle — reward, robot geometry, robot control, scene, and the
semantic policy graph — is genuinely load-bearing through the OUTER canonical runtime, sentinel-proven, discounted-
reward-aligned, and closed under one bundle gate. The frozen deploy stack reproduces unchanged on both the historical
and the canonical v2 runtime. No RL training was run.

## Gates (all PASS, all committed with tests)

| Gate | Verdict | Evidence |
|---|---|---|
| Robot physical equivalence | `GALAMBOS_PLANAR_HYMEKO_EQUIVALENCE_PASS` | `test_galambos_planar_v2_parity` (5) |
| Control-contract authority | `GALAMBOS_CONTROL_CONTRACT_RUNTIME_PASS` | 6 sentinels — spec→MJCF→MjModel, hard-fail no fallback |
| Scene authority | `HYMEKO_SCENE_RUNTIME_PASS` | 8 sentinels — spec→from_hymeko→runtime, no silent DEFAULT_ENV |
| Semantic graph equivalence | `GALAMBOS_PLANAR_GRAPH_CONTRACT_EQUIVALENCE_PASS` | 6 tests — v2 (10 bodies) → legacy 6-vertex/48-dim, Δ<1e-6 |
| Frozen-checkpoint compatibility | all `CHECKPOINT_CANONICAL_V2_COMPATIBLE` | 3/3 load unchanged, step-zero Δ0.0 |
| Discounted reward alignment | `COIN_DISCOUNTED_REWARD_ALIGNMENT_PASS` | strict +13.80 ≫ every failure (−18.1…−21.1), no farming |
| Complete bundle gate | `HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS` | 10/10 checks, one combined hash, no Python fallback |
| Reproduction (§10) | 4/7 `REPRODUCED_BOTH_RUNTIMES`, 3/7 `ARTIFACT_NOT_PRESENT` | frozen stack loads+acts identically on legacy & v2 |

49/49 gate tests pass together.

## Canonical contract (frozen)

- physical MuJoCo model: HyMeKo v2, **10 bodies**;
- semantic policy graph: legacy-compatible **6 vertices** (`sem:469094de…`), actor input **48**;
- control: `galambos_control` block (joint ±4, ctrl ±4, damping 1.5, kp 40, kv 4) — hard-fail if absent;
- scene: `galambos_env.hymeko` via `EnvSpec.from_hymeko`;
- reward: `galambos_task_deliver_v3.hymeko` (`d42c9cbf…`), K=6 shared strict-held-dwell driving reward/success/
  termination/certificate;
- frozen checkpoints load unchanged (step-zero Δ0.0).

## Discounted alignment (γ = 0.99, read from SACConfig + OffPolicyConfig)

Strict K=6 delivery (deterministic reference: reaches centered∧settled∧robot-touched, holds the dwell, fires the v3
terminal — 3/3 seeds terminate at dwell 6) earns discounted +13.80. Every one of 10 non-success classes returns
−18.1…−21.1; no failure dominates strict; every repeatable loop's infinite-horizon upper bound (−22.9…−23.5) stays
below strict → no farming. The strict predicate that fires the reward terminal IS the certificate predicate.

## §10 reproduction (two columns)

Present (4): neutral bridge (`8955e8db`), POINT zero-shot (`7dbbf1a7`, graph fp match), residual SAC (d4a legacy-mode,
flat 48), frozen transport (`8bd73d8c`) — all REPRODUCED_BOTH_RUNTIMES (step-zero Δ0.0). Absent (3): relay_bridge,
`HANDOFF_CORRECTED_V1`, `BC_FULL_ACTION_PHYSICAL_V1` — ARTIFACT_NOT_PRESENT (quarantined full-action lineage; reported,
not invented).

## Honest scope notes

- The strict-delivery reference is a **controlled demonstration** (arm retracted, coin placed at the zone at rest,
  robot-attribution latched by the acquisition grasp) that measures the reward a delivery EARNS. No scripted or frozen
  controller reaches the strict state in the transport env under its own dynamics — the documented contact-mechanics
  wall — and §9 forbids reading that as reward misalignment. The alignment result is about the reward FUNCTION, not
  policy competence.
- The 4 pre-existing test failures (`test_topology_zoo[*]`, two `test_planar_grasp_env` cases, one
  `test_coin_delivery_rl` case) are unrelated to this work (verified identical on the pre-graph baseline).

## Remaining (gated, per directive §11 — no training yet)

`FULL_ACTION_BC_COMPETENCE_PASS` → frozen-actor critic calibration → one SAC smoke → one TD3 smoke → 5-seed campaign.
These require the absent full-action artifacts to be rebuilt (a training phase the user has explicitly gated).
