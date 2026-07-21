---
campaign: COIN bridge-relay (frozen transport policy + learned bridge policy + readiness relay)
title: The bridge-relay does not fire its handoff — momentary readiness + terminate-on-entry reward leave the relay ≈ bridge-alone (no-effect, well-diagnosed)
date: 2026-07-21
branch: exp/coin-bridge-relay
source_commit: 80d76cb
verdict: NO_EFFECT — trained relay does not exceed transport-alone; the handoff never fires (0/24 all checkpoints) and progressive training forgot the one marginal band
---

# Bridge-relay — solving the F21 distribution-connectivity problem

**Created-at:** 2026-07-21 04:55 JST. Motivation (not re-run): F21 (`reports/2026-07-21-coin-f11-f21-contact-actor-bank.md`)
showed the TRANSPORT actor was **starved** (1.6% occupancy) — the controller almost never enters a transport-ready
state, so a dedicated transport policy never runs. The bridge-relay attacks exactly that: a **frozen TRANSPORT_POLICY**
owns transport; a learned **BRIDGE_POLICY** owns getting *into* the transport basin; a rule-based **relay** hands off.

## Frozen TRANSPORT_POLICY (Phase 1, verified)
`experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt`, **sha256 `39551de3…`**. On the +0.0253 strong
state (hash `04870b0e0357ecb5`, signed clearance **+0.0253**): **greedy strict 10/10**, zero-action 0/10 — reproduced via
the canonical deterministic rollout. Frozen for the whole campaign.

## Empirical TRANSPORT_READY basin (Phase 2)
Candidate states from successful greedy transport trajectories (stride 2) + STAGE-1 held states → **769 candidates**.
Each labelled by the frozen transport policy: **DEPLOY-MATCHED** (the relay deploys the greedy policy, which certifies
only when greedy — under its own stochastic noise it degrades strict→loose, so stochastic labelling would mislabel
solid states). Result: **39 TRANSPORT_READY**, 244 LOOSE_READY, 94 CONTACT_ONLY, 392 NOT_READY. Detector = nearest-
ready-state kNN over 15 **named** public fields, enter 0.373 / exit 0.746 (self-excluded calibration).

## Reverse curriculum (Phase 3) + progressive training (Phase 6)
Non-ready collected states bucketed by distance-to-basin (near→far) + the frozen clear-start corpora:
B0_ready 39 · B1_near 243 · B2_mid 243 · B3_far 244 · B4_clear_start 192. Bridge warm-started from the transport
policy, trained on the bridge reward (potential toward the basin + contact bonuses + dominant terminal READY bonus;
delivery-v2b/strict/env untouched), budgets 8k/12k/15k/15k/25k.

## Per-band relay eval (held STAGE-1 / STAGE-2, 24 each)
| band | held1 strict | held1 ready-entry | held1 **handoff** | held2 strict |
|---|---|---|---|---|
| B0_ready (8k) | **5/24** | 0.167 | **0.0** | 0/24 |
| B1_near (12k) | 1/24 | 0.167 | 0.0 | 0/24 |
| B2_mid (15k) | 0/24 | 0.167 | 0.0 | 0/24 |
| B3_far (15k) | 0/24 | 0.167 | 0.0 | 0/24 |
| B4_clear_start (25k) | 0/24 | 0.167 | 0.0 | 0/24 |

## §7 causal comparison (best band checkpoint, identical states)
| checkpoint | held1 relay strict | handoff | ready-entry |
|---|---|---|---|
| **transport-alone** | 4/24 | — | — |
| untrained relay (transport clone) | 4/24 | 0/24 | 4/24 |
| **trained relay — B0 (best band)** | **5/24** | **0/24** | 4/24 |
| trained relay — B1 | 1/24 | 0/24 | 4/24 |
| trained relay — final (B4) | 0/24 | 0/24 | 4/24 |
| zero-action | 0/24 | — | — |

held2 (far STAGE-2): every cell **0/24**.

## Verdict: **NO_EFFECT** (well-diagnosed) — two concrete mechanism failures
1. **The handoff never fires (0/24, every checkpoint).** The detector reaches "ready" on ~4/24 held states
   (ready-entry 0.167) but **never for 3 consecutive steps**, so the relay never switches to the frozen transport
   policy. The relay therefore reduces to **bridge-alone**. The bridge reward *terminates on first basin entry*, so the
   bridge is never trained to ENTER-AND-HOLD a ready state — it learns to touch the basin, not to dwell in it. The
   3-step hysteresis and the terminate-on-entry reward are mutually defeating.
2. **Progressive training forgot the one marginal band.** B0 gave held1 5/24 (marginally > transport-alone 4/24), then
   B1→B4 monotonically destroyed it (5→1→0). The §6 best-checkpoint selection recovers the 5/24, but even that +1 is
   **within noise and is bridge-alone acting** (B0 bridge ≈ the transport clone it warm-started from; handoff inert),
   not a relay-driven gain.

So the trained relay does **not** reproducibly exceed transport-alone, and the handoff mechanism — the whole point — is
inert. This is **not** BRIDGE_POSITIVE (no relay-driven certified gain) nor BRIDGE_CONTACT_POSITIVE (ready-entry did not
*improve* over the transport clone — both sit at 4/24). It is a bounded NO_EFFECT, **not an impossibility**.

## Measured vs inferred vs hypothesis
**Measured:** the frozen transport policy certifies the +0.0253 basin 10/10; a 39-state READY basin exists; the bridge
reaches it momentarily (ready-entry 0.167) but never holds it 3 steps; the trained relay ≤ transport-alone; progressive
training forgets. **Inferred:** the handoff hysteresis + terminate-on-entry reward prevent sustained readiness, so the
relay is inert. **Still hypothesis (not closed):** whether a bridge *rewarded to dwell in the basin* (hold readiness N
steps before terminating) with a **1-step handoff** and **retention guards** across bands produces a real relay-driven
gain — untested; this is the evidenced next iteration, not a refutation of the bridge idea.

## §8 far-start demo — gated on BRIDGE_POSITIVE, not produced
No certified clear-start relay result exists, so no far-start demo/video/closure. Honest diagnostic: the best relay
(B0) certifies 5/24 STAGE-1 states as **bridge-alone**, handoff inert; STAGE-2 remains 0/24.

## §11 next iteration (SPEC ONLY, not run)
- Bridge reward: **reward dwelling in the basin** (hold detector-ready K steps) instead of terminating on first entry;
  drop the handoff hysteresis to 1 step (readiness is momentary, so 3 is unreachable).
- **Retention guards** across bands (the §6 rollback the driver logs but the forgetting shows is needed), or train all
  bands mixed rather than sequentially.
- A **denser basin near clear-start** (the 39 READY states are near-goal; the bridge must traverse the whole task to
  reach one — the same wall). The basin needs reachable intermediate ready states.

## Files touched
- `hymeko_rl/experiments/coin_bridge_relay.py` (NEW) — basin, detector, relay, bridge-reward env, curriculum.
- `hymeko_rl/experiments/coin_bridge_relay_run.py` (NEW) — driver: basin→curriculum→progressive→causal→classify.
- `hymeko_rl/tests/test_coin_bridge_relay.py` (NEW, 8 tests).
- Data: `experiments/2026_07_21_coin_bridge_relay/` (freeze manifest, 6 band checkpoints, result JSON).

## CORE.YAML items touched
None. No dependencies added.

## Test results & performance
8 bridge unit tests pass; 27 SAC/bank tests unchanged. Full run wall ≈ 12 min (75k training steps + evals), peak RSS
~0.45 GB (16 GB cap ✓). RL not bit-reproducible (§3) — the verdict rests on the causal-comparison margins, which are ≤ 0.

## Provenance
Git SHA `80d76cb`; frozen transport sha `39551de3`, strong state `04870b0e` (+0.0253); basin 39 READY; host Apple M5
Pro, torch 2.12.0 / mujoco 3.10.0.
