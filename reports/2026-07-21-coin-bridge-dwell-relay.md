---
campaign: COIN bridge-dwell-relay — correct the train/handoff contract + causal relay test
title: The relay-plumbing fix makes the relay causally alive (real handoffs + completions, 4→9); the three-step-dwell training added nothing
date: 2026-07-21
branch: exp/coin-bridge-dwell-relay
source_commit: 3688d7b
classification: RELAY_HANDOFF_POSITIVE (from the handoff-plumbing fix; the dwell TRAINING itself was NO_EFFECT; +0.030 not reached)
---

# Bridge-dwell-relay — correcting the train/handoff contract

**Created-at:** 2026-07-21 05:30 JST. Continues the frozen bridge NO_EFFECT (`5019d4b`) on branch
`exp/coin-bridge-dwell-relay` (previous six commits unamended). Frozen inputs: TRANSPORT_POLICY sha `39551de3`,
verified state `04870b0e` (+0.0253, greedy 10/10), transport-alone 4/24, prev best B0 5/24, prev handoffs **0/24**.

## Classification: **RELAY_HANDOFF_POSITIVE** — with a precise, honest split
The iteration corrected **two** defects; one worked, one didn't:
- **Handoff-plumbing fix → the relay is now causally ALIVE** (the real positive).
- **Three-step-dwell TRAINING → NO_EFFECT** (no 3-step handoffs on held states).

## 2. One-step handoff diagnostic (best B0 bridge, detector frozen, no tuning) — the decisive finding
| dwell | readiness entries | handoffs | post-handoff strict | fallbacks | false-positive handoffs |
|---|---|---|---|---|---|
| 3 (previous) | 4/24 | **0** | 0 | 0 | 0 |
| **1 (diagnostic)** | 4/24 | **4** | **0** | 4 | 4 |

**Interpretation (per the §2 decision tree):** the previous zero handoffs WERE the dwell mismatch (dwell=1 → 4
handoffs). But post-handoff strict was **0** and each handoff was immediately followed by a fallback — a **second,
deeper defect**: the relay fell back the instant the coin left the ready region, which the transport policy **must** do
to transport. The fallback punished exactly the transport it was meant to run.

## Relay-plumbing correction (the load-bearing fix)
`RelayController`: once handed off, the frozen transport policy is **trusted to complete** — fall back only on a
**genuine failure** (arm-body shove, or a real targetward stall), never on "not ready" / "bilateral lost". (Also fixed
a latent bug: the stall counter was never incremented.) With the fix, the **same** best B0 bridge:

| relay | held1 strict | handoffs | post-handoff strict |
|---|---|---|---|
| transport-alone | 4/24 | — | — |
| previous eager-fallback, dwell=1 | 5/24 | 4 | **0** |
| **fixed relay, dwell=1** | **9/24** | 4 | **4/4** |

→ **strict 4/24 → 9/24, 4 real handoffs, all 4 completing post-handoff.** The relay mechanism is causally alive; the
readiness states ARE deploy-solvable (so **not** DETECTOR_MISMATCH — I checked the exact branch the diagnostic pointed
at).

## 3–7. Dwell-aware training (the part that did NOT work)
`BridgeRewardEnv` made dwell-aware: episode succeeds only on holding readiness **dwell_target=3** consecutive steps (=
the relay handoff condition, so train==eval), with an increasing streak bonus, a dominant 3-step terminal, and
leave-ready / negative-progress penalties. Progressive curriculum B0→B4 (25k/30k/40k/50k/50k) with **≥30% B0/B1
rehearsal** and **lexicographic checkpoint selection**. Frozen: delivery-v2b reward, strict predicate, transport
policy, detector labels, eval states.

Per-band held eval — **every band: handoff 0, three-step dwell 0.0** on held clear-start states:
| band | held1 strict | handoff | 3-step dwell |
|---|---|---|---|
| B0 (25k) | 0/24 | 0 | 0.0 |
| B1 (30k) | 0/24 | 0 | 0.0 |
| B2 (40k) | 4/24 | 0 | 0.0 |
| B3 (50k) | 0/24 | 0 | 0.0 |
| B4 (50k) | 0/24 | 0 | 0.0 |

The dwell-trained bridge **never achieves a 3-step readiness dwell on a held clear-start state** — it holds readiness on
the B0 *training* states (already ready) but does not generalise the *approach-then-hold* to clear-start held states
(the approach-to-basin is the same wall F21 hit). The dwell objective did not add capability and (B2 aside) did not even
match the bridge-alone 5/24.

## 8–9. Causal controls (identical held states)
| arm | held1 | held2 |
|---|---|---|
| transport-alone | 4 | 0 |
| prev bridge, 3-step relay (bridge-alone) | 5 | 0 |
| **prev bridge, 1-step FIXED relay** | **9** | 0 |
| dwell bridge, 3-step relay | 4 | 0 |
| zero-action | 0 | 0 |

Max reproducibly certified initial clearance (dwell bridge, relay-strict): **+0.0182** (< +0.030).

## Why RELAY_HANDOFF_POSITIVE (and the honest caveats)
Per the §10 gate — *"actual handoffs and post-handoff transport completions improve reproducibly, but the +0.030
certified threshold is not yet reached"* — the corrected relay does exactly that: handoffs 0→4, post-handoff
completions 0→4, held strict 4→9, reproducible (deterministic frozen policies + detector). **But three caveats,
stated plainly, not buried:**
1. The gain is from the **handoff-plumbing fix**, not the dwell training.
2. It is measured at the **diagnostic 1-step dwell**, which the spec calls "not the final robust controller."
3. The **three-step-dwell TRAINING was NO_EFFECT** (0 three-step handoffs on held states), and **no ≥+0.030** clear-start
   is certified (max +0.0182). Not RELAY_POSITIVE; not DETECTOR_MISMATCH (post-handoff transport DOES complete).

## 11. Demonstration — not met, not produced
No held state reaches clearance ≥+0.030 with a dwell-relay ≥8/10 + 3-step handoff (max +0.0182, 0 three-step handoffs).
No video/demo fabricated. Coin Delivery remains open.

## Next iteration (SPEC only)
The relay works; the blocker is the bridge reaching **AND holding** a *deploy-solvable* ready state from clear-start.
(a) The dwell objective fought approach — separate "approach to basin" from "hold in basin" (two sub-skills / phase the
reward). (b) The readiness basin's near-goal states demand a full approach; add reachable *intermediate* dwell targets.
(c) Deploy the fixed relay at a **1-step handoff with a short confirm window** rather than a strict 3-step dwell the
bridge can't produce — the mechanism already completes when it fires.

## Files touched / tests / provenance
- `hymeko_rl/experiments/coin_bridge_relay.py` — RelayController fallback contract fixed; `BridgeRewardEnv` dwell-aware; `RelayLog.max_ready_streak`; coin→target named fields.
- `hymeko_rl/experiments/coin_bridge_dwell.py` (NEW) — dwell driver: rehearsal, lexicographic selection, 5-arm causal, §10 classification.
- `hymeko_rl/tests/test_coin_bridge_relay.py` — dwell-reward test updated. 8 bridge tests + 38 in the regression sweep pass. No CORE.YAML items; no deps.
- **Shared policy/trainer unchanged this iteration** → transfer smokes NOT rerun (§13): P&P `d2da720a`, Beni `4630b537` preserved.
- Data: `experiments/2026_07_21_coin_bridge_dwell/run_s0/` (5 band checkpoints + `dwell_bridge_best.pt` sha `13e7122d` + `dwell_result.json` sha `b5c5f22d`). Source `3688d7b`; host Apple M5 Pro, torch 2.12.0. Full run wall ≈ 8 min (195k steps), RSS ~0.45 GB.
