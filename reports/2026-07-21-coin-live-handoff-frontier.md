---
campaign: COIN live-handoff frontier + backward transport expansion
title: No earlier deploy-solvable handoff frontier exists — the frozen transport basin is narrowly near-goal and cannot be extended backward
date: 2026-07-21
branch: exp/coin-live-handoff-frontier
source_commit: 2ebe726
classification: NO_EARLY_FRONTIER (confirmed by the live oracle AND the failed bounded backward expansion)
---

# Live-handoff frontier — is there an earlier deploy-solvable handoff?

**Created-at:** 2026-07-21 11:25 JST. The option-chain result (`2ebe726`) localized the wall to CAPTURE: the empirical
TRANSPORT_READY bank was dominated by LATE near-goal states, so reaching it demanded transport-like motion. This
iteration asks the deploy-matched question directly and answers it with two independent measurements.

## Live-state handoff oracle (Phase 2)
During a real APPROACH→CAPTURE rollout, at a swept post-contact timestep, control is handed to the frozen TRANSPORT
policy **in-rollout** (no reset — the live MuJoCo + wrapper state, contact history and phase intact) under the sticky
ownership contract; frozen transport runs to termination and strict delivery is recorded. The earliest solvable step
is the earliest deploy-solvable handoff frontier. (For a deterministic greedy transport policy the "8/10" is
degenerate → deploy-matched greedy pass/fail.)

## Measurement 1 — the frontier is NOT earlier (Phase 3–4)
| bank | n | handoff dtz (coin→target) |
|---|---|---|
| OLD near-goal ready bank | 39 | median 0.074, range **0.046 – 0.085** |
| NEW live-solvable handoffs (30 clear-start trajs) | 6 solvable | median 0.056, range **0.031 – 0.061** |

The target footprints touch at dtz **0.060**. The live-solvable handoffs sit at dtz 0.031–0.061 — **at or inside the
target footprint**, i.e. *not earlier* than the old bank; they are near-goal states CAPTURE already pushed the coin
close to. Only 6/30 clear-start trajectories reach any solvable handoff, all near-goal. **There is no meaningfully
earlier deploy-solvable frontier.**

## Measurement 2 — the frozen transport basin cannot be extended backward (Phase 10 contingency)
Per §4, on finding no earlier frontier, I executed the bounded backward expansion: fine-tune a **COPY** of the frozen
policy (original untouched) on the 68 earlier bilateral-contact states (`C1`) using the env's native delivery-v2b
reward, with **35% original-basin rehearsal**; accept only if 04870b0e stays ≥8/10 **and** earlier states become
solvable.

| | state 04870b0e | earlier C1 solvable |
|---|---|---|
| frozen transport | **10/10** | **0/68** |
| expanded copy (30k steps, 35% rehearsal) | **0/10** | 6/68 |

**Both gates fail:** the copy catastrophically **forgot** the original competence (10→0) despite rehearsal, while
gaining only 6/68 on earlier states. `retained=False, gained=False, accept=False`.

## Classification: **NO_EARLY_FRONTIER**
Two independent measurements agree: (1) the live oracle finds solvable handoffs only near-goal (dtz ≤ 0.061, at/inside
the footprint); (2) the frozen policy solves 0/68 earlier bilateral states and cannot be fine-tuned to solve them
without destroying its near-goal competence. **No earlier deploy-solvable handoff states exist under the frozen
transport policy, and its basin cannot be cheaply extended backward.**

## What this means (measured → inferred)
**Measured:** the frozen transport policy has a *narrow near-goal basin of attraction* (dtz ≲ 0.06); it delivers only
when the coin is already at the target footprint. CAPTURE's target ("reach a transport-solvable state") is therefore
**near-goal by necessity** — nearly the whole delivery — which is why CAPTURE was the option-chain bottleneck.
**Inferred:** this is the **contact-mechanics wall** of the whole arc restated at the option boundary — sphere-on-
cylinder point contact makes *pushing the coin from far* unreliable, so no policy (frozen or a backward-expanded copy)
holds a large transport basin. The decomposition did not create a solvable sub-problem because the sub-problem
("push far → near") is the hard one.

## Primary target / demo — not met, not produced
No held state reaches ≥+0.030 with an earlier real handoff + ≥8/10 chain-strict (max solvable dtz is near-goal). No
video/demo/command fabricated. **Coin Delivery remains open**, honestly.

## Where the arc now stands (spec only, not run)
Six local/structural interventions (replay, gate, n-step, semantic critic, two-actor, bridge/dwell/option-chain) and now
the live-frontier probe all bottom out on the **same contact-mechanics wall**: the coin (a cylinder) cannot be reliably
transported far by fingertip push, so the transport-solvable basin is narrowly near-goal. The evidenced next lever is
**not** another controller/curriculum on the current embodiment — it is the **contact geometry** (a fingertip
pad / gripper that achieves a stable grasp rather than point contact), which the arc's ledger already flagged as the
terminal reading (`F-COIN-*`, CLAMP-ORACLE). That is an embodiment change, out of scope for the frozen-policy relay line.

## Files / tests / provenance
- `hymeko_rl/experiments/coin_live_frontier.py` (NEW) — live-state handoff oracle + earliest-frontier mapping.
- `hymeko_rl/experiments/coin_transport_backward.py` (NEW) — §10 bounded backward expansion (fine-tune a copy + rehearsal + accept gate).
- No CORE.YAML items; no deps. Frozen transport `39551de3` untouched (reverified 10/10). Shared trainer unchanged → transfer + APPROACH checkpoints preserved (P&P `d2da720a`, Beni `4630b537`, APPROACH `94601ea4`).
- Data: `experiments/2026_07_21_coin_live_frontier/` (frontier measurement + `backward/backward_result.json` sha `ab19d3fc`). 30 tests pass. Source `2ebe726`; host Apple M5 Pro, torch 2.12.0. Wall: frontier map ≈ 2 min, backward expansion ≈ 2 min; RSS ~0.45 GB.
