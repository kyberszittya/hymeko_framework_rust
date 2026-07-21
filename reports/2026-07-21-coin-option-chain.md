---
campaign: COIN APPROACH→CAPTURE→frozen-TRANSPORT option chain
title: The option chain localizes the wall to CAPTURE — APPROACH is solved, converting contact→TRANSPORT_READY is the bottleneck
date: 2026-07-21
branch: exp/coin-approach-capture-relay
source_commit: 6292431
classification: APPROACH_POSITIVE (first-contact solved; CAPTURE→TRANSPORT_READY remains the limiting boundary; no ≥+0.030 certified)
---

# Approach–capture–transport option chain

**Created-at:** 2026-07-21 11:05 JST. Continues the dwell-relay result (`6292431`) on branch
`exp/coin-approach-capture-relay`. The dwell iteration proved the relay is causally alive under the **sticky** handoff
contract (frozen transport runs to completion; fall back only on genuine physical failure, never on loss of readiness),
but a monolithic bridge could not reach the ready basin from clear-start. This iteration splits the learned work into
two options to **localize** where the wall is.

## Frozen transport (Phase 1)
TRANSPORT_POLICY sha `39551de3`, reverified **10/10 strict** on state `04870b0e` (+0.0253). Parameters unchanged; it is
no longer the experimental variable.

## Design (Phases 2–3)
Two learned options behind explicit **named-field** boundaries, then the frozen transport under the sticky contract:
- **APPROACH** — clear-start → first valid fingertip contact. Boundary APPROACH→CAPTURE: `(left ∨ right) ∧ ¬body`.
- **CAPTURE** — contact → bilateral bracket → one-step TRANSPORT_READY. Boundary CAPTURE→TRANSPORT:
  `detector.is_ready` for one step (irreversible handoff).
- **TRANSPORT** (frozen) — owns the rest; falls back to CAPTURE only on body-shove / sustained stall, **never** on loss
  of readiness.
Reachable-state banks (Phase 4) from the labelled candidates: A1 first-contact 197, C1 bilateral 68, T0 ready 39.

## Per-option training (Phases 5–6, 45k steps each, warm-started from transport)
- APPROACH: terminate on first valid fingertip contact; reward closing fingertip→coin distance, corridor, first
  contact; penalise body-shove / coin drifting from target.
- CAPTURE: start from A1/C1; terminate on first TRANSPORT_READY; reward one-sided→bilateral, symmetry, toward-ready,
  first-ready; penalise all-contact loss, body-shove, left↔right oscillation.

## Full-chain evaluation by clearance band (held STAGE-1+2, §8 per-option failure diagnosis)
| band | n | first-contact | handoff | strict | failure breakdown |
|---|---|---|---|---|---|
| +0.018–0.030 | 13 | 0.92 | **0** | 1/13 | CAPTURE 11, APPROACH 1 |
| +0.030–0.045 | 8 | 0.62 | **0** | 0/8 | CAPTURE 5, APPROACH 3 |
| +0.045–0.060 | 15 | 0.87 | **0** | 0/15 | CAPTURE 13, APPROACH 2 |
| +0.060–0.080 | 1 | 1.00 | **0** | 0/1 | CAPTURE 1 |

**Every band: 0 transport handoffs; the dominant failure is CAPTURE_FAILURE (30 of 34 failures).**

## §9 causal controls (held STAGE-1, identical states)
| arm | strict |
|---|---|
| transport-alone | 4/24 |
| approach-only | 0/24 |
| **full chain (APPROACH+CAPTURE+frozen TRANSPORT)** | 3/24 |
| zero-action | 0/24 |

Max reproducibly certified clearance: **+0.0182** (< +0.030). The full chain's 3 successes have **0 handoffs** — they are
CAPTURE-alone incidental deliveries (the capture policy pushed the coin in), not relay-driven, and 3 < transport-alone 4.

## Classification: **APPROACH_POSITIVE** — the decomposition localized the wall
- **APPROACH is solved.** First-contact coverage is 0.62–1.0 (~85% pooled) from clear-start — the approach sub-problem
  the monolithic bridge conflated with everything else is cleanly handled by a dedicated option.
- **CAPTURE is the definitive bottleneck.** 0 transport handoffs across all bands; CAPTURE_FAILURE is the dominant class
  (contact occurs, but the policy never converts it into a TRANSPORT_READY handoff). Not CAPTURE_POSITIVE (that needs
  handoffs to increase — they stayed at 0); not CHAIN_POSITIVE (no ≥+0.030, chain 3 < transport 4, 0 handoffs).
- **TRANSPORT_FAILURE is absent** (0), consistent with the dwell iteration's 4/4 post-handoff completions: *when* a
  handoff fires, transport delivers. The chain simply never reaches a handoff here.

## Root cause (measured + inferred)
The **empirical TRANSPORT_READY basin is near-goal** (39 states where the frozen greedy transport policy *finishes* — by
construction late-trajectory). To reach it, CAPTURE must perform transport-like motion from a clear-start contact state
— i.e. traverse most of the task. So "convert contact → TRANSPORT_READY" is not a short local skill; it is nearly the
whole delivery, which is exactly the wall five prior interventions hit. APPROACH is genuinely local (fingertip→coin) and
is solved; CAPTURE inherits the near-goal-basin structural difficulty.

## Primary target / demo — not met, not produced (Phases 10, 13)
No held state reaches ≥+0.030 with ≥8/10 chain-strict + a real handoff (0 handoffs, max +0.0182). No video/demo
fabricated. Coin Delivery remains open. No matched replication (Phase 11) — nothing solved beyond the incidental capture
band.

## Next iteration (SPEC only, not run)
The bottleneck is now precisely located: **the ready basin must contain reachable, non-near-goal intermediate states.**
Concretely: (a) build the CAPTURE target as a *reachable* sub-basin (states a short bilateral-bracketing motion away
from contact, verified transport-solvable), not only the transport policy's late-trajectory states; or (b) certify
readiness under **deploy-matched handoff** (the transport policy must finish from the *live* handed-off state, not a
clean restore) so CAPTURE's target is achievable; or (c) give CAPTURE a curriculum of contact→ready micro-transitions
close to the basin. APPROACH is done and reusable.

## Files / tests / provenance
- `hymeko_rl/experiments/coin_option_chain.py` (NEW) — ApproachRewardEnv, CaptureRewardEnv, OptionChainController, banks, train_option.
- `hymeko_rl/experiments/coin_option_chain_run.py` (NEW) — driver: banks → train APPROACH+CAPTURE → chain eval by band → per-option failure diagnosis → causal → §12 classify.
- `hymeko_rl/tests/test_coin_option_chain.py` (NEW, 6 tests). No CORE.YAML items; no deps. Shared trainer unchanged → transfer smokes (P&P `d2da720a`, Beni `4630b537`) NOT rerun, preserved.
- Checkpoints: `approach.pt` sha `94601ea4`, `capture.pt` sha `23e92153`, `chain_result.json` sha `2a54c0e4`. Source `6292431`; host Apple M5 Pro, torch 2.12.0; full run wall ≈ 5 min (90k steps), RSS ~0.45 GB.
