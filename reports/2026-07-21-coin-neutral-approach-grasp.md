---
campaign: COIN neutral approach/grasp — recover the approach policy, compose with transport
title: GRASP_POSITIVE — the recovered E_valselect approach solves neutral→bilateral grasp (7/9); transport integration from that handoff is the remaining gap (neutral delivery 0/9)
date: 2026-07-21
branch: exp/coin-neutral-start-delivery
source_commit: daa8676
classification: GRASP_POSITIVE (approach+grasp learned/recovered; transport-from-neutral-grasp integration incomplete)
---

# Neutral approach/grasp: recover the approach, compose with the proven transport

**Created-at:** 2026-07-21 16:21 JST. The prior NO_EFFECT run diagnosed the missing skill as the neutral→grip
APPROACH. This iteration recovers the real approach policy, gates it, composes with the frozen transport, and
fine-tunes the transport on the shifted handoff. No monolithic sparse-reward neutral SAC.

## §1 PEDC approach RECOVERED — it loads and executes
`sample_precontact_snapshots(env, mlp, …)` builds the bank by rolling a **frozen approach MLP** to pre-contact. That
MLP is `_load_e()` → `experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt` (md5 **b822a660** — the user-frozen
deploy checkpoint), a `DeterministicMLPMultiActor` driven by `node_features` → `action_mean`. It **loads and executes**
on the exact new E0 `NeutralCoinDeliveryEnv`. Evaluated from true canonical neutral (arm [0,0,0,0], no contact, 9
headline states):

| recovered approach (E_valselect) from neutral | rate |
|---|---|
| first-contact | **9/9** |
| bilateral grasp | **7/9** |

**This passes the §6 gate** (first-contact ≥8/9, bilateral ≥6/9). The neutral→grip approach/grasp skill is real and
recovered — no from-scratch approach training was needed. (Explicit neutral→grasp trajectories are regenerable
deterministically with E; they are NOT reinserted as a hidden reset prefix.)

## §7 Composition: E-approach+grasp → frozen TRANSPORT
E-approach until stable bilateral grasp (hold 3), then the frozen transport (`learned_delivery_positive.pt`, 8bd73d8c):

| composed on neutral (9) | grasp | strict delivery |
|---|---|---|
| E-approach + frozen transport | **6/9** | **0/9** |

The grasp forms (6/9) but delivery is 0/9 — the frozen transport sees a **shifted handoff distribution**.

## §8 Fine-tune a transport COPY on the learned E-handoff states (+ rehearse original bank)
Regenerated **56 E-handoff snapshots** (neutral→bilateral-grasp states) + 40% original-bank rehearsal → mixed bank;
fine-tuned a copy of transport (SAC, α-floor, bc-anchored), 24k steps. **Neutral delivery stayed 0/9 at every eval**
(grasp 6/9; contact-prepared retention oscillated 2–9/9).

**Why it fails (diagnosed, seed 1011):** at the E-grasp handoff the coin sits at dtz≈0.139 with both pads in contact,
but the transport **holds the grip and does not move the coin** — dtz stays ~0.14 over 120 steps (coin speed
~0.01–0.06). The E-approach grips the coin from a *neutral-approach geometry* (arm config + grip orientation) that is
out-of-distribution for the transport, and its push does not transport from there. The fine-tune could not adapt: its
BC anchor is the **bank-based** transport demos (there are no successful E-handoff transport demos to imitate), so it
was pulled back toward bank behavior; RL from the sparse delivery reward alone did not bridge the gap.

## §9 Causal comparison (true neutral, 9 states)

| policy | grasp | strict delivery |
|---|---|---|
| zero-action | 0/9 | 0/9 |
| transport-only | 0/9 | 0/9 |
| old monolithic neutral policy | 0/9 | 0/9 |
| **recovered E approach+grasp only** | **7/9** | (no transport) |
| E approach + frozen transport | 6/9 | **0/9** |
| E approach + fine-tuned transport | 6/9 | **0/9** |

## §10 Classification: **GRASP_POSITIVE**
Stable bilateral grasp from the canonical neutral pose is achieved (recovered E approach, 6–7/9, passes the gate), but
transport integration from that handoff remains incomplete — neutral strict delivery is 0/9 with both the frozen and
the fine-tuned transport. Not NEUTRAL_DELIVERY_POSITIVE (0/9). Not APPROACH_POSITIVE (bilateral grasp *is* learned).
Not NO_EFFECT (grasp is a clear positive over the 0/9 baseline).

## What would close it (specific, not a monolith)
The gap is a transport that works from the E-approach grip geometry. The fine-tune failed because it had no E-handoff
transport demos and a bank-based BC anchor. Concrete next step: generate E-handoff transport demos (e.g. a scripted
targetward push from the E-grasp state, validated by the strict predicate) to BC-seed the transport fine-tune, OR
retrain the approach to hand off in the *same* grip geometry the transport already owns (align the handoff, don't just
fine-tune the transport). Either is a targeted step, not a monolithic neutral SAC.

## §11 Demo gate: not met (0/9) — no `coin_delivery_neutral_full_*` video produced.

## §12 Provenance
- Code: `hymeko_rl/experiments/coin_neutral_start.py` (+ `_e_approach_actor`, `collect_e_handoff_bank`, `eval_composed`,
  `finetune_transport_on_handoff`); reuses `exp_v3_handoff_gate._load_e`, `train.sac`, `delivery_certificate`,
  `coin_delivery_e0_campaign`. ruff clean.
- Recovered approach: `E_valselect_v2.pt` (md5 **b822a660**, read-only user-frozen). Transport: `learned_delivery_
  positive.pt` (8bd73d8c). Fine-tuned copy: `experiments/2026_07_21_coin_neutral_ft/transport_ft_best.pt` (2112ccf8),
  `run.json` (469b2c65). No retrain of the approach; transport fine-tuned as a COPY.
- Commits: env/baseline `f6890fa`, `daa8676`; this iteration additive. Branch `exp/coin-neutral-start-delivery`.
  Preserved: transport 8bd73d8c, P&P d2da720a, Beni 4630b537, E_valselect b822a660 (read-only).
- Host Apple M5 Pro; threads pinned; RL not bit-reproducible (BLAS); claims rest on deterministic per-state eval.

## Honest bottom line
The diagnosed missing skill (neutral→bilateral grasp) is **solved by a recovered policy** — a real advance over the
prior 0/9. The remaining gap is narrow and named: the transport does not carry the coin from the E-approach grip
geometry, and adapting it needs handoff-matched demos or an aligned handoff, not more sparse-reward RL.
