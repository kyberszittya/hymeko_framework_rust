---
campaign: COIN Delivery task-semantics fix — separate delivery from grasp; reproduce E0 oracle; learned campaign
title: DELIVERY_ORACLE_POSITIVE — scripted E0 push/coast delivery reproducibly certified (9/9 states, up to +0.070); learned policy partial & unstable
date: 2026-07-21
branch: exp/coin-wristed-pad-delivery-integration
source_commit: ac391f4
classification: DELIVERY_ORACLE_POSITIVE (scripted E0 delivery reproducibly certified; learned policy partial/unstable, below the >=8/10 bar, pending)
---

# Coin Delivery ≠ Grasp Delivery — E0 oracle reproduction + learned campaign

**Created-at:** 2026-07-21 14:25 JST. Correcting a task-semantics drift: the prior report treated the E0 strict
successes as "phantom" because they lacked force closure. They are not phantoms. **Coin Delivery** — move the coin
from a visibly-outside start into the zone and leave it stably — is validly solved by a **robot-caused push/coast**;
force closure is required only for the separate, harder **Grasp Delivery**. This report makes the two certificates
first-class, reproduces the E0 delivery oracle rigorously, and runs the learned campaign.

## §1–§2 Two named, separated certificates (`hymeko_rl/coin_delivery/delivery_certificate.py`, 8 tests)
- **COIN_DELIVERY_STRICT** — positive initial footprint clearance ∧ zone entry ∧ settle velocity ∧ dwell ≥ 6 ∧
  robot/mechanism attribution (a fingertip touched the coin) ∧ clean mechanism (no body-shove, illegal impulse ≤ tol).
  **No force closure.** This is the existing delivery predicate, now named.
- **COIN_GRASP_DELIVERY_STRICT** — COIN_DELIVERY_STRICT **plus** sustained bilateral contact with per-side force ≥
  min over the dwell (a force-closure interval). Separate, optional, **not** the primary Coin-Delivery criterion.
- Historical E0 results re-labeled: **valid COIN_DELIVERY_STRICT successes**, `grasp=False`. The FORCE_CLOSURE_BLOCKED
  finding applies only to COIN_GRASP_DELIVERY_STRICT (prior report amended in place, data unchanged).

## §4 E0 delivery oracle — reproduced, causally validated (frozen_states.json)
9 clear-start headline states (clearance ≥ +0.030, up to **+0.0698**), each under 10 deterministic restores + 10
zero-action controls:

| seed | clearance | scripted COIN_DELIVERY_STRICT | zero-action | footprints disjoint | grasp |
|---|---|---|---|---|---|
| 1011 | +0.033 | **10/10** | 0/10 | yes | no |
| 1045 | +0.0386 | **10/10** | 0/10 | yes | no |
| 1164 | +0.0367 | **10/10** | 0/10 | yes | no |
| 1174 | **+0.0698** | **10/10** | 0/10 | yes | no |
| 1202 | **+0.0698** | **10/10** | 0/10 | yes | no |
| 1278 | +0.0698 | **10/10** | 0/10 | yes | no |
| 1358 | +0.0367 | **10/10** | 0/10 | yes | no |
| 1447 | +0.033 | **10/10** | 0/10 | yes | no |
| 1568 | +0.0367 | **10/10** | 0/10 | yes | no |

**Zero-action delivers 0/10 on every state** → the delivery is **robot-caused** (the push is causal, not gravity/drift).
Footprints disjoint (positive clearance). The oracle even certifies **above the preferred +0.060** (three states at
+0.0698). This is a solid **DELIVERY_ORACLE_POSITIVE**. Model hash `cdae5951…`; 34 clear-start deliverable states
found in seeds 1000–1700 (delivery reachable in the near band + D1 + sparse D3; D2 empty — E0's push reach is limited).

## §5–§6 Learned campaign — demos → BC-init → SAC (direct-action E0)
Demonstrations recorded from the certified scripted push/coast trajectories (obs + executed action, canonical schema);
BC-init the actor; then canonical `train.sac` (`build_sac`/`train_sac`/`SACConfig.stable`) with the demo anchor. Two
configs, both on the identical 9 headline states:

| policy | headline delivery /9 | note |
|---|---|---|
| scripted oracle | **9/9** | reproduced 10/10 each |
| zero-action | **0/9** | robot-attribution control |
| BC-init | **2/9** | supervised floor |
| SAC (static bc_coef=1.0) | best **6/9**, final **0/9** | wild 0–6/9 oscillation, then collapse |
| SAC (competence-gated) | best **2/9**, final **0/9** | BC-locked at the floor, then collapse |

**Learned delivery is partial and UNSTABLE, and never reaches the deployable ≥8/10 bar.** The static-bc run oscillated
0–6/9 across evals (the "6/9" is one sample of an unstable policy, not robust acquisition) and then collapsed to 0/9
once α annealed; the competence-gated run stayed at the BC floor (2/9, 3283 demo pairs → BC dominates) and also
collapsed. Both collapses coincide with α→0.005 — the known BC-anchor/entropy-drift instability. This is consistent
with the arc's standing result that **local policy-improvement caps at (here, below) the supervised ceiling**.

Note on the trainer: the `[sac] crit=nan` print in the first ~500 warmup steps is a **transient** of the shared
trainer (recovers to finite crit≈0.5–1.4; the committed canonical `coin_two_arm_sac` shows the identical transient and
still trains) — not a persistent divergence. Env obs/reward are finite (0/800 bad under random actions).

## §7 Causal validity (identical states)
zero-action 0/9 ≪ BC-init 2/9 < SAC-best 6/9 (unstable) ≪ scripted 9/9; SAC-final 0/9. The scripted push is causal;
the learned policy acquires *some* transfer but not stably and not to the bar. The final trajectory a deployable
policy would use is therefore the **scripted oracle** (positive, reproducible), with the learned policy **pending** a
stabilized trainer (persistent bc-anchor floor + no α→0 collapse, or early-stop capture).

## §10 Classification: **DELIVERY_ORACLE_POSITIVE**
Scripted E0 push/coast delivery is reproducibly certified (9/9 states, 10/10 each, zero-action 0/10, up to +0.070);
the learned policy is partial/unstable and pending. Not LEARNED_DELIVERY_POSITIVE (no stable ≥8/10). Not
NO_LEARNING_TRANSFER (there is partial transfer, 6/9 peak ≫ zero-action). Force closure not required and not claimed.

## §9 Artifacts
- `reports/figures/2026-07-21-coin-delivery-e0/coin_delivery_oracle_clear_start.gif` — scripted E0 delivering a
  **+0.070** clear start: contact → push → coast → zone entry → dwell → CERTIFIED (not grasping).
- `reports/figures/2026-07-21-coin-delivery-e0/e0_delivery_causal.png` — eval curves (both SAC configs) + causal bars.
- Code: `hymeko_rl/coin_delivery/delivery_certificate.py` (NEW, two certificates, 8 tests);
  `hymeko_rl/experiments/coin_delivery_e0_campaign.py` (NEW, reproduce/demos/BC+SAC/eval); reuses
  `coin_wristed_delivery.make_wristed_delivery_env`, `train.sac.*`, `coin_delivery.provenance.snapshot_hash`.
- Data: `experiments/2026_07_21_coin_e0_learned/` — `frozen_states.json` (`691e06f1`), `run.json` (gated, `1a29f23a`),
  `run_static_bc.json` (`ca64bf8b`), `bc_init.pt`, `sac_actor_best.pt`, `sac_actor_final.pt`.
- Tests: 15 pass (8 certificate + 7 pad-actuation). ruff clean. No CORE.YAML. No deps. Preserved: transport
  `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`.
- Host Apple M5 Pro; threads pinned OMP/MKL/OPENBLAS=1; SAC ~400 steps/s; RSS < 1 GB. RL not bit-reproducible (BLAS);
  claims rest on the deterministic scripted oracle + matched-state eval, per the discipline.

## Honest scope
The delivery oracle is rock-solid and correctly separated from grasp. The learned result is a **first bounded CPU
pass** with an unstable trainer config — per the no-first-pass-verdict rule it is **partial/pending**, not "learning
fails." The concrete next step is a stabilized learned run (persistent bc floor, α floor or early-stop at the peak),
which is the only missing piece for LEARNED_DELIVERY_POSITIVE — not more oracle or embodiment work.
