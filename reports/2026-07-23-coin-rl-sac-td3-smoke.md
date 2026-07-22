# REWARD_DRIVEN_RL_SMOKE_VALID (TD3) + SAC_ACTOR_COLLAPSE — the RL entry infrastructure works, but neither algorithm improves the fragile 3/9 init in the bounded smoke

**Created-at:** 2026-07-23 00:20 JST · branch recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62`
· obs contract node_features 48 · actor clip-squash init SHA `1902454c`. No final-test access; no squash/clip/reward/
gamma/obs/init change. No multi-seed launched.

## Verdict

**Primary: `REWARD_DRIVEN_RL_SMOKE_VALID`** — the entry infrastructure and reward-driven learning are valid (TD3
demonstrates it) but neither algorithm improves over the init in 10000 updates. **Secondary: `SAC_ACTOR_COLLAPSE`**
— SAC destabilises grasp. Neither smoke is positive → no multi-seed (§10).

## Parity (§1)

Both begin from the identical shared clip-actor tensor (update-0 reproduces headline 3/9, validation 2/30, grasp
9/9). Identical initial replay (demo-certified transitions, seed 0) — replay SHA `9031053d` (same for both). Critic
warmup 6000 (frozen actor) so the actor learns against a calibrated critic (per `CRITIC_CALIBRATION_PASS`). Gamma
0.99, tau 0.005, actor/critic LR 3e-4, batch 256, checkpoints 0/1/10/100/1000/5000/10000. SAC stochastic init set to
the **target entropy** (log_std −2.4; the deterministic action_mean = the 3/9 reproduction is unchanged).

## TD3 (kato14, 10000 updates)

| upd | 0 | 1 | 10 | 100 | 1000 | 5000 | 10000 |
|---|---|---|---|---|---|---|---|
| headline | **3** | 1 | 2 | 0 | 0 | 1 | 0 |
| validation | **2** | 2 | 1 | 1 | 0 | 2 | 0 |
| grasp | 9 | 9 | 9 | 9 | 8 | 9 | 8 |
| actor Δparam | 0 | 2e-4 | 8e-4 | 6e-3 | 0.015 | 0.027 | 0.039 |
| Q-scale | 31.8 | — | — | 31.8 | 31.3 | 47.2 | 64.1 |

**Valid, not positive.** The actor changes measurably (Δparam→0.039), the critic stays finite/bounded (Q 31→64, no
divergence), grasp holds (9/9, brief 8/9). But **delivery never exceeds the init** — best headline 3/9 and best
validation 2/30 are both at update 0; the policy wanders (HL 0–2, VAL 0–2) without a reproducible improvement.
`TD3_REWARD_DRIVEN_SMOKE_POSITIVE` fails (no validation checkpoint above 2/30; no sustained transport-phase gain).

## SAC (kato14, stopped at upd 1000)

| upd | 0 | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|
| headline | **3** | 0 | 0 | 0 | 0 |
| grasp | 9 | 8 | 9 | **2** | **2** |
| actor Δout | 0 | 0.02 | 0.20 | 2.16 | 3.06 |
| alpha | 1.0 | 1.0 | 1.0 | 0.96 | 0.69 |

`SAC_ACTOR_COLLAPSE`. Even at the target-entropy init, SAC's stochastic actor drifts far (Δout 2.2–3.1) and grasp
collapses to 2/9 by update 100, sustained → the two-consecutive grasp<6/9 stop fires. Q stays finite (119→72) and
alpha adapts down (1.0→0.69), so it is an **actor** collapse (the higher-variance stochastic updates leave the fragile
init faster than the deterministic TD3), not a critic divergence.

## Mechanism (measured, consistent with the whole arc)

The 3/9 init is a **chaos-fragile local optimum**: a tiny action change (Δ 0.02 at update 1) already drops all three
chaos-marginal deliveries (HL 3→0/1). RL cannot make *gradual* delivery progress from it — any step falls off the
knife-edge, and neither the deterministic (TD3) nor the stochastic (SAC) update recovers a *better* policy in 10000
updates. Grasp (the robust competence) is preserved by TD3 but destroyed by SAC's larger steps. This is the same
fragility the covariate-shift and matched-BC pilots found, now confirmed under reward-driven optimization.

## Correctable, narrow blocker (§10-B) — NOT a change to the scientific question

- **Demo-replay dominance**: the initial replay is demo-heavy (~11k demo transitions vs 10k fresh over the run), so
  the critic is fit mostly on the BC's own action distribution — a `DEMO_REPLAY_DOMINATES_FRESH_DATA` pressure that
  biases the actor gradient toward the (fragile) demo actions.
- **No init anchor**: nothing keeps the actor near the 3/9 init while it explores — a BC-anchored objective (TD3+BC
  behaviour-cloning regularizer) would let it improve *around* the init instead of wandering off it.
- **Short budget** (10000) + **fragile optimum**: gradual improvement of a chaos-marginal metric needs either more
  updates or a smoother objective.

None of these change the reward, gamma, obs, init, or bundle.

## Decision (§10)

Neither smoke positive → **no multi-seed launched**. I stop for a decision. The evidence-backed correctable options
(one to pick, all keeping the scientific question): (1) TD3+BC anchor to the frozen init; (2) rebalance replay toward
fresh on-policy data / down-weight demo; (3) extend the budget. SAC needs the actor-collapse addressed (smaller actor
LR / stronger entropy control / BC anchor) before it is informative.

## Provenance

`coin_rl_smoke.py` (harness), `td3_smoke.json` / `sac_smoke.json` (+ logs). Both on kato14 (kato15's deployed source
was stale — missing `coin_neutral_start`'s dep chain — so both ran on kato14: isolated processes, identical CPU-torch
runtime, functionally equivalent to the host split). Replay SHA `9031053d`. kato14/kato15 left clean.
