# POST_WARMUP_CRITIC_CALIBRATION_FAILURE — the TD3 smoke's warmed critic valued failures ABOVE successes during the collapse window; the TD3 negative is invalidated

**Created-at:** 2026-07-23 00:45 JST · branch recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62`
· actor init SHA `1902454c` · replay SHA `9031053d`. Instrumented, checkpoint-persisting TD3 replication on kato14.
No corrective change (no TD3+BC / rebalance / longer budget / LR change / new seed).

## Verdict

`POST_WARMUP_CRITIC_CALIBRATION_FAILURE`. The critic the TD3 actor learned against was **miscalibrated** — on the
held-out calibration panel its success-minus-failure Q separation was **negative** through the exact window the actor
collapsed. Per §3 I stop before interpreting TD3 actor learning: **the TD3 "no improvement" result is invalidated**,
just as the SAC collapse was (`SAC_HARD_CLIP_LOGPROB_MISMATCH`). Neither smoke supports any conclusion about whether
reward-driven RL improves the actor.

## The smoking gun — critic calibration trace (kato14, held-out panel)

| checkpoint | Spearman | **success−failure Q sep** | OOD boundary pref | Q-scale |
|---|---|---|---|---|
| **post-warmup** (pre-actor) | +0.41 | **−2.50** | 0.0 | 32 |
| update 100 | +0.38 | **−2.90** | 0.0 | 32 |
| update 1000 | +0.14 | **−0.73** | 0.0 | 28 |
| update 5000 | +0.48 | +15.69 | 0.0 | 50 |
| update 10000 | +0.59 | +23.28 | 0.0 | 71 |

The separation is **negative** post-warmup and through ~update 1000 — the critic ranked *failing* trajectories
**above** *delivering* ones. The actor collapsed (HL 3→0 by update 100) precisely while learning `−mean Q(s,actor(s))`
against this inverted critic, so it moved *toward* the critic's (wrong) high-value region. The critic self-corrects
by update 5000 (+15.7), but far too late — the actor already left the delivering init.

## Why the standalone gate passed but the smoke's warmed critic failed

The earlier `CRITIC_CALIBRATION_PASS` (sep +17.5) fit and evaluated on the **same** distribution (headline+validation
BC/zero/away trajectories). The smoke's warmup fits on the **demo replay** (train_query 6000–6029 BC + failures) for
6000 steps, then is evaluated on the **held-out** headline+validation panel — where it does **not** generalize (sep
−2.5). The directive's warning was exactly right: *"do not assume the earlier standalone calibrated critic and the
smoke's warmed critic are equivalent."* They are not.

## Implementation checks that PASS (the rest of TD3 is sound)

- **§4/§8 target init + evolution:** online actor == target actor at update 0 (SHA-equal, max out diff 0.0); target
  evolves by Polyak (online-target distance 0→0.005, monotone), never resynced to update-0.
- **§6 Bellman:** implemented target matches an independent reference elementwise; no bootstrap after `terminated`;
  target actions clipped to [−4,4]; no raw ±63 reaches a target critic.
- **§6/§8 update-1:** one actor step increases mean Q1 on its batch (correct DPG direction), reproducible.
- **§9 evaluation identity:** every checkpoint's actor hash is distinct (`0a754e→3a9d9c→…→f0ba66`); eval loads the
  checkpoint's actor — no silent update-0 reload.

## Replay sampling (§4, measured not inferred)

Actual sampled composition: **76.2% demo / 23.8% fresh**; update-1's batch was **256 demo / 0 fresh** (no fresh data
yet). So early actor updates are driven entirely by demo transitions. This is a real demo-lean, but it is **secondary**
to the calibration failure — the §3 gate fails first and gates interpretation.

## Consequence

Both RL-entry smokes are now invalidated by implementation defects, not science:
- **SAC** — incoherent log-probability (`SAC_HARD_CLIP_LOGPROB_MISMATCH`).
- **TD3** — actor learned against a miscalibrated warmed critic (this).

The correctable requirement (for a future audited re-run, **not** run now): the actor must be authorized to learn
only against a critic that **independently passes calibration on the held-out panel** — i.e. warm up longer / on a
broader state distribution / gate the actor loop on the calibration panel — before any actor step. This changes the
*warmup/authorization procedure*, not the reward/gamma/obs/init/bundle.

## Provenance

`coin_rl_td3_audit.py` (instrumented harness), `td3_audit_kato14.json`. kato14, replay SHA `9031053d`, actor init
`1902454c`. Original smoke `097810b` preserved and its TD3 negative retracted. kato14 left clean.
