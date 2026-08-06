# TD3_CRITIC_FINAL_AUDIT_PASS → guarded actor smoke: VALID_BUT_CHAOS_SENSITIVE — with a validated critic the 3/9 init still collapses under a correct update; TD3+BC now justified

**Created-at:** 2026-07-23 01:05 JST · branch recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62`.

## Chain

1. **Both original smoke negatives were implementation defects** (audit): SAC `SAC_HARD_CLIP_LOGPROB_MISMATCH`
   (log-prob of the unclipped sample), TD3 `POST_WARMUP_CRITIC_CALIBRATION_FAILURE` (warmed critic sep −2.5 on held-out).
2. **Critic-first recovery → `TD3_CRITIC_FINAL_AUDIT_PASS`** + `CRITIC_DISTRIBUTION_COVERAGE_WAS_LOAD_BEARING`. The
   original warmup lacked delivering trajectories; replaying the certified deliveries flips the held-out separation
   −2.5 → +14. Ordered ablations: A demo-only ✗, B +frozen ✓ (+18), C +crude zero/away ✗ (−13, too-extreme), D
   +perturb ✓ (+12). Authorized at two consecutive checkpoints; untouched final audit sep +13.93.
3. **Guarded actor-entry smoke (frozen 3/9 actor + the AUDITED critic):** the critic stays valid throughout (sep
   +11.5…+12.3, Spearman +0.56, OOD 0.0 — no drift), yet the actor loses all three chaos-marginal deliveries by
   update **5** under a tiny action change (aΔ 0.066). Guard fired: `all_initial_successes_lost`.

## Interpretation (now trustworthy)

With the log-prob bug fixed (SAC quarantined) and the critic independently validated (authorization + untouched final
audit), the collapse is no longer an implementation artifact. The actor follows a **correctly ranked** Q-direction and
the 3/9 init **still** collapses → `VALID_BUT_CHAOS_SENSITIVE_UPDATE`. The init is a genuine chaos-fragile local
optimum (the whole arc's finding, now confirmed under a valid critic).

## Consequence — TD3+BC is now justified (per the §12 preconditions, all three met)

1. critic passes authorization + final audit ✓; 2. early actor updates follow correctly ranked Q directions ✓ (critic
sep +12, DPG follows it); 3. empirical outcomes still collapse because the initialization is excessively sensitive ✓.
The next intervention is a **TD3+BC anchor** (keep the actor near the frozen 3/9 init while it improves *around* it) —
NOT a further critic fix and NOT replay rebalance. Reward/gamma/obs/init/bundle unchanged.

## Provenance

`coin_rl_critic_recovery.py` (+ `critic_recovery_result.json`), `coin_rl_guarded_smoke.py` (+ `guarded_smoke_result.json`).
Mac, actor init `1902454c`. SAC stays quarantined. No BC anchor run yet (awaiting authorization).
