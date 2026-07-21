# QUARANTINE — full-action RL result is UNVERIFIED (2026-07-22)

The five-SAC / five-TD3 numerical observation (all RL checkpoints below the BC on strict) is REAL, but the causal
interpretation is NOT established. Per user directive (2026-07-22), the verdict is relabelled:

    FULL_ACTION_RL_REGRESSION  ->  UNVERIFIED_FULL_ACTION_RL_REGRESSION

Quarantined commits (do NOT amend or delete): 0ca6853 (BC), 90e323c (SAC/TD3 runs), 0a46d5e (report + bridge).

A runtime-identity + calibration audit is in progress (diagnostics 1-5). No new five-seed campaign until the
failure mechanism is identified. See `reward_identity_audit.json` and `reports/2026-07-22-full-action-rl-audit.md`.
