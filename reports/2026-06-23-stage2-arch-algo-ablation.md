# Stage 2 — PPO wired into the off-policy comparison harness; arch×algo campaign in flight

2026-06-23 · hymeko_rl · plan `docs/plans/2026-06-23-rl-scenario-ablation-entropy/` (Stage 2)

## Summary
PPO now joins `offpolicy_eval._ALGOS`, so the HSiKAN-vs-MLP architecture ablation runs across **{PPO, DDPG, TD3,
SAC}** on **one** comparison surface (curve-max median/IQR/worst), rather than a parallel on-policy runner
(CLAUDE.md §6.5 #1). The corrected Stage-2 gap was narrow — TD3 and the grid runner already existed — so this is
**adapter wiring only, no new train loop**. The full 40-cell campaign on Galambos (real topology) is launched
and running.

## Files touched (all non-core)
- **Edit** `hymeko_rl/ppo.py` (+62) — off-policy-harness adapter: `PPOEvalConfig` (step→iteration budget map),
  `build_ppo` (ActorCritic via `build_policy`, conforms to the off-policy `build` signature, returns empty
  critics), `train_ppo_eval` (runs `train_ppo`, scores the **same greedy curve** as the off-policy members via
  the shared `eval_fn`, one point per `eval_every`). Imported `build_policy`.
- **Edit** `hymeko_rl/offpolicy_eval.py` (+2) — registered `"ppo"` in `_ALGOS` (CLI `--algo` picks it up
  automatically).
- **New** `hymeko_rl/tests/test_ppo_offpolicy.py` (+44) — registry membership, build (both backbones, empty
  critics), and an end-to-end `compare_offpolicy` smoke yielding a finite curve-max.

## CORE.YAML items touched
None. `hymeko_rl` is non-core; no new dependencies; existing trainers (`train_ppo`, off-policy) unmodified.

## Test results
- `test_ppo_offpolicy.py`: **3/3 pass** (12.8 s, CPU-only). `ruff` + `mypy --strict` clean on `ppo.py`.
- Existing PPO/off-policy paths untouched (additive functions only).

## Performance / sizing (production-scale smoke, §3)
- **Smoke-mode** PPO on Galambos (both backbones) ran clean.
- **Full-budget** PPO on Galambos, 1 seed, both backbones: **418 s ⇒ ~3.5 min/cell**. Params-matched
  (`hsikan@64≈15368 ~ mlp@98≈15296`, SAC-actor basis). HSiKAN −224.6 vs MLP −223.4 — **tied at 1 seed**, as
  expected (architecture parity on these tasks needs the multi-seed grid; standing rule).
- **Campaign sizing:** 4 algos × 2 backbones × 5 seeds = 40 cells. PPO ~3.5 min/cell (measured); SAC/DDPG/TD3
  ~15 min/cell (prior campaign, memory `project-rl-algorithm-roadmap`). Total **≈ 8 h**.

## In-flight experiment (verifiable artifacts)
- **Command:** `python -m hymeko_rl.offpolicy_eval --task galambos --algo ppo sac ddpg td3 --mode full
  --journal reports/2026-06-23-stage2-arch-algo.jsonl --out reports/2026-06-23-stage2-arch-algo.json`
  (CPU-only, `CUDA_VISIBLE_DEVICES=-1`, `torch.set_num_threads(1)`).
- **PID:** 10568 (RSS 638 MB at launch — under the 16 GB cap).
- **Log:** `reports/2026-06-23-stage2-arch-algo.log` (growing). **Journal:**
  `reports/2026-06-23-stage2-arch-algo.jsonl` (per-cell, **resumable** — rerun the same command to resume).
- **Output (on completion):** `reports/2026-06-23-stage2-arch-algo.json` (aggregate curve-max stats).

## Honest notes / caveats
- **Params-match basis:** `match_mlp_hidden` matches the **SAC actor**; PPO's ActorCritic also carries a critic
  backbone, so PPO absolute param counts are ~2× the SAC-actor figure (MLP PPO ≈ 29.5 k). The HSiKAN-vs-MLP
  **ratio** is approximately preserved (both double), so the PPO comparison stays fair; the absolute numbers are
  not cross-algo comparable. To be revisited if a strict PPO-actor match is wanted.
- **Metric consistency:** PPO scores the same **greedy** curve as the off-policy members (`eval_fn`), so curve-max
  is apples-to-apples within the campaign. The eval reuses the training env (the shared-env eval the off-policy
  trainers already do; one stale post-eval transition, negligible).
- **No §6.5 anti-patterns:** the harness was extended (PPO into `_ALGOS`), not forked; no new train loop; TD3 was
  **not** re-added (it lives in `ddpg.py` config — caught earlier via the RL memory).

## Follow-up
1. On campaign completion: `offpolicy_tables.py --journal reports/2026-06-23-stage2-arch-algo.jsonl` → the
   HSiKAN-vs-MLP × {PPO,DDPG,TD3,SAC} table; read the per-(algo,backbone) median gap vs cross-seed IQR (a gap
   inside the IQR is a tie). Update this report with the verdict.
2. Optionally extend to `arm6dof` (second real-topology task) for a morphology axis.
3. Stage 3: `star_entropy.py` (H★) into the exploration seat; discriminating A/B vs policy-entropy &
   critic-disagreement.
