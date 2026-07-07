---
name: project-rl-algorithm-roadmap
description: RL-algorithm line on HyMeKo cart-pole — PPO baseline + DDPG done; TD3/SAC next; safe-RL queued; survey + architecture plan on disk
metadata: 
  node_type: memory
  type: project
  originSessionId: 3060e292-680f-4645-82c1-156ce78e537c
---

The RL-on-HyMeKo algorithm track (testbed: [[project-cartpole-hsikan-testbed]]; storage:
[[reference-policy-as-hymeko-storage]]). Architecture/workflow plan:
`docs/plans/2026-06-21-rl-algorithm-architecture/` (P1 exploration-vocab → P2 DDPG → P3 TD3 → P4 storage[done]).
Survey reference: `reports/2026-06-21-offpolicy-rl-survey.pdf` (DDPG/TD3/SAC + REDQ/TQC/DroQ/CrossQ/TD7 +
design-axis taxonomy + HyMeKo mapping + safe-RL section).

**Status (2026-06-21):**
- **PPO = saved baseline** (on-policy, `ppo.py`; vec rollout; 192±15 cart-pole). DONE.md.
- **DDPG = DONE** (`hymeko_rl/{ddpg.py,replay.py}`): replay ring + Q-critic + deterministic actor + Polyak
  targets + DPG + Gaussian noise; same swappable backbone (`build_ddpg(kind,...)`). mlp DDPG 27→199 upright,
  **solved in ~8k env steps vs PPO ~2M** (~250× sample-efficiency; PPO wins WALL, being vectorised). CLI
  `python -m hymeko_rl.ddpg --policy {mlp,hsikan,signedkan}`. `eval_balance` now typed to a `GreedyPolicy`
  Protocol so PPO+DDPG share one eval. `reports/2026-06-21-ddpg-offpolicy.md`.
- **TD3 = DONE** (`ddpg.py`, unified as 3 config axes of the DDPG core, §6.5#1 — NOT a separate td3.py):
  `n_critics=2` clipped-double-Q + `policy_delay=2` + `target_noise` smoothing; `OffPolicyConfig`,
  `td3_config()` preset, `run_td3`, `--algo`. **Honest non-result on cart-pole:** TD3 no better than DDPG
  (single-seed final 114 vs DDPG 199, oscillates) — TD3's robustness benefit needs multi-seed/harder task;
  cart-pole can't show it. `reports/2026-06-21-td3.md`.
- **SAC = DONE** (`hymeko_rl/sac.py`: squashed-Gaussian actor + twin soft-Q + auto-α; separate trainer per
  §6.5#8 since stochastic+entropy is structural, reuses replay/QCritic/_backbone/_polyak/eval_balance). Strongest
  learner: solves cart-pole ~4k steps (curve 196·200·200·200·122; **final-snapshot metric is NOISY under late
  dips — use the curve / curve-max, not the single step-20k eval**; all of PPO/DDPG/TD3/SAC hit 200 on cart-pole
  so it can't separate them — need multi-seed + harder task). `reports/2026-06-21-sac-and-entropy-seat.md`.
- **ENTROPY-FEEDBACK SEAT (the research seam, user-raised):** SAC's `α·H(π)` is the seat; 3 interchangeable
  signals wired via the exploration-vocab source×site — (1) policy-entropy [SAC, done], (2) critic-ensemble
  disagreement `Var_k Q` = "TD-k" [`n_critics` config already builds k critics+min; REDQ adds subset+UTD],
  (3) **structural entropy** `hymeko entropy` [the NOVEL bet; ppo.py's flagged seat]. Discriminating test: does
  (3) beat (1) on a real-topology task? NEVER cart-pole.
- **NEXT options:** fairer eval (curve-max/early-stop) + multi-seed PPO/DDPG/TD3/SAC; or TD-k/REDQ; or the
  structural-entropy-feedback experiment; or **safe RL** (`meta_constraint` + Lagrangian, queued).
- **MULTI-SEED OFF-POLICY HARNESS = BUILT (2026-06-22, not yet run).** `hymeko_rl/offpolicy_eval.py`:
  env-agnostic `compare_offpolicy(task, algos, backbones, seeds, budget)` over `TASKS`×`_ALGOS` Strategy
  registries; **curve-max** (median/IQR/worst) is the metric (final-snapshot is noisy — proven again in the
  smoke: sac/mlp curve-max −193 vs final −664). Seam = additive `eval_fn=None` on `train_sac`/`train_offpolicy`
  (default → unchanged `eval_balance`; grasp injects `greedy_return_eval`). **Fixed a real bug:** off-policy
  `_backbone` dropped `hidden` for the MLP → baseline pinned at width 64, un-matchable; now forwarded.
  **Params-match for Galambos SAC: `--hidden 64 --mlp-hidden 96`** (hsikan@64=14728 ≈ mlp@96=14792). Smoke:
  118s/2-cell @2000 steps → full 10-cell (2 backbones×5 seeds) ≈ **2.5 h SAC**. Run command + 3 open decisions
  in `reports/2026-06-22-offpolicy-multiseed-eval.md`; plan `docs/plans/2026-06-22-offpolicy-multiseed-eval/`.
  Also: artifact timestamps now baked on GIFs + result-table dicts (`evaluate.now_stamp`/`_stamp_frames`).
- **CAMPAIGN IN FLIGHT (2026-06-22, ~9h, SAC-only 5-seed × 4 tasks × 2 backbones = 40 cells).** Expanded to 4
  tasks in `TASKS`: cartpole(2-vtx,15k steps)/galambos(6-vtx)/arm6dof(6-DOF anthropomorphic)/quadruped(14-vtx);
  `--task` multi, **per-cell resume journal**, **auto params-match** (`match_mlp_hidden`), table generator
  `offpolicy_tables.py` (md + booktabs, verdict = gap vs IQR). **Verifiable in-flight artifacts:** bg task
  `btc6x5tiv`, log `reports/2026-06-22-offpolicy-campaign.log`, journal `reports/2026-06-22-offpolicy-campaign.jsonl`.
  When done: run `offpolicy_tables --journal …`; TD3 arm deferred (rerun with `--algo td3`, journal keeps SAC).
  Stale `test_strategy_spec.py` FIXED (300/200). Kato handout: `reports/2026-06-22-sac-td3-hsikan-overview.pdf`.
- **QUEUED: safe RL** (user-requested direction) — CMDP + Lagrangian on the off-policy update; the HyMeKo angle
  is a declarative `meta_constraint` cost vocabulary (same shape as `meta_reward`; joint-limit/self-collision/
  out-of-bounds terms already exist as reward penalties). Survey §8.

**NEXT-IMMEDIATE PLAN (2026-06-23, user-directed, "step lightly"):** `docs/plans/2026-06-23-rl-scenario-ablation-entropy/`
(4 artifacts built). 3 gated stages: **(1) HyMeKo as RL scenario description** — `ScenarioSpec.from_hymeko`
(scene+task+reward+HTL spec in one `.hymeko`), generalizes the parity-tested RewardSpec, consolidates the
hand-tuned scene thrash (bin/two-tables/disturbance) into declarations; gate = env parity vs Python `fanuc_pick_env`.
**(2) HSiKAN/MLP × {PPO,DDPG,TD3,SAC} ablation** on arm/Galambos (NEVER cart-pole) — **corrected after consulting
this memory: TD3 + the grid runner ALREADY EXIST** (TD3 = ddpg.py OffPolicyConfig axes, NOT a new td3.py;
grid = `offpolicy_eval.compare_offpolicy` + `offpolicy_tables`); the ONLY gap is **wiring PPO (on-policy) into the
`_ALGOS` registry** so all 4 algos share one curve-max comparison surface — no new train loops. **(3) Star-expansion
STRUCTURAL entropy = the seated novel bet** (signal (3) of the ppo.py entropy-feedback seat, alongside (1) policy-
entropy [SAC] + (2) critic-disagreement [TD-k]); `star_entropy.py` computes H★ from the star-expansion activation,
fed through the EXISTING seat (not a new mechanism); discriminating gate = does (3) beat (1) AND (2) on real
topology + a coverage proxy. Companion to the FSM-structured-RL plan [[project-fsm-structured-rl]] (Stage 1 shares
the TaskMachine). NOTE: nearly-duplicated td3.py/ablation_algos.py in the first plan draft — caught by reading this
memory (§6.1/§6.5#15); the shallow file-grep ("no td3.py") was the trap, TD3 lives inside ddpg.py.

**Standing rules (don't repeat past mistakes):** architecture (hsikan vs mlp) verdicts ONLY on a real-topology
task (6-DOF arm / Galambos), NEVER cart-pole (2-vtx, structure not load-bearing); always run the
params-matched control before crediting structure; truncation stored as non-terminal in off-policy backups
(bootstrap past the time limit). DDPG known-fragile (didn't show here on easy cart-pole) → TD3 is the fix.
