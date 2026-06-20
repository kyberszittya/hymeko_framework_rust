# hymeko_rl Phase 2 — debug checkpoint (PAUSED mid-diagnosis)

**Date:** 2026-06-18 · **Status:** ⏸️ paused. PPO does not yet improve past BC; the
diagnosis is well advanced and several fixes are in the tree but **unverified** (a
verification run was in flight at pause). This note saves the trail so the next session
resumes cleanly. **Do not treat the env/policy changes below as confirmed.**

## UPDATE (later 2026-06-18) — the action interface, resolved
The user flagged the control interface as the likely culprit ("torque or velocity
controlled?"). It was: `arm_world` was a **position** servo, but I had hacked a **delta on
position** action = an incoherent pseudo-velocity. **Resolution:** the env is now
parameterised by `control_mode ∈ {torque, position, velocity}` (`arm_world.make_arm_mjcf`),
**torque is the default/headline** (the standard MuJoCo-RL + real-robot interface, and what
the emitted `<motor>` arms use once B-005's axes are fixed). The torque expert is proper
**inverse dynamics** `τ = M(q)·a_des + bias` (the earlier version dropped the `M` scaling and
saturated). Verified experts: **torque 0.069 (13/16), position 0.082 (13/16), velocity
0.134 (9/16)** — torque is now the *best*. This sets up the **control-mode × architecture
ablation** the user asked for. The delta-action muddle (below) is **superseded** — ignore it.
**Still open:** the PPO truncation-bootstrap bug (hypothesis 1) — fix that next, then run
the ablation; then repair the test suite for the new env API (`control_mode`, torque default,
8-dim obs).

## UPDATE 2 — the truncation bootstrap, FIXED (PPO now learns)
`_collect` collapsed `terminated` and `truncated` into one `done`, zeroing the value
bootstrap on every **time-limit truncation** (most episodes) → corrupted value targets →
PPO degraded good policies. **Fixed:** on truncation (not termination) fold `γ·V(next_obs)`
into that step's reward. Result (HSiKAN, torque, BC→PPO, warm-started): PPO now **improves
monotonically with iterations** — reach 0.363 (BC) → 0.363 (50 it) → 0.347 (80) → **0.344
(150)** — instead of degrading (was 0.31 → 0.38 pre-fix). **The core Phase-2 PPO bug is
resolved.** Caveats: improvement is *slow*, and **torque control is hard** (BC reaches only
0.36 vs 0.28 on position; expert 0.069) — torque is the principled interface but a much
harder RL problem on this light arm. `run_ppo` now takes `control_mode`, so the
control-mode × architecture ablation is a clean switch. **Still TODO:** repair the test
suite for the new env API; run the ablation; (optional) more compute / method tweaks to
close the torque reach gap.

## The problem
Phase-2 PPO (`reports/2026-06-18-hymeko-rl-phase2-ppo.md`) was an honest negative: PPO
from scratch barely learns, and BC→PPO **degraded** a good BC policy. The goal of this
debugging pass: find *why* and make PPO improve past the BC reach floor.

## Diagnostic trail (what we learned, in order)
1. **Cold-critic, confirmed + FIXED.** BC trains only the actor; the critic is random, so
   under the dense-negative reward (`−dist`/step) early advantages `≈ −24 − 0` are
   uniformly bad and the value loss corrupts the shared backbone → wrecks the BC actor.
   **Fix shipped:** `ppo.py` `_warmup_critic` (value-only updates on a *frozen* backbone,
   MC returns) + `PPOConfig.value_warmup`. Result: HSiKAN no longer degrades (0.35 →
   0.295, holds BC level). **MLP still degrades (0.25 → 0.37)** — an asymmetry (structural
   policy stable, MLP not), single seed, unconfirmed.
2. **BC reach floor ≈ 0.27 is covariate shift, not a bug.** BC fits expert actions at
   MSE 7e-4 but reaches min ≈ 0.27 (expert ≈ 0.16). Adding the **live EE error to the obs**
   and switching to **delta actions** *each* left BC at ~0.27 — so neither was the BC
   problem; ~0.27 is just the imitation floor (PPO/on-policy is the intended fix).
3. **Action std too high — the live PPO blocker.** Stochastic-rollout reach (0.48) ≫
   deterministic-mean reach (0.31): the exploration noise dominates. With **delta actions**
   bounded to [−0.5, 0.5], `log_std_init = −0.5` (std 0.61) is *larger than the action
   range* → PPO samples garbage and can't exploit the BC mean. Lowered to `−1.6` (std 0.2).

## Changes in the tree (UNVERIFIED — verify or revert on resume)
- `hymeko_rl/env/arm_reach_env.py`:
  - `_NODE_FEAT` 5 → **8**: `node_features` now appends the live EE error `target − ee`.
  - **Delta actions**: `delta_max = 0.5`; `action_space = Box(−0.5, 0.5)`; `step` applies
    `ctrl = clip(qpos + delta, limits)`; `expert_action` returns `clip(Δq, ±delta_max)`
    (was the absolute target `q + Δq`).
- `hymeko_rl/policy.py`: `ActorCritic` default `log_std_init` −0.5 → **−1.6** (std 0.2).
- `hymeko_rl/ppo.py`: `value_warmup` + `_warmup_critic` + `_mc_returns` (the verified fix).
- `hymeko_rl/tests/test_ppo.py`: `test_value_warmup_runs_*` added.

**⚠️ Tests are likely BROKEN** by the obs (8-dim) + action-space changes — the suite has
**not** been re-run since. `test_reach_bc.py` (action bounds / obs assumptions) and the
BC numbers need updating once the design is confirmed.

## Verification result (run `bqzzuh729`, completed at pause) — STILL NOT FIXED
HSiKAN, delta + low-std + critic-warmup, 60 iters: **PPO reach 0.379 vs BC pretrain
0.307 → NOT improved** (still degrades). Worse: with **delta actions the BC floor
collapsed** — untrained 0.304 vs BC 0.307, i.e. BC barely beats doing-nothing. So the
delta-action change *hurt* and should be **reverted**; low-std + critic-warmup were
necessary-but-insufficient. PPO degrading a good policy persists → there is a deeper bug.

**Stale, disregard:** runs `b2275g0x9` / the earlier ablations ran on the *old*
absolute-action, 5-dim-obs env; their numbers no longer apply.

## Strongest remaining hypotheses (untested — the real next leads)
1. **Time-limit truncation treated as a true terminal (likely THE bug).** `_collect` sets
   `done = terminated or truncated`, and `_gae` zeroes the bootstrap on `done`. But
   reaching is rare, so **most episodes *truncate* at `max_steps`** — and for a time-limit
   cut the value must **bootstrap `V(s_T)`**, not assume 0 future. Treating truncation as
   terminal systematically under-estimates returns for the last steps of nearly every
   episode → corrupted value targets → bad advantages → PPO wrecks a good policy. **Fix:**
   track `terminated` vs `truncated` separately; on truncation add `gamma · V(next_obs)`
   to the last reward (or don't zero the bootstrap). This is a classic PPO correctness bug
   and fits the symptom exactly.
2. **Shared actor-critic backbone** lets the (large, dense-negative) value-loss gradient
   corrupt the actor's features — the diagnosis behind the critic-warmup, only partially
   mitigated. **Fix:** separate actor/critic networks, or stop-gradient the value head into
   the backbone, or drop `vf_coef`.
3. **Reward** — try a sparse success bonus + small `−dist`, and return/advantage
   normalisation, to de-risk the dense-negative scale.

## Next steps (in order — revised after the verification result)
1. **Fix the truncation bootstrap (hypothesis 1).** Separate `terminated`/`truncated` in
   `ArmReachEnv.step` plumbing and `_collect`; bootstrap `V(next_obs)` on truncation. This
   is the single most likely cause and a small, surgical change. Re-test BC→PPO: PPO should
   then *preserve/improve* the BC reach.
2. **Revert the delta-action change** (it collapsed the BC floor); go back to absolute
   joint-target actions + the 5-dim obs (or keep the EE-error obs but absolute actions —
   test both). **Keep** the critic-warmup and the lower `log_std_init`.
3. If (1)+(2) still degrade: **separate the actor/critic networks** (hypothesis 2) so the
   value loss can't corrupt the actor backbone; then reward shaping/normalisation (hyp 3).
4. **Repair the test suite** for whatever final obs/action config is kept; re-baseline the
   BC numbers (`reports/2026-06-18-hymeko-rl-phase1-reaching-bc.md` is now stale).
5. **Re-run the matched-capacity ablation multi-seed** — only meaningful once PPO reliably
   beats BC. Check the HSiKAN-stable / MLP-degrades asymmetry across seeds.
6. **Then** the algebraic-entropy-feedback test (`structural_inductive_entropy_note.tex` §5).

## Verified vs not (for honesty on resume)
- **Verified:** the critic cold-start diagnosis + the critic warm-up fix (HSiKAN stops
  degrading). The action-std-too-high observation (stochastic ≫ deterministic reach).
- **Not verified:** that delta-actions + EE-error-obs + low-std together make PPO *improve*
  past BC (the `bqzzuh729` run answers this); the MLP-degrades asymmetry (single seed).
