---
title: Compiled + rate-decoupled SAC update — the profile-corrected way to make the structural actor feasible
date: 2026-07-17
slug: sac-compiled-update
status: implemented + GPU-verified on kato61 (RTX 4090, sm_89): 5.79× update / 3.2× end-to-end, trains finite
core_yaml_touched: none
branch: integration/fanuc-pick-place-canonical
---

# Compiled + rate-decoupled SAC update

## Why (the profile that flipped the plan)

Goal: make the **structural (hsikan) SAC actor** affordable enough to run the Aibo/humanoid walking grid at
800k steps (it was ~40 steps/s on kato15 → ~5.5 h/cell). The working assumption — carried from the memory's
HSiKAN arc and my own framing — was that the structural actor is **B=1 rollout-launch-bound**, so the fix is a
vectorized rollout (`n_envs`). **A per-step profile refuted that** (`scratchpad/profile_sac_split2`, CPU, Aibo,
hidden 256, B=256):

| actor | physics | rollout (B=1) | update (B=256) | update share |
|---|---|---|---|---|
| flat | 0.09 ms | 0.05 ms | 5.76 ms | 97.7% |
| **structural** | 0.09 ms | **0.39 ms** | **389 ms** | **99.9%** |

Off-policy SAC does a **B=256 gradient update every env step**, and for the hsikan net that update is **99.9%**
of the per-step cost; the B=1 rollout forward is **0.1%**. So vectorizing the rollout optimizes the non-bottleneck
(§6.5 #18) — it was *not* done. The lever is the **update**, exactly as `train_offpolicy` already solved it:
`torch.compile` the update hot path into CUDA graphs (measured **8.25×** there on the analogous TD3 update; a tiny
net is per-op-dispatch-bound on GPU, the graph amortizes the dispatch).

## What changed

Two config knobs on `train_sac`, both back-compat-default (`compile=False`, `update_every=1` ⇒ the current loop,
byte-identical):

- **`compile: bool`** — extract `_critic_loss` / `_actor_terms` closures + a shared `_update_once(step)`, and
  `torch.compile(..., mode="reduce-overhead")` the two closures when `compile and dev.type=="cuda"`, with
  `cudagraph_mark_step_begin()` between the critic and actor graph replays (mirrors `train_offpolicy` exactly).
  Pure speedup: **same update math, same update-to-data ratio ⇒ the learned policy is unchanged**. The α update
  stays eager (uses `logp` returned from the actor closure); BC/DAgger anchors stay eager and **disable compile**
  (compile covers the pure-SAC-from-scratch path — the campaign; anchor experiments aren't the perf-critical grid).
  CUDA-only; on CPU it is a no-op (verified byte-identical to eager).
- **`update_every: int`** — one update per `N` env steps via UTD-preserving `pending` accounting (`train_offpolicy`
  pattern). `N=1` = current behaviour. `N>1` is **fewer gradient steps** — a *sample-efficiency* change (§6.5 #19),
  so it stays default-1 and is only used behind an explicit A/B, never a silent default.

Plus the CIP anti-bounce reward A/B that motivated the campaign (the 2026-07-17 teacher CIP discovery: Aibo trot
3.7× bounce-dominated, humanoid CpG 3.0×): reward-weight **factories** `cip_goal_reward(bounce)` /
`legged_cip_reward(bounce)` (back-compat constants preserved), a `bounce` param on `CipAiboEnv`, and a
`bounce` axis + `--bounce-ab` mode on the campaign (config-driven; the default 30-cell grid still reproduces).

## Files touched

| file | change |
|---|---|
| `hymeko_rl/train/sac.py` | +2 `SACConfig` fields; `train_sac` update extracted to compiled closures + `_update_once` + `pending`/`update_every` accounting (~+55 / −45 LOC) |
| `hymeko_rl/experiments/exp_aibo_cip_walk.py` | `cip_goal_reward(bounce)` factory; `CipAiboEnv(bounce=...)` (~+18) |
| `hymeko_rl/experiments/exp_cip_verification_campaign.py` | `legged_cip_reward(bounce)` factory (~+14) |
| `hymeko_rl/experiments/exp_sac_walk_campaign.py` | bounce axis; `compile`/`update_every` threaded; `--bounce-ab`/`--with-structural`/`--no-compile`/`--update-every` CLI (~+35) |
| `scripts/kato15/run_sac_walk.sh` | `smoke\|gpu-smoke\|bounce-ab\|full` modes; sync manifest incl. `train/sac.py` |
| `hymeko_rl/tests/test_sac_compiled_update.py` | **new**, 4 tests |
| `hymeko_rl/tests/test_aibo_walk.py` | +2 tests (bounce factory + campaign scenario threading) |
| `docs/plans/2026-07-17-sac-compiled-update/` | plan.{tex,pdf,tikz,mmd} |

## CORE.YAML items touched

**None.** `train/sac.py` is not listed in `CORE.YAML` (verified). No dependency added (`torch.compile` is stdlib torch).

## Test results

- `test_sac_compiled_update.py` — **4 passed** (9.3 s): refactor determinism (bit-identical params, same seed);
  `compile=True`-on-CPU == `compile=False` (guard parity); `update_every=2` does ½ the updates (via a `_polyak`
  counter); `update_every=0` raises.
- `test_aibo_walk.py` — **7 passed** (incl. the 2 new bounce-factory/scenario-threading tests).
- Regression sweep (every test importing `train_sac`/`build_sac`/`SACConfig`) — **110 passed** (65 s). The refactor
  changed no downstream behaviour.
- ruff — clean on all touched files.
- **CPU campaign smoke** (Aibo flat, bounce {3,8}, 20k, compile-default): both cells trained + CIP-diagnosed +
  aggregated (`flat@bounce3` / `flat@bounce8` separated); live `[sac]` logging healthy, loss finite. dx values are
  negative/noisy — 20k is a plumbing check, not a walking measurement.

## Performance results

- **Profile (the finding):** structural per-step = 99.9% update / 0.1% rollout (table above). Vectorizing the
  rollout would buy ~0; compiling the update is the lever.
- **Measured GPU win (kato61, RTX 4090, sm_89 = same Ada arch as the RTX 6000 Ada grid, free box):**
  - Isolated structural update: eager **9.42 ms** → compiled **1.63 ms** = **5.79×** (both finite). Close to
    `train_offpolicy`'s 8.25× on its TD3 update.
  - Real `train_sac` (hsikan, hidden 256, cartpole, 8k steps, campaign-style no-op eval): eager **81.9 s** →
    compiled **25.5 s** = **3.2× end-to-end** *including* the one-time compile cold-start; both `params_finite=True`
    (no divergence, no CUDA-graph crash — the reparam-RNG-in-graph concern is retired). At 800k the cold-start
    amortizes and end-to-end approaches the ~5.8× update figure. Projected: structural ~40 → ~230 steps/s on the
    RTX 6000 Ada → 800k ≈ 58 min/cell (grid feasible).
- **Bug found + fixed by the GPU smoke (CPU could not catch it — compile is a CPU no-op):** the first two kato61
  runs raised *"accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run."* Two causes,
  both fixed to match `train_offpolicy`: (1) I `mark_step_begin()` **between** the critic and actor graphs — a mark
  mid-update roots a fresh step and corrupts the shared critic's buffers; the mark must be **once per update**
  before both. (2) I read `c_loss.item()` at the **end** of `_update_once`, after the actor graph replayed over its
  pool — the loss must be consumed to a host float **immediately after its own backward**. This is precisely the
  §3 "production-scale smoke on real hardware" rule earning its keep.
- Back-compat path (`compile=False, update_every=1`) is byte-identical (verified); flat path unchanged.
- Peak RSS: single-env SAC ≪ 16 GB cap (§4).

## Open issues / follow-up

1. ~~GPU speedup + correctness verification~~ **DONE** on kato61 (RTX 4090, sm_89): 5.79× update, 3.2× end-to-end,
   trains finite. Bug found + fixed (above). kato85's GT 1030 is sm_61, incompatible with the cu128 build; kato61's
   4090 is the same Ada arch as the RTX 6000 Ada grid target, so the number transfers.
2. **The grid remains to launch:** `bash scripts/kato15/run_sac_walk.sh bounce-ab` — `{flat,structural}×{bounce 3,8}
   ×{Aibo,humanoid}×5 seeds×800k`, compiled. ~40 cells; structural ~58 min/cell projected → ~30–35 h on one box,
   so split across free Adas (kato61 now + kato15/kato14 when they free) or drop to 3 seeds. Answers: does
   anti-bounce=8 help the tall bodies walk, and does structural beat flat once the body walks.
3. `update_every>1` sample-efficiency A/B remains unrun (deliberately — it changes the policy; only if wall is still tight).

## Provenance

Branch `integration/fanuc-pick-place-canonical`, working tree dirty (this change). Profile + tests: Python 3.11
`.venv`, torch (CPU), macOS Apple-Silicon, seed 0. GPU target: kato15 RTX 6000 Ada, `.venv_stand` torch 2.11+cu128,
`HYMEKO_DEVICE=cuda`. No experiment data mutated; no long run launched from here.
