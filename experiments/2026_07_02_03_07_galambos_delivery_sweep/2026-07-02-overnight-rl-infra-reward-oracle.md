# Overnight: off-policy RL infrastructure, the reward-oracle, and a delivering galambos policy

**Date:** 2026-07-02 (session ran ~14:00 2026-07-01 → 03:00 2026-07-02)
**Git SHA:** `09e8894` — **working tree DIRTY** (this session's changes uncommitted; see *Files touched*).
**Author:** Aiko (Claude Code), for Dr. Cs. Hajdu.

## Summary

Pushed the off-policy (TD3+BC) RL line on galambos toward an actual **watchable delivery**, built the supporting
infrastructure, and — the standout — a **reward-alignment planner-oracle** that certifies a declared reward before
any RL. Several confident hypotheses were **falsified by measurement** (recorded honestly below); the eval turned
out to be noise-dominated, which reframed the whole effort. First overnight result: **at difficulty 0.0 the arms
deliver the coin at 0.42** (clone 0.10), with a GIF.

## Results (measured / inferred / hypothesis, kept distinct)

- **[measured] Delivering policy (the goal).** galambos difficulty 0.0, seed 0: BC clone **0.10 → TD3+BC refine
  0.42** (SA-HSiKAN actor + MLP-less; critic LayerNorm; vec n_envs=8), delivering-episode GIF rendered
  (`reports/gifs/difficulty_sweep/`). The arms grip and carry the coin into the zone. (Full 3-seed × 3-difficulty
  sweep in flight overnight.)
- **[measured] Off-policy beats PPO, modestly.** 3-seed Kato (difficulty 0.3, 1e5): TD3+BC median **0.125** (IQR
  0.083–0.208) vs PPO's collapse to 0.042. Directionally validates the *ditch-PPO* pivot; far from the 0.33 teacher.
- **[measured, FALSIFYING] The devolution is training instability, not reward-hacking.** A proof-run compared the
  devolved policy vs its clone: devolved episode-reward **−142 vs −45** (LOWER) at equal delivery. So the RL did
  *not* farm a misaligned reward — it destabilized (Q-overestimation). My earlier "the reward is the bug" was
  **wrong**; the proof the user insisted on caught it.
- **[measured, LIMITING] The galambos eval is noise-dominated.** At 24 episodes, p≈0.1, binomial noise is ±0.06 —
  larger than every lever's effect (SA-HSiKAN vs MLP critic, dual-critic, LayerNorm all gave sub-noise deltas). This
  is why nothing at difficulty 0.3 was reproducible. Fix in flight: 100-episode eval (±0.03).
- **[measured] LayerNorm bounds the critic (mechanism), delivery effect within noise.** crit peak ~3 (LN) vs ~16
  (no LN); delivery 0.125 vs 0.083 = one episode/24 = noise.
- **[measured] Reward-oracle works.** The declared galambos reward, planned in ms with zero RL, is **not
  cleanly deliverable** (optimum farms the in-zone annuity by oscillating); min terminal bonus for clean delivery =
  **+1329**. Reproduces the reward's structural flaw without training.

## Infrastructure shipped (all tested, lint+mypy clean)

- **Reward-Alignment Planner-Oracle** (`hymeko_rl/reward_oracle.py`, 5 tests, 4-artifact plan): certifies whether a
  *declared* reward's optimum delivers, at planning speed — the reward is a separable, plannable artifact (the same
  `RewardSpec` scores the live env and the planner). **The Kato-presentable capability.**
- **GPU + `torch.compile(reduce-overhead)`** in `train_offpolicy`: CUDA-graph the update. 8.25× on the update in
  isolation; **~1.5× end-to-end** (Amdahl: env physics + B=1 action-select dominate). Opt-in, nets return CPU,
  checkpoints stay clean. Persistent Inductor cache pinned.
- **Vectorized off-policy rollout** (`n_envs>1`, 4-artifact plan): batched B=N action-select + `add_batch`,
  UTD-invariant, shared `_update_once`. **~1.7× measured** (74→125 steps/s; update becomes the bound). Single-env
  path preserved (14 regression tests pass) + 2 vec tests.
- **Critic LayerNorm** (opt-in), **UTD knob** (`update_every`, 2.5×/4.6×), **live logging** (step/loss/steps-per-s/
  ETA + galambos preamble — killed the blind runs), **heterogeneous dual-critic** (`build_hetero_offpolicy` +
  `critic_update_every`).
- **Defaults flipped to the pivot**: `run_galambos_bc` → `algo="td3"`, `kind="sa_hsikan"`, TD3+BC anti-collapse
  anchor wired (`bc_coef` + `offline_data`).

## Negative / reverted (kept for the record)

- **Dual-critic (MLP-fast + SA-HSiKAN-slow)**: built + tested, but **falsified** on delivery (worse than either
  single critic, and slower — the hetero path is uncompiled).
- **Compiling the B=1 action-selection**: 3.2× in isolation, **net-negative end-to-end** (56 vs 75 steps/s,
  torch.compile shape-guard thrash) → **reverted**.
- **"MLP critic >> SA-HSiKAN critic"**: a one-seed signal (0.208 vs 0.083) that the next seed **contradicted**
  (0.042 vs 0.208) — seed variance, not a critic effect. Retracted.

## CLAUDE.md additions (this session's lessons, codified)

§3 **Evaluation-metric integrity** (measure the ceiling; horizon-match probes; demo-filter ≡ eval-metric; guard
metrics a failure can inflate; single-seed ≠ verdict) and **Live observability — never run blind**. §6.5 anti-patterns
**17** (overlapping runs / orphan procs / Windows page-file), **18** (profile the real bottleneck first), **19**
(default the proven optimization; withhold the unproven model-change).

## Plans on disk (4 artifacts each)

- `docs/plans/2026-07-02-reward-planner-oracle/`
- `docs/plans/2026-07-02-vec-offpolicy/`
- `docs/plans/2026-07-02-clifford-fir-rotary-spikes/` — **Gömb-Soma → RL**: rotary-spike mapping + Clifford-FIR (the
  Gömb membrane) → step 1; full stack adds CPML grade-0 control readout + Soma quadtree spatial-tree. Flagged
  Cl(0,1)↔SO(3) bridge; toy-validate first.

## Files touched (uncommitted)

`hymeko_rl/`: `ddpg.py` (device/compile, vec, LayerNorm, UTD, hetero, live-log, `_update_once` refactor),
`galambos_bc.py` (defaults + knobs + preamble logging), `policy.py` (`deploy_policy`), `reward_oracle.py` (new),
`evaluate.py` (metrics, earlier), `benchmark.py`, `exp_collaborative.py`, `ppo.py` (vec-default + backbone-agnostic
fix), `multichannel_ctde.py`, `tree_channel.py`. Tests: `test_reward_oracle.py` (new), `test_offpolicy_framework.py`
(+UTD/deploy/LayerNorm/hetero/vec), `test_ppo_vec.py`, `test_multichannel_ctde.py`, `test_sa_hsikan_backbone.py`.
CLAUDE.md, MEMORY (2 new). **CORE.YAML items touched: none.**

## Overnight campaign (in flight)

`bcxqzc2v7` — galambos difficulty sweep (0.0/0.15/0.3 × 3 seeds, SA-HSiKAN TD3+BC+LayerNorm+vec, 100-ep eval,
delivering GIF per cell). First cell: **diff 0.0 = 0.42 delivery**. On completion, auto-chains to a clean
pick-and-place campaign (LiftPlaceMetric divergence guard; lift+place GIF). If delivery weak: add the
oracle-certified terminal bonus + retune (autonomy granted).

## Open items

- Full difficulty curve + pick-and-place (overnight).
- High-power (100-ep, multi-seed) LayerNorm answer to settle it above the noise floor.
- Reward fix: implement the oracle-certified terminal bonus declaratively in `galambos_task.hymeko` (note: it
  addresses reward-farmability, **not** the training instability that caused the devolution — separate levers).
- Commit the working tree (large, tested change set).
