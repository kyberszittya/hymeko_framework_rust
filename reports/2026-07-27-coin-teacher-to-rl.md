# Coin Teacher-to-RL — 6-D torque-θ option space (update-0 gate)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-teacher-to-rl` (from tag `coin-physical-feasibility-closed`,
commit `a3459629`) · **Report head:** `62d09235`

## Verdict (headline)

- **Physical baseline stands.** The frozen robust CEM teacher delivers and frozen-K6 settles the coin from all four
  certified cradles (dev s1,s3; held-out s4,s7) — reproduced 4/4 in this campaign.
- **Update-0: `UPDATE_ZERO_REGRESSES_HELD_OUT`.** A learned actor (BC on the 6-D θ) reproduces the teacher on **both
  development cradles (2/2)** and is **load-bearing** (informed 2/4 vs uninformed box-centre 0/4), but does **not**
  generalize to the held-out cradles (0/2). **`authorises_rl = false` → Stages 5–7 (SAC/TD3) were NOT run** (correctly
  gated).
- **Blocker: `INSUFFICIENT_DATASET_COVERAGE_2_DEV_CRADLES`** — isolated from a framework search defect by an oracle
  control (teacher θ + the same search → 4/4).

The primary question ("can a learned actor reproduce the robust teacher at update-0, then improve via SAC/TD3?") is
answered at the first gate: **not yet** — the reproduction fails on held-out cradles, and the cause is data coverage
(two development cradles), not the pipeline. SAC/TD3 remain unauthorised.

## Physical baseline & 6-D θ semantics

Base tag `coin-physical-feasibility-closed` @ `a3459629`. The HyMeKo-structured slew-admissible option
(`forward_displacement.py`) maps `θ = (squeeze_mag, forward_mag, balance, ramp_steps, release_step, brake_gain)` per
control step to a slew-admissible Δτ (|Δτ|≤τ̇·dt) built from live geometry, in three phases:

- **PUSH** (t ≤ ramp_steps): ramped forward push toward the zone + grip + L/R steer.
- **BRAKE** (ramp_steps < t ≤ release_step): **velocity-feedback** braking, Δτ ∝ brake_gain·‖v_coin‖ opposing the coin
  velocity through the contacts.
- **RELEASE** (t > release_step): relax grip, let the coin settle.

Delivery config (frozen 4/4): `lo=(0,0,-0.10,1,4,0)`, `hi=(0.25,0.30,0.10,28,48,4)`, horizon 60. Frozen K6:
CENTER_TOL 0.02 m, SETTLE_VEL 0.06 m/s, HELD_DWELL 6, touched, motion contract held. Full freeze in
`reports/2026-07-27-coin-teacher-to-rl/option_semantics.json`.

**Canonical action provenance (anti-aliasing):** actor emits proposal centre θ_0 → fixed bounded structured search selects
θ_exec → env executes θ_exec. **Bellman action = θ_0**; θ_exec is provenance only. Enforced structurally
(`ThetaProvenance`, `OptionReplayBuffer`) and tested (`test_replay_buffer_bellman_action_is_center_not_selected`).

## Teacher bank (Stage 1) — provenance

Re-acquired each certified cradle at its frozen seed; reproduced the canonical delivering θ with the frozen deterministic
CEM; recorded snapshot hashes, a per-step physical trace, K6 dwell, and motion-contract metrics. **Gate PASS: 4/4 deliver
+ replay deterministically** (`teacher_bank.json`, wall 196 s, RSS 0.23 GB).

| cradle | split | seed | canonical θ source | dwell | dtz_end |
|--------|-------|------|--------------------|-------|---------|
| s1 | dev | 14250 | `cem_best` | 24 | 18.6 mm |
| s3 | dev | 14750 | `basin_repair` | 18 | 5.2 mm |
| s4 | held-out | 15000 | `cem_best` | 10 | 14.5 mm |
| s7 | held-out | 15750 | `cem_best` | 18 | 18.1 mm |

**Finding (recorded):** s3's CEM-optimum θ is *dynamically fragile* — a ~5e-5 θ perturbation from 4-decimal rounding
swings the coin ~7 mm and flips K6, whereas s1/s4/s7 are robust (≤0.03 mm). s3's canonical was repaired from its own dev
basin (a reproducing delivering θ); the fragile CEM optimum is preserved in the record (`cem_optimum_theta`). Held-out
cradles are never refined (eval-only, no basin augmentation).

## Dataset (Stage 2) — structured causal features + splits

`dataset_contract.json`: 42-D causal feature vector read at the frozen handoff (t=0, before any option action) + an
8×6 θ-independent passive-hold causal history probe. Target = proposal θ_0 (normalised). Frozen splits:

- train/val = development delivering θ (s1: 17, s3: 5 labels; every 4th → val).
- eval = 4-state panel canonical θ (held-out s4,s7 eval-only).

Leakage guards verified: no K6 in inputs, no θ_exec as a label, no post-option measurement, held-out labels eval-only,
split isolation OK, re-acquired snapshot hashes match the teacher bank (`all_hashes_match`). **The dataset has only two
development feature-points** — this is the crux the update-0 result exposes.

## BC proposal (Stage 3)

Three RL-ready flat-obs `DetActor` proposals fit identically (dataset/budget/seed): **B0** features, **B1** + flattened
history, **B2** + LSTM temporal embedding. **B0 ≡ B1 ≡ B2 (identical metrics).** Mechanistic reason: each cradle is a
single decision point with identical within-state features, so the causal-history probe is a deterministic function of
the snapshot and adds no discriminative signal — per the campaign rule, no more complex architecture is introduced.
Selected **B0** (held-back-dev normalised θ-error 0.119, phase err ~2.5 steps, bounded validity 1.0). Offline regression
is not the update-0 gate (`bc_results.json`).

## Update-0 no-regression (Stage 4) — the gate

Deploy: causal features → BC θ_0 → centre-inclusive fixed search (budget 8) → frozen physical option → frozen K6.
Three conditions at the same budget on the frozen panel (`update_zero.json`, `update_zero_panel.png`):

| condition | dev K6 | held-out K6 | total |
|-----------|:-----:|:-----:|:-----:|
| uninformed (box-centre θ_0) | 0/2 | 0/2 | **0/4** |
| **informed (BC θ_0)** | 2/2 | 0/2 | **2/4** |
| oracle (teacher θ, diagnostic only) | 2/2 | 2/2 | **4/4** |

`‖θ_0 − teacher‖` (normalised): **s1 0.139, s3 0.320 → deliver ; s4 0.546, s7 0.753 → miss.** The held-out θ_0 collapse
to ≈ the centroid of the two dev θ. The held-out failures are clean under-deliveries (coin 45–51 mm short, dwell 0), not
motion-contract breaches (peak qdot 1.1–1.4 < 3, peak coin speed < 1.5). GIFs: `update0_deploy_s1.gif` (delivers),
`update0_deploy_s4.gif` (misses).

**Chain of evidence (diagnosis):**
1. Oracle (teacher θ) + the same search → 4/4 ⇒ the physical option, K6 monitor, and the centre-inclusive search are
   correct; the narrow teacher basins are preserved.
2. Informed 2/4 > uninformed 0/4 ⇒ the BC actor is genuinely load-bearing (not a search artifact).
3. Informed dev 2/2 ⇒ the actor reproduces the teacher on both development cradles.
4. Informed held-out 0/2 with the oracle at 4/4 ⇒ the held-out miss is **not** a search/physics defect; it is the
   actor's θ_0 falling in the wrong basin — a **coverage** limit of two development cradles.

**A framework bug was found and separated (see commit `85a95b66`).** The first update-0 pass had the oracle at 2/4:
`option_rl.FixedBudgetSearch` scores only the *jittered neighbours*, never the centre, so an std=0.15 jitter escaped the
narrow coin basins and a delivering θ_0 was discarded for a worse neighbour. Making the search **centre-inclusive**
(budget = 1 centre + budget-1 jitters, applied uniformly to all conditions) fixed this — the oracle then went 2/4 → 4/4,
cleanly isolating the *remaining* held-out miss as the genuine learning/coverage limit. The two effects are committed
separately so it is traceable which result the bugfix changed.

## RL action & semi-MDP contract (Stages 5–7) — NOT run

The RL contract is *designed* (the `CoinThetaOptionEnv` would give: state = causal features; Bellman action = θ_0;
fixed centre-inclusive search → θ_exec; full PUSH→BRAKE→RELEASE consequence; τ = horizon; one terminal option per episode
⇒ `smdp_target` degenerates to a contextual bandit, terminal ⇒ target = reward). But **update-0 did not pass**, so per
the campaign these stages were **not executed** — no `rl_contract.json`, `sac_td3_smoke.json`, or `sac_td3_multiseed.json`
were produced. Running SAC/TD3 now would be scientifically meaningless: with two development cradles the actor cannot be
expected to generalize a 6-D delivering θ to geometrically-distinct held-out cradles, and the teacher itself does not
generalize — it re-searches each cradle from scratch (≈560 rollouts).

## Files touched (all new, non-core; CORE.YAML items touched: none)

`hymeko_rl/coin_delivery/theta_option/{__init__,semantics,search,teacher_bank,dataset,proposal,deploy}.py` (new package),
`hymeko_rl/experiments/coin_theta_rl_benchmark.py` (one harness, mode flags),
`hymeko_rl/tests/test_coin_theta_option.py` (24 tests). Artifacts under `reports/2026-07-27-coin-teacher-to-rl/`.
No edits to V2 physics, V4 motion contract, τ-rate limit, actuator limits, morphology, K6 monitor, zone geometry, reward
semantics, the frozen split, the successful teacher trajectories, or `CORE.YAML`. No new external dependencies.

## Test results

`pytest -p no:randomly hymeko_rl/tests/test_coin_theta_option.py` → **24 passed** (21 fast in ~1 s + 3 `@slow` live-physics
in ~110 s). Layers: unit (θ normaliser, semantics, anti-aliasing, feature/history determinism, split isolation, BC
determinism/round-trip, centre-inclusion budget-exactness), integration (teacher-bank replay, held-out reproduction,
update-0 artifact controls, oracle live delivery). `option_rl` engine tests unaffected (6 passed). `ruff check` clean on
all new code; `mypy --strict` clean on new files except the inherent residuals of calling the frozen untyped harness
(`_load_frozen`/`_setup`/`acquire_certified_straddle`) and the numpy/torch/mujoco stub idiom shared with the subtree.

## Performance

Host: macOS arm64 (Apple Silicon), Python 3.11.15, torch 2.12.0 (CPU), mujoco 3.10.0, numpy 2.4.6, `torch.set_num_threads(1)`
for deterministic eval. Peak RSS ≤ 0.24 GB throughout (hard cap 16 GB — never approached). Wall: teacher bank 196 s,
dataset build 130 s (acquisition-bound), BC fit ~5 s (after cached dataset), update-0 136 s (acquisition-bound). Seeds:
CEM 20260727 (frozen); certified cradle seeds 14250/14750/15000/15750; BC seed 0; search seeds fixed per state.

## Remaining blocker & exact recommended next action

**Blocker:** `INSUFFICIENT_DATASET_COVERAGE_2_DEV_CRADLES`. Two development feature-points cannot support generalization
of a 6-D delivering θ to geometrically-distinct held-out cradles. Basin candidates describe the *acceptable action-set of
one state* — they do not add state-space coverage. No honest dev-only repair restores held-out 4/4 without violating the
frozen split.

**Next campaign (NOT SAC/TD3 yet):**
1. **UNIQUE DEVELOPMENT CRADLE EXPANSION** — the real learning curve is `N unique development cradles = 2, 4, 8, 16, 32`
   with a **cradle-level** train/validation split (not more basin candidates per state). Acquire and freeze more certified
   straddle cradles as development states.
2. **Multimodal / acceptable-set proposal** — represent each state's delivering basin as an acceptable action-set;
   route strategy modes (the `MultimodalBudgetSearch` the engine already provides).
3. **Frozen update-0 retry** at each N — measure held-out K6 vs N (the coverage curve).
4. **Only then SAC/TD3** — once update-0 reaches 4/4 at some N, run the matched SAC/TD3 comparison as originally scoped.

## Provenance

Git head `62d09235` (working tree clean apart from documented pre-existing untracked artifacts + this campaign's `.gif`
outputs, which are gitignored repo-wide). Artifacts: `option_semantics.json`, `teacher_bank.json`, `dataset_contract.json`,
`theta_dataset.npz`, `bc_results.json`, `bc_B0.pt`, `update_zero.json`, `update_zero_panel.png`, `update0_deploy_{s1,s4}.gif`,
`k6_delivery_{s1,s3,s4,s7}.gif` (teacher). Plan: `docs/plans/2026-07-27-coin-teacher-to-rl/` (plan.md/tex/pdf/tikz/mmd).
