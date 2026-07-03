# 2026-07-03 — Collaborative CTDE win, HyMeKo control-substrate toy, standing objective, Galambos env

**Date:** 2026-07-03 (JST) · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu (with Galambos-sensei's task feedback)
**Branch:** `fix-hsikan` · **Base SHA:** `09e8894` (working tree dirty — see *Files touched*)
**Env:** torch 2.12.0+cu132, CUDA 13.2, RTX 3070 Laptop; MuJoCo planar/quadruped; page-file enlarged 2026-07-03 → concurrent torch runs permitted.

## Headline results

| thread | result |
|---|---|
| **GPU `torch.compile` cudagraph fix** | crash fixed; **~5×** restored (185 vs 41 steps/s). `reports/2026-07-03-quadruped-gpu-cudagraph-fix.md` |
| **Collaborative coin-toss (off-policy CTDE)** | **peak-delivery median 0.40 [0.20, 0.40, 0.44] vs joint baseline 0.34** — coordination *beats* joint |
| **HyMeKo declarative-MDP toy** | reach-rate **median 0.96** [0.96, 0.74, 1.0] (P-controller floor 0.55) — the substrate thesis in miniature |
| **Standing objective fix** | reward now metric-aligned (proven by test); pure TD3 still 0.0 → needs BC warm-start (necessary, not sufficient) |
| **Collision + fingertip reward** | 2 new terms, 17 reward tests; armcol-2.0 A/B in-flight (early: crashes drop, delivery suppressed — the historical tension) |
| **Galambos env changes** | fingertip-only coin contact (validated) + two-arm-force `frictionloss` mechanism; old demonstrator now needs re-tuning |

## 1. Collaborative coin-toss under off-policy TD3+BC (the structural win)

The joint single-actor TD3+BC already delivers 0.34 at budget (round-coin, 2026-07-02). Open question: does **explicit
two-arm coordination** help? Built the off-policy CTDE the retired-PPO collaborative path couldn't run:
- `DeterministicMultiChannelActor` — two per-arm backbones coordinated by the existing `MultiTreeChannel` (arms talk
  through the coin/zone couplings) → per-arm deterministic heads → joint action. The coordination lives in the
  *reasoning*, not the output heads (per-arm heads over a shared backbone are algebraically a single head).
- `build_collaborative_offpolicy` — that actor + centralized twin `QCritic`s (LayerNorm). Reuses `MultiTreeChannel`/`QCritic`.
- `train_offpolicy` generalized additively: `action_dim` property + backbone-optional shared-trunk guard (joint path untouched).

**Result:** 3 seeds × 200k, best-checkpoint, same env/reward/budget as the joint baseline → **peak delivery median 0.40,
per-seed [0.20, 0.40, 0.44]**, beating joint 0.34. Caveat: not param-matched (collab has two actor backbones), so this is
"coordination helps," not "free lunch." Late Q-collapse persists (best-checkpoint caught the peaks). Plot:
`reports/figures/collab_coin_delivery.png`; GIF `experiments/2026_07_03_17_15_collab_coin_offpolicy/gifs/best_s2.gif`.

## 2. HyMeKo as a declarative control substrate (the toy)

`data/robotics/toy_reach.hymeko` declares a point-mass reach MDP **end-to-end**: state dims, observed channels,
dynamics params, target, and objective (`reward_spec` reusing `reach_distance`+`action_cost`). `HymekoReachEnv.from_hymeko`
reads it via the existing `read_scene_fields`/`RewardSpec.from_hymeko` regex readers — no new parser. Solved by the *same*
off-policy TD3 as the robots: **reach-rate median 0.96** (untrained ~0.10, P-controller floor 0.55). One `.hymeko` =
state + observation + dynamics + objective, algorithm-agnostic backend. Plot `reports/figures/hymeko_toy_reach.png`.

## 3. Standing objective — fixed, but not sufficient alone

The prior `STAND_REWARD` paid an unconditional `alive` bonus + upright, so a crouched/collapsed-but-not-inverted robot
scored ≈ a standing one (`flip_cos=−0.2` rarely terminates) and height was dominated — the reward did not *require*
standing, so `stand_rate` stayed ~0 while the policy "learned." Fix: the dominant term is now `standing` (+1 iff
`upright>stand_cos AND |z−h|<tol` — the exact DwellMetric), `alive` removed, height up-weighted; `step()` reordered so
the term reads the current pose. Regression test proves standing beats a crouch by >4.0 (old reward: ~0.3 — would fail).
**But** the 100k pure-TD3 smoke on the fixed reward stayed at `stand_rate` 0.0 — the objective is right (necessary) yet
pure TD3 can't *discover* balancing from a free-falling base; the **BC warm-start (PD-hold-q0 demo)** is the next lever.
Plot `reports/figures/standing.png`.

## 4. Collision + fingertip reward (Galambos-motivated, reward side)

Two new terms in `reward.py` (+ galambos_task.hymeko + meta): `arm_body_collision` (−1 on upper-arm crash, **excludes the
fingertip pinch** so it won't fight the grasp — unlike whole-arm `arm_collision` at 2.0 which drove grasp 0.615→0.0) and
`finger_contact` (+1 per fingertip on the coin, graded). Weights set collision-forward (armcol 2.0, fingertouch 1.5) per
user choice. Tests: `test_arm_body_collision_term` (asserts 0 on the pinch), `test_finger_contact_term`, updated regression.
**A/B in-flight** (baseline arm-crash median ~4.5% at delivery 0.40): early seed-0 shows arm-crash dropping to ~1% **but
delivery collapsing 0.14→0.0** — the historical suppression reappearing even with the exclusion, at armcol 2.0.

## 5. Galambos-sensei's env changes (physics side — the better route)

Galambos's feedback on the first GIF: (1) *only the yellow fingertip should contact the cylinder — the arm must not be able
to*; (2) *it should take two robots' force to move the cylinder*.
- **(1) Fingertip-only contact — done, validated.** Collision bitmasks: coin on bit 2 (arm links, MuJoCo-default 1/1,
  cannot touch it), a yellow fingertip geom on conaffinity 3 (can), floor opened to bit 3 (coin still rests). Added to both
  the hand-authored scene and the emitted-arm path (`with_fingertip_sites` now injects the geom). Test
  `test_only_fingertip_can_touch_the_coin` asserts arm↔coin = no, fingertip↔coin = yes.
- **(2) Two-arm force — mechanism done.** `coin_frictionloss` = dry (Coulomb) friction on the coin's slide joints — a real
  force threshold (not just damping's linear drag), opt-in, tested to reach the joint. The threshold value needs tuning
  against the new fingertip push force.
- **Finding:** (1) *breaks the old scripted demonstrator* — it had been herding the coin with the arm *bodies*; with
  fingertip-only contact its strategy is mis-aimed (moves the coin the wrong way). So the task is now genuine fingertip
  manipulation: the demonstrator + trained policies need re-tuning, and (2)'s threshold can't be dialed in until a working
  fingertip controller exists. **The physics route (arm literally can't touch the coin) is the principled solution to the
  knock/collision problem that the reward-penalty route (§4) pays the old suppression price for.**

## Test coverage (every structural change)

| change | test(s) |
|---|---|
| GPU cudagraph fix | `test_cuda_compile_interleaved_graphs_no_overwrite` (verified fails pre-fix) |
| off-policy CTDE actor/builder | 7 in `test_multichannel_ctde` + `action_dim` regression |
| HyMeKo declarative MDP | 5 in `test_hymeko_mdp` |
| standing reward fix | `test_stand_reward_strongly_prefers_standing_over_crouch` (fails on old reward) |
| collision + fingertip reward | `test_arm_body_collision_term`, `test_finger_contact_term`, updated regressions |
| fingertip-only physics | `test_only_fingertip_can_touch_the_coin`, `test_coin_frictionloss_...` |

All targeted suites green: `test_multichannel_ctde` (11), `test_offpolicy_framework` (18), `test_reward` (17),
`test_planar_grasp_env` (28), `test_hymeko_mdp` (5), `test_quadruped_standing` (updated), `test_ddpg` (7).

## Files touched (all non-core; CORE.YAML items touched: none)

- `hymeko_rl/ddpg.py` (cudagraph fix + `action_dim` + shared-trunk guard), `hymeko_rl/multichannel_ctde.py` (new actor +
  builder), `hymeko_rl/env/hymeko_mdp.py` (new), `hymeko_rl/env/reward.py` (standing + 2 collision/contact terms),
  `hymeko_rl/env/quadruped_env.py` (STAND_REWARD + step reorder), `hymeko_rl/env/planar_grasp_env.py` (fingertip geom +
  collision masks + `coin_frictionloss`).
- `data/robotics/toy_reach.hymeko` (new), `data/robotics/galambos_task.hymeko` (collab reward).
- Tests: `test_multichannel_ctde`, `test_offpolicy_framework`, `test_hymeko_mdp` (new), `test_quadruped_standing`,
  `test_reward`, `test_planar_grasp_env`.
- Plan `docs/plans/2026-07-03-collaborative-coin-toss-offpolicy/` (4 artifacts).
- `CLAUDE.md` §6.5 #17 (page-file rule deleted per user — overlapping torch runs now permitted).

## Artifacts

- Experiments: `experiments/2026_07_03_17_15_collab_coin_offpolicy/` (collab, gif+policies+curves),
  `.../19_17_hymeko_reach_toy/`, `.../19_19_quad_stand_fixedreward_smoke/`, `.../19_..._collab_coin_armcol_ab/` (A/B, in-flight).
- Figures: `reports/figures/{collab_coin_delivery,hymeko_toy_reach,standing}.png`.

## Open / next

1. **armcol A/B (in-flight):** fold in the full 3-seed before/after (delivery vs 0.40 while arm-crash drops). Early signal
   says armcol 2.0 suppresses delivery → likely back off to the conservative weight, OR prefer Galambos's physics fix.
2. **Re-tune the Galambos demonstrator for fingertip manipulation** (item 1 broke it), then dial `coin_frictionloss` so one
   arm can't move the coin but two can — the two-arm-force validation.
3. **Standing:** BC warm-start (PD-hold-q0 demo) → TD3+BC on the fixed reward.
4. The collab CTDE win warrants a param-matched follow-up (collab vs a widened joint actor).

## Provenance

Seeds 0–2 throughout; torch 2.12.0+cu132, CUDA 13.2, RTX 3070 Laptop; MuJoCo. Multi-seed medians reported (RL variance).
Working tree dirty (this session's changes, listed above). Peak RSS well under the 16 GB cap (tiny nets).
