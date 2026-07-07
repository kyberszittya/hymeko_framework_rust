# FANUC pick-place TD3+BC — staged build & validation (2026-07-06)

**Author:** Aiko (agent) · **Host:** kato15 (RTX 6000 Ada, torch 2.11+cu128, `MUJOCO_GL=egl`) ·
**Plan:** `docs/plans/2026-07-06-fanuc-pick-td3bc/plan.md`

## Objective (conservative, per user directive)

Not "solve pick-place with RL." First target: a **correct** TD3+BC entrypoint that reuses the existing
`Campaign`/`td3_bc_config` wiring exactly — wrapper compiles, env builds, demos load, BC floor reproducible,
TD3+BC starts from the BC floor, and **best-checkpoint preserves the BC floor**. Reported separately from the
coin-collab TD3+BC-collapse finding (that was a contact-rich cooperative task; **pick-place is a distinct task,
tested cleanly**).

New files (no core edits, no new deps): `hymeko_rl/experiments/pick_place_td3bc.py` (config + closures over
`Campaign`), `hymeko_rl/tests/test_pick_place_td3bc.py`.

**Oracle note:** `reward_oracle.certify()` models the galambos grasp-and-deliver farming MDP only and cannot
score the procedural pick reward, so the pre-queue certify gate is **N/A** here (pick-place is not a
dense-annuity farming trap). Recorded explicitly in the run summary rather than faked green. Safety instead:
LiftPlaceMetric divergence guard (a blow-up is never a counted success) + best-checkpoint holding the BC floor.

## Staged results

### Stage A — env smoke (PASS)
`fanuc_pick_env()` under `MUJOCO_GL=egl`: reset + 30 steps of the scripted expert.
- obs `(9,10)` == observation_space; action `(7,)`, `n_actions=7`.
- reward finite (30-step sum −11.37, expected early shaping cost); obs finite; no crash.
- info keys present: `approach_contact, both_contact, death, lifted, obj_to_target, on_ground, reached`
  (demos filter on `lifted`/`reached`).

### Stage B — demo + BC smoke, tiny (PASS as path check)
`pick_place_bc --kind hsikan --algo bc --n-demos 6 --n-epochs 20`:
- demo collection OK (2444 samples from 6 successful demos); BC trains with **live per-epoch logging**
  (`[bc] epoch k/20 | loss | ep/s | ETA`), final loss 1.7e-3.
- untrained place 0.0, BC place 0.0 — expected at 6 demos (far below the 18-demo baseline). Confirms the
  pipeline runs; not a meaningful floor.

### D-plumbing test — `test_pick_td3bc_runs_and_preserves_bc_floor` (PASS, 3.7 s)
Tiny end-to-end (`kind=mlp`, 200 steps, 2 demos, 1 epoch): env→demos→BC→TD3+BC→measure→artifacts. Asserts
`results.json`+`run.log` written, `select=="place"`, and **curve[0].stage == "bc_step0"** — the BC warm-start
floor is evaluated and raced by best-checkpoint before any RL update (the anti-collapse anchor). Also a pure
`resolve()` budget test (smoke caps a fully-specified config, never expands it).

### Stage C — full BC baseline, 3 seeds (PASS — reproduces cached)
`pick_place_bc --kind hsikan --algo bc --n-demos 18 --n-epochs 80`, seeds {0,1,2}, saved
`checkpoints/fanuc/bc_hsikan_s{0,1,2}.pt`. Box result: **place 0.375 / 0.625 / 0.75** (seed 0/1/2),
**median 0.625, mean 0.583**; untrained 0.0 on all. Reproduces the cached reference (local,
post-divergence-guard, `checkpoints/pick_place_bc_fixed/hsikan_s0.json`, 2026-06-30: place **0.625**).
So the box + the emitted FANUC arm reproduce BC faithfully. **BC floor = 0.625 (median).** n_samples ≈ 8.4–8.9k.

### Stage D — TD3+BC production (3 seeds done)
`pick_place_td3bc --kind hsikan --seeds 1 --steps 100000` (18 demos, 80 BC epochs, 1e5 off-policy steps,
GPU + compile + n_envs=8, 144 s wall, 253→950 steps/s). Artifacts:
`experiments/2026_07_06_18_01_fanuc_pick_td3bc_hsikan/`.

- **BC floor (step-0 eval):** lift **1.0**, place **0.75**.
- **TD3+BC refined (every eval 12.5k→100k):** lift **0.0**, place **0.0** — **immediate, total collapse**,
  from the first refined eval onward.
- **Best-checkpoint held the floor:** `peak[place]=0.75` = the BC step-0 checkpoint (the saved policy is the
  BC one; RL never beat it, so the artifact does not regress).
- **Critic-loss behavior:** finite throughout (crit 0.5–1.6, no NaN/inf — *not* a numerical blow-up). But Q
  drifts monotonically negative (−8.9 → −57.9) and actor loss climbs (9.2 → 56.2), while the BC-anchor term
  stays negligible (bc-loss ≈ 0.002–0.015; `bc_coef 2.5 × ≈0.005` ≪ |Q|≈56). The fixed BC anchor only binds
  on *demo* states; as the visited distribution drifts off the demos there is no anchor, so the actor chases
  the drifting Q and leaves the grasp manifold. **Value-drift collapse.**

**Preserve-or-destroy verdict (seed 1): TD3+BC DESTROYS the BC policy** (0.75 → 0.0). Best-checkpoint
preserves the *artifact* at the BC floor, but RL refinement provides no gain and, run to `final`, wipes it out.

*Scope note:* this is reported on pick-place's own terms. The mechanism (BC anchor too weak vs a drifting Q)
_resembles_ the coin-collab TD3+BC collapse, but the conclusions are kept separate per directive — pick-place
was tested cleanly and independently reproduces collapse.

**3-seed confirmation** (`--seeds 0 2`, `experiments/2026_07_06_18_06_fanuc_pick_td3bc_hsikan/`):
seed0 BC place 0.25 (lift .625) → final 0.0; seed2 BC place 0.25 (lift .75) → final 0.0. Same monotonic Q-drift
(seed0 −7.8→−78.8, seed2 −8.5→−62.3), critic finite (rising 0.9→2.4 / 0.86→4.6, no NaN). **All 3 seeds collapse
to 0.0 at final.** ~140 s/seed.

## Reporting table (final, 3-seed)

| metric | BC floor (Stage C) | TD3+BC best-ckpt (peak) | TD3+BC final | verdict |
|---|---|---|---|---|
| place | 0.375 / 0.625 / 0.75 — **median 0.625** | 0.25 / 0.75 / 0.25 (= held BC floor) | **0.0 / 0.0 / 0.0** | **COLLAPSE**; best-ckpt preserves the floor |
| lift  | ~0.625–1.0 | held at BC | **0.0 / 0.0 / 0.0** | **COLLAPSE** |

(Stage-D step-0 BC floors differ slightly from Stage C — different eval seed/`n_eval`, BC eval variance — but
the collapse to 0.0 at `final` is identical across all seeds.)

## Verdict & follow-up

- **Preserve or destroy?** TD3+BC **DESTROYS** the BC policy on FANUC pick-place — final place 0.0 on all 3
  seeds. Best-checkpoint **preserves the artifact** at the BC floor (the saved `.pt` is the BC policy, place
  0.25–0.75), so there is no regression in what you deploy, but **RL refinement provides zero gain and, run to
  the end, wipes the policy out.**
- **Mechanism (measured, not inferred):** value-drift. Q trends monotonically negative (to −60…−80), actor
  loss ≈ |Q| climbs to 60–80, and the fixed BC anchor (`bc_coef 2.5 × bc-loss ≈0.005`) is ~4 orders below the
  Q term and binds only on demo states — so off the demo manifold the actor is unconstrained and leaves the
  grasp. Critic stays finite (no NaN blow-up); this is drift, not explosion.
- **Recommendation:** **BC (place ≈0.625 median) is the ceiling for this recipe; do NOT re-run vanilla TD3+BC
  expecting a gain.** If RL improvement is wanted, the discriminating lever is the *anchor/critic*, tested as an
  A/B, not more of the same: (a) `adaptive_bc` (schedule the anchor up on eval regress — already a preset);
  (b) an anchor over the *visited* distribution, not just demos; (c) a much larger / normalized `bc_coef`.
  Deploy path today = the best-checkpoint BC policy.
- **Independence:** reported on pick-place's own 3-seed evidence. The mechanism resembles the coin-collab
  TD3+BC collapse but the conclusions are kept separate (distinct task, tested cleanly).

## Artifacts (§9)

- `experiments/2026_07_06_18_01_fanuc_pick_td3bc_hsikan/` (seed 1) and `…18_06…` (seeds 0,2): each with
  `results.json`, `policies/*.pt` (best-checkpoint = BC floor), `gifs/` (best-seed rollout), `run.log`.
- Stage C checkpoints: `checkpoints/fanuc/bc_hsikan_s{0,1,2}.pt`.

CORE.YAML touched: none. New deps: none. Tests: `test_pick_place_td3bc.py` (2 pass). Kept distinct from
coin-collab conclusions.
