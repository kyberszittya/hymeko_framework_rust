# Aibo quadruped as a HyMeKo single-source control substrate (P0 + P1)

**Date:** 2026-07-07 (JST) · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**Base SHA:** `4320202` (working tree dirty — see *Files touched*) · torch 2.12.0+cu132, MuJoCo 3.9.0, RTX 3070 Laptop, Win11
**Plan:** `docs/plans/2026-07-07-aibo-quadruped-hymeko-substrate/` (tex/pdf/tikz/mmd, updated to the 22-DOF design)

## Summary

Rebuilt the HyMeKo quadruped as a **faithful Sony Aibo ERS-1000 (exactly 22 actuated DOF)** and wired the
**standing (balance) task to be driven end-to-end from a single `.hymeko`** — the declarative-control-substrate
thesis (`project-hymeko-as-control-substrate`) on a real 22-DOF robot. Three phases landed:

- **P0 — the plant.** `data/robotics/quadruped.hymeko` is now the ERS-1000 DOF layout (legs 3×4, head 3, neck 1,
  waist 1, mouth 1, ears 2, tail 2 = 22), with a rounded dog body (overlapping chest/belly/haunch spheres),
  silver ball-joint shoulders/hips, a natural bent-leg stance, a long dachshund snout, and a cute face
  (eyes/nose/muzzle/ears/paws/tail-tip as fixed cosmetic links). Screenshot: `reports/figures/aibo_quadruped.png`.
- **Held-joint mechanism (correctness, not just tractability).** Full leg-scale torque on the ~10 g expressive
  joints blew the reward to −50 M / QACC NaN. The 10 expressive DOF are now **passively spring-held at neutral**
  (motor stripped + `stiffness`/`damping`/`armature` — armature is the cure for a stiff hold on a light link),
  so **RL controls only the 12 legs** (action space 12) over the faithful 22-DOF plant. Reward is bounded
  (−2..−0.6 under random action) and stable.
- **P1 — the single source of truth.** `data/robotics/quadruped_stand.hymeko` declares the whole MDP: the
  `scene` tuning, the `reward_spec` (the gated `standing` predicate + dense shaping, ≡ the DwellMetric), and the
  `experiment_spec` training strategy (BC warm-start → TD3+BC). `QuadrupedGoalEnv.from_hymeko` builds the env from
  it (reusing `read_scene_fields` + `RewardSpec.from_hymeko`); `TrainingSpec.from_hymeko` reads the strategy from
  the *same* file; the `quadruped_stand` registry entry now builds via `from_hymeko`. Extends the existing reader
  stack — no new parser, no CORE change.

## Measured facts (not inferred)

- Plant: `nu=22` raw emit → **12 leg motors + 10 spring-held expressive**; `njnt=23` (22 hinges + promoted
  freejoint). `hymeko validate` clean (only the benign `base→world` frame warning).
- Rest pose upright (cos ≈ 1) at nominal height 0.25 m; scores standing at rest (reward ≈ 5.9, `standing` term
  dominates → reward ≡ metric).
- **Task is non-trivial:** a null (zero-torque) policy holds standing only **104/250** frames then sinks
  (torso 0.25 → 0.095) — well under the 200-frame dwell, so balancing requires active control. (Adding the
  lateral abduction DOF is what makes it fallable; the old sagittal-only box was artificially self-stable.)
- Round-trip: `from_hymeko` env reward = the declared `reward_spec` = the code `STAND_REWARD`
  `(standing 5, torso_height 3, upright 1, stand_still 0.1, joint_velocity 0.001)` — a test fails if the
  declarative and programmatic defaults ever drift.
- **Encoding-damage fix:** `data/robotics/meta_experiment.hymeko` began with a UTF-8 BOM (the FABLE-era damage
  CLAUDE.md's quarantine flags) — it lex-errored any importer. Stripped (3 bytes); no other BOM in `data/robotics`.

## Files touched (all non-core; CORE.YAML items touched: none)

| file | change |
|---|---|
| `data/robotics/quadruped.hymeko` | rewrite → 22-DOF Aibo (+165/−64) |
| `data/robotics/meta_reward.hymeko` | +4 standing term declarations (+11) |
| `data/robotics/meta_experiment.hymeko` | strip UTF-8 BOM (−3 bytes) |
| `data/robotics/quadruped_stand.hymeko` | **new** — single-source standing scenario |
| `hymeko_rl/env/quadruped_env.py` | held-joint mechanism + `from_hymeko` + `_tune_legs` regex (+111/−?) |
| `hymeko_rl/eval/tasks.py` | `quadruped_stand` factory → `from_hymeko` (+7/−4) |
| `hymeko_rl/tests/test_quadruped_aibo_plant.py` | **new** — 7 structural P0 tests |
| `hymeko_rl/tests/test_quadruped_from_hymeko.py` | **new** — 7 P1 round-trip tests |
| `hymeko_rl/tests/test_quadruped_env.py` | updated goal-task contracts (12 DOF / bounded reward / 60-step) |
| `hymeko_rl/tests/test_scenario_sanity.py` | quadruped spec → 33 vertices / 12 actions |

## Test results

- `test_quadruped_aibo_plant` (7) + `test_quadruped_from_hymeko` (7) + `test_quadruped_standing` (13) +
  `test_quadruped_env` (14) + `test_scenario_sanity` (11 across scenarios) → **52 passed** (10.2 s).
- Shared-file consumers (BOM/vocab): `test_training_spec` + `test_hymeko_mdp` + `test_reward` + `test_bc`
  → **36 passed** (11.1 s). No regressions.
- **Gates:** `ruff check` clean; `mypy --strict hymeko_rl/env/quadruped_env.py` — only the pre-existing
  package-wide `mujoco` missing-stubs (no new suppressions, no new errors).
- Perf (asserted in tests): stand env 200-step median < 2 s, tracked peak < 256 MB (≪ 16 GB cap).

## §6.5 anti-patterns

None introduced. The held-joint mechanism is one method on the shared env (config, not a class-per-variant); the
substrate reuses `read_scene_fields`/`RewardSpec`/`TrainingSpec.from_hymeko` and the `PlanarGraspEnv`/`HymekoReachEnv`
`from_hymeko` pattern rather than a new reader (§6.1 discovery pass done); no globals; no `_v2` file proliferation
(the plant was edited in place). One waiver: `mujoco` untyped-import (pre-existing, package-wide).

## P2 — PD-hold-q₀ demonstrator + DART BC (done, on kato15)

- **PD-hold-q₀ demonstrator** (`QuadrupedGoalEnv.expert_action`): leg torques `τ = −kp·(q−q0) − kd·q̇`, holding the
  legs at the standing stance. Label-sanity: holds standing **1.0** at every gain (kp 40–200) — a clean demo ceiling.
- **Root-caused BC failure, then fixed it.** Naive BC of the demonstrator gave standing **0.0**: the standing pose
  has near-vertical legs bearing weight *axially*, so the holding torques are tiny (mean|a|=0.008) and the clone's
  approximation error (~0.007) is comparable to the signal → it drifts and sinks (pinned by diagnostics, not guessed).
  Fix: **DART noisy demos** — perturb the *applied* action (σ=0.22, init-noise 0.10), label with the *clean* expert
  correction, so the clone learns the recovery feedback law over off-nominal states (mean|a|→0.17). BC floor **0.0 →
  dwell 0.625** (48-demo check) → **0.79** at full budget (200 demos × 200 epochs).

## P3 — the run (kato15 RTX 6000 Ada, 3 seeds × 150k, BC→TD3+BC)

**Reward certificate (pre-launch gate, standing analogue of `reward_oracle.certify`): `delivers=True`** — the
demonstrator scores the reward 5.92 (standing 1.0) vs a collapsing zero-policy 2.81 (0.42), quoted at launch and in
the run summary.

| seed | BC warm-start (step-0) | TD3+BC (25k–150k, all checkpoints) | best-checkpoint |
|---|---|---|---|
| 0 | **0.792** | 0.0 | 0.792 |
| 1 | 0.292 | 0.0 | 0.292 |
| 2 | 0.458 | 0.0 | 0.458 |

- **[measured] Standing dwell median 0.458, best 0.79** — a standing controller learned **end-to-end from the one
  `quadruped_stand.hymeko`** (plant + reward + obs + strategy). GIF (best seed) stands upright the full episode:
  `experiments/2026_07_07_17_32_quadruped_stand/gifs/quadruped_stand_s0.gif`; plot `reports/figures/quadruped_stand_result.png`.
- **[measured] TD3+BC collapses standing to 0.0 on every seed** (0.79/0.46/0.29 → 0.0 by 25k). Best-checkpoint banked
  the BC floor. Q stayed bounded (+7–8.7, huber critic), so this is the actor-BC-anchor-vs-Q divergence, **not** a
  numeric blow-up.
- **[inferred, matches prior record] BC (imitation) is the standing ceiling; off-policy RL subtracts** — exactly the
  FANUC/galambos pattern (`project-fanuc-offpolicy-collapse`, `project-galambos-reward-fixed-rl-below-demo`). The
  lever past a BC ceiling is **imitation (DAgger)**, not TD3+BC. Standing is thus **solved by BC** (was UNSOLVED by
  pure TD3 at any budget — `project-quadruped-standing-td3-diverges`), and the framework-substrate goal is met.
- **Wall:** ~2.2 min/seed RL (**~1135 steps/s** on the RTX 6000 Ada vs ~28 local — the gradient update was the
  bottleneck, not physics), ~2.7 min/seed incl. BC; whole 3-seed run < 10 min.

Result wording (mandatory framing): `scripted_pdhold ~1.0`, `bc_clone median 0.458 / best 0.79`, `td3bc_refine 0.0
(collapse)`, `framework_substrate = WORKING` (one `.hymeko` → learned standing controller).

## Files added (P2/P3)

`hymeko_rl/env/quadruped_env.py` (+`expert_action` PD-hold + pd gains), `hymeko_rl/experiments/quadruped_stand_train.py`
(new — the campaign: reward certificate + DART demos + `Campaign` reuse). Artifacts under
`experiments/2026_07_07_17_32_quadruped_stand/` (results.json + run.log + gifs + policies).

## CORE.YAML / dependency note

**Repo pin untouched.** The run used **torch 2.11.0+cu128 in a remote uv scratch venv on kato15** (`.venv_stand`) —
kato15's driver is CUDA 12.8 and torch 2.12.0 only ships a CUDA-13.2 GPU build, so the pinned build cannot use its
GPU. This is a *remote-venv* build choice for a **stochastic RL** run (§3 RL carve-out — not bit-exact reproducible,
not an RTL-parity validation), user-authorized (venv-only, no system/driver change); `CORE.YAML`'s `torch==2.12.0`
is unchanged and the local/CI stack stays on the pin.

## Provenance

Base SHA `4320202`, working tree dirty (this session). Seeds 0/1/2. **kato15**: NVIDIA RTX 6000 Ada 48 GB, driver
570.153.02 (CUDA 12.8), torch 2.11.0+cu128, MuJoCo 3.10.0, Python 3.12. Local dev/tests: torch 2.12.0+cu132, MuJoCo
3.9.0, RTX 3070 Laptop, Win11. Peak RSS ≪ 16 GB (tiny nets). No persistent repo state mutated; the remote venv is
scratch.
