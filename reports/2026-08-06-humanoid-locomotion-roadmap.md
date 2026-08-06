# Continuation roadmap — humanoid locomotion + the HOTARU planning layer

**Date:** 2026-08-06 · **Branch:** `research/humanoid-com-lyapunov` (this session's work) → for merge to `master`.

This is the *continuation plan* for the two arcs this session built, after the duplication was removed and
the code integrated into the framework (`cem.py` now holds the shared CEM + policy; `run_humanoid_walk_sac`
exposes `train_walk_sac`; the reward knobs live on the env configs, default-off).

---

## Where things stand (measured, committed)

**Locomotion.** The forward-speed ceiling was a **control-scheme** limit, not a learner one: the
position-servo anchors to the standing pose, so it can only lunge ~8 cm. **Torque control breaks it** →
a sustained forward walk (**0.71 m, 2.0 s, 0.36 m/s**, `cbb5721a`). Every reward-shaping refinement on top
— uprightness hinge, periodic-gait prior (peak **0.83 m/s**), learned-viability gate, co-adaptation — buys
**speed for survival** or **conservatism for distance**; **none beat plain torque for *sustained* walking.**
The honest conclusion: *sustained fast bipedal walking here is a control/model problem, not a reward one.*

**HOTARU / AKOIRE (the successful arc).** A real A\* planner over the implicit HIVE-delta space kicks off
HOTARU; a `SearchProblem`/`solve` framework; a Kyosei arity filter; a PyO3 binding (`hymeko.astar_plan`)
that runs the *same* `akoire::astar` from Python; and **HOTARU planning over the HyMeKo-described graph
elements → a semantic plan** (grounded `add_edge` ops toward a topological goal), the LLM-produces-spec /
HOTARU-verifies seam. This arc is clean and extensible.

## The levers that remain (locomotion) — structural, not reward

Ranked by expected payoff × effort. Each is a separate build with its own plan + report.

1. **Stronger low-level controller the RL rides on (highest payoff).** The torque policy learns balance
   *and* gait from scratch. Give it a stabilising inner loop — a **whole-body/ID controller** (the
   `wbc.py` WBC already exists) or a **PD-around-a-moving-reference** — and let SAC output a *bounded
   residual* (the coin-R8 regime that worked elsewhere). The RL then shapes a gait on a stable base
   instead of re-discovering balance. Reuse: `wbc.WholeBodyController`, `run_humanoid_walk_sac.train_walk_sac`.
2. **A non-toppling foot/ankle + contact tuning.** The gaits fall by pitching over small, stiff feet.
   Widen the foot, add an actuated ankle/toe with proper contact friction (`humanoid_toe*.hymeko` is a
   start), and re-measure the survival ceiling before more RL. This is a **model** change (data/robotics),
   not a controller one — cheap to try, potentially large.
3. **Imitation / reference gait (DeepMimic-style).** Track a nominal cyclic joint reference (phase already
   in the obs via `periodic_gait`) with an imitation reward, rather than reward-from-scratch. Turns the
   open-ended "discover a gait" into "match this gait", which is what published bipeds do. Reuse the phase
   clock; add a reference trajectory + a tracking term.
4. **Far more training.** This is CPU SAC at ~10⁵ steps; published bipeds use 10⁷–10⁸. A GPU run (kato15,
   per the memory) at 10× the steps is the brute-force check — do it *after* 1–2 above so it is not
   wasted on the wrong controller.

**Do 1 + 2 first** (a residual RL on the WBC + a better foot): they attack the actual failure (balance +
toppling) at the structural level. 3 and 4 are refinements on a working base.

## HOTARU continuation (the extensible arc)

- **`LlmSynthesizer`**: wire intent → `(GraphGoal, candidate_edges)` from an LLM (sketched in
  `akoire/synthesize.rs`) — the open-ended front-end HOTARU verifies.
- **Richer topological goals**: cycle-freeness, k-connectivity, a required subgraph — new `GraphGoal`
  variants over `GraphView`.
- **`add_node` + signed/directed ops**: grow the vertex set and use `SignedRef` for signed/arc goals.
- **Bridge to motion**: the shared `astar_plan` binding lets a Python planner reuse the akoire engine —
  the one planner framework across structure (HIVE-delta) and motion (the footstep planner already runs on it).

## Framework state after this session

- **Duplication removed**: `cem.py` (shared CEM + linear policy) replaces the 3× copied CEM loop;
  `train_walk_sac` is the shared SAC core (CLI + co-adapt driver reuse it). Reward knobs are env-config
  fields (default-off, obs unchanged when off), not forked scripts.
- **All reusable**: envs (`balance_env` torque/gait/gate, `footstep_env` WBC), planners (`footstep_planner`,
  `stepping_stone_demo` on the shared A\*), certificates (`viability_gate`, `neural_certificate`), and the
  akoire HOTARU planner.

## Merge

This branch carries the HOTARU planner + PyO3 binding + graph-semantic planner + the full locomotion arc,
all with default-off knobs, no CORE.YAML edits, no new dependencies (except the approved `akoire` path-dep
in `hymeko_py`). Proposed: merge `research/humanoid-com-lyapunov` → `master` via PR.
