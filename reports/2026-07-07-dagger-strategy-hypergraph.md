# The training STRATEGY as a signed dataflow hypergraph — DAgger declared in HyMeKo, run on Aibo standing

**Date:** 2026-07-07 (JST) · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**Base SHA:** `4320202` (working tree dirty) · **Plan:** `docs/plans/2026-07-07-dagger-strategy-hypergraph/` (tex/pdf/tikz/mmd)
**Follows:** `reports/2026-07-07-aibo-quadruped-hymeko-substrate.md` (the standing substrate; BC 0.458, TD3+BC 0.0)

## Summary

Reframed the training **strategy** as a **signed dataflow hypergraph** — the strategy-side of "HyMeKo as a
declarative substrate". Where the plant/reward/observation were already hypergraph-described, the strategy was a
flat `experiment_spec` (an `algorithm "dagger"` string + scalar knobs). Now DAgger is declared as a cyclic dataflow
graph: **training stages are nodes**, **data dependencies are signed hyperedges** (`+` producer, `-` consumer), and
the **relabel loop** (`retrain → rollout`) is the cycle. A reader **classifies the algorithm from the topology**
(cycle ⇒ `dagger`, acyclic ⇒ `bc`) and dispatches to the existing `Dagger` executor. **The algorithm's identity is
its dataflow shape, not a string.**

Then ran it on the Aibo standing task (the motivated lever past the BC ceiling; TD3+BC collapsed to 0.0):
**DAgger standing median 0.958, best 1.0** — it recovers the imitation ceiling, lifting even the worst BC seed
(0.167 → 1.0).

## Result (kato15 RTX 6000 Ada, 3 seeds)

| method | standing dwell median | per-seed | note |
|---|---|---|---|
| scripted PD-hold (demonstrator) | 1.0 | — | the ceiling |
| BC clone (from `.hymeko`) | 0.458 | [0.29, 0.46, 0.79] | covariate shift limits it |
| **TD3+BC (refine)** | **0.0** | collapse | off-policy value drift |
| **DAgger (hypergraph), `D0`=40** | **0.958** | **[0.958, 0.958, 1.0]** | recovers the ceiling |

- **[measured] DAgger fixes the covariate shift.** Round-1/2 relabelling on learner-visited states lifts every seed
  to ≥0.958 at its peak; seed-2's dreadful BC start (0.167) reaches **1.0**. This is exactly the predicted lever —
  *imitation* (DAgger), not off-policy RL, is what beats a BC ceiling (`project-fanuc-offpolicy-collapse`).
- **[measured] `D0` size matters — and the fix was a one-field edit to the DECLARED GRAPH.** The first run used
  `n_demos 200` (49k DART samples); each DAgger round adds only ~3k relabels, which are **drowned** → noisy rounds,
  best-checkpoint median 0.792. Editing the `source` stage `n_demos 200 → 40` (10k `D0`) let the relabels bite →
  median **0.958**. The smoke (`D0`≈6k) had already gone 0.33 → 1.0 in one round, evidencing the diagnosis.
- **[measured] The rounds are noisy** (each seed peaks at round 1–2 then later rounds collapse toward 0 — fresh-reBC
  variance on a growing aggregate). The **best-checkpoint** (standard DAgger practice) banks the peak; the peak is
  robust (all seeds ≥0.958). Not clean monotonic convergence — a `warm_start` / aggregate-balancing follow-up could
  smooth it, but the result already recovers the ceiling.

## Design (the hypergraph strategy)

- **`data/robotics/meta_strategy_graph.hymeko`** (new vocab): `@stage` (a training-stage node; `role` ∈
  {source, bc, rollout, label, aggregate, eval} selects the primitive, + scalar knobs), `@flow` (a signed dataflow
  hyperedge `(+ producer, - consumer …)`), `@strategy_graph` (the bundle + `iterate` + `seeds`).
- **`data/robotics/quadruped_stand_dagger.hymeko`**: DAgger as 7 stages + 6 flows; `f_loop = (+ retrain, - rollout,
  - evaluate)` is the relabel cycle. **`data/robotics/quadruped_stand_bc_graph.hymeko`**: the acyclic counterpart
  (BC as a graph) — proves the topology distinction concretely.
- **`hymeko_rl/train/strategy_graph.py`**: `StrategyGraph.from_hymeko` (parse stages/flows/iterate via the existing
  `_profile` readers), `classify()` (DFS cycle detection ⇒ `dagger`/`bc`), `to_dagger_config()` (knobs from the
  declared stages), `run_dagger_graph()` (verify topology IS dagger — a graph missing the loop is **rejected** —
  then dispatch to `Dagger`; **reuses the loop, §6.5 #3, no re-implementation**).
- **`hymeko_rl/experiments/quadruped_stand_dagger.py`**: binds the quad env + the stateless PD-hold expert + DART
  `_demos`; the MDP is the sibling `quadruped_stand.hymeko`.
- **`hymeko_rl/train/dagger.py`**: `label_sanity` gained a **stateless-expert carve-out** (`has_latch`) — the PD-hold
  expert has no `_lift_xy` latch, so the "reached the latched phase" requirement is vacuous for it (FANUC path
  unchanged: it still requires `committed_steps > 0`).

## The complete three-topology family (all algorithms told apart by dataflow shape)

`classify()` now distinguishes **all three** RL algorithms by topology alone — no string labels:

| graph | topology signature | classifies |
|---|---|---|
| `quadruped_stand_bc_graph.hymeko` | acyclic `source → bc → eval` | `bc` |
| `quadruped_stand_dagger.hymeko` | relabel **cycle** (no critic) | `dagger` |
| `quadruped_stand_td3bc_graph.hymeko` | a **`critic`** node + Q-target edges | `td3_bc` |

The `critic` node is the discriminator — the same off-policy value estimator whose **value drift collapsed
standing to 0.0**. `run_td3bc_graph` (symmetric with `run_dagger_graph`) verifies the critic topology and dispatches
to the existing `Campaign` executor; `quadruped_stand_strategy.py` is one entry that classifies a graph and routes
to the matching runner. **Verified:** the general entry on the td3_bc graph classified `td3_bc` from the critic node,
ran the Campaign, and reproduced the collapse (0.33 → 0.0). So `bc`/`dagger`/`td3_bc` are three *shapes* of one
dataflow substrate.

## `warm_start` smoothing — declared, tested, honest negative

The DAgger `retrain` stage gained a `warm_start` knob (continue re-BC from the prior round's weights vs textbook
fresh-reBC), read from the graph into `DaggerConfig.warm_start`. Re-run on kato15: **best-checkpoint median 0.958
either way** — `warm_start` [0.79, 0.958, 0.958] vs fresh-reBC [0.958, 0.958, 1.0]. It makes the mid-round decay more
gradual (seed-0 round-2 held 0.917 vs 0.25) but does **not** raise the peak and does not prevent the late-round
collapse — matching the prior FANUC record ("warm_start did not smooth DAgger").

Also tried the **`aggregate_cap`** knob (keep `D0` + only the last 2 rounds' relabels; declared on the `aggregate`
stage): the cap *works* (aggregate plateaus at 15.7k instead of growing) but the result is **identical** — median
0.958, `[0.958, 0.958, 1.0]`, same late-round collapse (d3/d4 → 0.0). So the over-aggregation hypothesis is
**falsified**: the collapse is fresh-reBC **variance** on this task, not aggregate growth. **Verdict:** the DAgger
round-to-round curve is inherently noisy; the **best-checkpoint (0.958) is the robust deliverable** regardless — do
not chase a monotonic curve via warm_start / aggregate-cap (both measured no-help, both now declared graph knobs
for provenance).

## Tests / gates

- **`test_strategy_graph.py` (12)**: parse (7 stages/6 flows/iterate/seeds), the loop flow, classify `dagger`/`bc`/
  **`td3_bc`** from the three graphs + a constructed cyclic-vs-acyclic pair (topology, not label), `to_dagger_config`
  + `to_campaign_config` match the declared knobs, `warm_start` read from the retrain stage, and **both** runners
  reject a wrong-topology graph. + `test_dagger.py` (3, incl. the `label_sanity` change) → **15 passed**.
- Both graphs `hymeko validate` clean. `ruff` clean; `mypy --strict` on the new files clean.
- Local smoke (1 seed, 1 round): the graph classified `dagger`, `label_sanity ok` with `has_latch=false`,
  0.33 → 1.0 — the full path exercised before the kato15 run.

## Files (all non-core; CORE.YAML: none)

New: `meta_strategy_graph.hymeko`, `quadruped_stand_dagger.hymeko`, `quadruped_stand_bc_graph.hymeko`,
`quadruped_stand_td3bc_graph.hymeko`, `hymeko_rl/train/strategy_graph.py`,
`hymeko_rl/experiments/{quadruped_stand_dagger,quadruped_stand_strategy}.py`, `hymeko_rl/tests/test_strategy_graph.py`,
the plan bundle, this report + `reports/figures/dagger_standing_result.png`.
Modified: `hymeko_rl/train/dagger.py` (stateless-expert `label_sanity` carve-out),
`data/robotics/quadruped_stand_dagger.hymeko` (`n_demos 200→40`, `warm_start 1` on retrain).

## Artifacts

`experiments/2026_07_07_18_49_quadruped_stand_dagger/` (`D0`=40, median 0.958; gif of the standing Aibo + curve +
results.json), `experiments/2026_07_07_18_28_quadruped_stand_dagger/` (`D0`=200, median 0.792). Plot
`reports/figures/dagger_standing_result.png`.

## Provenance

kato15 RTX 6000 Ada 48 GB (driver 570.153.02 / CUDA 12.8), **torch 2.11.0+cu128 in the `.venv_stand` uv scratch
venv** (repo `CORE.YAML` pin untouched; §3 RL carve-out; user-approved venv-only). MuJoCo 3.10.0, Python 3.12,
seeds 0/1/2. Local dev/tests: torch 2.12.0+cu132, MuJoCo 3.9.0, RTX 3070, Win11. No persistent repo state mutated.
