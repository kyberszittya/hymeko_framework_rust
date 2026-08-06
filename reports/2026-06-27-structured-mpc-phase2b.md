# Phase 2b — structured MPC (model-predictive control over the matched topology)

**When:** 2026-06-27 18:46 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu & Prof. Kato
**Design:** `docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/phase2b-structured-mpc.md`
**Builds on:** Phase 2 (`reports/2026-06-27-structured-control-phase2.md`).

## Summary

Added MPC to the topology framework. The honest design point first: **for an unconstrained linear-quadratic
problem MPC ≡ LQR**, so on the Phase-2 plant MPC would change nothing. MPC differs only under constraints, so we
add **input saturation** `|u_i| ≤ u_max = 0.6` and a challenging initial condition (`x₀ = 3·1`, large enough to
saturate). MPC is *model*-predictive, so the topology enters as the controller's **prediction model** coupling:
`MPC(H)` predicts `x_{k+1} = A_d(H)x_k + B_d u_k` over the horizon. The true plant has coupling `G_plant`.

## Result (measured)

**The matched model is the best controller for all 8 plants** (clean diagonal — including complete→complete),
under constrained MPC. Cost ratio `J/J_d` (`J_d` = discrete-LQR cost-to-go; 1.0 = unconstrained optimum):

| true\model | chain | ring | star | tree | grid | small_w | random | complete |
|---|---|---|---|---|---|---|---|---|
| chain   | **1.16** | 1.18 | 1.18 | 1.17 | 1.17 | 1.17 | 1.17 | 1.20 |
| star    | 1.09 | 1.08 | **1.07** | 1.07 | 1.09 | 1.10 | 1.09 | 1.09 |
| grid    | 1.11 | 1.14 | 1.14 | 1.11 | **1.07** | 1.10 | 1.10 | 1.14 |
| …       | matched (diagonal) is the row-min for every plant (full matrix in `mpc_map.json`) |

- **Diagonal dominance is now clean (8/8).** Unlike Phase 2's gain-sparsity framing (where chain tied with the
  `random` superset), the model-mismatch framing makes the matched model *unambiguously* best for every plant —
  a mismatched model mispredicts, and the error compounds over the horizon.
- **The mismatch penalty (2–7%, `worst_penalty` 0.020–0.069) is modestly larger than Phase 2's (2–5%).** The
  constraint *does* make topology-match bite more — directionally confirming the hypothesis — but the magnitude
  is still modest on this benign plant.
- **MPC beats saturated-LQR by ~1%** (constraint-aware optimisation vs naive clipping). Small, because the
  benign plant rarely saturates hard.

Figure: `reports/structured_mpc/mpc_map.png`.

## Correctness guards (oracle before claims)

1. **Unconstrained matched MPC = discrete-LQR.** With `u_max=∞` and the matched model, the closed-loop cost
   equals `x₀ᵀP_d x₀` to `rel<1e-4` for every plant — validates the condensed prediction + DARE terminal cost.
2. **Constrained MPC ≤ saturated-LQR** (matched) — MPC's reason to exist; holds for all plants.
3. **Input box respected** — the QP-feasible control satisfies `|u|≤u_max` (tested).

## Honest reading

This *strengthens* the topology-match story qualitatively (clean 8/8 diagonal vs Phase 2's near-ties) while
honestly confirming the magnitude is still modest: constraints amplify topology-match a little, not a lot, on a
fully-actuated stable plant. The strong regime remains **Phase 3** (under-actuation — control must route through
the communication topology — and open-loop instability — only covering topologies stabilise). The three-part
picture for Kato is now: *representation strongly topology-dependent (Phase 1, 100×); unconstrained control
weakly (Phase 2, ~3%); constrained MPC slightly more and cleanly diagonal (Phase 2b, 2–7%)* — a coherent gradient
of "structure matters more as the control problem gets harder."

## Files touched

- `hymeko_rl/structured_mpc.py` — **new**, ~200 LOC: `discretize`, `StructuredMPC` (condensed box-QP, DARE
  terminal, warm-start), `simulate_closed_loop`, `SaturatedLQR`, `run_mpc_topology_map`, `plot_mpc_map`, CLI.
- `hymeko_rl/tests/test_structured_mpc.py` — **new**, 7 tests (oracle, MPC≤sat-LQR, box, discretise-stability,
  matched-best).
- `reports/structured_mpc/{mpc_map.json, mpc_map.png}` — **new** artifacts.

## Tests & provenance

- **7/7 pass**, ruff clean. Oracle unconstrained-MPC = J_d `rel<1e-4`.
- CORE.YAML: none. New deps: none (`scipy.optimize.minimize` L-BFGS-B for the box-QP; scipy already present).
- N=9, ε=0.9, dt=0.1, horizon 12, steps 60, u_max 0.6, x₀=3·1, graph_seed 0.
- §6.5: no anti-patterns (reused `make_plant`/`unconstrained_lqr`/`topology_zoo`; the bicycle `MPCController` is
  single-input nonlinear — noted, not reused).
