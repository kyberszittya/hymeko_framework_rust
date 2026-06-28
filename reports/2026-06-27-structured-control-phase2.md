# Phase 2 — structured control (`u = −Kx`, gain-sparsity = H)

**When:** 2026-06-27 18:00 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu & Prof. Kato
**Design:** `docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/phase2-structured-control.md`
**Phase 1:** `reports/2026-06-27-topology-performance-map.md`.

## Summary

The control-theory leg of Kato's program: a controller topology H constrains a state-feedback gain K to a
**sparsity pattern** (`K_ij ≠ 0` only if `i=j` or `(i,j)∈H`) — a distributed controller whose communication
graph is H. For a networked plant `ẋ = Ax + Bu` (`A = −I + ε·Ŝ(G_plant)` Hurwitz, `B=I`, `Q=R=I`), we
synthesise the **structured-LQR** gain per topology (projected gradient, warm-started from the masked CARE
optimum) and measure the **suboptimality** `ρ = J(K_H)/J*`.

## Result (measured, reliable regime)

**The matched controller minimises ρ for all 7 plants** (best sparse controller: chain→chain, ring→ring, …,
random→random) — diagonal dominance, the same *qualitative* signal as Phase 1.

| plant\controller | chain | ring | star | tree | grid | small_w | random | complete |
|---|---|---|---|---|---|---|---|---|
| chain   | **1.00** | 1.00 | 1.04 | 1.04 | 1.01 | 1.00 | 1.00 | 1.00 |
| star    | 1.02 | 1.02 | **1.00** | 1.02 | 1.02 | 1.02 | 1.02 | 1.00 |
| grid    | 1.04 | 1.04 | 1.05 | 1.04 | **1.01** | 1.03 | 1.02 | 1.00 |
| …       | matched is the row-min sparse for every plant (full matrix in `structured_map.json`) |

**But the effect is small.** The worst mismatched sparse controller is only **2–5% above optimal**
(`worst_penalty` 0.022–0.047). Control of these benign, fully-actuated, stable plants is **weakly
topology-dependent** — in honest contrast to Phase 1, where the matching topology was **100×** better at
*representation*. Figure: `reports/structured_control/structured_map.png`.

### Interpretation (measured vs inferred)

- **Measured:** matched topology = best sparse controller (7/7); penalties 2–5%; framework numerically exact
  (oracle below).
- **Inferred:** the small margin is the known control-theoretic fact that a *fully-actuated, stable, symmetric*
  LTI plant barely needs off-diagonal feedback — diagonal regulation is already near-optimal. So topology being
  weakly load-bearing here is *expected*, not a defect. The contrast Phase-1-strong / Phase-2-weak is itself the
  finding: **structure governs representation strongly, control-of-easy-plants weakly** — the same lesson the RL
  results taught (structure is load-bearing only when the problem is hard).

## A real bug, found and fixed (per the contract — a regression is a bug, not a finding)

At strong coupling (ε≈0.98) the **complete** topology first read ρ=1.5–11.9 — *worse* than sparse topologies,
which is impossible (complete is a superset, it can always achieve J*). This was **not** a finding but a
**solver failure**: projected gradient from `K=0` underflowed the line search near the stability boundary before
reaching the optimum. Fix: **warm-start from the masked CARE optimum** `K*⊙mask` (when it stabilises). After the
fix, complete reads ρ=1.000 exactly at all coupling strengths (regression test
`test_complete_topology_is_optimal`). The oracle (full-mask → J*) and monotonicity (denser never worse) both
hold.

## Honest scope & next (Phase 3)

The regime where topology should *strongly* gate control is **under-actuation** (sparse B — actuators must route
control through the communication topology) and **open-loop instability** (only covering topologies stabilise →
ρ=∞ for mismatched). Both were probed and confirmed weak/strong respectively at the design stage; the
under-actuated + unstable plant (needing a stabilising init, e.g. the Lin–Fardad–Jovanović augmented Lagrangian)
is Phase 3. This Phase-2 deliverable is the validated framework + the honest "benign plants are weakly
topology-dependent" result — not an overclaim of strong control dependence.

## Files touched

- `hymeko_rl/structured_control.py` — **new**, ~190 LOC: `make_plant`, `unconstrained_lqr` (scipy CARE),
  `lqr_cost`, `structured_lqr` (warm-started projected gradient), `mask_from_topology`, `run_structured_map`,
  `plot_structured_map`, CLI.
- `hymeko_rl/tests/test_structured_control.py` — **new**, 10 tests (oracle, complete-regression, monotonicity,
  stability gate, mask, Hurwitz, matched-best).
- `reports/structured_control/{structured_map.json, structured_map.png}` — **new** artifacts.

## Tests & provenance

- **10/10 pass**, ruff clean. Oracle full-mask→J* `rel<1e-3`; complete ρ=1.000.
- CORE.YAML: none. New deps: none (scipy already present — same solver as `control/controllers.py:LQRController`).
- N=9, a=1.0, ε=0.95, Q=R=I, graph_seed 0, projected-gradient iters 1000.
- §6.5: no anti-patterns (reused `topology_zoo` + scipy CARE/CALE; Strategy registry; no duplication).
