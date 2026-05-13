# Robot communication cliques — demo plan

**Date:** 2026-05-14
**Audience:** Niitsuma presentation (signed-graph balance theory applied
to multi-robot communication).
**Status:** plan v1.

## Scope

A new tab in the demo GUI: **"Robot communication cliques"**.

A multi-robot communication network is a signed graph:
- vertices = robots,
- edges = pairwise communication attempts,
- sign = `+` reliable (high SINR, in range, trusted) or `−`
  (jammed, distrusted, dropped).

**Structural-balance theory** (Cartwright-Harary 1956) says a signed
graph is *balanced* iff every cycle has an even number of negative
edges. A balanced clique on robots is a **stable communication team**:
no internal conflicts, σ-product around any sub-cycle equals `+1`.
HSiKAN learns σ-products in its cycle pool by construction, so this is
exactly the inductive bias the network wants.

## CORE.YAML items touched

**Empty list.** All work lives in `signedkan_wip/src/demo/` and
new tests. No CORE crate, no pinned-dep changes.

## Affected files

New:

- `signedkan_wip/src/demo/cliques.py` — synthetic robot-network
  generator (`make_robot_network`), balanced-clique enumerator
  (`enumerate_balanced_cliques`), `RobotNetworkBundle` dataclass.
- `signedkan_wip/src/demo/cliques_plotting.py` — 2-D spatial scatter,
  edges colour-coded by sign, balanced cliques shaded.
- `signedkan_wip/tests/test_demo_cliques.py` — generator determinism,
  balance check on hand-built cases, clique enumeration limits.

Modified:

- `signedkan_wip/src/demo/gui.py` — add `"Robot communication cliques"`
  tab between `"Kinematic graph"` and `"How to use these predictions"`.
- `signedkan_wip/src/demo/README.md` — describe the new tab + use case
  framing for Niitsuma.

## Interface changes

New public API in `demo.cliques`:

- `make_robot_network(n_robots, area_size=10.0, comm_range=3.5,
   noise_prob=0.1, seed=0) -> RobotNetworkBundle`
  - Places robots uniformly in 2-D, edges between robots within
    `comm_range`. Each edge gets `+` by default, flipped to `−` with
    probability `noise_prob` (jamming / distrust events).
- `enumerate_balanced_cliques(bundle, min_size=3, max_size=6,
   limit=20) -> list[Clique]`
  - Returns balanced cliques sorted by size (largest first).
  - Cliques are detected on the positive subgraph then verified
    balanced over their signed edges.
- `RobotNetworkBundle` — vertices, edges, signs, 2-D positions,
  metadata (seed, noise_prob, comm_range).

## Test strategy

- **Unit:** generator determinism (same seed → identical bundle); known
  balanced 3-cycle hand-built case detected; known unbalanced
  3-cycle case rejected; clique enumeration respects `limit`.
- **Smoke:** generate a 20-robot network, enumerate cliques, render
  spatial figure without error.
- **No performance test** in v1 — the network sizes (≤ 50 robots) are
  trivial.

## Performance budget

- Generator: O(N²) edge enumeration; trivial for N ≤ 50.
- Clique enumeration: NetworkX `find_cliques`, sub-second for
  N ≤ 50 with `comm_range` set so the average degree is ≤ 8.
- Plot render: < 200 ms.

## Rollback path

Self-contained — drop the new modules + the tab from `gui.py`.
Existing kinematic + signed-link demos unaffected.

## Risk anticipation

- **NetworkX `find_cliques` is exponential in clique number**: with a
  dense `comm_range` and many small cliques, enumeration explodes.
  Mitigation: `limit` parameter + a hard cap on `max_size`.
- **Balance heuristic** treats `+` / `−` symmetrically. If we later
  weight by SINR / RSSI, the threshold logic changes; pin the binary
  semantics in v1, lift to weighted in v2.
- **Two layouts collide** with the kinematic spring layout — both use
  matplotlib + NetworkX. Use separate function names + separate
  figure objects to avoid state leaking between tabs.

## Out of scope for v1

- Training HSiKAN on the synthetic network (v0.5 — `predict_edge_sign`
  given vertex positions only). This would let the demo predict
  signs on unseen networks; descriptive v1 ships without it.
- Real RSSI / SINR-derived edge signs from a measurement log.
- Multi-network comparison (single network per render).
- Time-evolving networks (snapshot only in v1).

## Order of work

1. `cliques.py` — generator + bundle + balanced-clique enumeration.
2. `cliques_plotting.py` — spatial layout, sign-coloured edges,
   balanced-clique shading.
3. `test_demo_cliques.py` — unit + smoke tests.
4. `gui.py` — wire the new tab.
5. `README.md` — application framing for the Niitsuma narrative.
6. Visual smoke before reporting done.

## Why no TikZ/PDF/Mermaid plan

Same rationale as the kinematic demo (`docs/plans/2026-05-13-kinematic-demo/plan.md`):
exploratory applied demo, no CORE.YAML touched, small interface
surface. Upgrade to four-format if/when this becomes a research
artifact (paper figure, IROS / ICRA submission).
