# Stage 3 (started) — star-expansion structural entropy H★: design + foundation

2026-06-23 · hymeko_rl · plan `docs/plans/2026-06-23-rl-scenario-ablation-entropy/` (Stage 3)

## Status
Foundation implemented (the H★ signal + its activation source + boundary tests). The exploration-seat wiring and
the discriminating A/B are **deferred** until the Stage-2 campaign frees the CPU (no concurrent torch training —
contention). This note records the design so the rationale is on disk before the experiment.

## Definition (what H★ is)
For an HSiKAN policy, after the signed message-passing layers and **before** the mean-pool, each kinematic vertex
`v` has a hidden vector `h_v`. Define its activation energy `e_v = ‖h_v‖₂`, normalize over the vertices to a
distribution `p_v = e_v / Σ_u e_u`, and take the Shannon entropy

  **H★ = − Σ_v p_v log p_v**  (normalized by `log N` → [0, 1]).

H★ = 1 ⇒ every link equally engaged (diffuse structural reasoning); H★ = 0 ⇒ one link carries all the energy
(maximally concentrated). It reads the *structure of the policy's computation* over the robot hypergraph — not
its action distribution. It is differentiable in the activations, so it can sit exactly where `ent_coef·H(π)`
does in the objective.

## Why this signal (the bet)
It is the **third** signal of the already-seated exploration vocabulary (`project-rl-algorithm-roadmap`):
(1) policy-entropy `α·H(π)` [SAC, done]; (2) critic-ensemble disagreement `Var_k Q` [TD-k; `n_critics` builds
the ensemble]; (3) **structural entropy** [this — the novel bet]. The hypothesis (stated as hypothesis): a signal
derived from *where the policy is structurally attending* explores the configuration space differently from
action-noise entropy, and may help on branched morphologies. It may also do nothing — the A/B includes the
discriminating measurement.

## Files (all non-core)
- **New** `hymeko_rl/star_entropy.py` — `star_expansion_entropy(node_activations) → (B,)` and
  `mean_star_entropy(backbone, obs)`; a `StructuralBackbone` Protocol (the MLP baseline deliberately does not
  satisfy it — H★ is defined only over the hypergraph backbone).
- **Edit** `hymeko_rl/policy.py` (+11) — `HSiKANBackbone.node_activations(x) → (B, N, hidden)` exposes the
  pre-pool per-vertex activations; `forward` now delegates to it (DRY, behaviour unchanged).
- **New** `hymeko_rl/tests/test_star_entropy.py` — 6 tests: uniform→1, one-hot→0, normalized range +
  unnormalized `log N` nats, differentiability, rank guard, end-to-end from a tiny HSiKAN backbone.

## Tests / static analysis
6/6 pass (6.3 s, CPU-only); `ruff` + `mypy --strict` clean on the new module.

## Deferred (the experiment — needs CPU after the Stage-2 grid)
1. **Seat wiring:** add a structural-entropy coefficient to the exploration vocabulary (`StrategySpec`, the
   `.hymeko` exploration term) and apply `− struct_coef · H★` in the PPO / off-policy update, beside
   `− ent_coef · H(π)`. Typed config parsed at startup, passed explicitly (no env-var-at-call-site, §6.5 #11);
   one config field, not a Cartesian of variants (§6.5 #1).
2. **Discriminating A/B (real topology — Galambos/arm6dof, NEVER cart-pole):** {policy-entropy only,
   critic-disagreement only, H★ only, H★+policy-entropy}, ≥5 seeds, curve-max median/IQR **plus an exploration
   proxy (state coverage)** so a positive result is attributable to exploration, not variance (§3). A null result
   is a publishable finding.

## Hub nodes (hyperedges) — the star-expansion primitive (added)
The next natural step toward H★'s ideal source, done Python-side (no core edit): `HypergraphState.star_expansion()`
builds the **bipartite star-expansion** — each joint becomes a HUB node mediating its parent and child, so the
graph carries the original vertices **plus one hyperedge-hub per joint**, and message passing flows
`vertex → hub → vertex`. It returns a drop-in `HypergraphState`, so the existing `HSiKANBackbone` consumes it
unchanged; `node_activations` then span vertices+hubs and H★ reads the hub-augmented node set.

- **Files:** `hymeko_rl/hypergraph_state.py` (+34, `star_expansion`); tests `hymeko_rl/tests/test_star_expansion.py`
  (hub-per-joint count, exact hub connectivity + signs, drop-in HSiKAN + H★ over hubs, empty-graph guard). 4/4
  pass; with the 6 H★ tests, **10/10**; `ruff`/`mypy --strict` clean.
- **Honest scope:** each MJCF joint is binary, so every hub has degree 2 — this is the faithful star-expansion of
  the *current* source. True ``>2`` hyperedges (e.g. a grasp hyperedge over gripper+object, or `.hymeko`
  bundle-of-bundles) need the canonical `.hymeko` star-expansion; the form here carries them unchanged once the
  engine transitive-import resolver lands. Joints-as-nodes is already a real enrichment: the actuated DOF become
  first-class message-passing nodes rather than edges.
- **Deferred (with the A/B):** feed the star-expanded graph into the actual policy (the env must then emit
  per-hub/joint features — qpos/qvel are the natural fit), and A/B the vertex-graph vs the star-expanded graph for
  H★ feedback. Until then the primitive is exercised by the tests (synthetic obs), not yet wired into training.

## Caveat (measured, honest)
The current `hg_state` is the signed **vertex×vertex** kinematic adjacency, so H★ is the entropy over the
kinematic vertices. The canonical `.hymeko` **star-expansion** (with hyperedge-hub nodes) is the ideal source and
extends H★ to the full expanded node set *unchanged in form* — it is gated on the engine's transitive-import
resolver (memory `project-engine-transitive-imports`). The name "star-expansion entropy" anticipates that source;
today it is computed over the vertex set, which is the faithful approximation until the resolver lands.
