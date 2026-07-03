# HyMeKo arc weights — weights on the hyperedge, not the node

## The principle

A hyperedge is a set of **signed arcs** to its members. In HyMeKo a weight is a property of the *arc* (the
relationship), **not** an attribute on the member node. The same term/vertex can then participate in two
hyperedges with different weights, and the weight lives where the relationship is declared.

```hymeko
// weight on the ARC (the hyperedge defines it):
@grasp_reward: r.reward_spec {
    (+ approach 4.0, + pull 1.0, + both 3.0, + zone 10.0);
}
```

vs. the older node-attribute form (deprecated for this use):

```hymeko
@approach: rew.grasp_approach { weight 4.0; }   // weight on the NODE — avoid
@grasp_reward: r.reward_spec { (+ approach, + pull, ...); }   // unweighted arcs
```

## It is already in the grammar (no core change)

A signed arc reference is `[+|-|~] <tags> RefPath <anno>`, and the annotation's optional value (`OptValue`)
admits any `Value`, including `Value::Num` (`parser/src/hymeko.lalrpop`). So `(+ approach 4.0)` **parses today** —
the weight lands in `RefAtom.anno.value`. It is the same mechanism a joint arc uses for its transform payload
(`(+ world [[pos],[rpy]], - base)` is a `Value::List` annotation). **No `parser`/`hymeko_core` edit is needed.**

## Reading arc weights (the Python bridge)

Two entry points in `hymeko_rl/env/_profile.py`:

- **`read_bundle(profile, spec_kind)` → `(name, kind, body, arc_weight)`** — the typed-bundle reader
  (`reward_spec`, `observation_space`, `strategy_spec`, …). `arc_weight` is the numeric arc annotation
  (`None` when absent).
- **`read_arc_weights(profile, edge_name)` → `[(sign, member, weight)]`** — the **general** capability: the
  signed arcs + weights of *any* declared hyperedge, by name. Non-numeric arc payloads (e.g. a joint transform)
  read as weightless (`None`).

```python
from hymeko_rl.env._profile import read_arc_weights
read_arc_weights("data/robotics/galambos_task.hymeko", "grasp_reward")
# [("+", "approach", 4.0), ("+", "pull", 1.0), ("+", "both", 3.0), ("+", "zone", 10.0), ...]
```

## Where arc weights are used

- **Reward** — `reward.read_reward_terms` reads the arc weight (falls back to a legacy body `weight`, then
  `1.0`), so the scalar reward is `Σ wᵢ·termᵢ` with the weights on the bundle's arcs. Files:
  `data/robotics/{galambos_task,pick_place_task,arm_reach_task}.hymeko`.
- **HSiKAN incidence** — `hymeko_neuro.core` `incidence="weighted"` lifts the binary `{0,±1}` signed incidence to **free
  real arc weights** on the existing structure (init 1.0 → parity). This is the model-side counterpart: the arc
  weight is the *strength* of a signed message. See [`hsikan_overview.md`](hsikan_overview.md).
- **HSiKAN highway gate** — the legacy `cr_highway` mode threads per-arc weights into the gate
  (`hymeko_neuro`), i.e. the arc weight perturbs the highway transform gate.

## Why (the design rationale)

A *signed* hypergraph is only fully expressive when arc weights are **real values**, not `{0, +1, -1}`. Binary
incidence collapses the model to "connected or not, friend or foe" and discards relationship strength — the
limitation behind weak results on weight-sensitive tasks. Arc weights restore the real-valued relationship the
formalism is built on; putting them on the arc (not the node) keeps the term/vertex reusable and the weight
auditable where the relationship is declared.

## Status

- Grammar: arc weights parse (no core change). ✓
- Reward bundles: migrated to arc weights, parity-exact, tested. ✓
- General reader `read_arc_weights`: added + tested. ✓
- HSiKAN `incidence="weighted"`: real arc weights on the structural mask, tested. ✓
