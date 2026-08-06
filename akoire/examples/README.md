# AKOIRE RL Demo

This folder contains a tiny reinforcement-learning sketch that ties the current
architecture together:

- **AKOIRE** proposes candidate action refinements.
- **HyMeKo** gate-keeps the proposed `.hymeko` source.
- **HIVE** commits only accepted experience transitions as canonical
  signed-hypergraph transactions.

Run:

```bash
cargo run -p akoire --example rl_hive_policy
```

The demo includes one malformed proposal to show the accountability boundary:
HyMeKo rejects it, HIVE does not mutate, and the Q-table is updated only from
accepted transitions.

For a more game-like witness, run:

```bash
cargo run -p akoire --example solved_grid_game
```

That example solves a deterministic 4x4 escape game by value iteration, gates a
generated HyMeKo representation, commits the solved transition system and
optimal policy to HIVE, then prints the policy map, value map, and a Mermaid
policy graph.
