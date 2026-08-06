# Multidimensional (per-node) readout — don't collapse a per-joint output

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** under `docs/plans/2026-06-26-readout-ablation/` (this is its named next-step: an identity-preserving,
aggregation-capable readout) · **Status:** built, tested, run.
**Origin:** user insight — "use global pooling **and** multidimensional output; we have multiple joints anyway."

## The idea

The robot action is **per-joint** and the kinematic graph is **per-joint**, so the natural output is a vector
`y ∈ R^N` indexed by joint. The mean-pool readout collapses the N per-vertex embeddings into one vector, which
the actor then expands back to N actions — discarding the joint↔node correspondence. The fix: a **per-node
output head** — each joint's value read directly from *its own* message-passed node embedding — with global
pooling kept as broadcast **context**, not as the thing you expand from.

## Method

Same probe/graph, a new **vector target** `y_j = tanh(α·(B²x)_j)` for every node `j` (`pernode`). Four heads,
all `(B,N,1) → (B,N)`, 5 seeds, n_train=512:
- `pool_expand` — mean-pool `(B,N,H)→(B,H)` then `Linear(H,N)` (collapse-then-expand, the baseline);
- `per_node` — a shared `Linear(H,1)` applied per node (the multidimensional output, no collapse);
- `per_node_global` — per-node head over `[h_node ; mean-pool(h)]` (global pooling **and** per-node output);
- `mlp` — flat `Linear` baseline.

## Result

| readout | test MSE | params |
|---|---|---|
| pool_expand | 0.4840 | 3911 |
| **per_node** | **0.0001** | 3713 |
| **per_node_global** (winner) | **0.0001** | 3745 |
| mlp | 0.0033 | 5127 |

`pool_expand / per_node = 3723×`. Figure: `reports/structural_probe/multidim_probe.png`.

### Reading (measured vs inferred)

- **Measured.** Collapse-then-expand is **3723× worse** than the per-node head, despite *more* params. The
  per-node head fits the vector target near-perfectly and **beats the flat MLP 33×** with *fewer* params.
  Adding global context (`per_node_global`) costs +32 params and ties `per_node` for the win.
- **Inferred.** The mean-pool readout was the bottleneck twice over: it loses node-specific info (prior
  `readout-ablation`, 10.9×) **and** is structurally wrong for multi-output (collapse-then-expand, 3723×). The
  earlier HSiKAN-vs-MLP *ties on robot tasks* are consistent with the actor's pool-then-expand readout throwing
  away HSiKAN's per-node structural signal — give each joint its own node embedding and HSiKAN wins outright.
- **Caveat (honest).** The probe target is literally HSiKAN's own per-node computation (`B²x`), so the
  near-perfect HSiKAN fit is partly by construction; the load-bearing claim — *per-node ≫ collapse-then-expand,
  and global context is ~free* — is architecture-level and target-agnostic, but the **robot RL run is the real
  validation**.

## Implication & next step (the RL payoff test)

This closes the readout story and gives a concrete architecture for the actor:

> **Per-node (multidimensional) actor head with global-pool context** — each actuated joint's action from its
> own message-passed node embedding `[h_node ; mean-pool(h)]`, instead of mean-pool → `Linear(H, action_dim)`.

Next: wire this into the RL actor (`hymeko_rl.policy.ActorCritic` currently does `actor_mean = Linear(feat_dim,
action_dim)` over the pooled backbone) and re-run a robot task (galambos/quadruped) vs the MLP, params-matched.
Prediction: the tie flips to an HSiKAN win. The critic can keep a global pool (a scalar V genuinely needs an
aggregate); only the actor needs the per-node head. Map joint→vertex via the existing `_jnt_vtx` (the actuated
joints' child vertices).

## Files touched (CORE.YAML: none — `hymeko_neuro.core` consumed via `node_activations`)

- **Extended** `hymeko_rl/structural_probe.py` — `pernode` target; `_PoolExpand`/`_PerNode`/`_PerNodeGlobal`/
  `_MlpMultiDim` heads + `build_multidim_model`; `run_multidim_probe` + `plot_multidim`; CLI `--multidim`.
- **Extended** `hymeko_rl/tests/test_structural_probe.py` — vector target, the four heads, multidim smoke
  (**15 tests pass**; ruff clean).

## Provenance
- Reproduce: `python -m hymeko_rl.structural_probe --multidim --hidden 32 --seeds 5 --n-train 512 --epochs 300`.
  Deterministic (seed before each build). Git `fix-hsikan`, tree dirty. Windows 11, Python 3.12, torch CPU.
