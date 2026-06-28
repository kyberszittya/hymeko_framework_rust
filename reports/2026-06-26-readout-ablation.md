# Readout ablation — the mean-pool readout is HSiKAN's control liability (confirmed)

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-26-readout-ablation/` (tex/pdf/tikz/mmd) · **Status:** built, tested, run.
**Context:** follows `reports/2026-06-26-structural-probe.md`, which found HSiKAN's edge is its
per-node-activation + **mean-pool** (a Deep-Sets prior). Hypothesis tested here: on control, that mean-pool
is the *liability* — it collapses node identity, discarding the cross-joint coordination control needs
(which a flat MLP preserves).

## Method (supervised, isolates the readout)

Same fixed signed graph and probe as the structural probe, extended with:
- a **`local`** target `y = tanh(α·(B²x)_k)` for a fixed node `k=6` — a node-*specific* signed value that
  mean-pool (`(1/N)Σ_v`) structurally **cannot** isolate, but a node-preserving readout can;
- a **`concat`** readout: flatten the backbone's per-vertex activations `(B,N,H)→(B,N·H)` — the **same
  signed-conv backbone**, only the readout differs (so a gap isolates the pooling effect).

Three configs (HSiKAN·mean-pool, HSiKAN·concat, MLP) × three targets, 5 seeds, n_train=512, params-matched
(HSiKAN 3713, concat 3905 = +5% on the readout only, MLP 3697). Test MSE, lower better.

## Results

| target | HSiKAN·mean-pool | HSiKAN·concat | MLP |
|---|---|---|---|
| structural (pooled Σ) | **0.0291** | 0.1017 | 0.1489 |
| bag (pooled Σ, separable) | **0.0005** | 0.0010 | 0.0350 |
| **local (node-specific)** | 0.0267 | **0.0024** | 0.0021 |

`local` mean-pool / concat = **10.9×**. Figure: `reports/structural_probe/readout_ablation.png`.

### Reading

- **Measured.** On the node-specific target, mean-pool HSiKAN is **10.9× worse than concat HSiKAN with the
  identical backbone**, and ~13× worse than the MLP. On the two *pooled* targets, mean-pool is best (it
  matches the aggregate structure); concat is somewhat worse there, the MLP worst.
- **Inferred.** The effect is the **readout, not the backbone** (the backbone is byte-identical across the
  pool/concat columns; only the readout and its +5% params differ — far too small to explain 10.9×). Mean-pool
  is a permutation-invariant collapse: optimal for aggregates ("total energy", "any joint near limit"),
  structurally unable to carry "what is node *k* doing". The MLP, reading the flat per-node obs, never loses
  node identity — so it is unhandicapped on exactly the information control needs.
- **Trade-off the data reveals.** Naive `concat` is *worse* than mean-pool on purely-pooled targets
  (structural 0.10 vs 0.03) — it must relearn the aggregation flat. So the fix is **not** naive concat but an
  **identity-preserving *and* aggregation-capable** readout (attention pool, or a concat+pool hybrid).

## Verdict & implication for the robot tie

**Confirmed: HSiKAN's mean-pool readout is the bottleneck for node-specific / coordinated information** — the
exact regime robot control lives in. This is a coherent, evidence-backed cause for "HSiKAN ties/loses to MLP
on robot tasks" that requires no backbone bug (the structural probe already cleared the backbone) and no
wiring fault (the audit cleared the wiring). The chain is now closed:

1. wiring — clean (`hsikan-wiring-audit`);
2. backbone forward — correct (`structural-probe`);
3. **readout — collapses control-relevant node identity (this report).**

## Next step (RL, the payoff test)

Swap the RL backbone's mean-pool for a non-collapsing readout and re-run one robot task (galambos or
quadruped), params-matched vs the MLP. `signed_kan.SignedKANBackbone` exposes `node_activations` (the
non-collapsing seam) but `POOL_MODES` is only `("mean","sum")` — a non-collapsing/attention readout is a
small **non-core** addition in `signed_kan` (or a wrapper in `hymeko_rl`). Prediction: the robot tie narrows
or flips. If it does, the architecture story is complete; if not, the remaining suspect is RL optimisation
noise at the small obs sizes.

## Files touched (CORE.YAML: none — `signed_kan` consumed via `node_activations`, no edit)

- **Extended** `hymeko_rl/structural_probe.py` — `make_dataset` `kind="local"`; `_NodeConcat` readout +
  `readout=` on `build_model`; `run_readout_ablation` + `plot_ablation`; CLI `--ablation`.
- **Extended** `hymeko_rl/tests/test_structural_probe.py` — local target, concat readout, ablation smoke
  (**10 tests pass**; ruff clean).

## Provenance

- Reproduce: `python -m hymeko_rl.structural_probe --ablation --hidden 32 --seeds 5 --n-train 512
  --epochs 300`. Deterministic (seed before each build). Git branch `fix-hsikan`, tree dirty. Windows 11,
  Python 3.12, torch CPU. Seeds: data 1000+s, train s∈0..4.
