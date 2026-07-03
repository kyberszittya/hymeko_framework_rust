# Report — the signed-cycle transductive leak is a direct-label channel, and R_topo is a better protocol

**Date:** 2026-06-15
**Slug:** `cycle-leak-reachability-verdict`
**Author:** Csaba Hajdu
**Context:** Nature leakage paper · reachability-rules framework
(`docs/plans/2026-06-14-reachability-rules-audit-pgraph/`)

## Summary

The signed-cycle method (`cell_signed_graph` + HSiKAN, full-graph cycle
σ-products) reproduces the documented transductive leak — held-out AUROC stays at
**0.735** under label-shuffle. A new **R_topo** intervention (keep the test edge's
cycles in the pool as *topology* but mask its sign to the product identity in the
σ-products) isolates the mechanism: the leak collapses to **chance (0.467)**. So
the leak is the **direct σ-sign channel** (the held-out sign entering its own
cycle features), **not structural**. A second result falls out: cycle topology is
a *legitimate, non-leaking* feature (R_topo real 0.806), and the strict protocol
over-corrects by discarding it (0.500). **R_topo is therefore a better
protocol than strict** — it retains the legitimate structural signal while being
provably leak-free.

## Result (HSiKAN cycle method, bitcoin_alpha, 60 ep, max_k4=20000)

| protocol (reachability rule) | real AUROC | shuffled AUROC | reading |
|---|---|---|---|
| **strict** (exclude all test-edge cycles) | 0.500 | 0.500 | no features → degenerate |
| **topo** (keep test-edge cycles, sign withheld from σ) | **0.806** | **0.467** | strong feature, **no leak** |
| **full** (test-edge cycles with true signs) | 0.901 | **0.735** | **leaks** (+23.5 pp) |

The three rules instantiate the lattice $R_{\text{strict}} \sqsubseteq R_{\text{topo}}
\sqsubseteq R_{\text{full}}$: the leak (shuffled AUROC above chance) appears **only**
at $R_{\text{full}}$, where the held-out label is reachable in the σ-product.

## Interpretation

1. **The leak is direct, not structural.** Under shuffle, train signs are
   permuted but test signs stay intact in `g.signs`. A cycle through a test edge
   carries that edge's true sign in its σ-product; the model reads it back
   (0.735). Masking *only* that sign (R_topo) — topology unchanged — sends it to
   chance (0.467). The held-out sign was leaking through its own features.
2. **Topology is a clean feature.** R_topo real = 0.806: with the test edge's
   cycles present (so it *has* features) but their σ carrying no test sign, the
   model still learns well on real labels and is at chance under shuffle. So the
   cycle *structure* is legitimate, usable, and non-leaking.
3. **Strict over-corrects.** strict = 0.500 on *both* arms because it excludes
   every cycle through a test edge, leaving it featureless — it throws out the
   legitimate topology signal along with the leak.
4. **R_topo is the recommended protocol.** It is the lattice point that maximises
   legitimate reachability (topology) while keeping the label unreachable —
   0.806 real, leak-free. This is a *constructive* contribution, not just a
   diagnosis: a signed-link benchmark can use R_topo to admit cycle/structural
   features honestly.

## Connection to the broader findings

- **Readout locality (2026-06-14).** This cycle method has a *local* readout (a
  test edge's own σ-products are direct features), so it leaks at $R_{\text{full}}$
  where the diffuse node-embedding/attention baselines did not. But even this
  local-readout method does **not** leak at $R_{\text{topo}}$ — confirming the
  two-factor account: a leak needs the *label* reachable (not merely topology)
  **and** a local readout to expose it.
- **5-seed baseline R_topo (2026-06-14):** baselines show no structural leak
  (topo ≈ strict). Together: structural *reachability* (topology) is clean across
  the board; *label* reachability ($R_{\text{full}}$) is the leak, and only
  local-readout methods exploit it.

## Files touched

| Path | Action | Lines |
|---|---|---|
| `hymeko_neuro/runtime/runtime_config.py` | add `reach_topo` field + `HSIKAN_REACH_TOPO` env parse (mirrors `strict_protocol`) | +2 |
| `hymeko_neuro/experiments/runs/run_final_cell.py` | R_topo mask: `g.signs[te_idx]=1` after eval target captured; guard vs strict | +13 |

## CORE.YAML items touched

**None.** `hymeko_neuro/` is not in `CORE.YAML`. Additive, env-gated (default
off → all prior runs bit-identical), reversible.

## Method / verification

- `HSIKAN_STRICT_PROTOCOL=1` → strict; unset → full; `HSIKAN_REACH_TOPO=1` (strict
  off) → topo. Eval target `s_te`/`y_te` captured from true signs *before* the
  mask, so evaluation is unaffected; only cycle σ-products lose the test sign.
- `ruff` on the two edited files: the `runtime_config` change is clean; the
  `run_final_cell` additions are clean (the file's other ~70 ruff findings are
  pre-existing in this 1300-line legacy experiment script, not introduced here).
- Single seed (0), single dataset (bitcoin_alpha), 60 ep — a decisive *mechanism*
  demonstration. Open follow-up: 5-seed × multi-dataset R_topo for the cycle
  method to put error bars on the table (the env flags make this a grid sweep).

## Open issues / follow-ups

1. **Multi-seed / multi-dataset** R_topo for the cycle method (error bars).
2. **Paper wiring**: this three-way (strict/topo/full) is the figure that proves
   the leak is the σ-sign channel and motivates R_topo as the protocol — strong
   candidate for the Nature audit's central mechanism figure.

## Provenance

Git SHA: working tree dirty. Host: Windows 11, RTX 3070 Laptop, torch 2.12+cu132.
Runs: `cell_signed_graph(bitcoin_alpha, HSiKAN, hidden=16, n_epochs=60,
max_k4=20000, seed=0)` under the three env configs; shuffle = `--shuffle-train-signs`
(`seed+100003`).
