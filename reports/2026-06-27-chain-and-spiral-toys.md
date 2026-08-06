# Chain + spiral toys — the structural prior is real (even on a chain), and the highway is a structure-free spiral

**Date:** 2026-06-27 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Extends:** `docs/plans/2026-06-26-structural-probe/` (chain) and `docs/plans/2026-06-26-rotor-spikes-ablation/`
(spiral — its designed next-toy). Both supervised, deterministic (CLAUDE.md §3 strict carve-out).

## 1. Chain — even a chain's kinematic structure is exploitable (user's claim, confirmed)

HSiKAN (sparse signed reasoning over the chain adjacency) vs a params-matched MLP on the structural target,
swept over chain length (`hymeko_rl/structural_probe.py --chain`):

| chain length | HSiKAN | MLP | MLP/HSiKAN |
|---|---|---|---|
| 4 | 0.0007 | 0.0030 | 4.5× |
| 8 | 0.0047 | 0.0490 | 10× |
| 12 | 0.0044 | 0.100 | 23× |
| **16** | **0.0028** | **0.116** | **41×** |

- **HSiKAN beats the MLP on a chain at every length** — the maximally sparse topology and the audit's
  "structure-poor" worst case. So "a chain has no exploitable structure" was wrong.
- **HSiKAN's error is ~flat in length (~0.004); the MLP's explodes (0.003→0.116).** The sparse-reasoning
  signature: local signed messages compute the structural target with cost/error *independent of chain length*;
  the flat MLP brute-forces the `N→target` map and degrades. The advantage **grows ~10× per doubling**.
- Reframes the §0 audit: cart-pole (2 nodes) / arm (7) are *short* chains with non-structural objectives — that
  is why they looked like ties; longer chains with structural targets favour sparse HSiKAN, widening with length.

Figure: `reports/structural_probe/chain_probe.png`.

## 2. Spiral — the highway is a structure-free spiral skeleton (user's conjecture, confirmed)

A θ-graph with `K` parallel walks; per sample the connection `θ` varies and a source `x∈R²` is collected over
the walks: `y = mean_k R(Σ_{e∈W_k} θ_e)·x`. Three models (`hymeko_rl/spiral_probe.py`):

| K (walks) | spiral | highway_mlp | mlp |
|---|---|---|---|
| 1 | **0.0000** | 0.32 | 0.22 |
| 2 | **0.0000** | 0.56 | 0.62 |
| 4 | **0.0000** | 0.59 | 0.47 |
| 8 | **0.0000** | 0.35 | 0.28 |

- **The spiral fits the walk-holonomy exactly** (rotor-transport along the known walks + α-collect, a handful
  of params). The **plain highway fails (0.3–0.6) and is no better than the flat MLP.**
- That is the conjecture proven: the plain highway's *identity carry* carries **no structural/holonomy signal**,
  so it's null — exactly why `skip="highway"` did nothing on the coin task (§3 below). Recast the carry as
  rotor-transport-along-walks and the highway *becomes* the spiral, which nails the holonomy.
- `highway = spiral with identity connection + walk collapsed to layer-depth`. The spiral unifies the three
  loose pieces (per-node readout ✓, rotor connection, spike-walk timing) into one collector with the highway as
  its skeleton. This is the gauge plan's C1 realized *as the skip layer*.

Figure: `reports/spiral_probe/spiral_probe.png`.

## 3. Cross-reference — the galambos highway null (now explained)

The 4-config coin-toss A/B (`2026-06-26-pernode-actor`, single seed 20k): mlp 0.45 / pooled 0.00 / pernode 0.20
/ **pernode_hw (highway) 0.20** (return −158, +5.3k params, no delivery gain). The plain highway was null on
control — consistent with §2: a structure-free carry adds capacity, not signal. The implied fix is the *spiral*
(structure-carrying), not the plain highway.

## Files (CORE.YAML: none)
- `hymeko_rl/structural_probe.py` (`build_chain_graph`, `run_chain_probe`, `plot_chain_probe`, `--chain`),
  `hymeko_rl/spiral_probe.py` (new: θ-graph walks, `SpiralModel`/`HighwayMlp`/`FlatMlp`, `run_spiral_probe`;
  reuses `rotor_probe.rot_matrix`). Tests: `test_structural_probe.py` (+2), `test_spiral_probe.py` (5). ruff clean.

## Next
1. **Overnight:** 4-config coin-toss at 50k steps (`bbajhwgae`) — does longer training lift the conservative
   per-node HSiKAN toward the MLP's 45%?
2. **Spiral in HSiKAN** — the real prize: replace the plain highway carry with rotor-transport-along-walks
   (`CayleyRotor` + `hymeko_graph` walks + α-collect), test on a structural robot reward. Conjecture, untested.
3. Multi-seed the coin-toss for a proper verdict (needs the seed-axis; do it attended).

## Provenance
Reproduce: `python -m hymeko_rl.structural_probe --chain`; `python -m hymeko_rl.spiral_probe`. Git `fix-hsikan`,
tree dirty. Windows 11, Python 3.12, torch CPU, seeds 0–2.
