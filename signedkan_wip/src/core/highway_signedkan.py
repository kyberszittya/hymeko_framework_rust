"""Highway-SignedKAN (HSiKAN).

The named deployment architecture for Bitcoin OTC and similar
mid-scale signed-hypergraph link-sign prediction tasks. Wraps
:class:`MultiLayerSignedKAN` with the canonical configuration found
empirically:

  - $L\\!=\\!3$ stacked layers with **weight-sharing** (one shared
    SignedKANLayer applied 3 times — recurrent multi-layer; the
    "RHN-on-signed-hypergraphs" lineage).
  - **Highway gate** on the inner spline (Schmidhuber-style
    $T(x) = \\sigma(W_T x + b_T)$, bias initialised to $-2$ so $T \\approx 0$
    at training start). The outer spline is unskipped (heterogeneous
    skip placement, the "head-neck-spine" CV analogy).
  - **LayerNorm** on $\\mathbf{h}_v$ between layers.
  - **JK-concat** + **sum-pool** for inter-layer aggregation.
  - **Refined spectral-entropy regulariser** (KL-update normalised
    by $\\log_2(\\mathrm{rank})$, EMA momentum on $\\lambda_{\\mathrm{eff}}$).
    The refined schedule no longer fights the R2 regulariser.
  - **R2 vertex-degree participation regulariser** ($\\lambda\\!=\\!0.05$).
  - **EC** training recipe (early-stop + class-weighted BCE) as the
    foundation.

Empirical Bitcoin OTC numbers (median over 3 seeds): AUC 0.8738
($+0.0038$ over EC), macro-$F_1$ 0.7651 (tied with EC$+R_2$). First
single recipe to be near-best on both metrics simultaneously.

Note: this architecture is **deliberately not used on Bitcoin Alpha**.
The smaller fixture is too small to absorb the multi-layer capacity;
$L\\!=\\!1$ EC$+R_2$ is the recommended deployment recipe there.
The recipe-board in the paper is per-fixture.

Usage::

    cfg = HighwaySignedKANConfig(n_nodes=g.n_nodes, hidden_dim=32)
    model = HighwaySignedKAN(cfg)
    # Identical interface to MultiLayerSignedKAN; build M_vt as usual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .signedkan import (MultiLayerSignedKAN, MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)


@dataclass
class HighwaySignedKANConfig:
    """Canonical Highway-SignedKAN config. The defaults below are the
    empirically-validated deployment values; override only with a
    measured reason."""
    n_nodes: int
    hidden_dim: int = 32
    n_layers: int = 3
    grid: int = 5                     # G=5 stays — grid pruning HURTS OTC
    k: int = 3
    init_scale: float = 0.05          # 0.05 vs 0.1: +0.004 OTC AUC, +0.010 OTC F1m,
                                      # but -0.024 Alpha AUC. HSiKAN is for OTC-class
                                      # fixtures; use L=1 EC+R2 on Alpha-class.
    use_minus_branch: bool = True
    spline_kind: str = "bspline"


class HighwaySignedKAN(MultiLayerSignedKAN):
    """Highway-SignedKAN: the deployment architecture for Bitcoin OTC and
    similar mid-scale signed-hypergraph link-sign prediction.

    Inherits from :class:`MultiLayerSignedKAN`; the only role of this
    subclass is to lock in the canonical configuration so callers do
    not need to reproduce the seven-knob recipe every time.
    """

    def __init__(self, cfg: HighwaySignedKANConfig):
        ml_cfg = MultiLayerSignedKANConfig(
            n_nodes=cfg.n_nodes,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            grid=cfg.grid,
            k=cfg.k,
            use_minus_branch=cfg.use_minus_branch,
            init_scale=cfg.init_scale,
            spline_kinds=[cfg.spline_kind] * cfg.n_layers,
            # The locked-in pieces — the architecture's identity:
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            use_residual=True,         # vertex-side residual between layers
        )
        super().__init__(ml_cfg)

    @staticmethod
    def recommended_training_recipe() -> dict:
        """Return the kwargs dict for ``run_compare.run_one`` that
        reproduces the empirically-validated training recipe for
        Highway-SignedKAN: EC + R2 + refined-entropy + init_scale=0.05.

        Tuning sweep (`hsikan_hpsweep.json`) found ``init_scale=0.05``
        gives +0.004 OTC AUC and +0.010 OTC F1m over the original 0.1
        (cost: -0.024 Alpha AUC, but HSiKAN is OTC-deployment-only).
        ``lr`` stays at 5e-2 (insensitive in [1e-2, 1e-1])."""
        return dict(
            # EC.
            early_stopping=True,
            class_weighted=True,
            val_every=5,
            # R2.
            participation_lam=0.05,
            # Refined spectral-entropy regulariser (kl_normalized + momentum).
            entropy_lam0=0.01,
            entropy_target=0.5,
            entropy_eta=5.0,
            entropy_kl_normalized=True,
            entropy_momentum=0.9,
            # Architectural pieces — duplicated for direct invocation
            # via run_one() without instantiating HighwaySignedKAN.
            n_layers=3,
            spline_kinds=["bspline"] * 3,
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            # Embedding initialisation scale — measured optimum.
            init_scale=0.05,
        )
