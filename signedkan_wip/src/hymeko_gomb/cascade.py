"""Full HymeKo-Gömb cascade composer + ablation wrappers + mixed-arity.

Three model classes that consume the shells defined in `shells.py`:
  - `HymeKoGomb`     : the full three-shell cascade (the plan's
                      mainline architecture)
  - `GombNoOuter` / `GombNoMiddle` / `GombNoInner`  : one-shell-dropped
                      ablation models — separate model classes, not
                      forward-time flags, per plan §Sequencing
  - `MixedArityGomb` : one full stack per cycle arity in `cycle_ks`
                      with learned αₖ softmax fusion of edge logits
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_enumeration import SLOT_K
from .shells import InnerCPMLCore, MiddleHSiKAN, OuterFIRShell


@dataclass
class GombConfig:
    """HymeKo-Gömb configuration."""
    n_nodes: int = 0
    d_embed: int = 32
    d_outer: int = 16
    M_outer: int = 8
    d_middle: int = 32
    d_core: int = 32
    n_tiers: int = 3
    cycle_k: int = 3
    middle_grid: int = 5
    d_predictor_hidden: int = 32
    #: CPML readout inside ``InnerCPMLCore``: ``route`` (default) vs legacy
    #: ``pyramid`` (wider tier aggregators + larger activation tensors).
    cpml_topology: Literal["route", "pyramid"] = "route"
    #: How cycles are assigned to CPML tiers: hard **structural** incidence
    #: vs **capsule_soft** learned softmax routing (``route`` topology only).
    cpml_tier_organization: Literal["structural", "capsule_soft"] = "structural"
    #: Router MLP hidden dim when ``cpml_capsule_soft_router`` resolves to
    #: ``mlp_softmax`` (see ``CPMLConfig``).
    cpml_capsule_route_hidden: int = 64
    cpml_capsule_routing_iterations: int = 1
    cpml_capsule_soft_router: Literal[
        "auto", "mlp_softmax", "hypergraph_conv", "em_agreement",
    ] = "auto"
    cpml_capsule_hg_hidden: int = 64
    #: Cache ``D_v`` for ``hypergraph_conv`` when the same ``cycles`` tensor is reused.
    cpml_capsule_hg_cache_degrees: bool = True
    #: ``torch.compile`` the hypergraph routing head (PyTorch 2.x).
    cpml_torch_compile_hypergraph: bool = False

    # ─── Stacked-middle (2026-05-20) ───────────────────────────────
    #: Depth of the middle HSIKAN stack. 1 (default) = the original
    #: single-tier :class:`MiddleHSiKAN`. >= 2 dispatches to
    #: :class:`StackedMiddleHSiKAN` with the configured depth.
    middle_n_layers: int = 1
    #: Inner-skip kind for the stacked middle's per-layer gate
    #: (only used when ``middle_n_layers > 1``).
    middle_inner_skip: Literal[
        "highway", "cr_highway", "residual", "none", "auto",
    ] = "highway"
    #: JK aggregation across the stacked middle's L layers. ``last``
    #: keeps the output dim at ``d_middle``; ``concat`` widens to
    #: ``L * d_middle`` and the next-shell input dim adjusts.
    middle_jk_mode: Literal["last", "sum", "concat"] = "last"
    #: Share parameters across the stacked middle's L layers (saves
    #: params; matches the HSIKAN-Optuna SOTA's share_weights=True).
    middle_share_weights: bool = False

    # ─── Outer HSIKAN backbone (2026-05-20, Phase “outer HSIKAN”) ──
    #: Depth of the outer HSIKAN backbone that sits BEFORE Gömb's
    #: Clifford-FIR layer. 0 (default) = no outer HSIKAN (use the
    #: existing :class:`HymeKoGomb` class). $\geq 1$ creates a
    #: :class:`GombWithOuterHSIKAN` where the FIR layer sees an
    #: HSIKAN-refined embedding instead of a learned
    #: :class:`nn.Embedding`. (Note: this is a SEPARATE class —
    #: ``HymeKoGomb`` itself is unaware of the outer HSIKAN.)
    outer_hsikan_n_layers: int = 0
    outer_hsikan_inner_skip: Literal[
        "highway", "cr_highway", "residual", "none", "auto",
    ] = "highway"
    outer_hsikan_jk_mode: Literal["last", "sum", "concat"] = "last"
    outer_hsikan_share_weights: bool = False
    #: Wrap the outer HSIKAN's forward in
    #: ``torch.utils.checkpoint.checkpoint`` so backward recomputes
    #: the stack instead of keeping all per-layer intermediates
    #: alive. Necessary for outer_hsikan_n_layers=4 on Slashdot
    #: (where the autograd graph + Gömb cascade exceed 7.6 GiB).
    outer_hsikan_grad_checkpoint: bool = False


# ─── Full three-shell cascade ───────────────────────────────────────


class HymeKoGomb(nn.Module):
    """The three-shell cascade.

    Forward:
        x_embed → outer_shell → middle_shell → inner_core → edge_predictor

    Forward signature:
        cycles         : (M_c, k) long
        signs          : (M_c, k)
        tier_of        : (N,) long
        edges_to_score : (E, 2) long
        return         : (E,) edge sign logits
    """

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)

        self.outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        middle_in = cfg.d_embed + outer_out
        if cfg.middle_n_layers <= 1:
            self.middle = MiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in,
                d_layer=cfg.d_middle,
                cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
            )
            middle_out = cfg.d_middle
        else:
            from .shells import StackedMiddleHSiKAN
            self.middle = StackedMiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in,
                d_layer=cfg.d_middle,
                n_layers=cfg.middle_n_layers,
                cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
                inner_skip=cfg.middle_inner_skip,
                jk_mode=cfg.middle_jk_mode,
                share_weights=cfg.middle_share_weights,
            )
            middle_out = self.middle.d_out
        core_in = cfg.d_embed + outer_out + middle_out
        self.core = InnerCPMLCore(
            d_in=core_in, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
            topology=cfg.cpml_topology,
            tier_organization=cfg.cpml_tier_organization,
            capsule_route_hidden=cfg.cpml_capsule_route_hidden,
            capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
            capsule_soft_router=cfg.cpml_capsule_soft_router,
            capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
            capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
            torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
        )

    def forward(
        self,
        cycles: torch.Tensor,
        signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self.middle(x_for_middle, cycles, signs)
        x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
        scores, _ = self.core(x_for_core, cycles, signs, tier_of, edges_to_score)
        return scores

    def encode_per_vertex(
        self,
        cycles: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-vertex feature ``x_for_core`` that would
        normally feed the CPML core.

        Shape: ``(N, d_embed + M_outer · d_outer + d_middle)``.

        Used by :class:`GombBridgeGomb` to compose two Gömb
        cortices via an HSIKAN bridge: Gömb_1's
        ``encode_per_vertex`` output becomes the bridge input;
        the bridge's output feeds Gömb_2's input embedding.
        """
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self.middle(x_for_middle, cycles, signs)
        return torch.cat([x_embed, x_outer, x_middle], dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Outer HSIKAN → Clifford-FIR → Gömb (2026-05-20) ────────────────


class GombWithOuterHSIKAN(nn.Module):
    """Three-shell Gömb cascade preceded by a multi-layer HSIKAN
    backbone. The HSIKAN backbone refines a per-vertex embedding
    via :class:`MultiLayerSignedKAN`; its final per-vertex
    activation is fed into Gömb's Clifford-FIR layer as the
    "input embedding". Gömb's outer / middle / inner shells are
    used unchanged --- this is structurally different from the
    in-Gömb stacked-middle variant (which falsified on Bitcoin
    Alpha + Slashdot earlier today).

    Forward signature (matches :class:`HymeKoGomb`)::

        cycles          : (M_c, k) long
        signs           : (M_c, k)
        tier_of         : (N,) long
        edges_to_score  : (E, 2) long
        return          : (E,) edge sign logits
    """

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        if cfg.outer_hsikan_n_layers <= 0:
            raise ValueError(
                "GombWithOuterHSIKAN requires "
                "outer_hsikan_n_layers >= 1; got "
                f"{cfg.outer_hsikan_n_layers}. (Use HymeKoGomb for "
                "the no-outer-backbone case.)"
            )
        # Local imports to avoid circulars at module init.
        from ..core.signedkan import (
            MultiLayerSignedKAN, MultiLayerSignedKANConfig,
        )
        self.cfg = cfg

        # 1) Outer HSIKAN backbone — owns its own nn.Embedding,
        # outputs per-vertex features of shape (N, d_embed_eff).
        # With jk_mode="last" (default), d_embed_eff = cfg.d_embed.
        # With "concat", d_embed_eff = L * cfg.d_embed.
        self.outer_hsikan = MultiLayerSignedKAN(
            MultiLayerSignedKANConfig(
                n_nodes=cfg.n_nodes,
                n_layers=cfg.outer_hsikan_n_layers,
                hidden_dim=cfg.d_embed,
                grid=cfg.middle_grid, k=cfg.cycle_k,
                spline_kinds=["catmull_rom"] * cfg.outer_hsikan_n_layers,
                init_scale=0.05, pool_mode="sum",
                jk_mode=cfg.outer_hsikan_jk_mode,
                layer_norm_between=True,
                share_weights=cfg.outer_hsikan_share_weights,
                inner_skip=cfg.outer_hsikan_inner_skip,
                outer_skip="none",
                use_residual=True,
            )
        )
        # Effective d_embed at the FIR interface. The HSIKAN's
        # ``return_h_v=True`` always returns the LAST-LAYER vertex
        # embedding of shape ``(V, d_embed)`` — jk_mode only affects
        # the per-triad output inside the stack, which we don't use
        # here. So d_embed_eff = d_embed regardless of jk_mode. We
        # still keep jk_mode as a config knob since it affects the
        # stack's internal per-triad processing (which the per-layer
        # vertex update reads).
        d_embed_eff = cfg.d_embed
        self._d_embed_eff = d_embed_eff

        # ── Highway-gated residual composition (2026-05-20 v2) ──
        # The substitutive version (x_embed = HSIKAN_only) was null
        # on Bitcoin Alpha + negative on Slashdot d=2. The Clifford-
        # FIR layer was tuned for a learned embedding, so replacing
        # that embedding silently degenerates the architecture. The
        # fix is a HIGHWAY-GATED RESIDUAL:
        #     x_embed = (1 - g) · base + g · HSIKAN_refined
        # with per-channel learnable gate g ∈ (0, 1), biased low at
        # init (sigmoid(-3) ≈ 0.05). At step 0 the model is
        # effectively plain Gömb (base dominates); training can lift
        # g toward 1 per channel if HSIKAN's refinement helps.
        # This matches the inner_skip="highway" pattern that's been
        # the productive lever across phase 21/22 + arc-weight work.
        self.base_node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.base_node_embed.weight, std=0.1)
        self.hsikan_gate_logit = nn.Parameter(
            torch.full((cfg.d_embed,), -3.0, dtype=torch.float32),
        )

        # 2) Gömb cascade — same shells as HymeKoGomb, but d_in
        # adapts to the outer HSIKAN's output width.
        self.outer = OuterFIRShell(
            d_in=d_embed_eff, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        middle_in = d_embed_eff + outer_out
        if cfg.middle_n_layers <= 1:
            self.middle = MiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in,
                d_layer=cfg.d_middle,
                cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
            )
            middle_out = cfg.d_middle
        else:
            from .shells import StackedMiddleHSiKAN
            self.middle = StackedMiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in,
                d_layer=cfg.d_middle,
                n_layers=cfg.middle_n_layers,
                cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
                inner_skip=cfg.middle_inner_skip,
                jk_mode=cfg.middle_jk_mode,
                share_weights=cfg.middle_share_weights,
            )
            middle_out = self.middle.d_out
        core_in = d_embed_eff + outer_out + middle_out
        self.core = InnerCPMLCore(
            d_in=core_in, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
            topology=cfg.cpml_topology,
            tier_organization=cfg.cpml_tier_organization,
            capsule_route_hidden=cfg.cpml_capsule_route_hidden,
            capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
            capsule_soft_router=cfg.cpml_capsule_soft_router,
            capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
            capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
            torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
        )

    @property
    def node_embed(self):
        """Expose the BASE node embedding (the one Gömb's cascade
        directly consumes via the residual mix) for callers that
        read ``model.node_embed.weight``. The outer HSIKAN has its
        own ``nn.Embedding`` inside ``self.outer_hsikan``."""
        return self.base_node_embed

    def _outer_hsikan_h_v(self, cycles, signs):
        """Run the outer HSIKAN backbone and return its final
        per-vertex features ``(N, d_embed_eff)``.

        Caches ``M_vt`` (and the long-cast cycles/signs) by the
        cycles tensor's ``data_ptr`` so 60+ training forwards reuse
        the same scipy CSR + torch sparse tensor instead of
        rebuilding it every step. The cycles tensor is invariant
        during training, so the cache key is stable. The CPU
        round-trip (cycles → numpy → scipy CSR → torch sparse)
        otherwise happens once per forward — Python overhead per
        step plus GPU memory churn that fragments the allocator,
        squeezing peak headroom on deeper stacks.
        """
        cycles_l = cycles.long()
        device = self.outer_hsikan.node_embed.weight.device
        key = (cycles_l.data_ptr(), int(cycles_l.shape[0]),
               int(cycles_l.shape[1]))
        cache = getattr(self, "_m_vt_cache", None)
        if cache is None or cache[0] != key:
            from ..core.signedkan import build_vertex_triad_incidence
            cycles_np = cycles_l.detach().cpu().numpy()
            M_vt = build_vertex_triad_incidence(
                cycles_np, self.cfg.n_nodes, device, mode="sum",
            )
            signs_long = (signs.long() if signs.dtype != torch.long
                            else signs)
            self._m_vt_cache = (key, M_vt, signs_long)
        else:
            _, M_vt, signs_long = cache

        # Outer grad-checkpoint (Phase 22 pattern). When enabled and
        # in training mode with grad on, wrap encode_triads so
        # forward stores no intermediate activations and backward
        # recomputes the L-layer stack. Peak backward memory falls
        # from O(L · per-layer-state) to O(per-layer-state).
        use_outer_ckpt = (
            getattr(self.cfg, "outer_hsikan_grad_checkpoint", False)
            and self.training and torch.is_grad_enabled()
        )

        def _run(_anchor):
            # ``_anchor`` is the HSIKAN's node embedding weight, used
            # only to satisfy checkpoint's "at least one tensor arg
            # with requires_grad" requirement. The actual computation
            # uses closure-captured ``cycles_l, signs_long, M_vt``.
            _, h_v_local = self.outer_hsikan.encode_triads(
                cycles_l, signs_long, M_vt, return_h_v=True,
            )
            return h_v_local

        if use_outer_ckpt:
            anchor = self.outer_hsikan.node_embed.weight
            h_v = torch.utils.checkpoint.checkpoint(
                _run, anchor, use_reentrant=False,
            )
        else:
            h_v = _run(self.outer_hsikan.node_embed.weight)
        return h_v

    def forward(
        self,
        cycles: torch.Tensor,
        signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Outer HSIKAN: refine per-vertex embedding, then mix
        # with the learned base embedding via a per-channel highway
        # gate (sigmoid-bounded, biased low at init so plain Gömb
        # is the starting point).
        hsikan_h = self._outer_hsikan_h_v(cycles, signs)
        g = torch.sigmoid(self.hsikan_gate_logit).unsqueeze(0)  # (1, d_embed)
        x_embed = (1.0 - g) * self.base_node_embed.weight + g * hsikan_h
        # 2. Clifford-FIR layer (= Gömb's outer shell) — the
        # interface between the HSIKAN backbone and the cortical
        # cascade.
        x_outer = self.outer(x_embed, cycles, signs)
        # 3-4. Middle + inner shells.
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self.middle(x_for_middle, cycles, signs)
        x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
        scores, _ = self.core(
            x_for_core, cycles, signs, tier_of, edges_to_score,
        )
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Gömb → HSIKAN bridge → Gömb (two-stage cortex, 2026-05-21) ─────


class GombBridgeGomb(nn.Module):
    """Two Gömb cascades connected through an HSIKAN bridge.

    Stage 1: ``Gömb_1`` runs as :class:`HymeKoGomb` up to the
    pre-CPML per-vertex feature ``x_for_core_1`` (the CPML core
    of Gömb_1 is bypassed — its routing capacity is not used in
    this design; that's a deliberate trade-off, see the plan).

    Stage 2 (bridge): a multi-layer
    :class:`MultiLayerSignedKAN` refines the (projected)
    ``x_for_core_1`` into a per-vertex embedding of width
    ``cfg.d_embed``.

    Stage 3 (mix): highway-gated residual into Gömb_2's input
    embedding — same productive pattern as
    :class:`GombWithOuterHSIKAN`:
    ``x_embed_2 = (1-g)·base_2 + g·bridge_h``, ``g`` per-channel,
    init at sigmoid(-3) ≈ 0.05.

    Stage 4: ``Gömb_2`` runs the standard cascade from
    ``x_embed_2``; its CPML produces the final edge logits.

    Bridge depth is configured via the existing
    :class:`GombConfig` ``outer_hsikan_*`` fields — they
    semantically describe an HSIKAN backbone, which is what
    the bridge is.
    """

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        if cfg.outer_hsikan_n_layers <= 0:
            raise ValueError(
                "GombBridgeGomb requires outer_hsikan_n_layers >= 1 "
                "(the bridge depth). Use HymeKoGomb for no-bridge."
            )
        from ..core.signedkan import (
            MultiLayerSignedKAN, MultiLayerSignedKANConfig,
        )
        self.cfg = cfg

        # ── Stage 1: Gömb_1 (standard cascade) ──────────────
        # We use a plain HymeKoGomb but only invoke its
        # ``encode_per_vertex`` method — the CPML core's params
        # exist but never see gradient (unused). Saves the
        # parameters: build only the outer + middle shells.
        self.g1_node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.g1_node_embed.weight, std=0.1)
        self.g1_outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        g1_middle_in = cfg.d_embed + outer_out
        self.g1_middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=g1_middle_in,
            d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        # ``x_for_core_1`` has dim:
        g1_x_dim = cfg.d_embed + outer_out + cfg.d_middle

        # ── Stage 2: HSIKAN bridge ──────────────────────────
        # Project the wide x_for_core_1 down to cfg.d_embed for
        # the bridge's input.
        self.bridge_pre = nn.Linear(g1_x_dim, cfg.d_embed)
        self.bridge_hsikan = MultiLayerSignedKAN(
            MultiLayerSignedKANConfig(
                n_nodes=cfg.n_nodes,
                n_layers=cfg.outer_hsikan_n_layers,
                hidden_dim=cfg.d_embed,
                grid=cfg.middle_grid, k=cfg.cycle_k,
                spline_kinds=["catmull_rom"] * cfg.outer_hsikan_n_layers,
                init_scale=0.05, pool_mode="sum",
                jk_mode=cfg.outer_hsikan_jk_mode,
                layer_norm_between=True,
                share_weights=cfg.outer_hsikan_share_weights,
                inner_skip=cfg.outer_hsikan_inner_skip,
                outer_skip="none",
                use_residual=True,
            )
        )

        # ── Stage 3: highway-gated residual into Gömb_2 ─────
        self.g2_base_node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.g2_base_node_embed.weight, std=0.1)
        self.bridge_gate_logit = nn.Parameter(
            torch.full((cfg.d_embed,), -3.0, dtype=torch.float32),
        )

        # ── Stage 4: Gömb_2 (full cascade) ──────────────────
        self.g2_outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        g2_middle_in = cfg.d_embed + outer_out
        self.g2_middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=g2_middle_in,
            d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        g2_core_in = cfg.d_embed + outer_out + cfg.d_middle
        self.g2_core = InnerCPMLCore(
            d_in=g2_core_in, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
            topology=cfg.cpml_topology,
            tier_organization=cfg.cpml_tier_organization,
            capsule_route_hidden=cfg.cpml_capsule_route_hidden,
            capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
            capsule_soft_router=cfg.cpml_capsule_soft_router,
            capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
            capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
            torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
        )

    @property
    def node_embed(self):
        """Expose Gömb_2's base embedding (the one the cascade
        consumes via the residual mix)."""
        return self.g2_base_node_embed

    def _encode_g1_per_vertex(self, cycles, signs):
        x_e1 = self.g1_node_embed.weight
        x_o1 = self.g1_outer(x_e1, cycles, signs)
        x_for_mid_1 = torch.cat([x_e1, x_o1], dim=-1)
        x_m1 = self.g1_middle(x_for_mid_1, cycles, signs)
        return torch.cat([x_e1, x_o1, x_m1], dim=-1)

    def _bridge(self, x_for_core_1, cycles, signs):
        """Project + HSIKAN bridge → per-vertex (N, d_embed)."""
        from ..core.signedkan import build_vertex_triad_incidence
        bridge_in = self.bridge_pre(x_for_core_1)  # (N, d_embed)
        cycles_l = cycles.long()
        device = bridge_in.device
        # Cache M_vt by cycles.data_ptr (same pattern as
        # GombWithOuterHSIKAN; cycles are invariant across training).
        key = (cycles_l.data_ptr(), int(cycles_l.shape[0]),
               int(cycles_l.shape[1]))
        cache = getattr(self, "_bridge_m_vt_cache", None)
        if cache is None or cache[0] != key:
            cycles_np = cycles_l.detach().cpu().numpy()
            M_vt = build_vertex_triad_incidence(
                cycles_np, self.cfg.n_nodes, device, mode="sum",
            )
            signs_long = (signs.long() if signs.dtype != torch.long
                           else signs)
            self._bridge_m_vt_cache = (key, M_vt, signs_long)
        else:
            _, M_vt, signs_long = cache
        _, h_v = self.bridge_hsikan.encode_triads(
            cycles_l, signs_long, M_vt,
            initial_h_v=bridge_in, return_h_v=True,
        )
        return h_v

    def forward(self, cycles, signs, tier_of, edges_to_score):
        # 1. Gömb_1 → x_for_core_1.
        x_for_core_1 = self._encode_g1_per_vertex(cycles, signs)
        # 2. HSIKAN bridge → per-vertex (N, d_embed).
        bridge_h = self._bridge(x_for_core_1, cycles, signs)
        # 3. Highway-gated residual mix with Gömb_2's base embedding.
        g = torch.sigmoid(self.bridge_gate_logit).unsqueeze(0)
        x_e2 = (1.0 - g) * self.g2_base_node_embed.weight + g * bridge_h
        # 4. Gömb_2 cascade.
        x_o2 = self.g2_outer(x_e2, cycles, signs)
        x_for_mid_2 = torch.cat([x_e2, x_o2], dim=-1)
        x_m2 = self.g2_middle(x_for_mid_2, cycles, signs)
        x_for_core_2 = torch.cat([x_e2, x_o2, x_m2], dim=-1)
        scores, _ = self.g2_core(
            x_for_core_2, cycles, signs, tier_of, edges_to_score,
        )
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Ablation wrappers ──────────────────────────────────────────────


class GombNoOuter(nn.Module):
    """Cascade without the Outer FIR volume."""

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=cfg.d_embed, d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        self.core = InnerCPMLCore(
            d_in=cfg.d_embed + cfg.d_middle, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
            topology=cfg.cpml_topology,
            tier_organization=cfg.cpml_tier_organization,
            capsule_route_hidden=cfg.cpml_capsule_route_hidden,
            capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
            capsule_soft_router=cfg.cpml_capsule_soft_router,
            capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
            capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
            torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
        )

    def forward(self, cycles, signs, tier_of, edges_to_score):
        x_embed = self.node_embed.weight
        x_middle = self.middle(x_embed, cycles, signs)
        x_for_core = torch.cat([x_embed, x_middle], dim=-1)
        scores, _ = self.core(x_for_core, cycles, signs, tier_of, edges_to_score)
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GombNoMiddle(nn.Module):
    """Cascade without the Middle HSiKAN shell."""

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        self.core = InnerCPMLCore(
            d_in=cfg.d_embed + outer_out, d_layer=cfg.d_core,
            n_tiers=cfg.n_tiers, cycle_k=cfg.cycle_k,
            topology=cfg.cpml_topology,
            tier_organization=cfg.cpml_tier_organization,
            capsule_route_hidden=cfg.cpml_capsule_route_hidden,
            capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
            capsule_soft_router=cfg.cpml_capsule_soft_router,
            capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
            capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
            torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
        )

    def forward(self, cycles, signs, tier_of, edges_to_score):
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_core = torch.cat([x_embed, x_outer], dim=-1)
        scores, _ = self.core(x_for_core, cycles, signs, tier_of, edges_to_score)
        return scores

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GombNoInner(nn.Module):
    """Cascade without the Inner CPML core; plain MLP edge head."""

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg
        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.outer = OuterFIRShell(
            d_in=cfg.d_embed, d_layer=cfg.d_outer,
            M=cfg.M_outer, cycle_k=cfg.cycle_k,
        )
        outer_out = cfg.M_outer * cfg.d_outer
        self.middle = MiddleHSiKAN(
            n_nodes=cfg.n_nodes, d_in=cfg.d_embed + outer_out,
            d_layer=cfg.d_middle,
            cycle_k=cfg.cycle_k, grid=cfg.middle_grid,
        )
        final_dim = cfg.d_embed + outer_out + cfg.d_middle
        self.head = nn.Sequential(
            nn.Linear(2 * final_dim, cfg.d_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_predictor_hidden, 1),
        )

    def forward(self, cycles, signs, tier_of, edges_to_score):
        del tier_of   # no CPML tier dispatch in this ablation
        x_embed = self.node_embed.weight
        x_outer = self.outer(x_embed, cycles, signs)
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self.middle(x_for_middle, cycles, signs)
        x_final = torch.cat([x_embed, x_outer, x_middle], dim=-1)
        u = x_final[edges_to_score[:, 0]]
        v = x_final[edges_to_score[:, 1]]
        return self.head(torch.cat([u, v], dim=-1)).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Mixed-arity (k=3+k=4, k=4+k=5, …) ──────────────────────────────


class MixedArityGomb(nn.Module):
    """One full (outer, middle, inner) stack per cycle arity in `cycle_ks`,
    with learned αₖ softmax fusion of edge logits.

    Memory `project_phase9_k45_sweet_spot_2026_05_02`: k=4+k=5 mixed
    with learned αₖ Pareto-dominates single-arity k=3 on every signed
    dataset.

    Forward signature:
        cycles_by_k    : {k -> (M_c_k, k) long}
        signs_by_k     : {k -> (M_c_k, k) float}
        tier_of        : (N,) long
        edges_to_score : (E, 2) long
        return         : (E,) edge sign logits
    """

    def __init__(self, cfg: GombConfig, cycle_ks: tuple[int, ...] = (3, 4)):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        if len(cycle_ks) < 1:
            raise ValueError("cycle_ks must contain at least one arity")
        self.cfg = cfg
        self.cycle_ks = tuple(cycle_ks)

        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)

        outer_out = cfg.M_outer * cfg.d_outer
        middle_in = cfg.d_embed + outer_out
        core_in = cfg.d_embed + outer_out + cfg.d_middle

        self.outers = nn.ModuleDict()
        self.middles = nn.ModuleDict()
        self.cores = nn.ModuleDict()
        for k in self.cycle_ks:
            sk = str(k)
            self.outers[sk] = OuterFIRShell(
                d_in=cfg.d_embed, d_layer=cfg.d_outer,
                M=cfg.M_outer, cycle_k=k,
            )
            self.middles[sk] = MiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in, d_layer=cfg.d_middle,
                cycle_k=k, grid=cfg.middle_grid,
            )
            self.cores[sk] = InnerCPMLCore(
                d_in=core_in, d_layer=cfg.d_core,
                n_tiers=cfg.n_tiers, cycle_k=k,
                topology=cfg.cpml_topology,
                tier_organization=cfg.cpml_tier_organization,
                capsule_route_hidden=cfg.cpml_capsule_route_hidden,
                capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
                capsule_soft_router=cfg.cpml_capsule_soft_router,
                capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
                capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
                torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
            )

        self.alpha_logits = nn.Parameter(torch.zeros(len(self.cycle_ks)))

    def forward(
        self,
        cycles_by_k: dict[int, torch.Tensor],
        signs_by_k: dict[int, torch.Tensor],
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.node_embed.weight
        alpha = F.softmax(self.alpha_logits, dim=0)
        out = None
        for i, k in enumerate(self.cycle_ks):
            sk = str(k)
            cycles_k = cycles_by_k[k]
            signs_k = signs_by_k[k]
            x_outer = self.outers[sk](x_embed, cycles_k, signs_k)
            x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
            x_middle = self.middles[sk](x_for_middle, cycles_k, signs_k)
            x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
            scores_k, _ = self.cores[sk](
                x_for_core, cycles_k, signs_k, tier_of, edges_to_score,
            )
            contribution = alpha[i] * scores_k
            out = contribution if out is None else out + contribution
        assert out is not None
        return out

    def alpha(self) -> torch.Tensor:
        """Current αₖ (softmaxed) — diagnostic for per-arity weight tracking."""
        return F.softmax(self.alpha_logits, dim=0).detach()

class JointMixGomb(nn.Module):
    """Gömb with **joint_ba** tuple recipe: slots **c3, c4, w2, w3**.

    One full (outer, middle, inner) stack **per named slot** — same
    pattern as ``MixedArityGomb``, but keys are strings so **c3** and
    **w2** (both width 3) do not collide.  Learned softmax ``α`` fuses
    edge logits across slots (parallel to ``MixedAritySignedKAN``'s
    tuple-slot mixer in ``run_final_cell``).

    Forward:
        cycles_by_slot : {"c3","c4","w2","w3"} → (M, k) long tensors
        signs_by_slot  : same keys → (M, k) float/int8 broadcastable
    """

    SLOTS: tuple[str, ...] = ("c3", "c4", "w2", "w3")

    def __init__(self, cfg: GombConfig):
        super().__init__()
        if cfg.n_nodes <= 0:
            raise ValueError("GombConfig.n_nodes must be set (> 0)")
        self.cfg = cfg

        self.node_embed = nn.Embedding(cfg.n_nodes, cfg.d_embed)
        nn.init.normal_(self.node_embed.weight, std=0.1)

        outer_out = cfg.M_outer * cfg.d_outer
        middle_in = cfg.d_embed + outer_out
        core_in = cfg.d_embed + outer_out + cfg.d_middle

        self.outers = nn.ModuleDict()
        self.middles = nn.ModuleDict()
        self.cores = nn.ModuleDict()
        for slot in self.SLOTS:
            k = SLOT_K[slot]
            self.outers[slot] = OuterFIRShell(
                d_in=cfg.d_embed, d_layer=cfg.d_outer,
                M=cfg.M_outer, cycle_k=k,
            )
            self.middles[slot] = MiddleHSiKAN(
                n_nodes=cfg.n_nodes, d_in=middle_in, d_layer=cfg.d_middle,
                cycle_k=k, grid=cfg.middle_grid,
            )
            self.cores[slot] = InnerCPMLCore(
                d_in=core_in, d_layer=cfg.d_core,
                n_tiers=cfg.n_tiers, cycle_k=k,
                topology=cfg.cpml_topology,
                tier_organization=cfg.cpml_tier_organization,
                capsule_route_hidden=cfg.cpml_capsule_route_hidden,
                capsule_routing_iterations=cfg.cpml_capsule_routing_iterations,
                capsule_soft_router=cfg.cpml_capsule_soft_router,
                capsule_hg_hidden=cfg.cpml_capsule_hg_hidden,
                capsule_hg_cache_degrees=cfg.cpml_capsule_hg_cache_degrees,
                torch_compile_hypergraph=cfg.cpml_torch_compile_hypergraph,
            )

        self.alpha_logits = nn.Parameter(torch.zeros(len(self.SLOTS)))

    def forward(
        self,
        cycles_by_slot: dict[str, torch.Tensor],
        signs_by_slot: dict[str, torch.Tensor],
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.node_embed.weight
        alpha = F.softmax(self.alpha_logits, dim=0)
        out: torch.Tensor | None = None
        for i, slot in enumerate(self.SLOTS):
            cycles_k = cycles_by_slot[slot]
            signs_k = signs_by_slot[slot].float()
            x_outer = self.outers[slot](x_embed, cycles_k, signs_k)
            x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
            x_middle = self.middles[slot](x_for_middle, cycles_k, signs_k)
            x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
            scores_k, _ = self.cores[slot](
                x_for_core, cycles_k, signs_k, tier_of, edges_to_score,
            )
            contribution = alpha[i] * scores_k
            out = contribution if out is None else out + contribution
        assert out is not None
        return out

    def alpha(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0).detach()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "GombConfig",
    "HymeKoGomb",
    "GombNoOuter", "GombNoMiddle", "GombNoInner",
    "MixedArityGomb",
    "JointMixGomb",
]
