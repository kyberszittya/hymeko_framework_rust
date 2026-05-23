"""Side-stacked HSIKAN — Phase 17 (2026-05-20 overnight, "not over").

Phase 16's depth-stacking experiment falsified the ResNet-style
"deeper is better" hypothesis on Bitcoin Alpha; HSIKAN
monotonically degrades with depth at hidden=8 / n_epochs=30.

Phase 17 builds the architectural sister: **parallel branches
instead of depth**. Inspired by ResNeXt's cardinality dimension
and HSIKAN's own mixed-arity result
([[project_hsikan_mixed_arity_2026_05_01]], k=3+k=4+k=5 with
learned αₖ Pareto-dominates SGCN+balance on Bitcoin).

Architecture: $N$ independent :class:`SignedKAN` instances process
the same `(triad_v, triad_sigma)` input; their per-triad embeddings
are fused via one of {sum, mean, concat, learned-alpha,
attention}. Optionally each branch can be configured with a
different spline kind to give the ensemble diverse views of the
triad-activation manifold.

The empirical experiment in
``signedkan_wip/experiments/runs/run_side_vs_depth.py`` compares
side-stacked vs depth-stacked at the same parameter scaling on
Bitcoin Alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .signedkan import SignedKAN, SignedKANConfig


@dataclass
class SideSignedKANConfig:
    """Configuration for the side-stacked HSIKAN.

    `fusion ∈ {"sum", "mean", "concat", "learned_alpha", "attention"}`
    determines how the N branch outputs combine. `sum` and `mean`
    add no fusion parameters; `concat` quadruples the head input
    dimension; `learned_alpha` adds `N` parameters; `attention`
    adds `hidden_dim` parameters.

    `spline_kinds[i]` selects each branch's spline basis when
    provided; otherwise all branches use the same default
    (`"bspline"`). Branch-specific seeds let the user diversify
    initialisation across the ensemble.
    """

    n_nodes: int
    n_branches: int = 4
    hidden_dim: int = 8
    grid: int = 5
    k: int = 3
    use_minus_branch: bool = True
    init_scale: float = 0.1
    fusion: str = "mean"
    spline_kinds: list[str] | None = None
    branch_seeds: list[int] | None = None


class SideSignedKAN(nn.Module):
    """N parallel SignedKAN branches with a fusion head.

    Each branch is an independent :class:`SignedKAN` — they share
    no parameters but process the same triad input. After running
    every branch, the per-triad embeddings are combined according
    to `cfg.fusion`.

    Exposes the same `encode_triads(triad_v, triad_sigma)` API as
    bare `SignedKAN` so existing training harnesses
    (e.g.\\ `run_compare.run_one`) can swap it in.

    .. note::
        Branch outputs are per-triad embeddings of shape
        `(n_triads, hidden_dim)`. The `concat` fusion produces
        `(n_triads, n_branches × hidden_dim)`; all other fusions
        keep the output at `(n_triads, hidden_dim)`.
    """

    def __init__(self, cfg: SideSignedKANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_branches = cfg.n_branches
        self.hidden_dim = cfg.hidden_dim
        self.fusion = cfg.fusion

        # Branch configs — each independent. If branch_seeds is
        # supplied, seed the RNG before each branch's construction
        # so the init differs deterministically across branches.
        self.branches = nn.ModuleList()
        for i in range(cfg.n_branches):
            if cfg.branch_seeds is not None and i < len(cfg.branch_seeds):
                torch.manual_seed(int(cfg.branch_seeds[i]))
            spline_kind = (
                cfg.spline_kinds[i]
                if cfg.spline_kinds is not None and i < len(cfg.spline_kinds)
                else "bspline"
            )
            self.branches.append(SignedKAN(SignedKANConfig(
                n_nodes=cfg.n_nodes,
                hidden_dim=cfg.hidden_dim,
                grid=cfg.grid, k=cfg.k,
                use_minus_branch=cfg.use_minus_branch,
                init_scale=cfg.init_scale,
                spline_kind=spline_kind,
            )))

        # Fusion-head parameters (only for the parametric variants).
        if cfg.fusion == "learned_alpha":
            # One scalar weight per branch; softmax-normalised at
            # forward time.
            self.alpha = nn.Parameter(torch.zeros(cfg.n_branches))
        elif cfg.fusion == "attention":
            # Linear → 1 produces a per-(branch, triad) logit; the
            # softmax across branches gives a per-triad weighting.
            self.attn = nn.Linear(cfg.hidden_dim, 1)
        else:
            self.alpha = None
            self.attn = None

        # Edge classifier — matches the bare `SignedKAN.classifier`
        # interface that `run_compare.run_one` expects. Output dim
        # depends on the fusion mode: `concat` fuses to
        # `n_branches × hidden_dim`; everything else stays at
        # `hidden_dim`.
        clf_in = (cfg.n_branches * cfg.hidden_dim
                  if cfg.fusion == "concat"
                  else cfg.hidden_dim)
        self.classifier = nn.Linear(clf_in, 1)

    @property
    def node_embed(self):
        """First branch's node embedding (back-compat for callers
        that inspect `.node_embed`). Each branch has its own."""
        return self.branches[0].node_embed

    def encode_triads(
        self,
        triad_v: torch.Tensor,
        triad_sigma: torch.Tensor,
        return_h_v: bool = False,
    ):
        """Run every branch and fuse the per-triad outputs.

        Returns a tensor of shape `(n_triads, hidden_dim)` for
        sum/mean/learned_alpha/attention, or
        `(n_triads, n_branches × hidden_dim)` for concat.

        `return_h_v` is provided for back-compat with
        :class:`SignedKAN.encode_triads`'s signature (used by
        bilinear/attention heads in `run_compare`); when True the
        first branch's node embedding is returned as the
        h_v_final placeholder.
        """
        outs = [b.encode_triads(triad_v, triad_sigma)
                for b in self.branches]
        if self.fusion == "sum":
            fused = torch.stack(outs, dim=0).sum(dim=0)
        elif self.fusion == "mean":
            fused = torch.stack(outs, dim=0).mean(dim=0)
        elif self.fusion == "concat":
            fused = torch.cat(outs, dim=-1)
        elif self.fusion == "learned_alpha":
            stacked = torch.stack(outs, dim=0)                  # (N, T, d)
            weights = torch.softmax(self.alpha, dim=0)           # (N,)
            fused = (weights[:, None, None] * stacked).sum(dim=0) # (T, d)
        elif self.fusion == "attention":
            stacked = torch.stack(outs, dim=0)                  # (N, T, d)
            logits = self.attn(stacked).squeeze(-1)              # (N, T)
            weights = torch.softmax(logits, dim=0)               # (N, T)
            fused = (weights.unsqueeze(-1) * stacked).sum(dim=0) # (T, d)
        else:
            raise ValueError(f"unknown fusion mode: {self.fusion!r}")
        if return_h_v:
            return fused, self.branches[0].node_embed.weight
        return fused

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class MembraneSignedKANConfig:
    """Configuration for the membrane-coupled HSIKAN — Phase 18.

    Like :class:`SideSignedKANConfig` but adds a **shared membrane
    latent** that every parallel branch reads from + writes to.
    The membrane is a single per-triad latent (aggregate of all
    branches' first-pass outputs) plus a learned read gate that
    lets each branch ingest the shared signal before its
    contribution is fused.

    Biological analogy: each branch is a "cell"; the membrane is
    the extracellular space they share. One round of message
    passing through the membrane couples otherwise-independent
    branches.
    """

    n_nodes: int
    n_branches: int = 4
    hidden_dim: int = 8
    grid: int = 5
    k: int = 3
    use_minus_branch: bool = True
    init_scale: float = 0.1
    fusion: str = "mean"
    spline_kinds: list[str] | None = None
    branch_seeds: list[int] | None = None
    membrane_aggregator: str = "mean"   # {"mean", "max", "sum"}
    read_gate_init: float = 0.0           # bias on each read gate


class MembraneSignedKAN(nn.Module):
    """Membrane-coupled side-stacked HSIKAN — Phase 18.

    Architecture::

      Step 1:  outs_0[i] = SignedKAN_i.encode_triads(triad_v, triad_sigma)
      Step 2:  z         = aggregator(outs_0[0..N-1])      # (T, d)
      Step 3:  outs_1[i] = outs_0[i] + read_gates[i](z)
      Step 4:  fused     = fusion(outs_1[0..N-1])

    Read gates are per-branch ``nn.Linear(d, d)``; small-weight
    + zero-bias init so the model starts as plain
    :class:`SideSignedKAN` and learns the membrane coupling.
    """

    def __init__(self, cfg: MembraneSignedKANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_branches = cfg.n_branches
        self.hidden_dim = cfg.hidden_dim
        self.fusion = cfg.fusion

        self.branches = nn.ModuleList()
        for i in range(cfg.n_branches):
            if cfg.branch_seeds is not None and i < len(cfg.branch_seeds):
                torch.manual_seed(int(cfg.branch_seeds[i]))
            spline_kind = (
                cfg.spline_kinds[i]
                if cfg.spline_kinds is not None and i < len(cfg.spline_kinds)
                else "bspline"
            )
            self.branches.append(SignedKAN(SignedKANConfig(
                n_nodes=cfg.n_nodes,
                hidden_dim=cfg.hidden_dim,
                grid=cfg.grid, k=cfg.k,
                use_minus_branch=cfg.use_minus_branch,
                init_scale=cfg.init_scale,
                spline_kind=spline_kind,
            )))

        # One read gate per branch. Near-zero init means at step 0
        # the membrane is silent — the model starts as a plain
        # SideSignedKAN and learns the coupling.
        self.read_gates = nn.ModuleList([
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=True)
            for _ in range(cfg.n_branches)
        ])
        for gate in self.read_gates:
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.constant_(gate.bias, float(cfg.read_gate_init))

        if cfg.fusion == "learned_alpha":
            self.alpha = nn.Parameter(torch.zeros(cfg.n_branches))
        elif cfg.fusion == "attention":
            self.attn = nn.Linear(cfg.hidden_dim, 1)
        else:
            self.alpha = None
            self.attn = None

        # Edge classifier — matches the bare `SignedKAN.classifier`
        # interface (see :class:`SideSignedKAN`).
        clf_in = (cfg.n_branches * cfg.hidden_dim
                  if cfg.fusion == "concat"
                  else cfg.hidden_dim)
        self.classifier = nn.Linear(clf_in, 1)

    @property
    def node_embed(self):
        return self.branches[0].node_embed

    def _aggregate_membrane(self, stacked: torch.Tensor) -> torch.Tensor:
        """`stacked` has shape (N, T, d). Returns (T, d)."""
        agg = self.cfg.membrane_aggregator
        if agg == "mean":
            return stacked.mean(dim=0)
        if agg == "max":
            return stacked.max(dim=0).values
        if agg == "sum":
            return stacked.sum(dim=0)
        raise ValueError(f"unknown membrane_aggregator: {agg!r}")

    def encode_triads(
        self,
        triad_v: torch.Tensor,
        triad_sigma: torch.Tensor,
        return_h_v: bool = False,
    ):
        outs_0 = [b.encode_triads(triad_v, triad_sigma)
                  for b in self.branches]
        stacked_0 = torch.stack(outs_0, dim=0)            # (N, T, d)
        z = self._aggregate_membrane(stacked_0)            # (T, d)
        outs_1 = [
            outs_0[i] + gate(z)
            for i, gate in enumerate(self.read_gates)
        ]
        if self.fusion == "sum":
            fused = torch.stack(outs_1, dim=0).sum(dim=0)
        elif self.fusion == "mean":
            fused = torch.stack(outs_1, dim=0).mean(dim=0)
        elif self.fusion == "concat":
            fused = torch.cat(outs_1, dim=-1)
        elif self.fusion == "learned_alpha":
            stacked_1 = torch.stack(outs_1, dim=0)
            weights = torch.softmax(self.alpha, dim=0)
            fused = (weights[:, None, None] * stacked_1).sum(dim=0)
        elif self.fusion == "attention":
            stacked_1 = torch.stack(outs_1, dim=0)
            logits = self.attn(stacked_1).squeeze(-1)
            weights = torch.softmax(logits, dim=0)
            fused = (weights.unsqueeze(-1) * stacked_1).sum(dim=0)
        else:
            raise ValueError(f"unknown fusion: {self.fusion!r}")
        if return_h_v:
            return fused, self.branches[0].node_embed.weight
        return fused

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class StackedSideSignedKANConfig:
    """Configuration for the Phase-20 stacked-side HSIKAN.

    Each parallel branch is itself a :class:`MultiLayerSignedKAN`
    of depth `n_layers_per_branch`, optionally with highway gates
    (`inner_skip="highway"`, `use_residual=True`). Composes Phase
    16 (depth) + Phase 17 (side) + the existing
    :class:`HighwaySignedKAN` machinery into a single
    width × depth grid.

    Phase 19 found that pure depth degrades at L=8 (0.442) while
    pure side stays stable at any N (~0.65). Phase 20 asks: does
    *per-branch* depth help when paired with parallel-branch
    averaging?
    """

    n_nodes: int
    n_branches: int = 4
    n_layers_per_branch: int = 2
    hidden_dim: int = 8
    grid: int = 5
    k: int = 3
    use_minus_branch: bool = True
    init_scale: float = 0.1
    fusion: str = "mean"
    spline_kind: str = "bspline"
    inner_skip: str = "highway"   # "residual" | "highway" | "none"
    use_residual: bool = True       # vertex-side residual between layers
    layer_norm_between: bool = True
    jk_mode: str = "last"
    branch_seeds: list[int] | None = None


class StackedSideSignedKAN(nn.Module):
    """N parallel `MultiLayerSignedKAN` stacks fused at the
    per-triad embedding level.

    Architecture::

      branch i (i = 0..N-1)
          = MultiLayerSignedKAN(n_layers=L,
                                inner_skip=cfg.inner_skip,
                                use_residual=True,
                                layer_norm_between=True,
                                jk_mode='last')
      fused_h_t = fusion(branch_i.encode_triads(...) for i in 0..N-1)

    Highway support is baked in via `cfg.inner_skip="highway"`
    (the default) — matches :class:`HighwaySignedKAN`'s per-layer
    gate. Each parallel branch is then a HighwaySignedKAN-style
    stack; the fusion head averages them.

    Same interface as :class:`SideSignedKAN` /
    :class:`MembraneSignedKAN` (encode_triads + classifier +
    return_h_v + node_embed) so `run_compare.run_one` can call it
    via the same dispatch.
    """

    def __init__(self, cfg: StackedSideSignedKANConfig) -> None:
        super().__init__()
        # Local import to avoid a circular reference at module init.
        from .signedkan import MultiLayerSignedKAN, MultiLayerSignedKANConfig
        self.cfg = cfg
        self.n_branches = cfg.n_branches
        self.n_layers_per_branch = cfg.n_layers_per_branch
        self.hidden_dim = cfg.hidden_dim
        self.fusion = cfg.fusion

        self.branches = nn.ModuleList()
        for i in range(cfg.n_branches):
            if cfg.branch_seeds is not None and i < len(cfg.branch_seeds):
                torch.manual_seed(int(cfg.branch_seeds[i]))
            mcfg = MultiLayerSignedKANConfig(
                n_nodes=cfg.n_nodes,
                n_layers=cfg.n_layers_per_branch,
                hidden_dim=cfg.hidden_dim,
                grid=cfg.grid, k=cfg.k,
                use_minus_branch=cfg.use_minus_branch,
                init_scale=cfg.init_scale,
                spline_kinds=[cfg.spline_kind] * cfg.n_layers_per_branch,
                inner_skip=cfg.inner_skip,
                outer_skip="none",
                use_residual=cfg.use_residual,
                layer_norm_between=cfg.layer_norm_between,
                jk_mode=cfg.jk_mode,
                pool_mode="mean",
                share_weights=False,
            )
            self.branches.append(MultiLayerSignedKAN(mcfg))

        if cfg.fusion == "learned_alpha":
            self.alpha = nn.Parameter(torch.zeros(cfg.n_branches))
        elif cfg.fusion == "attention":
            self.attn = nn.Linear(cfg.hidden_dim, 1)
        else:
            self.alpha = None
            self.attn = None

        clf_in = (cfg.n_branches * cfg.hidden_dim
                  if cfg.fusion == "concat"
                  else cfg.hidden_dim)
        self.classifier = nn.Linear(clf_in, 1)

    @property
    def node_embed(self):
        return self.branches[0].node_embed

    def encode_triads(
        self,
        triad_v: torch.Tensor,
        triad_sigma: torch.Tensor,
        M_vt: torch.Tensor,
        return_h_v: bool = False,
    ):
        """`M_vt` is required because each branch is a
        `MultiLayerSignedKAN`, which uses the sparse triad→vertex
        incidence for inter-layer pooling. `return_h_v` matches
        the bare interface.
        """
        outs = [b.encode_triads(triad_v, triad_sigma, M_vt,
                                 return_h_v=False)
                for b in self.branches]
        if self.fusion == "sum":
            fused = torch.stack(outs, dim=0).sum(dim=0)
        elif self.fusion == "mean":
            fused = torch.stack(outs, dim=0).mean(dim=0)
        elif self.fusion == "concat":
            fused = torch.cat(outs, dim=-1)
        elif self.fusion == "learned_alpha":
            stacked = torch.stack(outs, dim=0)
            weights = torch.softmax(self.alpha, dim=0)
            fused = (weights[:, None, None] * stacked).sum(dim=0)
        elif self.fusion == "attention":
            stacked = torch.stack(outs, dim=0)
            logits = self.attn(stacked).squeeze(-1)
            weights = torch.softmax(logits, dim=0)
            fused = (weights.unsqueeze(-1) * stacked).sum(dim=0)
        else:
            raise ValueError(f"unknown fusion: {self.fusion!r}")
        if return_h_v:
            return fused, self.branches[0].node_embed.weight
        return fused

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class SideMixedAritySignedKANConfig:
    """Configuration for the Phase-21 side-stacked **mixed-arity**
    HSIKAN.

    Phase 17/18/19/20 established that parallel branches tighten σ
    and add a small AUC lift on c3-only HSIKAN; Phase 21 ports the
    pattern onto the mixed-arity family that produced the Bitcoin
    Alpha SOTA of 0.9959 ± 0.0011 with
    ``HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4`` (see
    ``project_bitcoin_optuna_best_10seed_2026_05_13``).

    Each branch is an independent :class:`MixedAritySignedKAN`
    (own ``arity_logits`` → own learned αₖ, own attention head,
    own ``inner_skip="highway"`` residuals through its
    :class:`MultiLayerSignedKAN` base). Fusion is at the edge-
    embedding level after every branch has produced its mixed-
    arity edge representation.

    The wrapper does NOT own per-arity inputs --- the same tuple
    list is forwarded to every branch. Branches diversify via
    independent stochastic init (no parameter sharing).
    """

    # The inner mixed-arity config is shared structurally across
    # branches but each branch gets its own freshly-initialised
    # parameters.
    base: "MixedAritySignedKANConfig"
    n_branches: int = 4
    fusion: str = "mean"     # "mean" | "sum"
    branch_seeds: list[int] | None = None
    # Gradient checkpointing: when True (default), each branch's forward
    # is wrapped in ``torch.utils.checkpoint`` so its intermediate
    # activations are dropped between branches and re-computed during
    # backward. Peak GPU memory per backward pass is then ~1 branch's
    # worth (plus the small N outputs) instead of N × forward state.
    # Necessary on 8 GB GPUs for N ≥ 4 with quaternion attention +
    # edge_cr highway, where the Slashdot SOTA config disables
    # cycle_batch_size and thus relies on per-branch activations
    # being released. Wall overhead: ~30% per branch.
    use_grad_checkpoint: bool = True
    # Outer-checkpoint mode (Phase 22 fix). When True, the *entire*
    # multi-branch forward (loop + accum + fusion) is wrapped in a
    # single ``torch.utils.checkpoint.checkpoint`` so that the forward
    # path holds **no** intermediates and the recompute during
    # backward runs branches sequentially — peak backward GPU memory
    # collapses from N × branch-forward to ~1 × branch-forward. The
    # cost: incompatible with the ``_attn_entropy_terms`` side channel
    # (the consumer reads it between forward and backward, but outer
    # checkpoint discards intermediate state); so when this is True we
    # also force ``collect_attn_entropy=False`` on every branch. Use
    # only when ``HSIKAN_ATTN_ENTROPY_LAMBDA == 0`` (default), i.e. the
    # Slashdot edge_cr SOTA config.
    outer_grad_checkpoint: bool = False


class SideMixedAritySignedKAN(nn.Module):
    """N parallel :class:`MixedAritySignedKAN` branches fused at
    the per-edge embedding level --- Phase 21.

    Exposes the same surface as a bare :class:`MixedAritySignedKAN`
    (``encode_edges``, ``classifier``, ``alpha()``, ``num_parameters``,
    ``node_embed``) so the existing ``run_one_mixed`` trainer can
    swap it in.

    Notes
    -----
    - ``alpha()`` returns the **mean** of each branch's softmaxed
      α over its arities (back-compat for callers that log a single
      ``alpha`` vector).
    - ``classifier`` is constructed at wrapper level (one per
      ``SideMixedAritySignedKAN``); we do not delegate to a single
      branch's classifier because the fused embedding's first dim
      depends on ``fusion`` (mean keeps d_jk; sum keeps d_jk;
      no concat path is provided to stay below the d_jk × N classifier
      cost --- the Phase 19/20 winners were mean fusion anyway).
    """

    def __init__(self, cfg: SideMixedAritySignedKANConfig) -> None:
        super().__init__()
        # Local import to avoid circular module init.
        from ..mixed_arity_signedkan import (
            MixedAritySignedKAN, MixedAritySignedKANConfig,
        )
        self.cfg = cfg
        self.n_branches = cfg.n_branches
        self.fusion = cfg.fusion
        # Validate fusion now to fail fast.
        if cfg.fusion not in ("mean", "sum"):
            raise ValueError(
                f"SideMixedAritySignedKAN fusion must be 'mean' or "
                f"'sum'; got {cfg.fusion!r}. (Concat path is not "
                f"provided --- it explodes the classifier dim by N.)"
            )

        self.branches = nn.ModuleList()
        for i in range(cfg.n_branches):
            if cfg.branch_seeds is not None and i < len(cfg.branch_seeds):
                torch.manual_seed(int(cfg.branch_seeds[i]))
            # Each branch gets its own freshly-instantiated copy of
            # the same config (so it gets its own freshly-initialised
            # parameters).
            self.branches.append(MixedAritySignedKAN(cfg.base))

        # Edge classifier --- matches the bare MixedAritySignedKAN
        # classifier interface that run_one_mixed expects.
        d = cfg.base.base.hidden_dim
        d_jk = (d * cfg.base.base.n_layers
                if cfg.base.base.jk_mode == "concat" else d)
        self.classifier = nn.Linear(d_jk, 1)

    @property
    def base(self):
        """First branch's base (back-compat for callers reading
        ``model.base.share_weights`` / ``model.base.jk_mode``)."""
        return self.branches[0].base

    @property
    def node_embed(self):
        """First branch's node embedding (back-compat for callers
        that inspect ``.node_embed.weight``)."""
        return self.branches[0].node_embed

    def alpha(self) -> torch.Tensor:
        """Mean α across branches (each branch's softmaxed α-over-
        arities, averaged). Same shape as a single branch's α."""
        alphas = torch.stack([b.alpha() for b in self.branches], dim=0)
        return alphas.mean(dim=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_signed_adjacency(self, A_pos, A_neg) -> None:
        """Forward to every branch (each owns its own direct-MP heads)."""
        for b in self.branches:
            if hasattr(b, "set_signed_adjacency"):
                b.set_signed_adjacency(A_pos, A_neg)

    def set_arity_mask(self, mask) -> None:
        """Forward to every branch (each owns its own α-mask)."""
        for b in self.branches:
            if hasattr(b, "set_arity_mask"):
                b.set_arity_mask(mask)

    def encode_edges(
        self,
        per_arity_inputs,
        query_edges=None,
        vertex_features=None,
        edge_features=None,
        edge_to_vertex=None,
        per_arity_arc_weights=None,
    ) -> torch.Tensor:
        """Run every branch on the same per-arity inputs and fuse.

        Memory-mode dispatch (Phase 22):

        - ``outer_grad_checkpoint=True``: wrap the entire branch
          loop in **one** checkpoint, holding zero intermediates
          across branches. Backward recomputes branches sequentially
          → peak ≈ 1 × branch-forward, not N ×. Each branch also
          runs with ``collect_attn_entropy=False`` (side-channel is
          discarded by the outer checkpoint anyway). Use when the
          attention-entropy regulariser is OFF (λ=0).
        - ``use_grad_checkpoint=True, outer_grad_checkpoint=False``:
          per-branch checkpoint. Saves intermediate activations but
          NOT the entropy graph — useful when λ > 0 (entropy reg on).
        - Both False: bare path, N × forward state alive.
        """
        n_branches = len(self.branches)
        outer_ckpt = (self.cfg.outer_grad_checkpoint
                       and self.training
                       and torch.is_grad_enabled())
        inner_ckpt = (self.cfg.use_grad_checkpoint
                       and self.training
                       and torch.is_grad_enabled()
                       and not outer_ckpt)

        def _run_branch(branch, anchor, collect_entropy: bool):
            return branch.encode_edges(
                per_arity_inputs,
                query_edges=query_edges,
                vertex_features=vertex_features,
                edge_features=edge_features,
                edge_to_vertex=edge_to_vertex,
                collect_attn_entropy=collect_entropy,
                per_arity_arc_weights=per_arity_arc_weights,
            )

        def _full_forward(_anchor):
            accum = None
            for branch in self.branches:
                # Outer mode: collection disabled (consumer can't read
                # it anyway across the checkpoint boundary). Inner /
                # bare mode: collection on, the consumer's side channel
                # works as designed.
                collect = not outer_ckpt
                if inner_ckpt:
                    out = torch.utils.checkpoint.checkpoint(
                        _run_branch, branch, branch.node_embed.weight,
                        collect,
                        use_reentrant=False,
                    )
                else:
                    out = _run_branch(branch, branch.node_embed.weight,
                                       collect)
                if accum is None:
                    accum = out
                else:
                    accum = accum + out
            if self.fusion == "mean":
                accum = accum / float(n_branches)
            return accum

        if outer_ckpt:
            # Anchor the outer checkpoint on a tensor that participates
            # in every branch's forward (any branch's node_embed weight
            # is fine; we pick branch 0's for stability).
            anchor = self.branches[0].node_embed.weight
            return torch.utils.checkpoint.checkpoint(
                _full_forward, anchor, use_reentrant=False,
            )
        return _full_forward(self.branches[0].node_embed.weight)

    @property
    def _attn_entropy_terms(self):
        """Concatenated per-edge attention entropy terms across all
        branches. Used by ``run_final_cell._aux_entropy_attention_alpha``
        when ``attn_entropy_lambda > 0``. Each branch fills its own
        list during ``encode_edges``; we just flatten.
        """
        out: list = []
        for b in self.branches:
            terms = getattr(b, "_attn_entropy_terms", None)
            if terms:
                out.extend(terms)
        return out

    def encode_graph(self, per_arity_inputs, query_edges=None,
                      pool: str = "mean", **kwargs) -> torch.Tensor:
        """Mirror of :meth:`MixedAritySignedKAN.encode_graph`."""
        edge_emb = self.encode_edges(
            per_arity_inputs, query_edges, **kwargs,
        )
        if pool == "mean":
            return edge_emb.mean(dim=0)
        if pool == "max":
            return edge_emb.max(dim=0).values
        if pool == "sum":
            return edge_emb.sum(dim=0)
        raise ValueError(f"unknown pool: {pool!r}")


__all__ = [
    "SideSignedKAN",
    "SideSignedKANConfig",
    "MembraneSignedKAN",
    "MembraneSignedKANConfig",
    "StackedSideSignedKAN",
    "StackedSideSignedKANConfig",
    "SideMixedAritySignedKAN",
    "SideMixedAritySignedKANConfig",
]
