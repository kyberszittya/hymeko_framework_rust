"""SignedKAN — Phase 2.6–2.10: Option C layer + full model.

Option C, per DECISIONS.md:
  - Three sub-aggregations per hyperedge, conditioned on sign
    σ ∈ {+1, -1, ~0}.
  - Inner spline φ_i^σ per (vertex, edge, sign) triple.
  - Outer spline φ_e^σ per (edge, sign) pair.
  - Final hyperedge embedding sums the three sub-aggregations.

Forward pass per hyperedge e = (v_1, ..., v_k) with σ assignments:

    h_e^{+} = Σ_{i: σ_i=+1} φ_e^{+}( φ_i^{+}( h_{v_i} ) )
    h_e^{-} = Σ_{i: σ_i=-1} φ_e^{-}( φ_i^{-}( h_{v_i} ) )
    h_e^{~} = Σ_{i: σ_i=~0} φ_e^{~}( φ_i^{~}( h_{v_i} ) )

    h_e = h_e^{+} + h_e^{-} + h_e^{~}

For Bitcoin link sign prediction, the model:
  1. Builds triad hyperedges (Phase 1).
  2. Computes h_e for every triad.
  3. Pools h_e across triads incident to the candidate edge.
  4. Linear classifier → P(sign = +1).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .hyperedges import SignedTriad
from .splines import (BSplineActivation, BatchedBSplineActivation,
                      DiagonalBatchedBSplineActivation,
                      BatchedCatmullRomActivation,
                      DiagonalBatchedCatmullRomActivation,
                      BatchedKochanekBartelsActivation,
                      DiagonalBatchedKochanekBartelsActivation)
from .bilinear_head import BilinearHead, LowRankBilinearHead


# Named canonical Kochanek-Bartels TCB triples.  Each gives a
# qualitatively different curve geometry without numerical fiddling:
#
#   smooth   (0, 0, 0)      Catmull-Rom equivalent — baseline
#   tense    (0.7, 0, 0)    sharper curve between knots, less overshoot
#   cusp     (0, 0.7, 0)    corner at control point, ReLU/elbow-like
#   skew     (0, 0, 0.5)    biased lean — asymmetric activation
#   sharp    (0.5, 0.5, 0)  combined tension + cusp
#   flat     (1.0, 0, 0)    near-linear segments (max tension)
#
# HSIKAN_KB_PRESET=<name> picks one; HSIKAN_KB_INIT_TCB="t,c,b" gives
# free numerical control. Preset takes precedence when both set.
_KB_PRESETS = {
    "smooth": (0.0, 0.0, 0.0),
    "tense":  (0.7, 0.0, 0.0),
    "cusp":   (0.0, 0.7, 0.0),
    "skew":   (0.0, 0.0, 0.5),
    "sharp":  (0.5, 0.5, 0.0),
    "flat":   (1.0, 0.0, 0.0),
}


def _resolve_kb_init_tcb() -> tuple[float, float, float] | None:
    """Resolve HSIKAN_KB_PRESET and/or HSIKAN_KB_INIT_TCB into a
    (t, c, b) tuple, or None if neither is set."""
    import os
    preset = os.environ.get("HSIKAN_KB_PRESET", "").strip().lower()
    if preset:
        if preset not in _KB_PRESETS:
            raise ValueError(
                f"unknown HSIKAN_KB_PRESET={preset!r}; "
                f"valid: {sorted(_KB_PRESETS.keys())}"
            )
        return _KB_PRESETS[preset]
    raw = os.environ.get("HSIKAN_KB_INIT_TCB", "")
    if raw:
        try:
            parts = [float(p) for p in raw.split(",")]
            if len(parts) == 3:
                return tuple(parts)
        except (ValueError, IndexError):
            pass
    return None


@dataclass
class SignedKANConfig:
    n_nodes: int
    hidden_dim: int = 32
    grid: int = 5
    k: int = 3
    use_minus_branch: bool = True
    use_zero_branch: bool = False        # ~0 unused on triad construction
                                         # since DECISIONS.md uses only ±1
    init_scale: float = 0.1
    spline_kind: str = "bspline"
    # spline_kind ∈ {"bspline", "catmull_rom",
    #                "bspline_cr"  — B-spline inner, CR outer,
    #                "cr_bspline"  — CR inner, B-spline outer}
    use_bilinear: bool = False           # add bilinear endpoint head
    bilinear_rank: int = 0               # 0 = full-rank, k>0 = rank-k
    spectral_init_eigvec: torch.Tensor | None = None  # (n_nodes, hidden_dim)
    spline_residual: bool = False        # f(x) = spline(x) + x (KAN-residual)
    spline_highway:  bool = False        # f(x) = T(x)*spline(x) + (1-T(x))*x
    # Per-position skip kind. Overrides the global flags above when
    # set. CV-analogy: "spine" (deeper representation) usually wants
    # skip; "head" (output projection) usually does not.
    inner_skip: str = "auto"             # {"auto", "none", "residual", "highway"}
    outer_skip: str = "auto"             # same options


class SignedKANLayer(nn.Module):
    """One Option C layer over signed-incidence triads.

    Inputs:
      x      : (n_nodes, hidden_dim) — vertex embeddings
      triads : list[SignedTriad]     — hyperedges produced by Phase 1.7
    Outputs:
      h      : (n_triads, hidden_dim) — per-hyperedge embeddings
    """
    def __init__(self, cfg: SignedKANConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        # Sign branches: $+1$, $-1$, $\sim 0$. Default config uses
        # $\{+1, -1\}$; the $\sim 0$ branch is left as a configurable
        # extension because the deterministic apex rule of §III emits
        # only $\pm 1$ on triads.
        n_branches = 1 + int(cfg.use_minus_branch) + int(cfg.use_zero_branch)
        self.n_branches = n_branches
        self.sign_values = [1] \
            + ([-1] if cfg.use_minus_branch else []) \
            + ([0]  if cfg.use_zero_branch  else [])
        # Register sign_values as a buffer so it moves with the module
        # (.to(device)) and we don't pay torch.tensor()-from-Python-list
        # cost on every forward — that single call was ~80% of forward
        # time on Bitcoin Alpha (cuda) before this fix.
        self.register_buffer(
            "_sign_vals",
            torch.tensor(self.sign_values, dtype=torch.long),
            persistent=False,
        )
        # Single batched spline pair: $S$ branches share the
        # Cox--de Boor basis (or Catmull-Rom segments), only the
        # coefficient tensor differs. Outer is diagonal-fused: each
        # per-sign aggregate is fed only into its own sign's outer
        # spline, skipping the $S^2 - S$ off-diagonal evaluations
        # the previous version computed and discarded.
        # Map spline_kind → (inner_kind, outer_kind).
        # Composite kinds use "_" as inner_outer separator
        # (e.g. "bspline_cr" → inner=bspline, outer=catmull_rom).
        kind = cfg.spline_kind
        _alias = {
            "bspline":          ("bspline",     "bspline"),
            "catmull_rom":      ("catmull_rom", "catmull_rom"),
            "kochanek_bartels": ("kochanek_bartels", "kochanek_bartels"),
            "bspline_cr":       ("bspline",     "catmull_rom"),
            "cr_bspline":       ("catmull_rom", "bspline"),
            "bspline_kb":       ("bspline",     "kochanek_bartels"),
            "kb_bspline":       ("kochanek_bartels", "bspline"),
            "cr_kb":            ("catmull_rom", "kochanek_bartels"),
            "kb_cr":            ("kochanek_bartels", "catmull_rom"),
        }
        if kind not in _alias:
            raise ValueError(f"unknown spline_kind: {kind}")
        inner_kind, outer_kind = _alias[kind]

        def _make_inner(k):
            if k == "bspline":
                return BatchedBSplineActivation(
                    n_branches, d, cfg.grid, cfg.k, cfg.init_scale,
                )
            if k == "catmull_rom":
                return BatchedCatmullRomActivation(
                    n_branches, d, cfg.grid, cfg.init_scale,
                )
            # KB: read per-experiment init from env var (sweep knob).
            # HSIKAN_KB_PRESET takes precedence over HSIKAN_KB_INIT_TCB.
            # Presets are named canonical TCB triples:
            #   "smooth": (0,0,0)     — Catmull-Rom equivalent
            #   "tense":  (0.7,0,0)   — sharper between-knot curve
            #   "cusp":   (0,0.7,0)   — corner at control point (ReLU-ish)
            #   "skew":   (0,0,0.5)   — biased lean
            #   "sharp":  (0.5,0.5,0) — combined tension + cusp
            init_tcb = _resolve_kb_init_tcb()
            return BatchedKochanekBartelsActivation(
                n_branches, d, cfg.grid, cfg.init_scale,
                init_tcb=init_tcb,
            )

        def _make_outer(k):
            if k == "bspline":
                return DiagonalBatchedBSplineActivation(
                    n_branches, d, cfg.grid, cfg.k, cfg.init_scale,
                )
            if k == "catmull_rom":
                return DiagonalBatchedCatmullRomActivation(
                    n_branches, d, cfg.grid, cfg.init_scale,
                )
            # KB outer: same env-var path as inner (preset > tcb).
            init_tcb = _resolve_kb_init_tcb()
            return DiagonalBatchedKochanekBartelsActivation(
                n_branches, d, cfg.grid, cfg.init_scale,
                init_tcb=init_tcb,
            )

        self.inner = _make_inner(inner_kind)
        self.outer = _make_outer(outer_kind)

        # Per-position skip kind: resolve "auto" via the global flags
        # for backward compatibility, otherwise honour the explicit
        # value. Highway gate (Srivastava–Greff–Schmidhuber 2015)
        # initialised with bias=-2 so $T \approx 0.12$ at start —
        # layer behaves like the identity and the optimiser opens the
        # gate only when the spline carries signal.
        def _resolve(per_pos_kind):
            if per_pos_kind != "auto":
                return per_pos_kind
            if cfg.spline_highway:  return "highway"
            if cfg.spline_residual: return "residual"
            return "none"
        self.inner_skip = _resolve(cfg.inner_skip)
        self.outer_skip = _resolve(cfg.outer_skip)
        if self.inner_skip == "highway":
            self.gate_inner = nn.Linear(d, d)
            with torch.no_grad():
                self.gate_inner.bias.fill_(-2.0)
        else:
            self.gate_inner = None
        if self.outer_skip == "highway":
            self.gate_outer = nn.Linear(d, d)
            with torch.no_grad():
                self.gate_outer.bias.fill_(-2.0)
        else:
            self.gate_outer = None

    def _can_use_triton_inner(self, x: torch.Tensor) -> bool:
        """Conditions for dispatching the inner forward to the Triton
        kernel.  Forward parity verified at ≤ 1e-7 vs the PyTorch path
        in signedkan_wip/tests/test_triton_kernels.py.

        Default OFF: the autograd backward currently routes through a
        PyTorch reference, which materialises a (T, k, 2, d) intermediate
        that can OOM on small GPUs at training time.  Set
        HSIKAN_TRITON_KERNEL=1 to opt in (recommended for inference,
        forward-only profiling, and any setup with ample VRAM)."""
        import os
        if int(os.environ.get("HSIKAN_TRITON_KERNEL", "0")) == 0:
            return False
        if not x.is_cuda:
            return False
        if self.n_branches != 2:
            return False
        if self.inner_skip not in ("none", "highway"):
            return False
        from .splines import BatchedCatmullRomActivation
        if not isinstance(self.inner, BatchedCatmullRomActivation):
            return False
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return True

    def _triton_inner_agg(
        self, x: torch.Tensor, triad_v: torch.Tensor,
        triad_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Triton fast-path: returns the per-sign mean aggregate
        ``agg`` of shape (T, 2, d), equivalent to steps 1–4 of the
        PyTorch path."""
        from .triton_kernels import (
            signedkan_inner_triton_autograd,
            signedkan_inner_highway_triton_autograd,
        )
        coef_pos = self.inner.coef[0]   # (d, G)
        coef_neg = self.inner.coef[1]
        if self.inner_skip == "highway":
            gate_w = self.gate_inner.weight.t().contiguous()  # (d, d)
            gate_b = self.gate_inner.bias
            return signedkan_inner_highway_triton_autograd(
                x, triad_v, triad_sigma, coef_pos, coef_neg,
                gate_w, gate_b, self.cfg.grid,
            )
        return signedkan_inner_triton_autograd(
            x, triad_v, triad_sigma, coef_pos, coef_neg, self.cfg.grid,
        )

    def forward(self, x: torch.Tensor,
                triad_v: torch.Tensor,    # (n_triads, k) vertex IDs
                triad_sigma: torch.Tensor # (n_triads, k) ∈ {+1, -1}
               ) -> torch.Tensor:
        """Aggregate per-sign sub-aggregations across each k-uniform
        hyperedge, fused as a single batched spline call.

        Generalised over ``k`` (arity): k=3 is the original triad case;
        k=4,5,... are Davis-1967 weakly-balanced n-tuples constructed
        by ``n_tuples.construct_k``.

        Memory mode: when ``HSIKAN_CHUNK_T`` is set in env, the
        T-dimension is processed in chunks of that size and the
        per-chunk outer-spline outputs are concatenated.  Reduces
        peak GPU memory roughly linearly with chunk count, at the
        cost of recomputing the inner+outer spline forward on each
        chunk.  Required for $|V| \\gtrsim 10^5$ datasets (Epinions)
        on $\\le 8$ GB GPUs.
        """
        import os
        chunk_t = int(os.environ.get("HSIKAN_CHUNK_T", "0"))
        if chunk_t > 0 and triad_v.shape[0] > chunk_t:
            chunks = []
            for s in range(0, triad_v.shape[0], chunk_t):
                e = min(s + chunk_t, triad_v.shape[0])
                chunks.append(self._forward_impl(
                    x, triad_v[s:e], triad_sigma[s:e]))
            return torch.cat(chunks, dim=0)
        return self._forward_impl(x, triad_v, triad_sigma)

    def _forward_impl(self, x: torch.Tensor,
                       triad_v: torch.Tensor,
                       triad_sigma: torch.Tensor) -> torch.Tensor:
        T, k = triad_v.shape[0], triad_v.shape[1]
        d = self.cfg.hidden_dim
        S = self.n_branches

        # Fast-path: gather + per-sign CR + (optional highway-skip) +
        # σ-mask + mean reduce as a single fused Triton kernel.  Parity
        # vs the PyTorch path verified at ≤ 1e-7 (see
        # signedkan_wip/tests/test_triton_kernels.py).  Set
        # HSIKAN_TRITON_KERNEL=0 to force the PyTorch path.
        if self._can_use_triton_inner(x):
            agg = self._triton_inner_agg(x, triad_v, triad_sigma)
            # Skip steps 1-4 below; proceed to step 5 (outer spline).
        else:
            # 1. Per-vertex embeddings.
            h_v = x[triad_v]                               # (T, k, d)

            # 2. Single batched inner spline call.
            inner_all = self.inner(h_v.reshape(-1, d))     # (T*k, S, d)
            inner_all = inner_all.view(T, k, S, d)         # (T, k, S, d)
            if self.inner_skip == "residual":
                inner_all = inner_all + h_v.unsqueeze(2)   # (T, k, S, d)
            elif self.inner_skip == "highway":
                T_inner = torch.sigmoid(self.gate_inner(h_v))   # (T, k, d)
                T_inner_b = T_inner.unsqueeze(2)                # (T, k, 1, d)
                x_b = h_v.unsqueeze(2)                          # (T, k, 1, d)
                inner_all = T_inner_b * inner_all + (1.0 - T_inner_b) * x_b

            # 3. Per-sign masks: M[t, i, s] = 1 iff triad_sigma[t, i] == sign_values[s]
            sign_vals = self._sign_vals
            if sign_vals.dtype != triad_sigma.dtype:
                sign_vals = sign_vals.to(triad_sigma.dtype)
            masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
            masks_e = masks.unsqueeze(-1)                  # (T, k, S, 1)

            # 4. Aggregate over the k vertices, per sign.
            counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (T, S, 1)
            agg = (inner_all * masks_e).sum(dim=1) / counts        # (T, S, d)

        # 5. Diagonal-fused outer: each per-sign aggregate row $s$ is
        # processed by branch $s$'s outer spline only. Saves an $S$
        # factor over the previous "compute $S^2$, keep diagonal $S$"
        # implementation.
        out_diag = self.outer(agg)                     # (T, S, d)
        if self.outer_skip == "residual":
            out_diag = out_diag + agg
        elif self.outer_skip == "highway":
            T_outer = torch.sigmoid(self.gate_outer(agg))   # (T, S, d)
            out_diag = T_outer * out_diag + (1.0 - T_outer) * agg

        # 6. Sum over signs.
        h_e = out_diag.sum(dim=1)                      # (T, d)
        return h_e


class SignedKAN(nn.Module):
    """Single SignedKAN layer + linear edge classifier.

    For link sign prediction on a target edge (u, v), we look up the
    triads incident to (u, v), compute h_e for each, mean-pool to get
    an edge representation, and apply a linear classifier.
    """
    def __init__(self, cfg: SignedKANConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.node_embed = nn.Embedding(cfg.n_nodes, d)
        if cfg.spectral_init_eigvec is not None:
            with torch.no_grad():
                self.node_embed.weight.copy_(cfg.spectral_init_eigvec)
        else:
            nn.init.normal_(self.node_embed.weight, std=cfg.init_scale)
        self.layer = SignedKANLayer(cfg)
        self.classifier = nn.Linear(d, 1)
        if cfg.use_bilinear:
            self.bilinear = (LowRankBilinearHead(d, rank=cfg.bilinear_rank)
                              if cfg.bilinear_rank > 0
                              else BilinearHead(d))
        else:
            self.bilinear = None

    def encode_triads(self, triad_v: torch.Tensor,
                      triad_sigma: torch.Tensor,
                      return_h_v: bool = False):
        """Return per-triad embeddings, (n_triads, d). With
        ``return_h_v=True`` also returns the (unmodified, single-layer)
        node embeddings — used by the bilinear endpoint head."""
        x = self.node_embed.weight     # (n_nodes, d)
        h_t = self.layer(x, triad_v, triad_sigma)
        return (h_t, x) if return_h_v else h_t

    def predict_edge_sign(self,
                          triad_emb: torch.Tensor,    # (n_triads, d)
                          edge_to_triads: list[list[int]]
                          ) -> torch.Tensor:
        """For each edge in `edge_to_triads`, mean-pool the embeddings
        of its incident triads and apply the classifier. Returns
        logits, (n_edges,)."""
        d = self.cfg.hidden_dim
        out_logits = []
        for tri_ids in edge_to_triads:
            if not tri_ids:
                # No triad incident: use zero embedding (back off to
                # baseline-like behaviour).
                emb = triad_emb.new_zeros(d)
            else:
                emb = triad_emb[tri_ids].mean(dim=0)
            out_logits.append(self.classifier(emb).squeeze(-1))
        return torch.stack(out_logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class MultiLayerSignedKANConfig:
    """Stacked SignedKAN: $L$ layers with per-layer spline kind, vertex-side
    triad pooling between layers, optional residual connections.

    Architecture (per layer $\\ell$, $\\ell = 1, \\ldots, L$):
        $\\mathbf{h}_t^{(\\ell)} = \\mathrm{SignedKANLayer}_\\ell
                                     (\\mathbf{h}_v^{(\\ell-1)},
                                       \\mathbf{T}_v, \\mathbf{T}_\\sigma)$
        $\\mathbf{h}_v^{(\\ell)} = \\mathbf{h}_v^{(\\ell-1)} +
                                     \\mathbf{M}_{vt} \\mathbf{h}_t^{(\\ell)}$
        (residual on vertex side; $\\mathbf{M}_{vt}$ scatter-means triad
        embeddings into vertex space)

    Final triad embeddings are taken from layer $L$ and pooled to
    edges via the existing edge-triad incidence in the outer code.
    """
    n_nodes: int
    n_layers: int = 2
    hidden_dim: int = 32
    grid: int = 5
    k: int = 3
    use_minus_branch: bool = True
    use_zero_branch: bool = False
    init_scale: float = 0.1
    use_residual: bool = True
    # Per-layer spline kind. If None, defaults to "bspline" for all layers.
    spline_kinds: list[str] | None = None
    # Inter-layer triad→vertex pooling. "mean" row-normalises (canonical
    # oversmoothing operator); "sum" preserves signal magnitude.
    pool_mode: str = "mean"          # {"mean", "sum"}
    # Jumping-Knowledge aggregation across layers (Xu et al., 2018) on the
    # final triad embeddings. "last" uses only the last layer (default,
    # original behaviour). "sum" sums all layers' triad embeddings before
    # the classifier (parameter-equivalent to L=1). "concat" concatenates,
    # classifier accepts L*d input.
    jk_mode: str = "last"            # {"last", "sum", "concat"}
    use_bilinear: bool = False       # add bilinear endpoint head
    bilinear_rank: int = 0           # 0 = full-rank, k>0 = rank-k
    spectral_init_eigvec: torch.Tensor | None = None
    spline_residual: bool = False    # f(x) = spline(x) + x (KAN-residual)
    spline_highway:  bool = False    # f(x) = T(x)*spline(x) + (1-T(x))*x
    inner_skip: str = "auto"
    outer_skip: str = "auto"
    layer_norm_between: bool = False # LayerNorm on h_v between layers
    share_weights: bool = False      # one SignedKANLayer applied L times


class MultiLayerSignedKAN(nn.Module):
    """Stacked SignedKAN with vertex-side scatter pooling between layers."""
    def __init__(self, cfg: MultiLayerSignedKANConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.node_embed = nn.Embedding(cfg.n_nodes, d)
        if cfg.spectral_init_eigvec is not None:
            with torch.no_grad():
                self.node_embed.weight.copy_(cfg.spectral_init_eigvec)
        else:
            nn.init.normal_(self.node_embed.weight, std=cfg.init_scale)

        spline_kinds = (cfg.spline_kinds
                        if cfg.spline_kinds is not None
                        else ["bspline"] * cfg.n_layers)
        if len(spline_kinds) != cfg.n_layers:
            raise ValueError(
                f"spline_kinds has {len(spline_kinds)} entries but "
                f"n_layers = {cfg.n_layers}"
            )

        if cfg.share_weights:
            # One shared SignedKANLayer applied $L$ times — recurrent
            # multi-layer with no extra parameters per layer. Spline
            # kind taken from the first entry; mixing kinds is
            # incompatible with weight sharing.
            shared = SignedKANLayer(SignedKANConfig(
                n_nodes=cfg.n_nodes, hidden_dim=d,
                grid=cfg.grid, k=cfg.k,
                use_minus_branch=cfg.use_minus_branch,
                use_zero_branch=cfg.use_zero_branch,
                init_scale=cfg.init_scale,
                spline_kind=spline_kinds[0],
                spline_residual=cfg.spline_residual,
                spline_highway=cfg.spline_highway,
                inner_skip=cfg.inner_skip,
                outer_skip=cfg.outer_skip,
            ))
            # Register once; we'll call it cfg.n_layers times in forward.
            self.shared_layer = shared
            self.layers = nn.ModuleList()  # empty, signals shared mode
        else:
            self.shared_layer = None
            self.layers = nn.ModuleList([
                SignedKANLayer(SignedKANConfig(
                    n_nodes=cfg.n_nodes, hidden_dim=d,
                    grid=cfg.grid, k=cfg.k,
                    use_minus_branch=cfg.use_minus_branch,
                    use_zero_branch=cfg.use_zero_branch,
                    init_scale=cfg.init_scale,
                    spline_kind=spline_kinds[i],
                    spline_residual=cfg.spline_residual,
                    spline_highway=cfg.spline_highway,
                    inner_skip=cfg.inner_skip,
                    outer_skip=cfg.outer_skip,
                )) for i in range(cfg.n_layers)
            ])

        # LayerNorm on h_v between layers — standard deep-net
        # stabilisation transposed to the multi-layer SignedKAN
        # forward path. Normalises the per-vertex embedding before
        # the next layer's spline evaluation.
        if cfg.layer_norm_between:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(max(cfg.n_layers - 1, 0))]
            )
        else:
            self.layer_norms = None
        # JK-concat needs a wider classifier; other modes match L=1.
        clf_in = d * cfg.n_layers if cfg.jk_mode == "concat" else d
        self.classifier = nn.Linear(clf_in, 1)
        if cfg.use_bilinear:
            self.bilinear = (LowRankBilinearHead(d, rank=cfg.bilinear_rank)
                              if cfg.bilinear_rank > 0
                              else BilinearHead(d))
        else:
            self.bilinear = None

    def encode_triads(self, triad_v: torch.Tensor,
                      triad_sigma: torch.Tensor,
                      M_vt: torch.Tensor,
                      return_h_v: bool = False):
        """Return per-triad embeddings according to ``cfg.jk_mode``:

          - ``"last"``  : last layer only — shape ``(T, d)``.
          - ``"sum"``   : sum over layers — shape ``(T, d)``.
          - ``"concat"``: concat over layers — shape ``(T, L*d)``.

        ``M_vt``: sparse incidence built by
        ``build_vertex_triad_incidence``. With ``cfg.pool_mode == "mean"``
        the rows are normalised; with ``"sum"`` they aren't.

        With ``return_h_v=True`` also returns the post-residual final
        vertex embeddings (after the $L-1$ inter-layer updates), used
        by the bilinear endpoint head.
        """
        h_v = self.node_embed.weight                     # (V, d)
        per_layer_t = []                                  # collect h_t per layer
        n_layers = self.cfg.n_layers
        for li in range(n_layers):
            layer = (self.shared_layer if self.shared_layer is not None
                     else self.layers[li])
            h_t = layer(h_v, triad_v, triad_sigma)        # (T, d)
            per_layer_t.append(h_t)
            if li < n_layers - 1:
                h_v_step = torch.sparse.mm(M_vt, h_t)     # (V, d)
                h_v = h_v + h_v_step if self.cfg.use_residual else h_v_step
                if self.layer_norms is not None:
                    h_v = self.layer_norms[li](h_v)

        jk = self.cfg.jk_mode
        if jk == "last":
            out = per_layer_t[-1]
        elif jk == "sum":
            out = torch.stack(per_layer_t, dim=0).sum(dim=0)
        elif jk == "concat":
            out = torch.cat(per_layer_t, dim=-1)         # (T, L*d)
        else:
            raise ValueError(f"unknown jk_mode: {jk}")
        return (out, h_v) if return_h_v else out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vertex_triad_incidence(triad_v_np, n_nodes: int,
                                 device: torch.device,
                                 mode: str = "mean") -> torch.Tensor:
    """Sparse $\\mathbf{M}_{vt} \\in \\mathbb{R}^{|V| \\times |T|}$.

    With ``mode == "mean"``,
    $\\mathbf{M}_{vt}[v, t] = 1 / |\\{t' : v \\in t'\\}|$, so each row
    sums to 1 for vertices appearing in at least one triad. This is
    the canonical oversmoothing operator (mean aggregation).

    With ``mode == "sum"``, $\\mathbf{M}_{vt}[v, t] = 1$ on incidence,
    preserving signal magnitude across the inter-layer pooling step.
    """
    # Vectorised numpy construction + CSR return. The previous
    # double-loop + COO path was both O(n_triads · k) Python overhead
    # at setup AND used the slower COO sparse_mm kernel at forward
    # time. CSR triggers the cuSPARSE fast path on cuda, matching the
    # M_e change made for `_build_edge_incidence_vertex_adj_scipy`.
    import numpy as _np
    triad_v_np = _np.asarray(triad_v_np, dtype=_np.int64)
    n_triads, k = triad_v_np.shape

    rows = triad_v_np.reshape(-1)                 # (n_triads * k,)
    cols = _np.repeat(_np.arange(n_triads, dtype=_np.int64), k)

    if mode == "mean":
        counts = _np.bincount(rows, minlength=n_nodes).astype(_np.float32)
        # Avoid divide-by-zero — vertices with zero incidence get 1 in
        # the denominator; the `vals` for those rows are simply zero
        # (no entries) so the substitution never appears in the output.
        inv = _np.where(counts > 0, 1.0 / _np.maximum(counts, 1.0), 0.0)
        vals = inv[rows]
    else:  # "sum"
        vals = _np.ones(rows.shape[0], dtype=_np.float32)

    # Build via scipy CSR for sorted indices, then bridge to torch.
    from scipy.sparse import coo_matrix
    M = coo_matrix((vals, (rows, cols)),
                    shape=(n_nodes, n_triads)).tocsr()
    crow = torch.from_numpy(M.indptr.astype(_np.int64)).to(device)
    col  = torch.from_numpy(M.indices.astype(_np.int64)).to(device)
    val  = torch.from_numpy(M.data).to(device)
    return torch.sparse_csr_tensor(
        crow, col, val, (n_nodes, n_triads),
    )


# ─── Smoke test (Phase 2.10) ──────────────────────────────────────────


def _smoke_test():
    """100-node subgraph, 50 random triads, forward + backward."""
    cfg = SignedKANConfig(n_nodes=100, hidden_dim=8, grid=5, k=3)
    model = SignedKAN(cfg)
    n_triads = 50
    triad_v = torch.randint(0, 100, (n_triads, 3))
    triad_sigma = torch.randint(0, 2, (n_triads, 3)) * 2 - 1   # {-1, +1}
    triad_emb = model.encode_triads(triad_v, triad_sigma)
    assert triad_emb.shape == (n_triads, 8), \
        f"unexpected triad emb shape: {triad_emb.shape}"
    # Mock 10 candidate edges, each with 2-5 triads.
    rng = torch.Generator().manual_seed(0)
    edges = [
        torch.randint(0, n_triads, (torch.randint(2, 6, (1,), generator=rng).item(),),
                      generator=rng).tolist()
        for _ in range(10)
    ]
    logits = model.predict_edge_sign(triad_emb, edges)
    assert logits.shape == (10,), f"unexpected logits shape: {logits.shape}"
    # Backward through a binary cross-entropy.
    target = torch.randint(0, 2, (10,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    n_params = model.num_parameters()
    print(f"  smoke test passed:  triads={triad_emb.shape}  "
          f"logits={logits.shape}  loss={loss.item():.4f}  "
          f"params={n_params}")
    return n_params


def main() -> None:
    print("SignedKAN smoke test (Phase 2.10):")
    n_params = _smoke_test()
    print(f"\nReference parameter count at hidden=8, grid=5, k=3: {n_params}")


if __name__ == "__main__":
    main()
