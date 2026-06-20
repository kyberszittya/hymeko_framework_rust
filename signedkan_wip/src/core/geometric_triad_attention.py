"""Geometric (quaternion + Clifford) attention pool of signed triads → vertices.

The leakage-free rotor-HSiKAN trails SiGAT by ~0.04 AUROC, and the gap is "partly
attention, not cycles": SiGAT's only edge is an attention readout. This module
gives HSiKAN a *stronger* readout — a signed-triad attention pool whose score is
a learned mix of two geometric channels over the same projected features:

  * **quaternion** channel — Hamilton-product real part
    ``Re(q ⊗ k) = q0 k0 − q1 k1 − q2 k2 − q3 k3`` (signature (+,−,−,−));
  * **Clifford** channel — Cl(2,0) geometric-product *scalar* part
    ``⟨q k⟩_0 = q0 k0 + q1 k1 + q2 k2 − q12 k12`` (signature (+,+,+,−)),

reusing ``signedkan_wip.src.sequence.clifford.geometric_product``. The two carry
*different* metric signatures, so the gate ``g = σ(γ)`` mixes genuinely distinct
inner products rather than two copies of one. The score is ``tanh`` (not softmax):
balanced triads vote ``+``, unbalanced ``−`` — the same signed-voting readout as
``core/attention.py``, generalised to a geometric score.

Plan: docs/plans/2026-06-17-geometric-attention-head/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..sequence.clifford import CL2_DIM, geometric_product

# Hamilton-product real-part coefficients: Re(q⊗k) = Σ coeff_c · q_c · k_c.
_HAMILTON_REAL = (1.0, -1.0, -1.0, -1.0)


class GeometricTriadAttentionPool(nn.Module):
    """Pool per-triad embeddings into per-vertex embeddings by geometric attention.

    # Preconditions
    ``hidden % 4 == 0`` (multivector blocks of 4); ``channel`` in
    ``{"both", "quaternion", "clifford"}``; ``inc_vertex`` and ``inc_triad`` are
    1-D, equal-length ``long`` tensors listing every (vertex ∈ triad) incidence.
    # Postconditions
    ``forward`` returns ``(n_nodes, d_out)``; vertices with no incident triad get
    a zero row. Magnitude-normalised (``Σ w·v / Σ|w|``) so the pool is bounded and
    invariant to a vertex's triad count, while staying signed.
    # Invariants
    Parameter count is independent of graph size (projections are per-feature).
    """

    def __init__(self, d_node: int, d_triad: int, hidden: int, d_out: int,
                 channel: str = "both", temperature: float = 1.0,
                 eps: float = 1e-6, score_init_scale: float = 0.1,
                 learn_scale: bool = False, sign_aware: bool = False) -> None:
        """``score_init_scale`` shrinks the ``W_q``/``W_k`` init (``0.1`` =
        legacy/back-compat; ``1.0`` = no suppression — gives the score dynamic
        range). ``learn_scale`` adds a learnable ``log``-scale on the score.
        ``sign_aware`` takes the vote direction from the per-incidence triad
        balance sign so the score learns only a non-negative relevance — see
        ``_weights``. The three default to the legacy behaviour so prior
        ``geom_attn`` numbers stay reproducible."""
        super().__init__()
        if hidden % CL2_DIM != 0:
            raise ValueError(f"hidden must be % {CL2_DIM} == 0; got {hidden}")
        if channel not in {"both", "quaternion", "clifford"}:
            raise ValueError(
                f"channel must be both|quaternion|clifford; got {channel!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        if score_init_scale <= 0:
            raise ValueError(f"score_init_scale must be > 0; got {score_init_scale}")
        self.hidden = int(hidden)
        self.n_blocks = hidden // CL2_DIM
        self.channel = channel
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.learn_scale = bool(learn_scale)
        self.sign_aware = bool(sign_aware)
        # Shared projections: both channels reinterpret the SAME (hidden) features
        # as n_blocks 4-tuples — as quaternions or as Cl(2,0) multivectors.
        self.W_q = nn.Linear(d_node, hidden, bias=False)
        self.W_k = nn.Linear(d_triad, hidden, bias=False)
        self.W_v = nn.Linear(d_triad, d_out, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))   # σ(0)=0.5 → equal channels
        # Learnable score amplitude (exp(0)=1 → identity at init): the direct,
        # well-conditioned knob the dead-score diagnostic showed was missing.
        if self.learn_scale:
            self.log_scale = nn.Parameter(torch.zeros(()))
        if score_init_scale != 1.0:
            with torch.no_grad():
                self.W_q.weight.mul_(score_init_scale)
                self.W_k.weight.mul_(score_init_scale)

    # ── geometric scores (per incidence) ────────────────────────────────
    def _quaternion_score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Re(q ⊗ k) summed over blocks. ``q``,``k``: (P, n_blocks, 4)."""
        coeff = torch.tensor(_HAMILTON_REAL, device=q.device, dtype=q.dtype)
        return ((q * coeff) * k).sum(dim=(-1, -2))

    def _clifford_score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Cl(2,0) scalar part ⟨q k⟩_0 summed over blocks. (P, n_blocks, 4)."""
        return geometric_product(q, k)[..., 0].sum(dim=-1)

    def _raw_score(self, q_flat: torch.Tensor, k_flat: torch.Tensor) -> torch.Tensor:
        """Channel-mixed, scaled geometric score per incidence, *pre*-nonlinearity.
        ``q_flat``,``k_flat``: (P, hidden). Folds the fixed ``n_blocks^-1/2``
        scale, the optional learnable ``exp(log_scale)``, and the temperature."""
        P = q_flat.shape[0]
        q = q_flat.view(P, self.n_blocks, CL2_DIM)
        k = k_flat.view(P, self.n_blocks, CL2_DIM)
        if self.channel == "quaternion":
            s = self._quaternion_score(q, k)
        elif self.channel == "clifford":
            s = self._clifford_score(q, k)
        else:
            g = torch.sigmoid(self.gate)
            s = g * self._quaternion_score(q, k) + (1.0 - g) * self._clifford_score(q, k)
        scale = self.n_blocks ** -0.5
        if self.learn_scale:
            scale = scale * torch.exp(self.log_scale)
        return s * scale / self.temperature

    def _score(self, q_flat: torch.Tensor, k_flat: torch.Tensor) -> torch.Tensor:
        """Legacy signed weight ``w = tanh(raw score)`` — the ``sign_aware=False``
        path; also the reference the scatter-pool unit test pins against."""
        return torch.tanh(self._raw_score(q_flat, k_flat))

    def _weights(self, q_flat: torch.Tensor, k_flat: torch.Tensor,
                 inc_balance: torch.Tensor | None) -> torch.Tensor:
        """Per-incidence signed pooling weight.

        * ``sign_aware=False`` → ``w = tanh(s̃)``: the score learns direction *and*
          magnitude (the diagnostic showed it cannot — born tiny, no pressure).
        * ``sign_aware=True``  → ``w = b_t · σ(s̃)``: the triad balance sign ``b_t``
          (per incidence) sets the vote direction, ``σ(s̃)∈(0,1)`` the learned
          geometric relevance. At init ``s̃≈0`` ⇒ relevance ≈0.5 ⇒ the pool is the
          (uniform) balance-signed value mean, a warm start geometry refines.

        # Preconditions
        ``sign_aware=True`` requires ``inc_balance`` (1-D, aligned with the
        incidences, values in ``{+1,-1}``); ``None`` raises.
        """
        if not self.sign_aware:
            return self._score(q_flat, k_flat)
        if inc_balance is None:
            raise ValueError("sign_aware=True requires inc_balance "
                             "(per-incidence triad balance sign)")
        return inc_balance * torch.sigmoid(self._raw_score(q_flat, k_flat))

    def forward(self, h_node: torch.Tensor, h_triad: torch.Tensor,
                inc_vertex: torch.Tensor, inc_triad: torch.Tensor,
                inc_balance: torch.Tensor | None = None) -> torch.Tensor:
        """``h_node``: (V, d_node) queries; ``h_triad``: (T, d_triad) keys/values;
        ``inc_vertex``/``inc_triad``: (P,) incidence index pairs; ``inc_balance``:
        (P,) per-incidence triad balance signs in ``{+1,-1}`` (required iff
        ``sign_aware``). Returns ``(V, d_out)``."""
        assert inc_vertex.shape == inc_triad.shape, "incidence arrays must align"
        n_nodes = h_node.shape[0]
        q_flat = self.W_q(h_node)[inc_vertex]          # (P, hidden)
        k_flat = self.W_k(h_triad)[inc_triad]          # (P, hidden)
        w = self._weights(q_flat, k_flat, inc_balance)  # (P,) signed weights
        v = self.W_v(h_triad)[inc_triad]               # (P, d_out)
        num = torch.zeros(n_nodes, v.shape[1], device=v.device, dtype=v.dtype)
        num.index_add_(0, inc_vertex, w.unsqueeze(-1) * v)
        den = torch.zeros(n_nodes, device=v.device, dtype=v.dtype)
        den.index_add_(0, inc_vertex, w.abs())
        return num / (den.unsqueeze(-1) + self.eps)


def build_vertex_triad_pairs(triad_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a ``(T, k)`` triad-vertex table into incidence index pairs.

    Returns ``(inc_vertex, inc_triad)`` each of shape ``(T*k,)``: for triad ``t``
    with vertices ``triad_v[t]``, emit one ``(v, t)`` pair per vertex. Mirrors the
    incidence ``build_vertex_triad_incidence`` builds as a sparse matrix, but as
    explicit pairs for the scatter-attention pool.
    """
    n_triads, k = triad_v.shape
    inc_vertex = triad_v.reshape(-1)
    inc_triad = torch.arange(n_triads, device=triad_v.device).repeat_interleave(k)
    return inc_vertex, inc_triad


@torch.no_grad()
def summarise_gate(pool: GeometricTriadAttentionPool,
                   h_node: torch.Tensor, h_triad: torch.Tensor,
                   inc_vertex: torch.Tensor, inc_triad: torch.Tensor,
                   inc_balance: torch.Tensor | None = None) -> dict:
    """Diagnostic snapshot of a *trained* pool — answers "is the readout used?".

    The 5-seed A/B found the geometric readout adds nothing net (tie on alpha,
    regress on otc). This distinguishes the two ways that can happen, so the next
    lever is chosen on evidence rather than the report's stated hypothesis:

    * **score collapse** — ``w_abs_mean`` → 0: the signed weights ``tanh(score)``
      saturate to 0, so the pool degrades to an (unweighted) mean of values and
      carries no triad-specific signal. ``w_frac_dead`` is the share with |w|<0.05.
    * **residual swamp** — ``pool_to_hv`` → 0: the pooled refinement is tiny next
      to the rotor node embedding it is *added* to, so ``h_v + pool ≈ h_v`` and the
      head is a no-op even if the weights are informative.

    ``gate_sigma = σ(gate)`` is the quaternion share (0.5 = init; →1 quaternion-
    only, →0 Clifford-only) — only meaningful when ``pool.channel == 'both'``.

    # Preconditions
    ``inc_vertex``/``inc_triad`` are aligned 1-D ``long`` tensors; all tensors on
    one device; ``pool`` is the trained module.
    # Postconditions
    Pure read — no parameter or tensor is mutated. Every value is a python scalar.
    """
    assert inc_vertex.shape == inc_triad.shape, "incidence arrays must align"
    q_flat = pool.W_q(h_node)[inc_vertex]
    k_flat = pool.W_k(h_triad)[inc_triad]
    w_abs = pool._weights(q_flat, k_flat, inc_balance).abs()       # (P,) |w|
    pooled = pool(h_node, h_triad, inc_vertex, inc_triad, inc_balance)  # (V,d_out)
    seen = torch.zeros(h_node.shape[0], dtype=torch.bool, device=h_node.device)
    seen[inc_vertex] = True
    pool_norm_mean = float(pooled[seen].norm(dim=-1).mean()) if bool(seen.any()) else 0.0
    hv_norm_mean = float(h_node[seen].norm(dim=-1).mean()) if bool(seen.any()) else 0.0
    return dict(
        channel=pool.channel,
        gate_sigma=float(torch.sigmoid(pool.gate)),
        w_abs_mean=float(w_abs.mean()),
        w_abs_std=float(w_abs.std(unbiased=False)),
        w_frac_saturated=float((w_abs > 0.9).float().mean()),
        w_frac_dead=float((w_abs < 0.05).float().mean()),
        pool_norm_mean=pool_norm_mean,
        hv_norm_mean=hv_norm_mean,
        pool_to_hv=(pool_norm_mean / hv_norm_mean) if hv_norm_mean > 0 else float("inf"),
        wq_norm=float(pool.W_q.weight.norm()),
        wk_norm=float(pool.W_k.weight.norm()),
        wv_norm=float(pool.W_v.weight.norm()),
        n_incidences=int(w_abs.numel()),
    )
