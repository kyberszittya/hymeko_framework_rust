"""Spectral-entropy Lyapunov-safe regulariser for SignedKAN.

Mirrors the `entropy_lyapunov` arm of
`python/benches/thesis_iv_hard/run_benchmark.py` (the regulariser
paper's §5 schedule), specialised for the SignedKAN node-embedding
matrix:

    H        = Shannon entropy of the singular-value^2 distribution
               of A = node_embed.weight, in bits
    H_norm   = H / log2(rank(A))                    in [0, 1]
    KL_step  = D_KL(p_{t-1} || p_t) of consecutive normalised
               spectra (in bits)
    lam_eff  = clamp(lam_0 * exp(-eta * KL_step), 0.1, 10.0)
    reg      = lam_eff * ( lam_a * (H_norm - H_target)^2
                         + lam_b *  H_norm )

The Lyapunov-safe schedule reduces the regularisation pressure
when the spectrum is moving fast (large KL step) and lets it
recover when the spectrum is settled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class EntropyRegConfig:
    lam_0: float = 0.01
    lam_a: float = 1.0
    lam_b: float = 1.0
    eta:   float = 5.0
    target_entropy: float = 0.5
    eps:   float = 1e-12
    # Refined-schedule knobs (default off for backward compat).
    kl_normalized: bool = False    # divide KL by log2(rank) → KL ∈ [0, 1]
    momentum: float = 0.0          # EMA on lam_eff; 0 = off, 0.9 = strong
    # Schedule-stride: only refresh KL_step / lam_eff every `stride`
    # calls. The reg term itself is computed every call (we still need
    # gradients through H_norm to A); only the *schedule update* is
    # strided. With momentum=0.9, the per-step refresh is largely
    # redundant — stride=5 saves ~80% of the schedule computation
    # without measurable behaviour change.
    stride: int = 1
    # Tier 5 / D: KL-to-target term. lam_KL > 0 adds the (normalised)
    # KL divergence from the spectral distribution to a uniform target,
    # i.e. lam_eff · lam_KL · KL(p ‖ uniform) / log₂(rank). Identity:
    # KL(p ‖ uniform) / log₂(rank) = 1 − H_norm. So lam_KL pushes
    # H_norm UP (spread-spectrum prior), opposite direction to lam_b
    # (which pushes H_norm DOWN). Combined with lam_a's
    # (H_norm − target)² attractor, this gives the GA a clean
    # signed control over the H_norm gradient.
    lam_KL: float = 0.0


def _spectral_distribution(A: torch.Tensor, eps: float) -> torch.Tensor:
    """Singular-value^2 normalised distribution. A: (n, d).

    Computed via eigvalsh on the smaller Gram side (Aᵀ A if n ≥ d, else
    A Aᵀ): eigenvalues of the Gram matrix are exactly the squared
    singular values of A. For typical n_nodes×hidden (e.g. 3700×32)
    this is 5–10× faster than svdvals(A).
    """
    n, d = A.shape
    gram = A.t() @ A if n >= d else A @ A.t()
    s_sq = torch.linalg.eigvalsh(gram)
    # eigvalsh returns ascending; clamp tiny FP-negative roundoff.
    s_sq = s_sq.clamp_min(0.0)
    p = s_sq / (s_sq.sum() + eps)
    return p


def _shannon_bits(p: torch.Tensor, eps: float) -> torch.Tensor:
    return -(p * (p.clamp_min(eps).log2())).sum()


def _kl_bits(p_prev: torch.Tensor, p_curr: torch.Tensor,
             eps: float) -> torch.Tensor:
    p_p = p_prev.clamp_min(eps)
    p_c = p_curr.clamp_min(eps)
    return (p_c * (p_c.log2() - p_p.log2())).sum()


class EntropyRegulariser:
    """Stateful regulariser. Carries `prev_spectrum` across calls
    so the KL feedback term has a previous-step reference."""

    def __init__(self, cfg: EntropyRegConfig):
        self.cfg = cfg
        self.prev_spectrum: torch.Tensor | None = None
        self.last_h_norm: float = float("nan")
        self.last_kl: float = 0.0
        self.last_lam_eff: float = float("nan")
        self.lam_eff_ema: float | None = None   # EMA on lam_eff for momentum
        self._call_count: int = 0               # for stride bookkeeping

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        """Compute the regularisation term for matrix A.
        Returns a scalar tensor that can be added to the task loss
        before .backward()."""
        cfg = self.cfg
        p = _spectral_distribution(A, cfg.eps)
        H = _shannon_bits(p, cfg.eps)
        rank = float(p.numel())
        H_max = math.log2(max(rank, 2.0))
        H_norm = H / H_max

        # Schedule update: only refresh KL_step / lam_eff every `stride`
        # calls. The reg term still uses the most-recent `last_lam_eff`
        # in between, so optimisation dynamics are smooth.
        do_schedule = (self._call_count % max(1, cfg.stride) == 0)
        self._call_count += 1
        if do_schedule:
            if self.prev_spectrum is not None and \
                    self.prev_spectrum.numel() == p.numel():
                kl_step = float(
                    _kl_bits(self.prev_spectrum, p.detach(), cfg.eps).item()
                )
            else:
                kl_step = 0.0
            self.prev_spectrum = p.detach()

            # Optional: normalise KL by log2(rank) → kl ∈ [0, 1]. With this
            # form, eta has a scale-invariant meaning across architectures.
            kl_for_schedule = (kl_step / max(H_max, cfg.eps)
                                if cfg.kl_normalized else kl_step)

            lam_eff = cfg.lam_0 * math.exp(-cfg.eta * kl_for_schedule)
            lam_eff = max(0.1 * cfg.lam_0, min(10.0 * cfg.lam_0, lam_eff))

            # Optional: EMA on lam_eff. Smooths the schedule across steps so
            # the regulariser does not thrash on transient spectral spikes.
            if cfg.momentum > 0.0:
                if self.lam_eff_ema is None:
                    self.lam_eff_ema = lam_eff
                else:
                    self.lam_eff_ema = (cfg.momentum * self.lam_eff_ema
                                        + (1.0 - cfg.momentum) * lam_eff)
                lam_eff = self.lam_eff_ema
            self.last_kl = kl_step
        else:
            # Reuse the schedule from the most-recent stride boundary.
            lam_eff = (self.last_lam_eff
                        if not math.isnan(self.last_lam_eff)
                        else cfg.lam_0)

        target = torch.tensor(
            cfg.target_entropy, device=A.device, dtype=H_norm.dtype
        )
        a_part = (H_norm - target).pow(2)
        b_part = H_norm
        # KL(p ‖ uniform) / log₂(rank) = 1 − H_norm; adds spread-prior
        # pressure when lam_KL > 0.
        kl_part = 1.0 - H_norm
        reg = lam_eff * (cfg.lam_a * a_part
                         + cfg.lam_b * b_part
                         + cfg.lam_KL * kl_part)

        # Bookkeeping for logging (no autograd tracking).
        self.last_h_norm = float(H_norm.item())
        self.last_lam_eff = lam_eff
        return reg


class SplineSmoothRegulariser:
    """Second-difference smoothness regulariser on spline coefficient
    tensors (Tier 6 / E in the gap-closing plan).

    For each ``Module`` in the host model with a 3-D ``coef`` Parameter
    of shape (S, C, G), computes the squared second-difference along
    the grid axis (the "discrete bending energy" of the spline
    control polygon):

        Δ²coef[..., g] = coef[..., g+2] − 2·coef[..., g+1] + coef[..., g]

        L = lam · mean_coef ( ‖Δ²coef‖² / (S · C · (G − 2)) )

    Discourages oscillatory splines — orthogonal to L1 (which
    collapses) and to coef-spectral-entropy (which targets the basis
    distribution, not the local control-polygon shape).

    Returns the **mean** reg term across discovered coef tensors so
    ``lam`` has the same magnitude interpretation regardless of how
    many splines exist.
    """

    def __init__(self, lam: float):
        self.lam = lam
        self.last_value: float = float("nan")

    def __call__(self, model) -> torch.Tensor:
        coefs = [m.coef for m in model.modules()
                 if hasattr(m, "coef")
                 and isinstance(m.coef, torch.nn.Parameter)
                 and m.coef.dim() == 3]
        if not coefs:
            return torch.zeros((), device=next(model.parameters()).device)
        terms = []
        for c in coefs:
            S, C, G = c.shape
            if G < 3:
                continue
            d2 = c[..., 2:] - 2.0 * c[..., 1:-1] + c[..., :-2]
            denom = float(S * C * (G - 2))
            terms.append(d2.pow(2).sum() / denom)
        if not terms:
            return torch.zeros((), device=next(model.parameters()).device)
        out = torch.stack(terms).mean()
        self.last_value = float(out.detach().item())
        return self.lam * out


class CoefEntropyRegulariser:
    """Spectral-entropy Lyapunov-safe regulariser on spline coefficient
    tensors (Tier 3 / A in the gap-closing plan).

    Targets every ``Module`` in the host model with a 3-D
    ``coef`` Parameter (the (S, C, G) spline tensor shared across the
    KAN branches). Each tensor is reshaped to ``(S·C, G)`` so the
    spectrum spans the grid axis (max rank = G ≈ 5, normalised by
    ``log₂(rank)``).

    Each coef tensor gets its own ``EntropyRegulariser`` instance so
    the per-tensor KL-step schedule and ``lam_eff_ema`` do not
    cross-contaminate. The forward pass returns the **mean** reg term
    across discovered tensors so ``lam_0`` has the same magnitude
    interpretation regardless of how many splines exist.
    """

    def __init__(self, cfg: EntropyRegConfig):
        self.cfg = cfg
        self._sub: dict[int, EntropyRegulariser] = {}
        self.last_h_norm: float = float("nan")
        self.last_lam_eff: float = float("nan")

    def _find_coefs(self, model) -> list[torch.Tensor]:
        return [m.coef for m in model.modules()
                if hasattr(m, "coef")
                and isinstance(m.coef, torch.nn.Parameter)
                and m.coef.dim() == 3]

    def __call__(self, model) -> torch.Tensor:
        coefs = self._find_coefs(model)
        if not coefs:
            return torch.zeros((), device=next(model.parameters()).device)
        regs = []
        h_norms = []
        for c in coefs:
            key = id(c)
            if key not in self._sub:
                self._sub[key] = EntropyRegulariser(self.cfg)
            S, C, G = c.shape
            A = c.reshape(S * C, G)
            regs.append(self._sub[key](A))
            h_norms.append(self._sub[key].last_h_norm)
        out = torch.stack(regs).mean()
        # Bookkeeping for logging.
        self.last_h_norm = sum(h_norms) / max(len(h_norms), 1)
        # Take lam_eff from the first sub-regulariser (they're all on
        # the same schedule given identical cfg).
        first = next(iter(self._sub.values()))
        self.last_lam_eff = first.last_lam_eff
        return out
