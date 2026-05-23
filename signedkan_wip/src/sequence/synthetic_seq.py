"""Synthetic Tier-1 benchmark for Sequential HSiKAN + CliffordFIR.

Generates mixed signal/information sequences:

  Class 0 ("signal-class"): the continuous channel raw[:, 0] carries
    a sinusoidal tone at FREQ_HZ cycles per length L; the
    information channel raw[:, 1] is pure white noise; σ is
    uninformative (random ±1).
    A pure-continuous detector (Fourier / dilated conv / TCN) can
    solve this class trivially.

  Class 1 ("info-class"): raw[:, 0] is pure white noise; raw[:, 1]
    is a noisy proxy of σ (cumulative-sign smoothed); σ follows a
    period-P pattern such that the windowed σ-product over a length-K
    sliding window is consistently +1 (or another predictable target).
    A σ-product-aware aggregator (HSiKAN) can solve this class
    near-perfectly; a same-param Fourier detector cannot
    preferentially.

The dual-path model should solve both classes. A single-path control
should solve only one. This is the discriminating Tier-1 benchmark.

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/ §6.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SynthConfig:
    """Generator hyperparameters."""
    n_samples: int = 1024
    L: int = 256
    freq_cycles: int = 24       # cycles per L → tone of class 0
    info_period: int = 7         # σ-pattern period for class 1
    sign_proxy_noise: float = 0.5  # std of noise added to raw[:, 1] in class 1
    signal_amplitude: float = 1.0  # amplitude of the class-0 tone
    noise_std: float = 1.0       # std of white noise
    seed: int = 0


def _make_class0(cfg: SynthConfig, gen: torch.Generator
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (raw, sigma) for one class-0 sample."""
    L = cfg.L
    t = torch.arange(L, dtype=torch.float32)
    tone = cfg.signal_amplitude * torch.cos(2.0 * torch.pi * cfg.freq_cycles * t / L)
    raw0 = tone + torch.randn(L, generator=gen) * cfg.noise_std
    raw1 = torch.randn(L, generator=gen) * cfg.noise_std
    raw = torch.stack([raw0, raw1], dim=-1)        # (L, 2)
    # σ random in {-1, +1}; uninformative for class 0.
    sigma = torch.where(
        torch.rand(L, generator=gen) > 0.5,
        torch.ones(L), -torch.ones(L),
    )
    return raw, sigma


def _make_class1(cfg: SynthConfig, gen: torch.Generator
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (raw, sigma) for one class-1 sample.

    σ follows a period-P pattern: σ[t] = sign(sin(2π t / P + φ)) where
    φ is drawn per-sample so the pattern phase varies but the windowed
    σ-product structure is preserved.
    """
    L = cfg.L
    t = torch.arange(L, dtype=torch.float32)
    phi = torch.rand(1, generator=gen).item() * 2.0 * torch.pi
    # Use a small offset to avoid the zero-crossing
    sigma_continuous = torch.sin(2.0 * torch.pi * t / cfg.info_period + phi)
    sigma = torch.sign(sigma_continuous + 1e-3)              # robust against zero
    # Make sure no zeros leak through.
    sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
    raw0 = torch.randn(L, generator=gen) * cfg.noise_std
    # raw1 is the "information channel": a smoothed noisy proxy of σ
    # so the model can derive σ from raw[:, 1] alone if it wants to.
    raw1 = sigma + torch.randn(L, generator=gen) * cfg.sign_proxy_noise
    raw = torch.stack([raw0, raw1], dim=-1)
    return raw, sigma


def make_dataset(cfg: SynthConfig = SynthConfig()
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``n_samples`` sequences split 50/50 between class 0 and 1.

    Returns
    -------
    raw : (N, L, 2)  per-position scalar features
    sigma : (N, L)   ground-truth sign stream (used for supervised-sign training)
    labels : (N,)    class labels in {0, 1}
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    raw_list = []
    sig_list = []
    lab_list = []
    for i in range(cfg.n_samples):
        cls = i % 2
        if cls == 0:
            r, s = _make_class0(cfg, gen)
        else:
            r, s = _make_class1(cfg, gen)
        raw_list.append(r)
        sig_list.append(s)
        lab_list.append(cls)
    raw = torch.stack(raw_list, dim=0)              # (N, L, 2)
    sigma = torch.stack(sig_list, dim=0)            # (N, L)
    labels = torch.tensor(lab_list, dtype=torch.long)
    # Shuffle so class 0 / 1 don't appear in a deterministic stride;
    # use a fresh seed-derived permutation so the split-by-seed pattern
    # is reproducible.
    perm = torch.randperm(cfg.n_samples, generator=gen)
    return raw[perm], sigma[perm], labels[perm]


# ─── Trivial oracle baselines for sanity checks ──────────────────────


def fourier_class0_oracle(raw: torch.Tensor, cfg: SynthConfig) -> torch.Tensor:
    """An oracle that detects the class-0 tone by computing the
    spectral energy at FREQ_HZ in raw[:, 0]. Used in tests to confirm
    the generator produces class-separable sequences.

    Returns a (N,) score; positive scores indicate class 0.
    """
    N, L, _ = raw.shape
    t = torch.arange(L, dtype=torch.float32)
    basis_c = torch.cos(2.0 * torch.pi * cfg.freq_cycles * t / L)
    basis_s = torch.sin(2.0 * torch.pi * cfg.freq_cycles * t / L)
    # Project raw[:, :, 0] onto the basis; the magnitude is the
    # tone energy.
    x = raw[..., 0]                                  # (N, L)
    cs = (x * basis_c).mean(dim=-1)
    sn = (x * basis_s).mean(dim=-1)
    return torch.sqrt(cs * cs + sn * sn)


def sigma_product_class1_oracle(
    sigma: torch.Tensor, K: int = 4,
) -> torch.Tensor:
    """An oracle that classifies based on the variance of the windowed
    σ-product. Class 1 has a periodic σ-pattern → low variance;
    class 0 has random σ → high variance of the windowed product.

    Returns a (N,) score; HIGH variance indicates class 0, LOW indicates class 1.
    """
    N, L = sigma.shape
    # Left-pad zero, unfold, take product along K dim.
    pad = torch.zeros(N, K - 1)
    s_pad = torch.cat([pad, sigma], dim=1)
    w = s_pad.unfold(dimension=1, size=K, step=1)   # (N, L, K)
    # Treat zeros as +1 (no effect).
    w_nz = torch.where(w == 0, torch.ones_like(w), w)
    prod = w_nz.prod(dim=-1)                        # (N, L) in {-1, +1}
    # Score = variance of prod across positions.
    return prod.var(dim=-1)
