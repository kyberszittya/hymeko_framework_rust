"""HSiKAN-style hypergraph convolution on vision (the "generalize the
hypergraph conv" version, not HGNN).

The user's correction on 2026-05-04 was that the overnight vision run
used HGNN (Feng 2019) directly instead of generalizing the
HSiKAN-style hypergraph conv we built for signed graphs. This file
ports HSiKAN to vision:

| HSiKAN (signed graphs)           | this file (vision)                    |
|----------------------------------|---------------------------------------|
| edge sign s ∈ {-1, +1}           | contrast polarity p = sign(I - mean)  |
| signed-branch sum over (+,-)     | polarity-branch sum over (above,below)|
| Catmull-Rom activation per branch| Catmull-Rom activation per branch     |
| arities k ∈ {3,4,5} (cycle size) | arities k ∈ {5,8,12} (RF size)        |
| αₖ learnable arity mixer         | αₖ learnable RF-scale mixer           |
| signed-incidence H_k             | signed-incidence H_k (vert × patch)   |

Architecture:

    image (B, 1, H, W)
        ↓
    pixel embed (1 → d)
        ↓
    HSiKANVisionLayer × n_layers
        for each arity k (RF size):
            split each patch into above-mean and below-mean branches
            signed-branch hypergraph conv with Catmull-Rom activation
        αₖ-mix the n_arity branch outputs
        ↓
    global mean pool → linear classifier

Run:
    python -m signedkan_wip.src.vision.hsikan_vision \
        --dataset mnist --hidden 32 --n-epochs 5 --arities 5,8,12
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# ─── Receptive-field incidence ──────────────────────────────────────


def build_rf_incidence(
    H: int, W: int, kernel: int, stride: int,
) -> tuple[torch.Tensor, int]:
    """Sparse (V × E) receptive-field incidence: each kernel×kernel
    receptive field becomes one hyperedge containing all kernel² pixels."""
    rows = list(range(0, H - kernel + 1, stride))
    cols = list(range(0, W - kernel + 1, stride))
    n_edges = len(rows) * len(cols)
    n_verts = H * W
    inc = torch.zeros(n_verts, n_edges, dtype=torch.float32)
    e_idx = 0
    for r in rows:
        for c in cols:
            for dr in range(kernel):
                for dc in range(kernel):
                    inc[(r + dr) * W + (c + dc), e_idx] = 1.0
            e_idx += 1
    return inc, n_edges


# ─── Catmull-Rom activation (lifted from signedkan_wip.src.splines) ─


class CRActivation(nn.Module):
    """Per-channel Catmull-Rom learnable activation (m control points
    on [-3, 3]). One activation per (channel, branch) = one set of
    control points."""
    def __init__(self, channels: int, n_branches: int, m: int = 8,
                 init_scale: float = 0.05):
        super().__init__()
        self.m = m
        # Control point y-values; x is uniform on [-3, 3].
        # Shape (n_branches, channels, m)
        self.cpts = nn.Parameter(
            init_scale * torch.randn(n_branches, channels, m)
        )
        x = torch.linspace(-3.0, 3.0, m)
        self.register_buffer("x_grid", x)

    def forward(self, x: torch.Tensor, branch_idx: int) -> torch.Tensor:
        """x: (..., channels), output: (..., channels)
        Catmull-Rom interpolation between control points."""
        # Find segment via clamp + bucketize.
        x_norm = (x.clamp(-3.0, 3.0) + 3.0) / 6.0 * (self.m - 1)  # in [0, m-1]
        i = x_norm.floor().long().clamp(0, self.m - 2)            # in [0, m-2]
        t = (x_norm - i.float()).clamp(0.0, 1.0)
        # Gather cpts for branches: (n_branches, channels, m) -> we need
        # cpts[branch_idx, ch, i] etc. with broadcasting over (...).
        cp = self.cpts[branch_idx]  # (channels, m)

        # Catmull-Rom needs cpts at i-1, i, i+1, i+2.
        i0 = (i - 1).clamp(0, self.m - 1)
        i1 = i
        i2 = (i + 1).clamp(0, self.m - 1)
        i3 = (i + 2).clamp(0, self.m - 1)

        ch_idx = torch.arange(cp.shape[0], device=x.device).expand_as(i)
        p0 = cp[ch_idx, i0]
        p1 = cp[ch_idx, i1]
        p2 = cp[ch_idx, i2]
        p3 = cp[ch_idx, i3]

        t2 = t * t
        t3 = t2 * t
        out = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        return out


# ─── Signed-branch hypergraph conv ──────────────────────────────────


class SignedBranchConv(nn.Module):
    """For one arity k:
       - patch-mean μ_e from H_k incidence
       - per-pixel polarity branch: above (s=+1) or below (s=-1) μ_e
       - Catmull-Rom activation per branch
       - aggregated back to vertices via H_k

    This is the HSiKAN structure ported to vision: the "sign" of a
    (pixel, patch) pair is determined by whether the pixel is above
    or below the patch mean."""
    def __init__(self, d_in: int, d_out: int, n_edges: int, m: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_out, bias=False)
        # Per-hyperedge scalar weight (analogous to W_e in HGNN).
        self.W_e = nn.Parameter(torch.ones(n_edges))
        # Two branches: 0 = above-mean (s=+1), 1 = below-mean (s=-1).
        self.activation = CRActivation(channels=d_out, n_branches=2, m=m)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor, inc: torch.Tensor,
                D_v_inv_sqrt: torch.Tensor, D_e_inv: torch.Tensor) -> torch.Tensor:
        """
        x: (B, V, d_in)
        inc: (V, E) sparse-friendly incidence (dense here for clarity)
        Returns: (B, V, d_out)
        """
        x_proj = self.proj_in(x)  # (B, V, d_out)
        # 1. compute per-patch mean (per channel): μ_e = (H^T x) / D_e
        h_t_x = torch.einsum("ve,bvd->bed", inc, x_proj)
        mu_e = h_t_x * D_e_inv.unsqueeze(0).unsqueeze(-1)   # (B, E, d_out)

        # 2. per (pixel, patch) determine polarity. We need to know,
        # for each pixel-patch pair, whether x_proj[pixel] > μ[patch].
        # To keep this dense-friendly: pull μ back to vertices weighted
        # by incidence (avg μ over patches the pixel belongs to).
        mu_v = torch.einsum("ve,bed->bvd", inc, mu_e) * D_v_inv_sqrt.unsqueeze(0).unsqueeze(-1) ** 2
        # polarity: above (+1) or below (0). Use float for differentiable α below.
        polarity = (x_proj - mu_v)   # (B, V, d_out)

        # 3. Compute the two branch contributions, gated by polarity sign.
        above = self.activation(polarity, branch_idx=0)
        below = self.activation(polarity, branch_idx=1)
        gate = torch.sigmoid(polarity * 4.0)   # smooth sign
        x_branch = gate * above + (1.0 - gate) * below   # (B, V, d_out)

        # 4. Hypergraph propagation H W_e D_e^{-1} H^T over the
        # branch-activated features.
        h_t_x_b = torch.einsum("ve,bvd->bed", inc, x_branch)
        h_t_x_b = h_t_x_b * (D_e_inv * self.W_e).unsqueeze(0).unsqueeze(-1)
        out = torch.einsum("ve,bed->bvd", inc, h_t_x_b)
        out = D_v_inv_sqrt.unsqueeze(0).unsqueeze(-1) * out
        return out + self.bias


# ─── HSiKAN-style multi-arity vision layer ──────────────────────────


class HSiKANVisionLayer(nn.Module):
    """For each receptive-field size k (an "arity"), run a
    SignedBranchConv. Combine arities with a learnable αₖ mixer
    (softmax-normalized so α sums to 1, as in MixedAritySignedKAN)."""
    def __init__(self, d_in: int, d_out: int, H: int, W: int,
                 arities: list[tuple[int, int]], m: int = 8):
        """arities: list of (kernel, stride) pairs. The default for
        28×28 input is [(5,2), (8,4), (12,4)] — a 3-scale RF mix."""
        super().__init__()
        self.arities = arities
        self.convs = nn.ModuleList()
        for (k, s) in arities:
            inc, n_e = build_rf_incidence(H, W, kernel=k, stride=s)
            D_v = inc.sum(dim=1).clamp(min=1)
            D_e = inc.sum(dim=0).clamp(min=1)
            conv = SignedBranchConv(d_in, d_out, n_e, m=m)
            self.convs.append(conv)
            self.register_buffer(f"inc_{k}_{s}", inc)
            self.register_buffer(f"Dv_{k}_{s}", D_v.pow(-0.5))
            self.register_buffer(f"De_{k}_{s}", D_e.pow(-1.0))
        # αₖ mixer (one weight per arity, softmax-normalized).
        self.alpha_logits = nn.Parameter(torch.zeros(len(arities)))

    def alpha_weights(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, d_in)
        alpha = self.alpha_weights()
        out = 0.0
        for i, (k, s) in enumerate(self.arities):
            inc = getattr(self, f"inc_{k}_{s}")
            D_v = getattr(self, f"Dv_{k}_{s}")
            D_e = getattr(self, f"De_{k}_{s}")
            out = out + alpha[i] * self.convs[i](x, inc, D_v, D_e)
        return out


class HSiKANVisionClassifier(nn.Module):
    """Embed → n HSiKANVisionLayer → mean-pool → linear head."""
    def __init__(self, H: int, W: int, n_classes: int,
                 hidden: int = 32, n_layers: int = 2,
                 arities: list[tuple[int, int]] | None = None,
                 m: int = 8):
        super().__init__()
        if arities is None:
            arities = [(5, 2), (8, 4), (12, 4)]
        self.embed = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            HSiKANVisionLayer(hidden, hidden, H, W, arities, m=m)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        for L in self.layers:
            v = L(v) + v   # residual; HSiKAN-style
        z = v.mean(dim=1)
        return self.head(z)

    def alpha_summary(self) -> dict:
        return {
            f"layer_{i}": dict(zip(
                [f"k={k}_s={s}" for (k, s) in L.arities],
                [round(a.item(), 4) for a in L.alpha_weights()],
            ))
            for i, L in enumerate(self.layers)
        }


# ─── Train + eval (mirrors neocog_hgnn.main) ────────────────────────


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion", "cifar10"],
                    default="mnist")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--m", type=int, default=8, help="Catmull-Rom control points")
    ap.add_argument("--arities", type=str, default="5_2,8_4,12_4",
                    help="comma-separated kernel_stride pairs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arities = []
    for tok in args.arities.split(","):
        k, s = tok.split("_")
        arities.append((int(k), int(s)))

    # Load dataset.
    transform = T.Compose([T.ToTensor()])
    data_root = Path("/tmp/torchvision_cache")
    data_root.mkdir(exist_ok=True)
    if args.dataset == "mnist":
        train_ds = torchvision.datasets.MNIST(data_root, train=True,
                                                download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST(data_root, train=False,
                                               download=True, transform=transform)
        H, W, n_classes = 28, 28, 10
    elif args.dataset == "fashion":
        train_ds = torchvision.datasets.FashionMNIST(
            data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.FashionMNIST(
            data_root, train=False, download=True, transform=transform)
        H, W, n_classes = 28, 28, 10
    elif args.dataset == "cifar10":
        transform = T.Compose([T.Grayscale(), T.ToTensor()])
        train_ds = torchvision.datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(
            data_root, train=False, download=True, transform=transform)
        H, W, n_classes = 32, 32, 10

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HSiKANVisionClassifier(
        H, W, n_classes, hidden=args.hidden, n_layers=args.n_layers,
        arities=arities, m=args.m,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0

    model.eval()
    correct = total = 0
    t1 = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    eval_time = time.time() - t1
    acc = correct / total

    print(json.dumps({
        "dataset": args.dataset,
        "model": "hsikan_vision",
        "arities": arities,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "m": args.m,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "test_accuracy": acc,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "n_params": n_params(model),
        "alpha": model.alpha_summary(),
    }))


if __name__ == "__main__":
    main()
