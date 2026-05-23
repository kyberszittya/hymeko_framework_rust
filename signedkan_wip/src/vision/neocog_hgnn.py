"""Hypergraph convolution on basic visual datasets with Neocognitron-style
receptive-field clustering.

Connection to the day's signed-cycle work: HSiKAN treats cycles as
hyperedges (sets of vertices joined by signed edges). Here we treat
**receptive-field patches as hyperedges** — each patch is a set of
pixel-vertices joined by spatial proximity. The Neocognitron (1980)
S-cell / C-cell hierarchy gives us the natural multi-scale grouping:

  - S₁ cells: small (5×5) overlapping receptive fields → first-layer hyperedges
  - C₁ cells: pooled patches at lower spatial resolution → second-layer hyperedges

Architecture:

    image (28×28)
        ↓
    pixel-vertex embedding (1 → d)
        ↓
    HypergraphConv₁ over S₁ receptive fields (5×5, stride 2)
        ↓
    HypergraphConv₂ over C₁ pooled patches (10×10 effective receptive field)
        ↓
    global pool → linear classifier → logits

Real (not stub) hypergraph convolution following Feng-You-Zhang-Ji 2019
(HGNN):

    X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ + b

where H is the (V × E) incidence matrix, D_v / D_e are diagonal
degree matrices, W_e is a learnable per-hyperedge weight, Θ is a
per-feature linear projection.

Run:
    python -m signedkan_wip.src.vision.neocog_hgnn \
        --dataset mnist --hidden 32 --n-epochs 5
    python -m signedkan_wip.src.vision.neocog_hgnn \
        --dataset fashion --hidden 32 --n-epochs 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ─── Real hypergraph convolution ────────────────────────────────────


def build_hyper_incidence(
    H: int, W: int, kernel: int, stride: int,
) -> Tuple[torch.Tensor, int]:
    """Build a sparse (V × E) incidence tensor for an `H × W` image
    with overlapping `kernel × kernel` receptive fields, stride
    `stride`.

    Each receptive field becomes one hyperedge containing all
    `kernel²` pixels inside it. Returns the dense incidence (V, E)
    plus the number of hyperedges E.
    """
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
                    pixel = (r + dr) * W + (c + dc)
                    inc[pixel, e_idx] = 1.0
            e_idx += 1
    return inc, n_edges


class HypergraphConv(nn.Module):
    """HGNN-style hypergraph convolution.

    Following Feng et al. 2019:
        X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ
    with W_e a learnable diagonal hyperedge weight, Θ a linear projection.
    """
    def __init__(self, d_in: int, d_out: int, n_edges: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=True)
        # Per-hyperedge learnable weight (diagonal).
        self.W_e = nn.Parameter(torch.ones(n_edges))

    def forward(self, x: torch.Tensor, inc: torch.Tensor,
                D_v_inv_sqrt: torch.Tensor, D_e_inv: torch.Tensor) -> torch.Tensor:
        """
        x: (B, V, d_in)
        inc: (V, E) incidence (precomputed; same for whole batch)
        D_v_inv_sqrt: (V,) diagonal D_v^{-1/2}
        D_e_inv: (E,) diagonal D_e^{-1}
        Returns: (B, V, d_out)
        """
        x_proj = self.proj(x)                         # (B, V, d_out)
        # X̃ = D_v^{-1/2} X
        x_norm = D_v_inv_sqrt.unsqueeze(0).unsqueeze(-1) * x_proj
        # H^T X̃ : (E, V) @ (B, V, d) -> (B, E, d) per-batch
        # Use einsum for clarity.
        h_t_x = torch.einsum("ve,bvd->bed", inc, x_norm)
        # D_e^{-1} W_e (H^T X̃)
        h_t_x = h_t_x * (D_e_inv * self.W_e).unsqueeze(0).unsqueeze(-1)
        # H (D_e^{-1} W_e H^T X̃) : (V, E) @ (B, E, d) -> (B, V, d)
        out = torch.einsum("ve,bed->bvd", inc, h_t_x)
        # D_v^{-1/2} on the left
        out = D_v_inv_sqrt.unsqueeze(0).unsqueeze(-1) * out
        return out


class NeocogHGNN(nn.Module):
    """Two-layer Neocognitron-inspired HGNN classifier."""
    def __init__(self, H: int, W: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.H, self.W = H, W
        # S₁: 5×5 receptive fields, stride 2
        inc1, n_e1 = build_hyper_incidence(H, W, kernel=5, stride=2)
        # Build degree vectors.
        D_v1 = inc1.sum(dim=1).clamp(min=1)
        D_e1 = inc1.sum(dim=0).clamp(min=1)
        self.register_buffer("inc1", inc1)
        self.register_buffer("D_v1_inv_sqrt", D_v1.pow(-0.5))
        self.register_buffer("D_e1_inv", D_e1.pow(-1.0))

        # C₁ analogue: bigger receptive fields (8×8, stride 4)
        inc2, n_e2 = build_hyper_incidence(H, W, kernel=8, stride=4)
        D_v2 = inc2.sum(dim=1).clamp(min=1)
        D_e2 = inc2.sum(dim=0).clamp(min=1)
        self.register_buffer("inc2", inc2)
        self.register_buffer("D_v2_inv_sqrt", D_v2.pow(-0.5))
        self.register_buffer("D_e2_inv", D_e2.pow(-1.0))

        self.embed = nn.Linear(1, hidden)
        self.conv1 = HypergraphConv(hidden, hidden, n_e1)
        self.conv2 = HypergraphConv(hidden, hidden, n_e2)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        # x_img: (B, 1, H, W)  →  (B, V=H*W, 1)  →  embed  →  conv x2  →  pool  →  head
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        v = F.relu(self.conv1(v, self.inc1, self.D_v1_inv_sqrt, self.D_e1_inv))
        v = F.relu(self.conv2(v, self.inc2, self.D_v2_inv_sqrt, self.D_e2_inv))
        # Global mean pool over vertices.
        z = v.mean(dim=1)
        return self.head(z)


# ─── Baseline: simple MLP and CNN ───────────────────────────────────


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.shape[0], -1))


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(1, hidden, 5, stride=2, padding=2)
        self.c2 = nn.Conv2d(hidden, hidden, 5, stride=2, padding=2)
        self.head = nn.Linear(hidden * 7 * 7, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.c1(x))
        z = F.relu(self.c2(z))
        return self.head(z.view(z.shape[0], -1))


# ─── Train + eval ───────────────────────────────────────────────────


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion", "cifar10"],
                    default="mnist")
    ap.add_argument("--model", choices=["hgnn", "mlp", "cnn"], default="hgnn")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Convert to grayscale for 1-channel HGNN compatibility
        transform = T.Compose([T.Grayscale(), T.ToTensor()])
        train_ds = torchvision.datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(
            data_root, train=False, download=True, transform=transform)
        H, W, n_classes = 32, 32, 10
    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "hgnn":
        model = NeocogHGNN(H, W, n_classes, hidden=args.hidden).to(device)
    elif args.model == "mlp":
        model = MLP(H * W, n_classes, hidden=args.hidden).to(device)
    elif args.model == "cnn":
        if H != 28 or W != 28:
            print("warn: TinyCNN currently hardcoded for 28x28; CIFAR may fail",
                  flush=True)
        model = TinyCNN(n_classes, hidden=args.hidden).to(device)

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

    # Eval
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
        "model": args.model,
        "hidden": args.hidden,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "test_accuracy": acc,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "n_params": n_params(model),
    }))


if __name__ == "__main__":
    main()
