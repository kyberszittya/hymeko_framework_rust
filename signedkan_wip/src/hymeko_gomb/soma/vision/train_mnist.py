"""GömbSoma MNIST benchmark — Phase 3-V-bench.

First hard number for the GömbSoma vision stack: train
WalkConvImageClassifier on MNIST for n_epochs with n_seeds different
random initialisations; report mean ± pstd test accuracy, parameter
count, and wall time.

A parameter-light Linear-baseline (Linear(784, 10) = 7 850 params)
trains alongside as a control: if GömbSoma's 2 010 params can't
beat a 7 850-param linear classifier on MNIST, the walks-only
sensorimotor hypothesis is in trouble. If it ties or wins at fewer
params, walks are doing real structural work.

Plan: docs/plans/2026-05-14-gomb-soma/.

Run via the orchestrator script:
    signedkan_wip/experiments/run_gomb_soma_mnist_bench_2026_05_14.sh
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    WalkConvImageClassifier,
)

DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "mnist"


class LinearBaseline(nn.Module):
    """Trivial reference: flatten + Linear → 10. ~7 850 params."""

    def __init__(self, image_h: int = 28, image_w: int = 28,
                 in_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels * image_h * image_w, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            return self.fc(images.reshape(-1))
        return self.fc(images.reshape(images.shape[0], -1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(n_train: int, n_test: int, batch_size: int, seed: int):
    """MNIST with sub-sampling for speed."""
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(
        str(DATA_ROOT), train=True, download=True, transform=tfm,
    )
    test = datasets.MNIST(
        str(DATA_ROOT), train=False, download=True, transform=tfm,
    )
    rng = np.random.default_rng(seed)
    if n_train < len(train):
        idx = rng.choice(len(train), size=n_train, replace=False)
        train = Subset(train, idx.tolist())
    if n_test < len(test):
        idx = rng.choice(len(test), size=n_test, replace=False)
        test = Subset(test, idx.tolist())
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader


def load_cluttered(n_train: int, n_test: int, batch_size: int, seed: int,
                   canvas: int):
    """Single-digit Cluttered-MNIST classification (digit at a random position
    on a canvas×canvas field). Train/test draw from disjoint MNIST splits."""
    from signedkan_wip.src.vision.cluttered_classification import (
        ClutteredMNISTClassification,
    )
    train = ClutteredMNISTClassification(
        n_samples=n_train, canvas=canvas, seed=seed, train=True)
    test = ClutteredMNISTClassification(
        n_samples=n_test, canvas=canvas, seed=seed + 10_000, train=False)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def build_model(model_type: str, device: torch.device,
                image_h: int = 28, image_w: int = 28) -> nn.Module:
    if model_type == "gomb_soma":
        # Sign-as-routing base-Soma (the 2026-06-15 falsified arm): dual
        # sign-branched banks + sign-blind sum-pool.
        m = WalkConvImageClassifier(
            image_h=image_h, image_w=image_w, patch_size=4,
            in_channels=1, d_hidden=16, n_classes=10,
        )
    elif model_type == "gomb_soma_holonomy":
        # Sign-as-connection: single message bank + σ-product holonomy pool
        # (M_v @ (σ ⊙ m)). The operator the 2026-06-15 ablation never tested.
        from signedkan_wip.src.hymeko_gomb.soma.hg_conv import Aggregation
        m = WalkConvImageClassifier(
            image_h=image_h, image_w=image_w, patch_size=4,
            in_channels=1, d_hidden=16, n_classes=10,
            use_sign_branching=False, aggregation=Aggregation.HOLONOMY,
        )
    elif model_type in ("gomb_soma_cheby", "gomb_soma_flat", "gomb_soma_cheby_flat"):
        # 2x2 cell × readout sweep over the base-Soma anchor (routing/SUM held
        # fixed). cell on = Chebyshev-CR patches + Chebyshev-CR messages;
        # readout flatten = position-preserving. See
        # docs/plans/2026-06-29-soma-cheby-cell-readout-sweep/.
        from signedkan_wip.src.hymeko_gomb.soma.hg_conv import MessageActivation
        from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
            PatchEncoder,
            Readout,
        )
        cheby = "cheby" in model_type
        flat = "flat" in model_type
        m = WalkConvImageClassifier(
            image_h=image_h, image_w=image_w, patch_size=4,
            in_channels=1, d_hidden=16, n_classes=10,
            message_activation=(MessageActivation.CHEBY_CR if cheby
                                else MessageActivation.GELU),
            patch_encoder=(PatchEncoder.CHEBY_CR if cheby else PatchEncoder.LINEAR),
            readout=(Readout.FLATTEN if flat else Readout.MEAN_POOL),
        )
    elif model_type in ("gomb_soma_attn", "gomb_soma_posattn", "gomb_soma_tree",
                         "gomb_soma_tree_static"):
        # Scalable position-aware readouts (out_dim independent of grid size).
        # ATTENTION = content-weighted set pool; POS_ATTENTION adds a learned
        # per-patch positional embedding; SPATIAL_TREE = dynamic quadtree-pyramid
        # pool (multi-scale position + learned per-cell gate). See
        # docs/plans/2026-06-29-soma-position-aware-readout-program/.
        from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
            Readout,
        )
        ro = {"gomb_soma_attn": Readout.ATTENTION,
              "gomb_soma_posattn": Readout.POS_ATTENTION,
              "gomb_soma_tree": Readout.SPATIAL_TREE,
              "gomb_soma_tree_static": Readout.SPATIAL_TREE_STATIC}[model_type]
        m = WalkConvImageClassifier(
            image_h=image_h, image_w=image_w, patch_size=4,
            in_channels=1, d_hidden=16, n_classes=10,
            readout=ro,
        )
    elif model_type.startswith("holo_"):
        # Holonomy-group ablation (flatten readout): none / routing / z2 / u1.
        # See signedkan_wip/.../soma/vision/holonomy_walk.py.
        from signedkan_wip.src.hymeko_gomb.soma.vision.holonomy_walk import (
            Holonomy,
            HolonomyClassifier,
        )
        mode = Holonomy(model_type.split("holo_", 1)[1])
        m = HolonomyClassifier(
            image_h=image_h, image_w=image_w, patch_size=4,
            in_channels=1, d_hidden=16, n_classes=10, mode=mode,
        )
    elif model_type == "linear":
        m = LinearBaseline(image_h=image_h, image_w=image_w)
    elif model_type.startswith("ricci_stim"):
        # RicciStim 3-branch backbone. Name suffixes (Phase 2 of the
        # position-aware-readout program): `_up` = upgraded aggregators;
        # `_enc` = encoder-only ablation (zero the walk/poly/tri branches —
        # the structural control); `_attn` = attention readout (vs mean-pool).
        from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_classifier import (
            RicciStimClassifier,
        )
        from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
            Readout,
        )
        up = "_up" in model_type
        enc = "_enc" in model_type
        attn = "_attn" in model_type
        tree = "_tree" in model_type
        readout = (Readout.SPATIAL_TREE if tree
                   else Readout.ATTENTION if attn
                   else Readout.MEAN_POOL)
        # Highway is ON for all arms: it carries the encoder features to the head,
        # so the encoder-only ablation (enc) is a *real* control (encoder skip
        # alone) rather than zero-input. full−enc then isolates the structural
        # branches' contribution. (See 2026-06-29 Phase 2 confound fix.)
        m = RicciStimClassifier(
            image_h=image_h, image_w=image_w, d_hidden=16, n_classes=10,
            max_depth=1, use_arity_mixer=up, use_highway=True, use_pyramid=up,
            ablate_structural_branches=enc,
            readout=readout,
            cache_geometry=True,  # deterministic per-index images → safe; ~2x faster
        )
    else:
        raise SystemExit(f"unknown model_type {model_type!r}")
    return m.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            n_correct += (pred == y).sum().item()
            n_total += y.shape[0]
    return n_correct / max(1, n_total)


def train_one_seed(
    model_type: str,
    seed: int,
    n_train: int,
    n_test: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    dataset: str = "mnist",
    canvas: int = 48,
) -> dict:
    set_seed(seed)
    if dataset == "cluttered":
        train_loader, test_loader = load_cluttered(
            n_train, n_test, batch_size, seed, canvas)
        image_h = image_w = canvas
    else:
        train_loader, test_loader = load_mnist(n_train, n_test, batch_size, seed)
        image_h = image_w = 28
    model = build_model(model_type, device, image_h, image_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        model.train()
        ep_loss, n_steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_steps += 1
        train_loss = ep_loss / max(1, n_steps)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"  [seed={seed}] epoch {epoch + 1}/{n_epochs} "
            f"loss={train_loss:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )
    wall = time.perf_counter() - t0
    final_acc = evaluate(model, test_loader, device)
    return {
        "model": model_type,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_params": n_params,
        "test_acc": final_acc,
        "wall_s": wall,
        "dataset": dataset,
        "canvas": (canvas if dataset == "cluttered" else 28),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("gomb_soma", "gomb_soma_holonomy",
                                          "gomb_soma_cheby", "gomb_soma_flat",
                                          "gomb_soma_cheby_flat", "gomb_soma_attn",
                                          "gomb_soma_posattn", "gomb_soma_tree",
                                          "gomb_soma_tree_static", "linear",
                                          "ricci_stim", "ricci_stim_up",
                                          "ricci_stim_attn", "ricci_stim_enc",
                                          "ricci_stim_enc_attn",
                                          "ricci_stim_tree", "ricci_stim_enc_tree",
                                          "holo_none", "holo_routing",
                                          "holo_z2", "holo_u1"),
                     default="gomb_soma")
    ap.add_argument("--dataset", choices=("mnist", "cluttered"), default="mnist",
                     help="centred MNIST or single-digit Cluttered-MNIST (random position)")
    ap.add_argument("--canvas", type=int, default=48,
                     help="canvas side for --dataset cluttered")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default=None,
                     help="cuda / cpu; auto-detect if omitted")
    ap.add_argument("--out-jsonl", default=None,
                     help="append per-seed records to this JSONL")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[bench] model={args.model} dataset={args.dataset} device={device} "
          f"n_train={args.n_train} n_test={args.n_test} "
          f"n_epochs={args.n_epochs}"
          + (f" canvas={args.canvas}" if args.dataset == "cluttered" else ""),
          flush=True)

    rows = []
    for seed in args.seeds:
        print(f"[bench] seed={seed}", flush=True)
        rec = train_one_seed(
            args.model, seed, args.n_train, args.n_test,
            args.n_epochs, args.batch_size, args.lr, device,
            dataset=args.dataset, canvas=args.canvas,
        )
        rows.append(rec)
        if args.out_jsonl:
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")

    accs = [r["test_acc"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    print(f"[bench] === summary ({args.model}, n={len(accs)} seeds) ===",
          flush=True)
    print(f"[bench]   n_params = {rows[0]['n_params']}", flush=True)
    print(f"[bench]   test_acc per seed: {[round(a, 4) for a in accs]}",
          flush=True)
    if len(accs) > 1:
        m = statistics.mean(accs)
        s = statistics.pstdev(accs)
        print(f"[bench]   mean = {m:.4f}  pstd = {s:.4f}", flush=True)
    print(f"[bench]   wall per seed: "
          f"{[round(w, 1) for w in walls]} s", flush=True)


if __name__ == "__main__":
    main()
