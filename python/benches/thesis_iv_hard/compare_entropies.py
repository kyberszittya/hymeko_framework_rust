"""Compare weight-side spectral entropy with activation-side Sanchez-
Giraldo Rényi-2 entropy across training.

Trains a baseline (no regulariser) model briefly, logging at each epoch:

  - H_weight(layer_l)    — Shannon entropy of the normalised-Laplacian
                           eigenvalues of the per-layer adjacency
                           (the quantity the existing 9 regulariser
                           arms operate on);
  - H_2(K_l) act         — Rényi-2 entropy of the trace-1 Gaussian-RBF
                           Gram of the layer's activations on a fixed
                           validation batch;
  - I_2(K_i; K_j) act    — pairwise cross-layer Rényi-2 mutual
                           information for adjacent layers.

Renders a 2x2 figure comparing the trajectories on MNIST plain MLP
(MNISTNetSmall) and on the spirals synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)
import run_benchmark as R

# Reuse the unit-tested helpers from the sanity script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sanchez_giraldo import (  # type: ignore
    gaussian_gram,
    hadamard_join,
    renyi2_entropy,
)


REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def weight_side_entropy(model, view="dataflow") -> float:
    """Existing-framework entropy: Shannon over normalised-Laplacian
    eigvals of the weight adjacency."""
    adj = model.spectral_adjacency(view=view)
    eigs = R.normalized_laplacian_eigvals(adj)
    if eigs is None:
        return float("nan")
    H = R.spectral_entropy_bits(eigs)
    return float(H.item())


def collect_activations(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Forward x through the model with hooks on each Linear that
    appears in spectral_weights(). Returns a list of activations
    (post-linear, pre-activation) in the same order as
    model.spectral_weights()."""
    # Identify the Linear modules whose `.weight` is in spectral_weights()
    spec_ws = model.spectral_weights()
    weight_ids = [id(w) for w in spec_ws]
    target_modules = []
    for m in model.modules():
        if hasattr(m, "weight") and id(m.weight) in weight_ids:
            target_modules.append(m)
    # Sort by the order in which the weights appear in spectral_weights()
    target_modules.sort(key=lambda m: weight_ids.index(id(m.weight)))

    captured: list[torch.Tensor] = []
    handles = []
    for mod in target_modules:
        def hook(_m, _i, out, store=captured):
            store.append(out.detach())
        handles.append(mod.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return captured


def activation_renyi(activations: list[torch.Tensor]) -> tuple[list[float], list[float]]:
    """Per-layer Rényi-2 entropy and pairwise (i, i+1) mutual info."""
    K_list = [gaussian_gram(a.flatten(start_dim=1)) for a in activations]
    H_per_layer = [float(renyi2_entropy(K).item()) for K in K_list]
    I_adjacent: list[float] = []
    for i in range(len(K_list) - 1):
        K_j = hadamard_join(K_list[i], K_list[i + 1])
        H_j = renyi2_entropy(K_j)
        I = (renyi2_entropy(K_list[i]) +
             renyi2_entropy(K_list[i + 1]) - H_j)
        I_adjacent.append(float(I.item()))
    return H_per_layer, I_adjacent


def train_and_log(dataset: str, n_epochs: int, batch_size: int = 128,
                   seed: int = 0) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_fn, model_cls = R.DATASETS[dataset]
    tr_loader, va_loader = loader_fn(seed, batch_size=batch_size)
    model = model_cls().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reserve a fixed batch for entropy probing — removes mini-batch noise.
    probe_batch = next(iter(va_loader))[0].to(device)

    log = {
        "epoch":           [0],
        "val_acc":         [],
        "H_weight":        [],
        "H_act_per_layer": [],
        "I_act_adjacent":  [],
    }

    def measure(epoch: int) -> None:
        log["epoch"].append(epoch)
        # Weight-side entropy (existing framework)
        log["H_weight"].append(weight_side_entropy(model, view="dataflow"))
        # Activation-side Rényi-2
        acts = collect_activations(model, probe_batch)
        H_per, I_adj = activation_renyi(acts)
        log["H_act_per_layer"].append(H_per)
        log["I_act_adjacent"].append(I_adj)
        # Quick val acc
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        log["val_acc"].append(correct / total)
        model.train()

    log["epoch"] = []  # reset so the first measure() call writes 0
    measure(0)
    for ep in range(1, n_epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optim.step()
        measure(ep)
    return log


def plot_panel(ax_top, ax_mid, ax_bot, log: dict, title: str) -> None:
    epochs = log["epoch"]
    # Top: weight-side entropy
    ax_top.plot(epochs, log["H_weight"], marker="o", color="0.3")
    ax_top.set_title(f"{title}\nWeight-side H(adjacency) [bits]", fontsize=10)
    ax_top.grid(alpha=0.3)
    # Middle: per-layer activation H_2
    arr = np.array(log["H_act_per_layer"])  # (T, L)
    L = arr.shape[1] if arr.ndim == 2 else 0
    for l in range(L):
        ax_mid.plot(epochs, arr[:, l], marker="s",
                    label=f"layer {l}", alpha=0.85)
    ax_mid.set_title("Activation-side H_2(K_l) [Rényi-2, nats]", fontsize=10)
    ax_mid.legend(fontsize=8, frameon=False)
    ax_mid.grid(alpha=0.3)
    # Bottom: pairwise adjacent MI
    arr_I = np.array(log["I_act_adjacent"])  # (T, L-1)
    L_pairs = arr_I.shape[1] if arr_I.ndim == 2 else 0
    for p in range(L_pairs):
        ax_bot.plot(epochs, arr_I[:, p], marker="^",
                    label=f"I(layer {p}; {p+1})", alpha=0.85)
    ax_bot.set_xlabel("epoch")
    ax_bot.set_title("Adjacent-layer I_2 (Sanchez-Giraldo) [nats]", fontsize=10)
    ax_bot.legend(fontsize=8, frameon=False)
    ax_bot.grid(alpha=0.3)


def main() -> None:
    mnist_log = train_and_log("mnist_small", n_epochs=5)
    print("mnist_small final val acc:", mnist_log["val_acc"][-1])
    spirals_log = train_and_log("spirals", n_epochs=20)
    print("spirals final val acc:    ", spirals_log["val_acc"][-1])

    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex=False)
    plot_panel(axes[0, 0], axes[1, 0], axes[2, 0], mnist_log,
               "MNIST plain MLP (mnist_small)")
    plot_panel(axes[0, 1], axes[1, 1], axes[2, 1], spirals_log,
               "Spirals (SyntheticMLP)")
    fig.suptitle(
        "Weight-side vs activation-side entropy trajectories — baseline arm",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_pdf = OUT_DIR / "fig_compare_entropies.pdf"
    out_png = OUT_DIR / "fig_compare_entropies.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
