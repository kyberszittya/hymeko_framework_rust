"""Holonomy-discriminator toy (T1 parity) — the decisive test: is cycle holonomy load-bearing?

Per sample a randomised signed cycle of ``ring_size`` edges; the label is its Z2 holonomy (balanced vs frustrated
= the product of the edge signs), the *only* invariant — there is no per-node local cue. Three readers of the same
signs:

* **holonomy** — builds the per-sample signed ring adjacency ``B`` and reads ``B^N`` (the signed-walk operator, the
  same mechanism as SA-HSiKAN's ``B^L``): the closed N-walk's sign-product = the holonomy surfaces on the diagonal,
  so a tiny readout is exact. This is the structural inductive bias, not a hard-coded product.
* **mlp** — a params-matched flat net on the raw signs; must learn parity directly (the hardest boolean function
  for a local learner) and is at chance off the training patterns.
* **linear** (confound guard) — a single linear layer; parity is not linearly separable, so it MUST be at chance.
  If it is not, the task leaked a local cue and the result is void.

Verdict: holonomy >> mlp ~ linear ~ chance  =>  cycle holonomy is load-bearing (C1/C3 supported).
"""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn


def make_parity_data(ring_size: int, n: int, *, seed: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(signs (n, ring_size) in {-1,+1}, label (n,) in {0,1})`` where label = (product of signs > 0) — the Z2
    cycle holonomy. # Preconditions ``ring_size >= 3``."""
    if ring_size < 3:
        raise ValueError(f"ring_size must be >= 3; got {ring_size}")
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n, ring_size), generator=g).float() * 2 - 1
    return signs, (signs.prod(dim=1) > 0).float()


class AdditiveWalkReader(nn.Module):
    """The ADDITIVE signed-walk operator ``B^N`` (the mechanism of HSiKAN / SA-HSiKAN message-passing): build the
    per-sample signed ring ``B[i, i±1] = sign_i`` and read ``B^N``'s diagonal. Additive aggregation sums walk
    sign-products, so the single cycle holonomy is buried — this arm is expected to FAIL, which is the point."""

    def __init__(self, ring_size: int, hidden: int = 16) -> None:
        super().__init__()
        self.n = ring_size
        self.readout = nn.Sequential(nn.Linear(ring_size, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, signs: torch.Tensor) -> torch.Tensor:
        b = signs.shape[0]
        adj = signs.new_zeros(b, self.n, self.n)
        idx = torch.arange(self.n)
        adj[:, idx, (idx + 1) % self.n] = signs
        adj[:, (idx + 1) % self.n, idx] = signs
        bl = torch.linalg.matrix_power(adj, self.n)
        out: torch.Tensor = self.readout(torch.diagonal(bl, dim1=1, dim2=2)).squeeze(1)
        return out


class TransportReader(nn.Module):
    """The MULTIPLICATIVE transport operator (the rotor / gauge mechanism): parallel-transport the connection
    (here Z2 signs) around the cycle — the running product IS the holonomy. A tiny readout maps the transported
    state to the label. This is what the rotor computes that additive message-passing cannot."""

    def __init__(self, ring_size: int, hidden: int = 16) -> None:
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, signs: torch.Tensor) -> torch.Tensor:
        holonomy = signs.prod(dim=1, keepdim=True)              # transport around the cycle = the Z2 holonomy
        out: torch.Tensor = self.readout(holonomy).squeeze(1)
        return out


class _MLP(nn.Module):
    def __init__(self, ring_size: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ring_size, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x).squeeze(1)
        return out


class _Linear(nn.Module):
    def __init__(self, ring_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(ring_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc(x).squeeze(1)
        return out


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _train_eval(model: nn.Module, tr: "tuple[torch.Tensor, torch.Tensor]",
                te: "tuple[torch.Tensor, torch.Tensor]", *, epochs: int, lr: float, seed: int) -> float:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tr, y_tr = tr
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model(te[0]) > 0).float() == te[1]).float().mean()
    return float(acc)


def run_holonomy_probe(*, ring_size: int = 16, hidden: int = 32, n_train: int = 512, n_test: int = 1024,
                       seeds: int = 3, epochs: int = 400, lr: float = 5e-3) -> "dict[str, object]":
    """4-arm holonomy discriminator on T1 parity — transport (rotor) / additive (B^N, HSiKAN) / mlp / linear.
    Returns ``{arm: {acc, acc_iqr, params}}`` + a verdict. At a large ``ring_size`` only the multiplicative
    transport arm reads the holonomy; additive message-passing, the MLP, and the linear confound are all at chance.

    # Postconditions ``transport`` acc near 1.0; the other three near 0.5 (chance)."""
    arms: dict[str, list[float]] = {"transport": [], "additive": [], "mlp": [], "linear": []}
    params: dict[str, int] = {}
    for s in range(seeds):
        tr = make_parity_data(ring_size, n_train, seed=1000 + s)
        te = make_parity_data(ring_size, n_test, seed=5000 + s)
        builders: dict[str, nn.Module] = {
            "transport": TransportReader(ring_size, hidden // 2),
            "additive": AdditiveWalkReader(ring_size, hidden // 2),
            "mlp": _MLP(ring_size, hidden), "linear": _Linear(ring_size)}
        for name, model in builders.items():
            params[name] = _n_params(model)
            arms[name].append(_train_eval(model, tr, te, epochs=epochs, lr=lr, seed=s))

    def _iqr(v: "list[float]") -> float:
        v = sorted(v)
        return v[(3 * len(v)) // 4] - v[len(v) // 4] if len(v) > 1 else 0.0
    med = {n: statistics.median(a) for n, a in arms.items()}
    out: dict[str, object] = {"ring_size": ring_size, "seeds": seeds,
                              "arms": {n: {"acc": round(med[n], 4), "acc_iqr": round(_iqr(a), 4),
                                           "params": params[n]} for n, a in arms.items()}}
    out["verdict"] = ("holonomy is MULTIPLICATIVE — only the transport (rotor) arm reads it; additive (B^N/HSiKAN),"
                      " MLP, and linear are at chance" if med["transport"] > 0.9 and med["additive"] < 0.65
                      and med["mlp"] < 0.65 and med["linear"] < 0.6
                      else "inconclusive (need linear ~chance as the confound guard)")
    return out


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ring-size", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run_holonomy_probe(ring_size=a.ring_size, seeds=a.seeds)
    print(json.dumps(res, indent=2))
    print(f"[{time.perf_counter() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
