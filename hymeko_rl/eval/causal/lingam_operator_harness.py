"""Synthetic LiNGAM/HSiKAN operator harness — does HSiKAN benefit from the signed causal operator ONLY when the
data-generating process has structure-rich *nonlinear* mechanisms? (H1), and does a degree/sign-preserving directed
scramble of the operator collapse that benefit? (H2).

This is the falsification the bridge (``hsikan_mechanism.py``) set up; the bridge alone proves nothing empirical.
Task: predict the **sink** variable (last in causal order) from the others, with the sink **masked** in the input
(leakage guard). Models: a fitted linear predictor, a params-matched flat MLP, a DeepSets/bag baseline, HSiKAN over
the correct signed operator ``(A⁺, A⁻) = split(B)``, and HSiKAN over the **scrambled** operator. Ground-truth ``B``
is the primary condition; DirectLiNGAM-estimated ``B`` is an optional robustness check.

Run small (``python -m hymeko_rl.eval.causal.lingam_operator_harness --seeds 3 --n 16 --samples 800``).

**Non-claims:** no general HSiKAN>MLP claim, no RL, no GNN comparison, no real-MetaWorld validation, no causal-
discovery claim unless the estimated-``B`` path is separately validated. Structural leverage is claimed only if the
scramble collapses the advantage (H2).
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from hymeko_rl.eval.causal.hsikan_mechanism import signed_adjacency_split
from hymeko_rl.experiments.exp_structural_leverage_ladder import DeepSetsBackbone
from hymeko_rl.experiments.incidence_scramble import scramble_signed_operator
from hymeko_rl.experiments.structural_probe import ProbeModel, mlp_backbone
from hymeko_neuro.core import SignedKANBackbone

_CONDITIONS = ("linear", "nonlinear", "flat")           # + optional "multihop"
_ALPHA = 1.2                                            # nonlinear-mechanism gain (tanh/sin saturate gently)


# ── synthetic SEM ────────────────────────────────────────────────────────────────────────────────────────
def generate_signed_dag(n: int, *, density: float = 0.35, sign_balance: float = 0.6,
                        weight_lo: float = 0.4, weight_hi: float = 1.0, seed: int = 0) -> np.ndarray:
    """A random acyclic signed operator ``B`` with ``B[effect, cause]``, **zero diagonal**, strictly lower
    triangular in the causal order ``0 (exogenous) … n-1 (sink)``.

    # Postconditions ``B`` is ``(n, n)``, ``diag(B) == 0``, ``B[i, j] == 0`` for ``j >= i`` (acyclic); nonzero
      entries have magnitude in ``[weight_lo, weight_hi]`` and sign ``+`` with probability ``sign_balance``.
    """
    if n < 3:
        raise ValueError("need n >= 3")
    rng = np.random.default_rng(seed)
    b = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):                              # effect i draws from earlier causes j < i
        for j in range(i):
            if rng.random() < density:
                sign = 1.0 if rng.random() < sign_balance else -1.0
                b[i, j] = sign * rng.uniform(weight_lo, weight_hi)
    return b


_NONLINEARITIES: tuple[Callable[[np.ndarray], np.ndarray], ...] = (
    lambda z: np.tanh(z), lambda z: np.sin(z),
    lambda z: np.sign(z) * z * z, lambda z: z / (1.0 + np.abs(z)))       # tanh / sin / signed-square / softsign


def _edge_nonlinearity(rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    """Pick a simple monotone-ish nonlinearity for one edge."""
    return _NONLINEARITIES[int(rng.integers(0, len(_NONLINEARITIES)))]


def sample_sem(b: np.ndarray, n_samples: int, *, kind: str, seed: int, noise_scale: float = 0.5) -> np.ndarray:
    """Sample ``(n_samples, n)`` from the SEM over ``b`` in causal order (``x_i`` after its parents).

    ``linear``: ``x_i = Σ_j B[i,j] x_j + ε_i``. ``nonlinear``: ``x_i = Σ_j sign(B[i,j]) g_ij(|B[i,j]|·x_j) + ε_i``
    with a per-edge nonlinearity ``g_ij`` (fixed by ``seed``). ``multihop``: same nonlinear rule but the deeper
    causal order composes mechanisms across more hops. Non-Gaussian noise (uniform) per LiNGAM.
    """
    n = b.shape[0]
    rng = np.random.default_rng(seed)
    g = {(i, j): _edge_nonlinearity(rng) for i in range(n) for j in range(i) if b[i, j] != 0.0}
    x = np.zeros((n_samples, n), dtype=np.float64)
    eps = rng.uniform(-1.0, 1.0, size=(n_samples, n)) * noise_scale     # non-Gaussian exogenous noise
    for i in range(n):
        acc = eps[:, i].copy()
        for j in range(i):
            w = b[i, j]
            if w == 0.0:
                continue
            if kind == "linear":
                acc += w * x[:, j]
            else:                                       # nonlinear / multihop
                acc += np.sign(w) * g[(i, j)](_ALPHA * abs(w) * x[:, j])
        x[:, i] = acc
    return x


def _sample_condition(condition: str, b: np.ndarray, sink: int, n_samples: int, *, sample_seed: int,
                      ) -> "tuple[np.ndarray, np.ndarray]":
    """Sample ``(X, y)`` for a condition from a FIXED ``b`` (so train/val/test share one SEM). ``flat`` keeps the
    structured features but **shuffles the target** (``y ⊥ x``): nothing can predict it, so the operator must give
    no signal and the scramble must do nothing — the clean 'operator should not fabricate signal' control."""
    x = sample_sem(b, n_samples, kind=("nonlinear" if condition == "flat" else condition), seed=sample_seed)
    y = x[:, sink].copy()
    if condition == "flat":
        np.random.default_rng(sample_seed + 777).shuffle(y)      # break the x → y relation
    return x, y


# ── models (all params-reported; HSiKAN/MLP/DeepSets params-matched) ───────────────────────────────────────
class _SinkReadout(nn.Module):
    """Read the **sink node's** message-passed activation (not a mean-pool) — the natural mechanism readout: the
    sink's embedding after message passing encodes its causes. Leakage-safe: the sink's own input feature is masked
    to 0, so ``W_self·0 = 0`` and its activation comes only from its parents via ``(A⁺, A⁻)``."""

    def __init__(self, backbone: SignedKANBackbone, sink: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.sink = sink

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.node_activations(x)[:, self.sink, :]      # (B, N, H) -> (B, H) at the sink


def build_hsikan_operator(a_pos: np.ndarray, a_neg: np.ndarray, hidden: int, n_layers: int, sink: int) -> ProbeModel:
    """HSiKAN over the raw weighted signed operator: ``SignedKANBackbone`` with ``(A⁺, A⁻)`` as its prior, read at
    the sink node (the mechanism-model readout)."""
    ap = torch.as_tensor(a_pos, dtype=torch.float32)
    an = torch.as_tensor(a_neg, dtype=torch.float32)
    return ProbeModel(_SinkReadout(SignedKANBackbone(1, hidden, n_layers, ap, an), sink), hidden)


def build_mlp(n_nodes: int, hidden: int, depth: int) -> ProbeModel:
    backbone, h = mlp_backbone(n_nodes, hidden=hidden, depth=depth)
    return ProbeModel(backbone, h)


def build_deepsets_model(hidden: int, depth: int) -> ProbeModel:
    return ProbeModel(DeepSetsBackbone(1, hidden, depth=depth), hidden)


def _match_hidden(builder: Callable[[int], ProbeModel], target_params: int,
                  search: range = range(8, 257)) -> int:
    return min(search, key=lambda h: abs(builder(h).n_params() - target_params))


# ── data prep (leakage guard) ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Prepared:
    x_masked: np.ndarray        # (n, N) standardised features, sink column zeroed
    x_linear: np.ndarray        # (n, N-1) non-sink features (sink dropped) for the linear predictor
    y: np.ndarray               # (n,) standardised target
    x_leaky: np.ndarray         # (n, N) sink NOT masked — the leakage probe only


def prepare(x_std: np.ndarray, y_std: np.ndarray, sink: int) -> _Prepared:
    """Mask the sink (leakage guard) and lay out per-model inputs from ALREADY-STANDARDISED arrays.
    ``x_masked``/``x_linear`` both withhold the sink value; ``x_leaky`` deliberately keeps it (probe only)."""
    xm = x_std.copy()
    xm[:, sink] = 0.0                                  # HSiKAN/MLP/DeepSets never see the target value
    non_sink = [k for k in range(x_std.shape[1]) if k != sink]
    return _Prepared(x_masked=xm, x_linear=x_std[:, non_sink], y=y_std, x_leaky=x_std.copy())


def _standardise_splits(xs: "list[np.ndarray]", ys: "list[np.ndarray]", sink: int) -> "list[_Prepared]":
    """Standardise train/val/test by TRAIN column stats (X) and TRAIN mean/std (y); mask the sink in each."""
    mu, sd = xs[0].mean(axis=0), xs[0].std(axis=0) + 1e-8
    ymu, ysd = float(ys[0].mean()), float(ys[0].std()) + 1e-8
    return [prepare((x - mu) / sd, (y - ymu) / ysd, sink) for x, y in zip(xs, ys)]


# ── train / eval ──────────────────────────────────────────────────────────────────────────────────────────
def _to_nodes(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x[:, :, None], dtype=torch.float32)


def _fit_torch(model: ProbeModel, tr: _Prepared, va: _Prepared, te: _Prepared, *, epochs: int, seed: int,
               leaky: bool = False) -> "dict[str, float]":
    """Adam/MSE fit; returns train/val/test MSE + test R². ``leaky`` feeds the unmasked features (probe only)."""
    torch.manual_seed(seed)
    key = "x_leaky" if leaky else "x_masked"
    x_tr, x_va, x_te = (_to_nodes(getattr(s, key)) for s in (tr, va, te))
    y_tr, y_va, y_te = (torch.as_tensor(s.y, dtype=torch.float32) for s in (tr, va, te))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    nb = x_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(nb, generator=gen)
        for i in range(0, nb, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        mse = {k: float(nn.functional.mse_loss(model(x), y))
               for k, x, y in (("train", x_tr, y_tr), ("val", x_va, y_va), ("test", x_te, y_te))}
    return {**mse, "r2_test": 1.0 - mse["test"] / (float(y_te.var(unbiased=False)) + 1e-12)}


def _fit_linear(tr: _Prepared, te: _Prepared) -> "dict[str, float]":
    """Least-squares linear predictor on the (sink-dropped) features — the linear/LiNGAM-loadings baseline."""
    a = np.concatenate([tr.x_linear, np.ones((tr.x_linear.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(a, tr.y, rcond=None)
    ate = np.concatenate([te.x_linear, np.ones((te.x_linear.shape[0], 1))], axis=1)
    pred = ate @ coef
    mse = float(np.mean((pred - te.y) ** 2))
    return {"train": float(np.mean((a @ coef - tr.y) ** 2)), "val": mse, "test": mse,
            "r2_test": 1.0 - mse / (float(np.var(te.y)) + 1e-12)}


# ── run: conditions × models × seeds ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HarnessConfig:
    seeds: int = 3
    n: int = 16
    samples: int = 800
    density: float = 0.35
    sign_balance: float = 0.6
    hsikan_hidden: int = 24
    n_layers: int = 2
    epochs: int = 200
    conditions: tuple[str, ...] = _CONDITIONS


def _score(builder: Callable[[], ProbeModel], tr: _Prepared, va: _Prepared, te: _Prepared, *, epochs: int,
           seed: int, leaky: bool = False) -> "dict[str, float]":
    torch.manual_seed(seed)                            # reproducible init
    model = builder()
    return {**_fit_torch(model, tr, va, te, epochs=epochs, seed=seed, leaky=leaky), "n_params": model.n_params()}


def _agg(per_seed: "list[dict[str, float]]") -> "dict[str, object]":
    tests = [d["test"] for d in per_seed]
    return {"test_median": round(statistics.median(tests), 5),
            "test_mean": round(statistics.fmean(tests), 5),
            "r2_median": round(statistics.median(d["r2_test"] for d in per_seed), 4),
            "n_params": int(per_seed[0]["n_params"]),
            "per_seed_test": [round(t, 5) for t in tests]}


def run_condition(condition: str, cfg: HarnessConfig) -> "dict[str, object]":
    """One condition across ``cfg.seeds`` replicates (fresh DAG + data + init each). Returns per-model aggregates,
    the gaps, the scramble collapse, and a leakage-guard check."""
    models = ("linear", "mlp", "deepsets", "hsikan_correct", "hsikan_scrambled")
    collected: "dict[str, list[dict[str, float]]]" = {m: [] for m in models}
    leak: "dict[str, float]" = {}
    for seed in range(cfg.seeds):
        b = generate_signed_dag(cfg.n, density=cfg.density, sign_balance=cfg.sign_balance, seed=seed)
        sink = cfg.n - 1
        a_pos, a_neg = signed_adjacency_split(b)
        a_pos_s, a_neg_s = scramble_signed_operator(a_pos, a_neg, seed=seed)
        xs, ys = zip(*(_sample_condition(condition, b, sink, cfg.samples, sample_seed=seed * 10 + k)
                       for k in range(3)))
        tr, va, te = _standardise_splits(list(xs), list(ys), sink)
        target = build_hsikan_operator(a_pos, a_neg, cfg.hsikan_hidden, cfg.n_layers, sink).n_params()
        mlp_h = _match_hidden(lambda h: build_mlp(cfg.n, h, cfg.n_layers), target)
        ds_h = _match_hidden(lambda h: build_deepsets_model(h, cfg.n_layers), target)
        collected["linear"].append({**_fit_linear(tr, te), "n_params": 0.0})
        collected["mlp"].append(_score(lambda: build_mlp(cfg.n, mlp_h, cfg.n_layers), tr, va, te,
                                        epochs=cfg.epochs, seed=seed))
        collected["deepsets"].append(_score(lambda: build_deepsets_model(ds_h, cfg.n_layers), tr, va, te,
                                            epochs=cfg.epochs, seed=seed))
        collected["hsikan_correct"].append(_score(
            lambda: build_hsikan_operator(a_pos, a_neg, cfg.hsikan_hidden, cfg.n_layers, sink), tr, va, te,
            epochs=cfg.epochs, seed=seed))
        collected["hsikan_scrambled"].append(_score(
            lambda: build_hsikan_operator(a_pos_s, a_neg_s, cfg.hsikan_hidden, cfg.n_layers, sink), tr, va, te,
            epochs=cfg.epochs, seed=seed))
        if seed == 0:                                  # leakage probe: an MLP that SEES the unmasked sink must cheat
            leak = {"leaky_test": _score(lambda: build_mlp(cfg.n, mlp_h, cfg.n_layers), tr, va, te,
                                         epochs=cfg.epochs, seed=seed, leaky=True)["test"],
                    "masked_test": collected["mlp"][0]["test"]}
    agg = {m: _agg(collected[m]) for m in models}
    hk = float(agg["hsikan_correct"]["test_median"])       # type: ignore[arg-type]
    scr = float(agg["hsikan_scrambled"]["test_median"])    # type: ignore[arg-type]
    return {
        "condition": condition, "models": agg,
        "gap_vs_mlp": round(float(agg["mlp"]["test_median"]) - hk, 5),          # type: ignore[arg-type]
        "gap_vs_linear": round(float(agg["linear"]["test_median"]) - hk, 5),    # type: ignore[arg-type]
        "scramble_collapse": round(scr - hk, 5),               # >0 ⇒ correct operator helps (MSE lower)
        "scramble_collapse_frac": round((scr - hk) / hk, 3) if hk > 1e-9 else None,
        "leakage": {**leak, "guard_ok": bool(leak.get("leaky_test", 1.0) < 0.05 <= leak.get("masked_test", 0.0))},
    }


_ADV_FRAC = 0.05        # HSiKAN must beat a baseline's MSE by ≥5% to count as an advantage
_COLLAPSE_FRAC = 0.20   # scramble must raise HSiKAN MSE by ≥20% to count as a collapse
# conditions whose target IS the sink (so the sink-masking leakage guard is meaningful — not 'flat').
_SINK_TARGET_CONDS = ("linear", "nonlinear", "multihop")


def _f(x: object) -> float:
    assert isinstance(x, (int, float)), f"expected numeric, got {type(x).__name__}"
    return float(x)


def _hk_median(row: "dict[str, object]") -> float:
    from typing import cast
    return _f(cast("dict[str, dict[str, object]]", row["models"])["hsikan_correct"]["test_median"])


def _verdict(rows: "dict[str, dict[str, object]]") -> "dict[str, object]":
    """H1 (advantage only on nonlinear-structured) + H2 (scramble collapses it), per the decision rules."""
    lin, nl, flat = rows.get("linear"), rows.get("nonlinear"), rows.get("flat")
    checks: "dict[str, bool]" = {}
    if lin is not None:                                  # linear SEM: linear baseline matches/beats HSiKAN
        checks["linear_baseline_matches_hsikan"] = _f(lin["gap_vs_linear"]) <= _ADV_FRAC * abs(_hk_median(lin) + 1e-9)
    if nl is not None:                                  # nonlinear structured: HSiKAN beats MLP AND linear
        hk = _hk_median(nl)
        checks["hsikan_beats_mlp_on_nonlinear"] = _f(nl["gap_vs_mlp"]) > _ADV_FRAC * hk
        checks["hsikan_beats_linear_on_nonlinear"] = _f(nl["gap_vs_linear"]) > _ADV_FRAC * hk
        checks["scramble_collapses_on_nonlinear"] = (nl["scramble_collapse_frac"] is not None
                                                     and _f(nl["scramble_collapse_frac"]) >= _COLLAPSE_FRAC)
    if flat is not None:                                # flat: operator irrelevant → scramble does ~nothing
        cf = flat["scramble_collapse_frac"]
        checks["flat_operator_irrelevant"] = cf is None or abs(_f(cf)) < _COLLAPSE_FRAC
    # leakage guard only where the target IS the sink (on 'flat' the sink is irrelevant → masking is vacuous).
    from typing import cast
    checks["leakage_guard_ok"] = all(
        bool(cast("dict[str, object]", rows[c]["leakage"])["guard_ok"])
        for c in _SINK_TARGET_CONDS if c in rows)
    label = "SUPPORTED" if all(checks.values()) else (
        "WEAKENED" if checks.get("hsikan_beats_mlp_on_nonlinear") and checks.get("scramble_collapses_on_nonlinear")
        else "FALSIFIED")
    return {"label": label, "checks": checks,
            "thresholds": {"advantage_frac": _ADV_FRAC, "collapse_frac": _COLLAPSE_FRAC}}


def run_harness(cfg: HarnessConfig | None = None) -> "dict[str, object]":
    cfg = cfg or HarnessConfig()
    torch.set_num_threads(1)
    rows = {c: run_condition(c, cfg) for c in cfg.conditions}
    return {"config": asdict(cfg), "conditions": rows, "verdict": _verdict(rows)}


def plot_harness(report: "dict[str, object]", out_path: "str | Path") -> Path:
    """Grouped-bar test MSE per condition × model, with the scramble collapse annotated (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from typing import cast
    rows = cast("dict[str, dict[str, object]]", report["conditions"])
    conds = list(rows)
    models = ("linear", "mlp", "deepsets", "hsikan_correct", "hsikan_scrambled")
    colors = ("#7f7f7f", "#9467bd", "#1f77b4", "#2ca02c", "#d62728")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.16
    for i, (m, c) in enumerate(zip(models, colors)):
        vals = [float(cast("dict[str, object]", rows[cd]["models"])[m]["test_median"]) for cd in conds]  # type: ignore[index]
        ax.bar(np.arange(len(conds)) + (i - 2) * width, vals, width, label=m, color=c, edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([f"{cd}\n(collapse {rows[cd]['scramble_collapse_frac']})" for cd in conds])
    ax.set_ylabel("held-out test MSE (log; lower = better)")
    ax.set_title(f"LiNGAM operator harness — {report['verdict']['label']}")  # type: ignore[index]
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--multihop", action="store_true", help="add the optional multi-hop nonlinear condition")
    ap.add_argument("--out-dir", default="reports/figures/2026_07_10_hsikan_lingam_operator_harness")
    a = ap.parse_args(argv)
    conds = (*_CONDITIONS, "multihop") if a.multihop else _CONDITIONS
    cfg = HarnessConfig(seeds=a.seeds, n=a.n, samples=a.samples, epochs=a.epochs, conditions=conds)
    report = run_harness(cfg)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "harness.json").write_text(json.dumps(report, indent=2, default=float))
    plot_harness(report, out / "harness")
    from typing import cast
    conditions = cast("dict[str, dict[str, object]]", report["conditions"])
    print(json.dumps({c: {"gap_vs_mlp": r["gap_vs_mlp"], "gap_vs_linear": r["gap_vs_linear"],
                          "scramble_collapse_frac": r["scramble_collapse_frac"]}
                      for c, r in conditions.items()}, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
