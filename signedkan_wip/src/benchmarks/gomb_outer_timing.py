"""Wall-time benchmark: legacy OuterFIRShell path vs batched forward.

Run from repo root (default **CUDA** when ``torch.cuda.is_available()``, else CPU)::

    python -m signedkan_wip.src.benchmarks.gomb_outer_timing
    python -m signedkan_wip.src.benchmarks.gomb_outer_timing --iters 40
    python -m signedkan_wip.src.benchmarks.gomb_outer_timing --device cpu

Real signed graphs (same cycle pool as ``run_gomb_smoke`` — train split, Rust top-K)::

    python -m signedkan_wip.src.benchmarks.gomb_outer_timing \\
        --datasets bitcoin_alpha bitcoin_otc

Optional **torch.compile** on a copy of the shell (CUDA recommended; extra warmup)::

    python -m signedkan_wip.src.benchmarks.gomb_outer_timing --torch-compile \\
        --device cuda --datasets bitcoin_alpha

Rust ``hymeko_graph::spine::CliffordFIR`` is **not** exposed via ``hymeko_py``;
the outer shell stays ``ClifFIRTierAggregator`` in PyTorch until a PyO3 + autograd
(or frozen-coeff) bridge exists.

For **val AUROC, recall, precision, F1, AP**, and **learnable parameter counts**
(including per-shell breakdown), train with::

    python -m signedkan_wip.experiments.runs.run_gomb_smoke --dataset bitcoin_otc --n-epochs 50

Uses fixed seed, warmup, then ``n_iters`` timed forwards (median / IQR / worst).
Legacy = Python loop over ``M`` banks + per-corner scatter loop (pre-optimization).
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import numpy as np
import torch

from signedkan_wip.src.datasets import load
from signedkan_wip.src.hymeko_gomb.shells import OuterFIRShell
from signedkan_wip.experiments.runs.run_gomb_smoke import _enumerate_cycles, _train_val_split

_DEFAULT_BENCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _scatter_mean_legacy_corners(
    per_cycle: torch.Tensor,
    cycles: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    """Pre-vectorization scatter: one index_add per corner index."""
    m_c, k = cycles.shape
    d = per_cycle.shape[-1]
    out = torch.zeros(n_vertices, d, device=per_cycle.device, dtype=per_cycle.dtype)
    counts = torch.zeros(n_vertices, device=per_cycle.device, dtype=per_cycle.dtype)
    c = cycles.long()
    for i in range(k):
        vidx = c[:, i]
        out.index_add_(0, vidx, per_cycle)
        counts.index_add_(0, vidx, torch.ones_like(vidx, dtype=per_cycle.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(-1)


def _outer_forward_legacy(
    shell: OuterFIRShell,
    x: torch.Tensor,
    cycles: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """M-bank Python loop + legacy scatter (matches old implementation)."""
    n = x.shape[0]
    cycles_l = cycles.long()
    signs_f = signs.float()
    parts: list[torch.Tensor] = []
    for m in range(shell.M):
        x_proj = shell.pre_projs[m](x)
        cv = x_proj[cycles_l]
        pc = shell.banks[m](cv, signs_f)
        parts.append(_scatter_mean_legacy_corners(pc, cycles_l, n))
    return torch.cat(parts, dim=-1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[list[float], torch.Tensor]:
    """Returns (times_ms, last_output).

    Wraps warmup and timed iterations in ``torch.inference_mode()`` so no
    autograd tape is built (inference-only micro-bench; avoids CUDAGraph
    "pending backwards" warnings on compiled paths).
    """
    with torch.inference_mode():
        for _ in range(warmup):
            y = fn()
        _sync(device)
        times: list[float] = []
        y_last = y
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            y_last = fn()
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times, y_last


def _summarize_ms(times_ms: list[float]) -> dict[str, float]:
    xs = sorted(times_ms)
    n = len(xs)
    mid = statistics.median(xs)
    lo = xs[max(0, (n - 1) // 4)]
    hi = xs[min(n - 1, (3 * (n - 1)) // 4)]
    return {
        "median_ms": mid,
        "iqr_lo_ms": lo,
        "iqr_hi_ms": hi,
        "worst_ms": xs[-1],
        "best_ms": xs[0],
    }


def _clone_and_compile_shell(shell: OuterFIRShell) -> torch.nn.Module:
    """Copy weights then ``torch.compile`` (dynamic shapes for ``M_c``)."""
    dev = next(shell.parameters()).device
    dt = next(shell.parameters()).dtype
    dup = OuterFIRShell(
        d_in=shell.d_in,
        d_layer=shell.d_layer,
        M=shell.M,
        cycle_k=shell.cycle_k,
    ).to(device=dev, dtype=dt)
    dup.load_state_dict(shell.state_dict())
    dup.train(False)
    return torch.compile(dup, dynamic=True, mode="reduce-overhead")


def _torch_compile_stats(
    shell: OuterFIRShell,
    x: torch.Tensor,
    cycles: torch.Tensor,
    signs: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
    *,
    fail_label: str,
) -> dict[str, float]:
    """Parity-check eager vs compiled on a slice, then timed compiled forward."""
    if device.type != "cuda":
        print(
            "  [torch-compile] warning: CUDA strongly recommended; "
            "CPU compile often regresses.",
            flush=True,
        )
    shell_c = _clone_and_compile_shell(shell)
    mc = int(cycles.shape[0])
    n_chk = min(_PARITY_SLICE, mc)
    if n_chk > 0:
        c0, s0 = cycles[:n_chk], signs[:n_chk]
        with torch.no_grad():
            y_e = shell(x, c0, s0)
            y_c = shell_c(x, c0, s0)
        if not torch.allclose(y_e, y_c, rtol=2e-3, atol=2e-3):
            max_err = (y_e - y_c).abs().max().item()
            raise SystemExit(
                f"{fail_label} torch.compile parity fail: max_err={max_err}",
            )
    warm_c = warmup + (15 if device.type == "cuda" else 5)

    def run_comp() -> torch.Tensor:
        return shell_c(x, cycles, signs)

    t_comp, _ = _bench(run_comp, device, warm_c, iters)
    return _summarize_ms(t_comp)


_PARITY_SLICE = 4096


def _bench_dataset_outer(
    *,
    dataset: str,
    device: torch.device,
    dtype: torch.dtype,
    d_in: int,
    d_layer: int,
    M: int,
    k: int,
    topk: int,
    seed: int,
    val_frac: float,
    warmup: int,
    iters: int,
    torch_compile: bool,
    cycle_abb_mode: str = "none",
    cycle_abb_fullness_gate: float = 0.25,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    t_load = time.perf_counter()
    g = load(dataset)
    n = g.n_nodes
    n_edges = int(g.edges.shape[0])
    print(
        f"\n[{dataset}] |V|={n} |E|={n_edges}  "
        f"(load {time.perf_counter() - t_load:.2f}s)",
        flush=True,
    )

    e_tr, s_tr, _e_va, _s_va = _train_val_split(
        g.edges, g.signs, val_frac, seed,
    )
    t_enum = time.perf_counter()
    cycles_np, cyc_signs_np = _enumerate_cycles(
        e_tr, s_tr, n, k=k, m_per_vertex=topk,
        abb_mode=cycle_abb_mode,
        abb_fullness_gate=cycle_abb_fullness_gate,
    )
    enum_s = time.perf_counter() - t_enum
    mc = int(cycles_np.shape[0])
    print(f"  [prep] train |E|={e_tr.shape[0]}  cycles Mc={mc}  enum_wall={enum_s:.2f}s", flush=True)

    shell = OuterFIRShell(
        d_in=d_in, d_layer=d_layer, M=M, cycle_k=k,
    ).to(device=device, dtype=dtype)
    shell.train(False)

    x = torch.randn(n, d_in, device=device, dtype=dtype)
    cycles = torch.from_numpy(cycles_np).to(device=device, dtype=torch.long)
    signs = torch.from_numpy(
        np.asarray(cyc_signs_np, dtype=np.float32),
    ).to(device=device, dtype=dtype)

    n_chk = min(_PARITY_SLICE, mc)
    if n_chk > 0:
        c0, s0 = cycles[:n_chk], signs[:n_chk]
        with torch.no_grad():
            y_new = shell(x, c0, s0)
            y_old = _outer_forward_legacy(shell, x, c0, s0)
        if not torch.allclose(y_new, y_old, rtol=1e-4, atol=1e-5):
            max_err = (y_new - y_old).abs().max().item()
            raise SystemExit(f"[{dataset}] parity fail on first {n_chk} cycles: max_err={max_err}")

    def run_legacy() -> torch.Tensor:
        return _outer_forward_legacy(shell, x, cycles, signs)

    def run_batched() -> torch.Tensor:
        return shell(x, cycles, signs)

    t_legacy, _ = _bench(run_legacy, device, warmup, iters)
    t_batched, _ = _bench(run_batched, device, warmup, iters)

    s_leg = _summarize_ms(t_legacy)
    s_new = _summarize_ms(t_batched)
    speedup = s_leg["median_ms"] / max(s_new["median_ms"], 1e-9)

    print(f"  device={device} dtype={dtype}  M={M} k={k}  d_in={d_in} d_layer={d_layer}", flush=True)
    print(f"  warmup={warmup} iters={iters}", flush=True)
    print(f"  legacy median_ms={s_leg['median_ms']:.4f}  eager median_ms={s_new['median_ms']:.4f}", flush=True)
    print(
        f"  legacy worst_ms={s_leg['worst_ms']:.4f}  eager worst_ms={s_new['worst_ms']:.4f}",
        flush=True,
    )
    print(f"  median speedup (legacy / eager): {speedup:.2f}x", flush=True)

    if torch_compile:
        s_comp = _torch_compile_stats(
            shell,
            x,
            cycles,
            signs,
            device,
            warmup,
            iters,
            fail_label=f"[{dataset}]",
        )
        ratio = s_new["median_ms"] / max(s_comp["median_ms"], 1e-9)
        warm_extra = 15 if device.type == "cuda" else 5
        print(
            f"  torch.compile median_ms={s_comp['median_ms']:.4f}  "
            f"worst_ms={s_comp['worst_ms']:.4f}  "
            f"(extra warmup +{warm_extra})",
            flush=True,
        )
        print(f"  eager / compiled median: {ratio:.2f}x", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device",
        default=_DEFAULT_BENCH_DEVICE,
        choices=("cpu", "cuda"),
        help=f"Compute device (default: {_DEFAULT_BENCH_DEVICE}).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help="If set (e.g. bitcoin_alpha bitcoin_otc), benchmark OuterFIRShell "
        "on Rust-enumerated train-split cycles for each dataset.",
    )
    p.add_argument("--N", type=int, default=512, help="vertices (synthetic mode only)")
    p.add_argument("--Mc", type=int, default=4096, help="cycles (synthetic mode only)")
    p.add_argument("--M", type=int, default=8, help="parallel FIR banks")
    p.add_argument("--d-in", type=int, default=32, dest="d_in")
    p.add_argument("--d-layer", type=int, default=16, dest="d_layer")
    p.add_argument("--k", type=int, default=3, help="cycle arity")
    p.add_argument("--topk", type=int, default=64, help="m_per_vertex cycle cap (dataset mode)")
    p.add_argument(
        "--cycle-abb-mode",
        default="none",
        choices=("none", "start_local", "global_min"),
        help="Rust cycle enumerator ABB mode (same as run_gomb_smoke).",
    )
    p.add_argument(
        "--cycle-abb-fullness-gate",
        type=float,
        default=0.25,
        help="global_min ABB fullness gate.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="After eager timings, clone shell and benchmark torch.compile (CUDA recommended).",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if args.datasets is not None:
        for name in args.datasets:
            _bench_dataset_outer(
                dataset=name,
                device=device,
                dtype=dtype,
                d_in=args.d_in,
                d_layer=args.d_layer,
                M=args.M,
                k=args.k,
                topk=args.topk,
                seed=args.seed,
                val_frac=args.val_frac,
                warmup=args.warmup,
                iters=args.iters,
                torch_compile=args.torch_compile,
                cycle_abb_mode=args.cycle_abb_mode,
                cycle_abb_fullness_gate=float(args.cycle_abb_fullness_gate),
            )
        return

    torch.manual_seed(0)

    shell = OuterFIRShell(
        d_in=args.d_in, d_layer=args.d_layer, M=args.M, cycle_k=args.k,
    ).to(device=device, dtype=dtype)
    shell.train(False)

    n, mc, k = args.N, args.Mc, args.k
    x = torch.randn(n, args.d_in, device=device, dtype=dtype)
    cycles = torch.randint(0, n, (mc, k), device=device, dtype=torch.long)
    signs = torch.where(torch.randint(0, 2, (mc, k), device=device) == 0, -1.0, 1.0)
    if dtype == torch.float64:
        signs = signs.to(dtype=torch.float64)

    with torch.no_grad():
        y_new = shell(x, cycles, signs)
        y_old = _outer_forward_legacy(shell, x, cycles, signs)
    if not torch.allclose(y_new, y_old, rtol=1e-4, atol=1e-5):
        max_err = (y_new - y_old).abs().max().item()
        raise SystemExit(f"parity fail: max abs err {max_err}")

    def run_legacy() -> torch.Tensor:
        return _outer_forward_legacy(shell, x, cycles, signs)

    def run_batched() -> torch.Tensor:
        return shell(x, cycles, signs)

    t_legacy, _ = _bench(run_legacy, device, args.warmup, args.iters)
    t_batched, _ = _bench(run_batched, device, args.warmup, args.iters)

    s_leg = _summarize_ms(t_legacy)
    s_new = _summarize_ms(t_batched)
    speedup = s_leg["median_ms"] / max(s_new["median_ms"], 1e-9)

    print(
        f"device={args.device} dtype={args.dtype} "
        f"N={n} Mc={mc} M={args.M} k={k} d_in={args.d_in} d_layer={args.d_layer}",
    )
    print(f"warmup={args.warmup} iters={args.iters}")
    print()
    print("legacy (M-loop + corner scatter):")
    for k2, v in s_leg.items():
        print(f"  {k2}: {v:.4f}")
    print()
    print("batched (current OuterFIRShell):")
    for k2, v in s_new.items():
        print(f"  {k2}: {v:.4f}")
    print()
    print(f"median speedup (legacy / batched): {speedup:.2f}x")

    if args.torch_compile:
        s_comp = _torch_compile_stats(
            shell,
            x,
            cycles,
            signs,
            device,
            args.warmup,
            args.iters,
            fail_label="synthetic",
        )
        ratio = s_new["median_ms"] / max(s_comp["median_ms"], 1e-9)
        print()
        print("torch.compile (shell copy):")
        for k2, v in s_comp.items():
            print(f"  {k2}: {v:.4f}")
        print()
        print(f"median speedup (batched / compiled): {ratio:.2f}x")


if __name__ == "__main__":
    main()
