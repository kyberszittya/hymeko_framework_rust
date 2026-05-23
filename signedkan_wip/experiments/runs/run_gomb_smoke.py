"""HymeKo-Gömb smoke runner — real Rust-enumerated cycle pool.

Loads a signed-graph dataset, enumerates cycles via the Rust
top-K per-vertex enumerator on **train** edges only, trains a small
HymeKoGomb for N epochs, logs **validation** ROC-AUC each epoch.

**Edge split (``--edge-split``):**

* ``80_20`` (default): random 80/20 train/val (``--val-frac``), same as
  earlier smoke runs — no held-out test set.
* ``80_10_10``: ``datasets.split`` — same **train/val/test** convention
  as ``run_final_cell.cell_signed_graph`` / ``run_hsikan_sota_gate``.
  After training, prints **val** and **test** AUROC/AP/F1 (threshold 0.5).
  The JSON summary also includes **inference** timing: one batched forward
  on all held-out edges (val ∪ test for ``80_10_10``, else val only),
  with an untimed CUDA warmup when on GPU; keys ``infer_wall_s``,
  ``infer_n_edges``, ``infer_edges_per_s``.

**Cycle enumeration ABB (``--cycle-abb-mode``):** optional Rust
``enumerate_cycles_rs`` branch-and-bound: ``none`` (default),
``start_local``, or ``global_min``; plus ``--cycle-abb-fullness-gate`` for
``global_min``. Joint-mix **c3/c4** slots use the same flags.

**What full HSiKAN / ``cell_signed_graph`` still has (non-exhaustive):**
optional ``MixedAritySignedKAN`` depth / attention / cycle-batch env knobs,
strict no-leakage protocol, entropy regulariser, etc.  With ``--joint-mix``,
Gömb uses the **same c3,c4,w2,w3 tuple pools** as ``joint_ba`` (train-edge
enumeration + walk enumeration), fused via learned ``α`` across four
``JointMixGomb`` stacks.

Usage:
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset bitcoin_otc --seed 0 --n-epochs 50 --device cpu
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset bitcoin_alpha --edge-split 80_10_10 --seed 0 --n-epochs 80
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset bitcoin_alpha --joint-mix --edge-split 80_10_10 --seed 0

Paired ABB vs baseline (same arch, multiple modes)::

    python -m signedkan_wip.src.benchmarks.run_gomb_cycle_abb_compare \\
        --dataset bitcoin_otc --edge-split 80_10_10 --device cpu \\
        --n-epochs 8 --topk 48 --modes none start_local
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import hymeko

from signedkan_wip.src.datasets import load, split
from signedkan_wip.src.hymeko_gomb import (
    GombConfig, HymeKoGomb, GombNoOuter, GombNoMiddle, GombNoInner,
    GombWithOuterHSIKAN, GombBridgeGomb,
    JointMixGomb, MixedArityGomb,
)
from signedkan_wip.src.hymeko_gomb.joint_enumeration import JOINT_BA_SLOTS, build_joint_ba_pools

# HTL monitor (pure-Python; lazy import inside _build_monitor so missing module
# never blocks a smoke run that doesn't pass --monitor).

_MODELS = {
    "gomb":              HymeKoGomb,
    "no_outer":          GombNoOuter,
    "no_middle":         GombNoMiddle,
    "no_inner":          GombNoInner,
    "outer_hsikan_gomb": GombWithOuterHSIKAN,
    "gomb_bridge_gomb":  GombBridgeGomb,
}

# HSiKAN on Slashdot uses cycle micro-batching; Gömb joint holds full tensors.
# Subsample each slot to this many rows (uniform, train-only pools already).
_DEFAULT_JOINT_SLOT_CAP_SNAP: int = 12_000


def _subsample_joint_pools(
    pools: dict[str, tuple[np.ndarray, np.ndarray]],
    cap: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Uniform subsample without replacement when a slot exceeds ``cap`` rows."""
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for slot, (cyc, sgn) in pools.items():
        m = int(cyc.shape[0])
        if m <= cap:
            out[slot] = (cyc, sgn)
            continue
        idx = rng.choice(m, size=cap, replace=False)
        out[slot] = (cyc[idx], sgn[idx])
    return out


def _enumerate_cycles(
    edges: np.ndarray, signs: np.ndarray, n: int,
    k: int = 3, m_per_vertex: int = 64,
    *,
    abb_mode: str = "none",
    abb_fullness_gate: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    eu = np.ascontiguousarray(edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(signs, dtype=np.int8)
    arr, _ = hymeko.enumerate_cycles_rs(
        eu, ev, es, n, k, m_per_vertex,
        score_kind="fraction_negative",
        pruner_kind="none",
        filter_kind="none",
        filter_min_degree=2,
        abb_mode=abb_mode,
        fullness_gate=float(abb_fullness_gate),
        tiers=[],
        adaptive_c=0.0,
        adaptive_m_min=0,
        adaptive_m_max=0,
    )
    cycles = np.asarray(arr, dtype=np.int64)
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(edges, signs):
        sign_of[(int(u), int(v))] = int(s)
        sign_of[(int(v), int(u))] = int(s)
    cyc_signs = np.zeros_like(cycles, dtype=np.int8)
    for ci, cycle in enumerate(cycles):
        for j in range(k):
            u, v = int(cycle[j]), int(cycle[(j + 1) % k])
            cyc_signs[ci, j] = sign_of.get((u, v), 1)
    return cycles, cyc_signs


def _train_val_split(edges, signs, val_frac, seed):
    rng = np.random.default_rng(seed)
    n = edges.shape[0]
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    return (edges[perm[n_val:]], signs[perm[n_val:]],
            edges[perm[:n_val]], signs[perm[:n_val]])


def _param_breakdown(module: nn.Module) -> dict[str, int]:
    """First-level child module parameter counts (rest is unclassified)."""
    by_child: dict[str, int] = {}
    for name, child in module.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            by_child[name] = int(n)
    total = sum(p.numel() for p in module.parameters())
    accounted = sum(by_child.values())
    if accounted < total:
        by_child["_other_direct"] = int(total - accounted)
    return by_child


def _heldout_edge_metrics(
    y_true: np.ndarray, probs: np.ndarray, label: str,
) -> dict[str, float]:
    """Binary edge-sign metrics at 0.5 threshold + AUROC / AP.

    ``label`` is ``'val'`` or ``'test'`` — keys are ``{label}_auroc``, etc.
    """
    y = y_true.astype(np.int32)
    pred = (probs >= 0.5).astype(np.int32)
    out: dict[str, float] = {}
    lk = label
    try:
        out[f"{lk}_auroc"] = float(roc_auc_score(y, probs))
    except ValueError:
        out[f"{lk}_auroc"] = float("nan")
    try:
        out[f"{lk}_average_precision"] = float(average_precision_score(y, probs))
    except ValueError:
        out[f"{lk}_average_precision"] = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average=None, labels=[0, 1], zero_division=0,
    )
    out[f"{lk}_precision_neg"] = float(prec[0])
    out[f"{lk}_recall_neg"] = float(rec[0])
    out[f"{lk}_f1_neg"] = float(f1[0])
    out[f"{lk}_precision_pos"] = float(prec[1])
    out[f"{lk}_recall_pos"] = float(rec[1])
    out[f"{lk}_f1_pos"] = float(f1[1])
    out[f"{lk}_f1_macro"] = float(
        f1_score(y, pred, average="macro", zero_division=0),
    )
    return out


def _benchmark_inference_wall_s(
    module: nn.Module,
    *,
    forward_edges: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    warmup: bool = True,
) -> tuple[float, int, float]:
    """One batched forward on held-out edges; wall seconds with CUDA sync.

    Returns ``(elapsed_s, n_edges, edges_per_s)``. When ``device`` is CUDA,
    uses ``torch.cuda.synchronize()`` around the timed region and one
    untimed warmup forward if ``warmup`` (captures compile / allocator).
    """
    module.eval()
    n = int(forward_edges.shape[0])
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        if warmup:
            _ = forward_fn(forward_edges)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_fn(forward_edges)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(float(time.perf_counter() - t0), 1e-9)
    return elapsed, n, float(n / elapsed)


def _degree_to_tier(degrees: np.ndarray, n_tiers: int) -> np.ndarray:
    n = degrees.shape[0]
    order = np.argsort(degrees, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n) / max(1, n - 1)
    cuts = np.linspace(0.0, 1.0, n_tiers + 1)
    tiers = np.zeros(n, dtype=np.int64)
    for i in range(n_tiers):
        lo = cuts[i]; hi = cuts[i + 1]
        mask = (ranks >= lo) & (ranks <= hi) if i == 0 else (ranks > lo) & (ranks <= hi)
        tiers[mask] = i
    return tiers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument(
        "--monitor",
        type=str,
        default=None,
        metavar="FORMULA",
        help="HTL (hypergraph temporal logic) formula evaluated each epoch on "
              "scalar training signals (val_auc, loss, best_auc). Examples: "
              "'G(val_auc > 0.85)', 'F(val_auc > 0.90)', "
              "'G(val_auc > 0.85) AND F(val_auc > 0.90)'.",
    )
    ap.add_argument(
        "--monitor-horizon",
        type=int,
        default=1024,
        help="Maximum history window for the HTL monitor (bounded ring buffer).",
    )
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--d-outer", type=int, default=16)
    ap.add_argument("--M-outer", type=int, default=8)
    ap.add_argument("--d-middle", type=int, default=32)
    ap.add_argument("--d-core", type=int, default=32)
    ap.add_argument("--n-tiers", type=int, default=3)
    ap.add_argument(
        "--cpml-topology",
        choices=("route", "pyramid"),
        default="route",
        help="Inner CPML readout: route (default, Option B) vs legacy pyramid.",
    )
    ap.add_argument(
        "--cpml-tier-organization",
        choices=("structural", "capsule_soft"),
        default="structural",
        help="Tier routing: structural (hard incidence) vs capsule_soft "
             "(learned softmax cycle→tier; requires --cpml-topology route).",
    )
    ap.add_argument(
        "--cpml-capsule-route-hidden",
        type=int,
        default=64,
        metavar="H",
        help="Hidden dim for capsule_soft router MLP when "
             "--cpml-capsule-routing-iterations is 1 (ignored if structural).",
    )
    ap.add_argument(
        "--cpml-capsule-routing-iterations",
        type=int,
        default=1,
        metavar="T",
        help="Capsule_soft: interpreted with --cpml-capsule-soft-router (see "
             "CPMLConfig); ignored if tier org is structural.",
    )
    ap.add_argument(
        "--cpml-capsule-soft-router",
        choices=("auto", "mlp_softmax", "hypergraph_conv", "em_agreement"),
        default="auto",
        help="Capsule_soft routing head: auto, MLP on corners, one HGNN-style "
             "step on cycles as hyperedges, or EM agreement (needs T>=2).",
    )
    ap.add_argument(
        "--cpml-capsule-hg-hidden",
        type=int,
        default=64,
        metavar="D",
        help="Hidden dim for hypergraph_conv router (ignored unless that router).",
    )
    ap.add_argument(
        "--cpml-capsule-hg-cache-degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For hypergraph_conv: cache vertex degree tensors when the same "
             "cycles tensor object is reused across forwards (default on).",
    )
    ap.add_argument(
        "--cpml-torch-compile-hypergraph",
        action="store_true",
        help="torch.compile the hypergraph_conv routing submodule (PyTorch 2.x; "
             "first forwards may compile).",
    )
    ap.add_argument("--k", type=int, default=3)
    # ─── Stacked-middle (2026-05-20) ────────────────────────────
    ap.add_argument(
        "--middle-n-layers", type=int, default=1,
        help="Depth of the middle HSIKAN stack. 1 (default) keeps "
              "the original single-tier MiddleHSiKAN; >= 2 dispatches "
              "to StackedMiddleHSiKAN.",
    )
    ap.add_argument(
        "--middle-inner-skip",
        choices=("highway", "cr_highway", "residual", "none", "auto"),
        default="highway",
        help="Per-layer skip kind for the stacked middle (ignored when "
              "--middle-n-layers <= 1).",
    )
    ap.add_argument(
        "--middle-jk-mode",
        choices=("last", "sum", "concat"),
        default="last",
        help="JK aggregation across the stacked middle's L layers.",
    )
    ap.add_argument(
        "--middle-share-weights", action="store_true",
        help="Share parameters across the stacked middle's L layers.",
    )
    # ─── Outer HSIKAN backbone (2026-05-20) ─────────────────────
    ap.add_argument(
        "--outer-hsikan-n-layers", type=int, default=0,
        help="Depth of the outer HSIKAN backbone that sits BEFORE "
              "Gömb's Clifford-FIR shell. 0 (default) = no outer "
              "HSIKAN (use --model gomb). >= 1 requires --model "
              "outer_hsikan_gomb.",
    )
    ap.add_argument(
        "--outer-hsikan-inner-skip",
        choices=("highway", "cr_highway", "residual", "none", "auto"),
        default="highway",
        help="Inner-skip kind for the outer HSIKAN's per-layer gates.",
    )
    ap.add_argument(
        "--outer-hsikan-jk-mode",
        choices=("last", "sum", "concat"),
        default="last",
        help="JK aggregation across the outer HSIKAN's L layers "
              "(affects internal per-triad processing; the per-"
              "vertex output passed to Clifford-FIR is always at "
              "d_embed regardless).",
    )
    ap.add_argument(
        "--outer-hsikan-share-weights", action="store_true",
        help="Share parameters across the outer HSIKAN's L layers.",
    )
    ap.add_argument(
        "--outer-hsikan-grad-checkpoint", action="store_true",
        help="Wrap the outer HSIKAN's forward in torch.utils."
              "checkpoint.checkpoint — necessary for d=4 on "
              "Slashdot where the L-layer autograd graph + Gömb "
              "cascade exceed 7.6 GiB.",
    )
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument(
        "--cycle-abb-mode",
        choices=("none", "start_local", "global_min"),
        default="none",
        help="Rust per-vertex cycle enumerator: ABB branch-and-bound "
             "(start_local / global_min) vs none (default, backward-compatible).",
    )
    ap.add_argument(
        "--cycle-abb-fullness-gate",
        type=float,
        default=0.25,
        metavar="G",
        help="Fullness gate for global_min ABB (ignored for none/start_local).",
    )
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW-style L2 (passed to torch.optim.Adam weight_decay).",
    )
    ap.add_argument(
        "--pos-weight-auto",
        action="store_true",
        help="Class-balanced BCE: pos_weight = n_neg/n_pos on **train** edges "
             "(same recipe as run_final_cell for non-Slashdot HSiKAN).",
    )
    ap.add_argument(
        "--edge-split",
        choices=("80_20", "80_10_10"),
        default="80_20",
        help="80_20: train/val via --val-frac (default 0.2). "
             "80_10_10: datasets.split (same convention as run_final_cell / "
             "run_hsikan_sota_gate); reports test AUROC at end.",
    )
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", choices=sorted(_MODELS), default="gomb")
    ap.add_argument(
        "--shuffle-train-signs", action="store_true",
        help="LABEL-SHUFFLE sanity check (ChatGPT audit, 2026-05-14). "
             "Permute signs on training edges in place; cycle/walk pools "
             "and BCE target both consume the shuffled view. Test edges "
             "untouched. Confirms strict protocol: expected AUC ≈ 0.5 "
             "since Gömb already enumerates cycles on train-edges only.",
    )
    ap.add_argument(
        "--unrestricted-cycles", action="store_true",
        help="UNRESTRICTED (transductive) PROTOCOL. Enumerate the cycle "
             "pool over ALL edges (train + val + test) with their real "
             "signs. The training loss still uses only training-edge "
             "labels; only the σ-product feature extraction sees test "
             "signs. Reproduces the canonical signed-link convention "
             "used by SiGAT / SDGNN / SGCN published numbers. Default "
             "is the strict protocol (train-only cycle pool).",
    )
    ap.add_argument(
        "--cycle-ks", default="",
        help="Comma-separated arities for MixedArityGomb (e.g. '3,4' or '4,5'). "
             "If non-empty, overrides --model with MixedArityGomb.",
    )
    ap.add_argument(
        "--joint-mix",
        action="store_true",
        help="JointMixGomb: four stacks (c3,c4,w2,w3) + α fusion — same tuple "
             "recipe as joint_ba HSiKAN. Mutually exclusive with --cycle-ks.",
    )
    ap.add_argument(
        "--max-walks-w2", type=int, default=50_000,
        help="Cap on length-2 simple walks for joint-mix slot w2.",
    )
    ap.add_argument(
        "--max-walks-w3", type=int, default=50_000,
        help="Cap on length-3 simple walks for joint-mix slot w3.",
    )
    ap.add_argument(
        "--joint-slot-cap",
        type=int,
        default=None,
        metavar="M",
        help="Max rows per joint-mix slot after pooling (uniform subsample). "
             "For slashdot/epinions default is %d when omitted (VRAM). "
             "Use 0 to keep full Rust pools (needs large GPU)." % (
                 _DEFAULT_JOINT_SLOT_CAP_SNAP,
             ),
    )
    args = ap.parse_args()
    cycle_ks: tuple[int, ...] = tuple(
        int(s) for s in args.cycle_ks.split(",") if s.strip()
    )
    if args.joint_mix and cycle_ks:
        raise SystemExit("--joint-mix cannot be combined with --cycle-ks")
    if (
        args.cpml_tier_organization == "capsule_soft"
        and args.cpml_topology != "route"
    ):
        raise SystemExit(
            "--cpml-tier-organization capsule_soft requires --cpml-topology route",
        )
    if args.cpml_capsule_routing_iterations < 1:
        raise SystemExit(
            "--cpml-capsule-routing-iterations must be >= 1",
        )

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    g = load(args.dataset)
    n = g.n_nodes
    print(f"[load] {args.dataset}: |V|={n}, |E|={len(g.edges)}", flush=True)

    three_way = args.edge_split == "80_10_10"
    if three_way:
        tr_idx, va_idx, te_idx = split(g, seed=args.seed)
        # Apply label-shuffle to TRAIN edges only (graph-level, before
        # cycle enumeration consumes g.signs / s_tr).
        if args.shuffle_train_signs:
            shuffle_rng = np.random.default_rng(args.seed + 100003)
            perm = shuffle_rng.permutation(len(tr_idx))
            g.signs[tr_idx] = g.signs[tr_idx][perm]
            print(
                f"[run_gomb_smoke] LABEL-SHUFFLE active: permuted signs "
                f"on {len(tr_idx)} training edges "
                f"(seed+100003={args.seed + 100003}).", flush=True,
            )
        e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
        e_va, s_va = g.edges[va_idx], g.signs[va_idx]
        e_te, s_te = g.edges[te_idx], g.signs[te_idx]
        print(
            f"[split] 80/10/10 train={len(tr_idx)} val={len(va_idx)} "
            f"test={len(te_idx)} seed={args.seed}",
            flush=True,
        )
    else:
        e_tr, s_tr, e_va, s_va = _train_val_split(
            g.edges, g.signs, args.val_frac, args.seed,
        )
        if args.shuffle_train_signs:
            shuffle_rng = np.random.default_rng(args.seed + 100003)
            perm = shuffle_rng.permutation(s_tr.shape[0])
            s_tr = s_tr[perm]
            print(
                f"[run_gomb_smoke] LABEL-SHUFFLE active (80/20): "
                f"permuted signs on {s_tr.shape[0]} training edges.",
                flush=True,
            )
        e_te = None
        s_te = None
        print(
            f"[split] train={e_tr.shape[0]} val={e_va.shape[0]} "
            f"(val_frac={args.val_frac}) seed={args.seed}",
            flush=True,
        )

    joint_mix = bool(args.joint_mix)
    mixed = len(cycle_ks) >= 2
    pools_joint: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    joint_slot_cap: int | None = None
    # Cycle-pool edge set: strict protocol (default) uses train edges
    # only; --unrestricted-cycles uses the full edge set so test-edge
    # signs participate in σ-products. The training/eval split itself
    # is unchanged either way.
    if args.unrestricted_cycles:
        e_cyc, s_cyc = g.edges, g.signs
        print(
            f"[protocol] UNRESTRICTED (transductive): cycle pool over "
            f"{e_cyc.shape[0]} edges (train+val+test).",
            flush=True,
        )
    else:
        e_cyc, s_cyc = e_tr, s_tr
        print(
            f"[protocol] STRICT: cycle pool over {e_cyc.shape[0]} "
            f"training edges only.",
            flush=True,
        )
    # k-pool enumeration: one (cycles, signs) per arity for mixed,
    # joint-mix dict for JointMixGomb, else a single pool.
    if joint_mix:
        pools_joint = build_joint_ba_pools(
            e_cyc, s_cyc, n,
            topk_c3=args.topk, topk_c4=args.topk,
            max_walks_w2=args.max_walks_w2, max_walks_w3=args.max_walks_w3,
            walk_seed=args.seed,
            subsample_walks_seed=args.seed,
            cycle_abb_mode=args.cycle_abb_mode,
            cycle_abb_fullness_gate=float(args.cycle_abb_fullness_gate),
        )
        if args.joint_slot_cap is not None and args.joint_slot_cap == 0:
            joint_slot_cap = None
        elif args.joint_slot_cap is not None and args.joint_slot_cap > 0:
            joint_slot_cap = int(args.joint_slot_cap)
        elif args.dataset in ("slashdot", "epinions"):
            joint_slot_cap = _DEFAULT_JOINT_SLOT_CAP_SNAP
        else:
            joint_slot_cap = None
        if joint_slot_cap is not None:
            pools_joint = _subsample_joint_pools(
                pools_joint, joint_slot_cap, args.seed + 91,
            )
            print(
                f"[joint-mix] per-slot row cap={joint_slot_cap} "
                f"(subsampled; HSiKAN-scale graphs)",
                flush=True,
            )
        n_cycles_total = 0
        for slot in JOINT_BA_SLOTS:
            m_i = int(pools_joint[slot][0].shape[0])
            n_cycles_total += m_i
            print(f"[joint-mix] slot {slot}: {m_i} tuples", flush=True)
        print(f"[joint-mix] total tuples (all slots)={n_cycles_total}", flush=True)
        ks_used: tuple[int, ...] = ()
    elif mixed:
        ks_used = cycle_ks
        cycles_by_k_np: dict[int, np.ndarray] = {}
        cyc_signs_by_k_np: dict[int, np.ndarray] = {}
        n_cycles_total = 0
        for k in ks_used:
            cyc_k, sgn_k = _enumerate_cycles(
                e_cyc, s_cyc, n, k=k, m_per_vertex=args.topk,
                abb_mode=args.cycle_abb_mode,
                abb_fullness_gate=float(args.cycle_abb_fullness_gate),
            )
            cycles_by_k_np[k] = cyc_k
            cyc_signs_by_k_np[k] = sgn_k
            n_cycles_total += cyc_k.shape[0]
            print(f"[cycles] k={k}: {cyc_k.shape[0]}", flush=True)
        print(f"[cycles] total={n_cycles_total} mixed={ks_used}", flush=True)
    else:
        ks_used = (args.k,)
        cycles_np, cyc_signs_np = _enumerate_cycles(
            e_cyc, s_cyc, n, k=args.k, m_per_vertex=args.topk,
            abb_mode=args.cycle_abb_mode,
            abb_fullness_gate=float(args.cycle_abb_fullness_gate),
        )
        n_cycles_total = int(cycles_np.shape[0])
        print(f"[cycles] {n_cycles_total} k={args.k}", flush=True)

    print(
        f"[cycles] abb_mode={args.cycle_abb_mode} "
        f"abb_fullness_gate={args.cycle_abb_fullness_gate}",
        flush=True,
    )

    degrees = np.zeros(n, dtype=np.int64)
    for (u, v) in e_tr:
        degrees[int(u)] += 1; degrees[int(v)] += 1
    tier_of_np = _degree_to_tier(degrees, args.n_tiers)

    cfg = GombConfig(
        n_nodes=n, d_embed=args.d_embed,
        d_outer=args.d_outer, M_outer=args.M_outer,
        d_middle=args.d_middle, d_core=args.d_core,
        n_tiers=args.n_tiers, cycle_k=args.k,
        cpml_topology=args.cpml_topology,
        cpml_tier_organization=args.cpml_tier_organization,
        cpml_capsule_route_hidden=args.cpml_capsule_route_hidden,
        cpml_capsule_routing_iterations=args.cpml_capsule_routing_iterations,
        cpml_capsule_soft_router=args.cpml_capsule_soft_router,
        cpml_capsule_hg_hidden=args.cpml_capsule_hg_hidden,
        cpml_capsule_hg_cache_degrees=args.cpml_capsule_hg_cache_degrees,
        cpml_torch_compile_hypergraph=args.cpml_torch_compile_hypergraph,
        middle_n_layers=args.middle_n_layers,
        middle_inner_skip=args.middle_inner_skip,
        middle_jk_mode=args.middle_jk_mode,
        middle_share_weights=args.middle_share_weights,
        outer_hsikan_n_layers=args.outer_hsikan_n_layers,
        outer_hsikan_inner_skip=args.outer_hsikan_inner_skip,
        outer_hsikan_jk_mode=args.outer_hsikan_jk_mode,
        outer_hsikan_share_weights=args.outer_hsikan_share_weights,
        outer_hsikan_grad_checkpoint=args.outer_hsikan_grad_checkpoint,
    )
    if joint_mix:
        gomb = JointMixGomb(cfg).to(device)
        model_label = "joint_mix_gomb[c3,c4,w2,w3]"
    elif mixed:
        gomb = MixedArityGomb(cfg, cycle_ks=cycle_ks).to(device)
        model_label = f"mixed_arity_gomb[{','.join(str(k) for k in cycle_ks)}]"
    else:
        gomb = _MODELS[args.model](cfg).to(device)
        model_label = args.model

    if joint_mix:
        assert pools_joint is not None
        cyc_t_by_slot = {
            s: torch.from_numpy(pools_joint[s][0]).to(device) for s in JOINT_BA_SLOTS
        }
        cyc_sgn_t_by_slot = {
            s: torch.from_numpy(pools_joint[s][1]).to(device) for s in JOINT_BA_SLOTS
        }
    elif mixed:
        cyc_t_by_k = {
            k: torch.from_numpy(cycles_by_k_np[k]).to(device) for k in ks_used
        }
        cyc_sgn_t_by_k = {
            k: torch.from_numpy(cyc_signs_by_k_np[k]).to(device) for k in ks_used
        }
    else:
        cyc_t = torch.from_numpy(cycles_np).to(device)
        cyc_sgn_t = torch.from_numpy(cyc_signs_np).to(device)
    tier_of = torch.from_numpy(tier_of_np).to(device)
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    s_tr_t = torch.from_numpy((s_tr > 0).astype(np.float32)).to(device)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    s_va_y = (s_va > 0).astype(np.float32)
    if three_way:
        assert e_te is not None and s_te is not None
        e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
        s_te_y = (s_te > 0).astype(np.float32)
    else:
        e_te_t = None
        s_te_y = None

    opt = torch.optim.Adam(
        gomb.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.pos_weight_auto:
        n_pos = float(s_tr_t.sum().item())
        n_neg = float((1.0 - s_tr_t).sum().item())
        pos_w = max(n_neg, 1.0) / max(n_pos, 1.0)
        bce_pw = torch.tensor(pos_w, device=device, dtype=torch.float32)
        print(
            f"[loss] pos_weight_auto  n_pos={int(n_pos)} n_neg={int(n_neg)} "
            f"pos_weight={pos_w:.4f}",
            flush=True,
        )
    else:
        bce_pw = None
    print(
        f"[model] {model_label} d_embed={args.d_embed} M_outer={args.M_outer} "
        f"d_outer={args.d_outer} d_middle={args.d_middle} "
        f"d_core={args.d_core} n_tiers={args.n_tiers} "
        f"cpml_topology={args.cpml_topology} "
        f"cpml_tier_organization={args.cpml_tier_organization} "
        f"cpml_capsule_soft_router={args.cpml_capsule_soft_router} "
        f"cpml_capsule_routing_iterations={args.cpml_capsule_routing_iterations} "
        f"n_params={gomb.n_params()}  wd={args.weight_decay}",
        flush=True,
    )

    def _fwd(edges_t):
        if joint_mix:
            assert cyc_t_by_slot is not None and cyc_sgn_t_by_slot is not None
            return gomb(cyc_t_by_slot, cyc_sgn_t_by_slot, tier_of, edges_t)
        if mixed:
            return gomb(cyc_t_by_k, cyc_sgn_t_by_k, tier_of, edges_t)
        return gomb(cyc_t, cyc_sgn_t, tier_of, edges_t)

    losses, val_aucs = [], []
    best = 0.0

    monitor = None
    monitor_trace: list[dict] = []
    if args.monitor:
        from signedkan_wip.src.htl import HtlMonitor, HypergraphEvent
        try:
            monitor = HtlMonitor(args.monitor, horizon=args.monitor_horizon)
            print(
                f"[htl] monitor active: formula={args.monitor!r} "
                f"horizon={args.monitor_horizon}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface ParseError at boundary
            print(f"[htl] monitor parse failed: {exc!r}", flush=True)
            raise

    for ep in range(args.n_epochs):
        gomb.train()
        scores = _fwd(e_tr_t)
        loss = F.binary_cross_entropy_with_logits(
            scores, s_tr_t, pos_weight=bce_pw,
        )
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gomb.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))

        gomb.eval()
        with torch.no_grad():
            v_scores = _fwd(e_va_t)
            v_probs = torch.sigmoid(v_scores).cpu().numpy()
        try:
            auc = float(roc_auc_score(s_va_y, v_probs))
        except ValueError:
            auc = float("nan")
        val_aucs.append(auc)
        best = max(best, auc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep:02d}  loss={loss.item():.4f}  "
                  f"val_auc={auc:.4f}  best={best:.4f}", flush=True)

        if monitor is not None:
            evt = HypergraphEvent(
                t=float(ep),
                scalar_signals={
                    "val_auc": float(auc) if auc == auc else -1.0,
                    "loss": float(loss.detach()),
                    "best_auc": float(best),
                },
            )
            rho = monitor.observe(evt)
            sat = monitor.satisfied()
            monitor_trace.append({"epoch": ep, "rho": rho, "satisfied": bool(sat)})
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [htl] ep {ep:02d}  rho={rho:.4f}  "
                      f"satisfied={'Y' if sat else 'N'}", flush=True)

    gomb.eval()
    with torch.no_grad():
        v_scores_final = _fwd(e_va_t)
        v_probs_final = torch.sigmoid(v_scores_final).cpu().numpy()
        if three_way and e_te_t is not None:
            te_scores_final = _fwd(e_te_t)
            te_probs_final = torch.sigmoid(te_scores_final).cpu().numpy()
        else:
            te_probs_final = None

    metrics_val = _heldout_edge_metrics(s_va_y, v_probs_final, "val")
    if three_way and te_probs_final is not None and s_te_y is not None:
        metrics_test = _heldout_edge_metrics(s_te_y, te_probs_final, "test")
        metrics = {**metrics_val, **metrics_test}
    else:
        metrics_test = None
        metrics = metrics_val
    params_by_module = _param_breakdown(gomb)

    infer_parts: list[torch.Tensor] = [e_va_t]
    if three_way and e_te_t is not None:
        infer_parts.append(e_te_t)
    e_infer = torch.cat(infer_parts, dim=0)
    infer_wall_s, infer_n_edges, infer_edges_per_s = (
        _benchmark_inference_wall_s(
            gomb,
            forward_edges=e_infer,
            forward_fn=_fwd,
            device=device,
        )
    )
    print(
        f"[inference] edges={infer_n_edges} wall_s={infer_wall_s:.4f} "
        f"edges_per_s={infer_edges_per_s:.1f}",
        flush=True,
    )

    alpha_dump = None
    if mixed or joint_mix:
        alpha_dump = gomb.alpha().cpu().tolist()  # type: ignore[union-attr]
    summary = {
        "dataset": args.dataset, "seed": args.seed,
        "edge_split": args.edge_split,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight_auto": bool(args.pos_weight_auto),
        "model": model_label,
        "joint_mix": joint_mix,
        "joint_slot_cap": int(joint_slot_cap) if joint_mix and joint_slot_cap else None,
        "cycle_ks": list(ks_used) if mixed and not joint_mix else None,
        "alpha_k":  alpha_dump,
        "M_outer": args.M_outer, "d_outer": args.d_outer,
        "d_middle": args.d_middle, "d_core": args.d_core,
        "n_tiers": args.n_tiers, "k": args.k, "topk": args.topk,
        "cycle_abb_mode": args.cycle_abb_mode,
        "cycle_abb_fullness_gate": float(args.cycle_abb_fullness_gate),
        "cpml_topology": args.cpml_topology,
        "cpml_tier_organization": args.cpml_tier_organization,
        "cpml_capsule_route_hidden": args.cpml_capsule_route_hidden,
        "cpml_capsule_routing_iterations": args.cpml_capsule_routing_iterations,
        "cpml_capsule_soft_router": args.cpml_capsule_soft_router,
        "cpml_capsule_hg_hidden": args.cpml_capsule_hg_hidden,
        "cpml_capsule_hg_cache_degrees": bool(args.cpml_capsule_hg_cache_degrees),
        "cpml_torch_compile_hypergraph": bool(args.cpml_torch_compile_hypergraph),
        "n_params": gomb.n_params(),
        "params_by_module": params_by_module,
        "n_cycles": n_cycles_total,
        "n_train_edges": int(e_tr.shape[0]),
        "n_val_edges": int(e_va.shape[0]),
        "loss_start": losses[0], "loss_end": losses[-1],
        "val_auc_start": val_aucs[0], "val_auc_end": val_aucs[-1],
        "val_auc_best": best,
        "htl_formula": args.monitor,
        "htl_final_robustness": (monitor_trace[-1]["rho"] if monitor_trace else None),
        "htl_final_satisfied": (monitor_trace[-1]["satisfied"] if monitor_trace else None),
        "htl_trace": monitor_trace if monitor_trace else None,
        "wall_s": time.perf_counter() - t0,
        "infer_wall_s": infer_wall_s,
        "infer_n_edges": infer_n_edges,
        "infer_edges_per_s": infer_edges_per_s,
        **metrics,
    }
    if three_way and e_te is not None:
        summary["n_test_edges"] = int(e_te.shape[0])
    if args.edge_split == "80_20":
        summary["val_frac"] = args.val_frac

    if three_way and metrics_test is not None:
        print(
            f"[metrics] val_AUROC={metrics_val['val_auroc']:.4f}  "
            f"val_AP={metrics_val['val_average_precision']:.4f}  "
            f"test_AUROC={metrics_test['test_auroc']:.4f}  "
            f"test_AP={metrics_test['test_average_precision']:.4f}  "
            f"R+_val={metrics_val['val_recall_pos']:.4f}  "
            f"R+_test={metrics_test['test_recall_pos']:.4f}  "
            f"F1_macro_val={metrics_val['val_f1_macro']:.4f}  "
            f"F1_macro_test={metrics_test['test_f1_macro']:.4f}",
            flush=True,
        )
    else:
        print(
            f"[metrics] AUROC={metrics_val['val_auroc']:.4f}  "
            f"AP={metrics_val['val_average_precision']:.4f}  "
            f"R+={metrics_val['val_recall_pos']:.4f}  R-={metrics_val['val_recall_neg']:.4f}  "
            f"P+={metrics_val['val_precision_pos']:.4f}  "
            f"F1_macro={metrics_val['val_f1_macro']:.4f}",
            flush=True,
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
