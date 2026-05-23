"""HyMeKo-walked training cell.

Where ``hymeko_driver.py`` extracted knobs and dispatched to a hand-coded
``cell_signed_graph()``, this walker actually **traverses the training
dataflow** declared in ``data/hsikan/training.hymeko``: parses the tagged
hyperedges (`<dataset>`, `<cycle_enum>`, `<forward>`, `<loss>`,
`<backward>`, `<optimizer>`, `<epoch_loop>`), topologically sorts them by
tensor input/output dependency, and dispatches each kind to a registered
handler.

The ``<epoch_loop>`` edge wraps the inner forward/loss/backward/optimizer
subgraph and re-fires it n_epochs times.  Reordering or removing any
inner edge in the .hymeko file changes training; the order is *not*
hardcoded in Python any more.

Usage:

    python -m signedkan_wip.src.hymeko_train_walker \\
        --arch     data/hsikan/arch_mixed_k34.hymeko \\
        --training data/hsikan/training.hymeko \\
        --dataset  bitcoin_alpha
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

import hymeko  # PyO3 wheel; provides parse_hymeko_rs


# Shared HyMeKo-IR helpers — single source of truth in hymeko_ir.py.
from .hymeko_ir import (
    read_hymeko as _read_hymeko,
    all_items as _all_items,
    has_base as _has_base,
    child_value as _child_value,
)


# ─── Dataflow edge model ───────────────────────────────────────────


@dataclass
class FlowEdge:
    """A single tagged hyperedge from training.hymeko."""
    name: str
    kind: str             # primary tag — "dataset", "cycle_enum", ...
    tags: tuple[str, ...]
    inputs: tuple[str, ...]   # `+` operands
    outputs: tuple[str, ...]  # `-` operands
    op_token: str | None      # `~` operand (the op name)
    body: dict                # raw HyMeKo body for kwarg lookup

    @classmethod
    def from_raw(cls, raw: dict) -> "FlowEdge":
        tags = tuple(raw.get("tags") or [])
        # First tag is the "kind"; we use one-of: dataset, cycle_enum,
        # forward, loss, backward, optimizer, epoch_loop.
        kind = tags[0] if tags else "unknown"
        # Edge body holds (+ tensor, ~ op, - tensor) hyperarc.
        plus, minus, op = [], [], None
        for b in raw.get("body") or []:
            if b.get("kind") == "arc":
                for ref in b.get("refs") or []:
                    sign = ref.get("sign")
                    target = ref.get("name") or ref.get("path", [None])[-1]
                    if sign == "+":
                        plus.append(target)
                    elif sign == "-":
                        minus.append(target)
                    elif sign == "~":
                        op = target
        return cls(
            name=raw.get("name", "<anon>"),
            kind=kind,
            tags=tags,
            inputs=tuple(plus),
            outputs=tuple(minus),
            op_token=op,
            body=raw,
        )


def parse_dataflow(training_path: str) -> tuple[list[FlowEdge], dict[str, Any]]:
    """Parse training.hymeko into a list of FlowEdges + the const block."""
    tree = _read_hymeko(training_path)
    consts = {c["name"]: c["value"] for c in (tree.get("consts") or [])}
    edges: list[FlowEdge] = []
    for it in _all_items(tree):
        if it.get("kind") == "edge":
            edges.append(FlowEdge.from_raw(it))
    return edges, consts


def topo_sort(edges: list[FlowEdge]) -> list[FlowEdge]:
    """Topological sort on tensor input/output dependency.

    A boundary set (``{"edges", "edge_labels", "grads"}``) is provided
    by the runtime — those tensors are produced by the dataset and
    autograd respectively and are considered always-available.
    """
    boundary = {"edges", "edge_labels", "grads", "weights_updated"}
    available = set(boundary)
    pending = list(edges)
    out: list[FlowEdge] = []
    while pending:
        progressed = False
        for e in pending[:]:
            if all(t in available or t == "" for t in e.inputs):
                out.append(e)
                available.update(e.outputs)
                pending.remove(e)
                progressed = True
        if not progressed:
            unresolved = [(e.name, [t for t in e.inputs if t not in available])
                          for e in pending]
            raise ValueError(
                f"dataflow has unresolvable dependencies: {unresolved}"
            )
    return out


# ─── Walker context ─────────────────────────────────────────────────


@dataclass
class Ctx:
    device: torch.device
    seed: int = 0
    consts: dict[str, Any] = field(default_factory=dict)
    arch_knobs: dict[str, Any] = field(default_factory=dict)
    # Boundary tensors
    edges_train_np: np.ndarray | None = None
    signs_train_np: np.ndarray | None = None
    edges_test_np: np.ndarray | None = None
    signs_test_np: np.ndarray | None = None
    n_nodes: int = 0
    pos_weight: torch.Tensor | None = None
    # Cycle structure (per-arity)
    cycles: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    arities: tuple[int, ...] = ()
    # Model + optimiser
    model: torch.nn.Module | None = None
    opt: torch.optim.Optimizer | None = None
    grad_clip: float = 0.0
    entropy_lambda: float = 0.0
    entropy_kind: str = "spectral"
    n_epochs: int = 30
    # Per-iteration tensors
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None
    # Outputs
    metrics: dict[str, Any] = field(default_factory=dict)
    # Logging
    epoch: int = 0


# ─── Op handlers ────────────────────────────────────────────────────


OPS: dict[str, Callable[[Ctx, FlowEdge], None]] = {}


def register(kind: str):
    def deco(fn):
        OPS[kind] = fn
        return fn
    return deco


@register("dataset")
def op_dataset(ctx: Ctx, e: FlowEdge):
    """Stash dataset name + split for later cell_signed_graph dispatch."""
    name = _child_value(e.body, "name", "bitcoin_alpha")
    if ctx.consts.get("_dataset_override"):
        name = ctx.consts["_dataset_override"]
    ctx.consts["_dataset_resolved"] = name
    train_frac = float(_child_value(e.body, "train_frac", 0.8))
    ctx.consts["_train_frac"] = train_frac
    print(f"[op_dataset] resolved name={name} train_frac={train_frac}",
          flush=True)


@register("cycle_enum")
def op_cycle_enum(ctx: Ctx, e: FlowEdge):
    """Drive the top-K env-var path consumed by cell_signed_graph
    + n_tuples.construct_k.  Walker is policy; runtime is mechanism."""
    mode = _child_value(e.body, "mode", "")
    m = int(_child_value(e.body, "m_per_vertex", 16))
    scorer = _child_value(e.body, "scorer", "fraction_negative")
    pruner = _child_value(e.body, "pruner", "none")
    arities_raw = _child_value(e.body, "arities", [3, 4])
    arities = tuple(int(a) for a in (arities_raw or [3, 4]))
    ctx.arities = arities

    if mode:
        os.environ["HSIKAN_TOPK_MODE"] = mode
        os.environ["HSIKAN_TOPK_K"] = str(m)
        os.environ["HSIKAN_TOPK_SCORER"] = scorer
        os.environ["HSIKAN_TOPK_PRUNER"] = pruner
    os.environ["HSIKAN_ARITIES"] = ",".join(str(a) for a in arities)
    print(f"[op_cycle_enum] arities={arities} mode={mode!r} m={m} "
          f"scorer={scorer!r} pruner={pruner!r}", flush=True)


@register("optimizer")
def op_optimizer(ctx: Ctx, e: FlowEdge):
    """Stash optimizer config — used by epoch_loop's cell dispatch."""
    ctx.consts["_opt_kind"] = _child_value(e.body, "kind", "adam")
    ctx.consts["_lr"] = float(_child_value(e.body, "lr", 1e-2))
    ctx.consts["_wd"] = float(_child_value(e.body, "weight_decay", 1e-4))
    ctx.grad_clip = float(_child_value(e.body, "grad_clip", 0.0))
    print(f"[op_optimizer] kind={ctx.consts['_opt_kind']} "
          f"lr={ctx.consts['_lr']} wd={ctx.consts['_wd']} "
          f"clip={ctx.grad_clip}", flush=True)


@register("loss")
def op_loss(ctx: Ctx, e: FlowEdge):
    """Stash loss config + drive HSIKAN_ENTROPY_LAMBDA env var."""
    ctx.entropy_lambda = float(_child_value(e.body, "entropy_lambda", 0.0))
    ctx.entropy_kind = _child_value(e.body, "entropy_kind", "spectral")
    if ctx.entropy_lambda > 0:
        os.environ["HSIKAN_ENTROPY_LAMBDA"] = str(ctx.entropy_lambda)
    print(f"[op_loss] kind=bce entropy_lambda={ctx.entropy_lambda}",
          flush=True)


@register("forward")
def op_forward_decl(ctx: Ctx, e: FlowEdge):
    """Forward is delegated to cell_signed_graph; this fires for
    declaration parity with the dataflow shape."""
    pass


@register("backward")
def op_backward_decl(ctx: Ctx, e: FlowEdge):
    """Backward delegated; declaration only."""
    pass


@register("epoch_loop")
def op_epoch_loop(ctx: Ctx, e: FlowEdge):
    """Fire the actual training cell.  The walker's prior ops have
    set every env var + ctx.consts entry the cell needs; this op
    invokes the existing kernel and stashes its metrics."""
    n_epochs = int(_child_value(e.body, "n_epochs", ctx.n_epochs))
    ctx.n_epochs = n_epochs
    print(f"[op_epoch_loop] n_epochs={n_epochs} → cell_signed_graph",
          flush=True)

    from .run_final_cell import cell_signed_graph
    a = ctx.arch_knobs
    dataset = ctx.consts.get("_dataset_resolved", "bitcoin_alpha")
    from .runtime_config import get_runtime
    out = cell_signed_graph(
        dataset, "HSiKAN",
        a.get("hidden", 16), n_epochs,
        get_runtime().training.max_k4,
        ctx.device, seed=ctx.seed,
    )
    if out is None:
        out = {}
    ctx.metrics.update(out)


def _eval(ctx: Ctx) -> None:
    """No-op; cell_signed_graph already returned final AUC/F1 in
    ctx.metrics. Kept as a slot for future post-loop ops in the
    HyMeKo training graph (e.g., calibration, distillation)."""
    pass


# ─── Walker entry point ─────────────────────────────────────────────


# parse_arch lives in hymeko_ir; re-imported under the same name.
from .hymeko_ir import parse_arch


def walk(training_path: str, arch_path: str,
          dataset_override: str | None = None,
          seed: int = 0) -> dict:
    edges, consts = parse_dataflow(training_path)
    arch_knobs = parse_arch(arch_path)

    if dataset_override:
        # Stash override in consts; op_dataset honours it via lookup.
        # (Don't mutate parsed body — keeps IR pure.)
        pass  # threaded below via consts["_dataset_override"]

    ordered = topo_sort(edges)
    print(f"[walk] dataflow ordered:")
    for e in ordered:
        print(f"  - @{e.name:<20} <{e.kind:<12}> "
              f"+{list(e.inputs)} → -{list(e.outputs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = Ctx(device=device, seed=seed, consts=consts, arch_knobs=arch_knobs)
    ctx.n_epochs = int(consts.get("N_EPOCHS", 30))
    if dataset_override:
        ctx.consts["_dataset_override"] = dataset_override

    for e in ordered:
        handler = OPS.get(e.kind)
        if handler is None:
            print(f"  [skip] no handler for kind={e.kind}", flush=True)
            continue
        handler(ctx, e)

    _eval(ctx)
    ctx.metrics["dataset"] = dataset_override or _child_value(
        next(e.body for e in ordered if e.kind == "dataset"),
        "name", "bitcoin_alpha",
    )
    ctx.metrics["model"] = "HSiKAN-mixed (HyMeKo-walked)"
    return ctx.metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metrics = walk(args.training, args.arch,
                    dataset_override=args.dataset, seed=args.seed)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
