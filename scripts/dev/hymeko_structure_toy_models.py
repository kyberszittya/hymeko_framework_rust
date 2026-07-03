"""Run HSIKAN, Gomb, Gomb-Soma, and FSR on a parsed HyMeKo toy source."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import hymeko
from hymeko_lm.sequence_mixer import FiberSpikeRotorMixer
from hymeko_neuro.core import CatmullRomActivation
from hymeko_neuro.graph.embeddings.cayley_rotor import cayley_to_unit_quat
from hymeko_neuro.models.hymeko_gomb import GombConfig, HymeKoGomb, MiddleHSiKAN
from hymeko_neuro.models.hymeko_gomb.soma import HypergraphConvConfig, WalkConvLayer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name("hymeko_structure_toy.hymeko")


@dataclass(frozen=True)
class ToyGraph:
    """Tensor-ready projection of a parsed toy HyMeKo AST."""

    names: list[str]
    features: torch.Tensor
    tier_of: torch.Tensor
    cycles: torch.Tensor
    cycle_signs: torch.Tensor
    walks: torch.Tensor
    walk_signs: torch.Tensor
    edges: torch.Tensor
    edge_signs: torch.Tensor
    sequence_order: list[int]


@dataclass(frozen=True)
class ToyModelSuite:
    """Parsed toy graph plus cached modules/tensors for hot-path timings."""

    source: Path
    graph: ToyGraph
    hsikan: torch.nn.Module
    fixed_hsikan: torch.nn.Module
    gomb: torch.nn.Module
    fixed_gomb: torch.nn.Module
    sa_hsikan: torch.nn.Module
    gomb_soma: torch.nn.Module
    fsr: torch.nn.Module
    walk_membership: torch.Tensor
    fsr_input: torch.Tensor


def _dense_cycle_pool(cycles: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Dense vertex-by-cycle mean-pool for tiny fixed-topology experiments."""
    n_cycles = int(cycles.shape[0])
    pool = torch.zeros(n_nodes, n_cycles, dtype=torch.float32)
    if n_cycles == 0:
        return pool
    rows = cycles.reshape(-1)
    cols = torch.arange(n_cycles).repeat_interleave(cycles.shape[1])
    pool.index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float32), accumulate=True)
    counts = pool.sum(dim=1, keepdim=True).clamp_min(1.0)
    return pool / counts


class FixedTopologyGomb(torch.nn.Module):
    """Gömb forward with fixed toy topology cached as dense incidence pools."""

    def __init__(self, model: HymeKoGomb, graph: ToyGraph) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("cycles", graph.cycles)
        self.register_buffer("cycle_signs", graph.cycle_signs)
        self.register_buffer("edges", graph.edges)
        self.register_buffer("cycle_pool", _dense_cycle_pool(graph.cycles, len(graph.names)))
        for ell in range(model.core.cpml.L):
            cycle_tiers = graph.tier_of[graph.cycles]
            mask = (cycle_tiers == ell).any(dim=1)
            cycles_ell = graph.cycles[mask]
            signs_ell = graph.cycle_signs[mask]
            self.register_buffer(f"cycles_ell_{ell}", cycles_ell)
            self.register_buffer(f"signs_ell_{ell}", signs_ell)
            self.register_buffer(f"pool_ell_{ell}", _dense_cycle_pool(cycles_ell, len(graph.names)))

    def _outer(self, x_embed: torch.Tensor) -> torch.Tensor:
        outer = self.model.outer
        w_stack = torch.stack([p.weight for p in outer.pre_projs], dim=0)
        b_stack = torch.stack([p.bias for p in outer.pre_projs], dim=0)
        x_all = torch.einsum("ni,mji->mnj", x_embed, w_stack) + b_stack.unsqueeze(1)
        bank_outputs = []
        signs = self.cycle_signs.float()
        for m, bank in enumerate(outer.banks):
            per_cycle = bank(x_all[m][self.cycles], signs)
            bank_outputs.append(self.cycle_pool.to(per_cycle.dtype) @ per_cycle)
        return torch.cat(bank_outputs, dim=-1)

    def _middle(self, x_for_middle: torch.Tensor) -> torch.Tensor:
        middle = self.model.middle
        x_proj = middle.pre_proj(x_for_middle)
        per_cycle = middle.agg(x_proj, self.cycles, self.cycle_signs)
        return self.cycle_pool.to(per_cycle.dtype) @ per_cycle

    def _core(self, x_for_core: torch.Tensor) -> torch.Tensor:
        cpml = self.model.core.cpml
        h_parts = []
        for ell in range(cpml.L):
            cycles_ell = getattr(self, f"cycles_ell_{ell}")
            if cycles_ell.shape[0] == 0:
                h_ell = torch.zeros(
                    x_for_core.shape[0],
                    cpml.cfg.d_layer,
                    device=x_for_core.device,
                    dtype=x_for_core.dtype,
                )
            else:
                pool = getattr(self, f"pool_ell_{ell}").to(x_for_core.dtype)
                per_cycle = cpml.aggregators[ell](x_for_core[cycles_ell])
                h_ell = pool @ per_cycle
            h_parts.append(h_ell)
        x_final = torch.cat([x_for_core, torch.cat(h_parts, dim=-1)], dim=-1)
        return cpml._edge_logits(x_final, self.edges)

    def forward(self) -> torch.Tensor:
        x_embed = self.model.node_embed.weight
        x_outer = self._outer(x_embed)
        x_for_middle = torch.cat([x_embed, x_outer], dim=-1)
        x_middle = self._middle(x_for_middle)
        x_for_core = torch.cat([x_embed, x_outer, x_middle], dim=-1)
        return self._core(x_for_core)


class FixedTopologyHSIKAN(torch.nn.Module):
    """HSIKAN with fixed toy cycle-to-vertex pooling cached as a dense operator."""

    def __init__(self, model: MiddleHSiKAN, graph: ToyGraph) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("cycles", graph.cycles)
        self.register_buffer("cycle_signs", graph.cycle_signs)
        self.register_buffer("cycle_pool", _dense_cycle_pool(graph.cycles, len(graph.names)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x_proj = self.model.pre_proj(features)
        if self.cycles.shape[0] == 0:
            return torch.zeros_like(x_proj)
        per_cycle = self.model.agg(x_proj, self.cycles, self.cycle_signs)
        return self.cycle_pool.to(per_cycle.dtype) @ per_cycle


class FixedHolonomyHSIKAN(torch.nn.Module):
    """SA-HSIKAN-style fixed signed-walk collapse for the parsed toy graph."""

    def __init__(self, graph: ToyGraph, hidden: int = 8, walk_len: int = 2) -> None:
        super().__init__()
        n_nodes = len(graph.names)
        adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
        for (src, dst), sign in zip(graph.edges.tolist(), graph.edge_signs.tolist()):
            adj[src, dst] = float(sign)
            adj[dst, src] = float(sign)
        self.register_buffer("bl", torch.matrix_power(adj, walk_len))
        self.node_head = torch.nn.Sequential(
            torch.nn.Linear(graph.features.shape[1], hidden),
            CatmullRomActivation(hidden),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        node_agg = self.bl.to(features.dtype) @ features
        return self.node_head(node_agg)


def _flatten_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        out.append(item)
        body = item.get("body")
        if isinstance(body, list):
            out.extend(_flatten_items(body))
    return out


def _path_name(ref: dict[str, Any]) -> str:
    path = ref["path"]
    return str(path[-1])


def _base_name(item: dict[str, Any]) -> str:
    bases = item.get("bases") or []
    if not bases:
        return ""
    return ".".join(str(p) for p in bases[0]["path"])


def _field_value(node: dict[str, Any], field: str, default: Any) -> Any:
    body = node.get("body")
    if not isinstance(body, list):
        return default
    for item in body:
        if item.get("kind") == "node" and item.get("name") == field:
            return item.get("value")
    return default


def _edge_refs(edge: dict[str, Any]) -> tuple[list[str], list[int]]:
    body = edge.get("body")
    if not isinstance(body, list) or not body:
        return [], []
    arc = next((item for item in body if item.get("kind") == "arc"), None)
    if arc is None:
        return [], []
    names = []
    signs = []
    for ref in arc["refs"]:
        names.append(_path_name(ref))
        signs.append(-1 if ref["sign"] == "-" else 1)
    return names, signs


def load_toy_graph(path: Path = DEFAULT_SOURCE) -> ToyGraph:
    """Parse a toy `.hymeko` file and project it into model tensors."""
    ast = hymeko.parse_hymeko_rs(path.read_text(encoding="utf-8"))
    items = _flatten_items(ast["items"])
    node_items = [
        item for item in items
        if item.get("kind") == "node" and _base_name(item).endswith("node")
    ]
    node_items.sort(key=lambda item: int(_field_value(item, "token", 0)))
    names = [str(item["name"]) for item in node_items]
    name_to_idx = {name: i for i, name in enumerate(names)}
    features = torch.tensor(
        [_field_value(item, "feature", [0.0, 0.0, 0.0, 0.0]) for item in node_items],
        dtype=torch.float32,
    )
    tier_of = torch.tensor(
        [int(_field_value(item, "tier", 0)) for item in node_items],
        dtype=torch.long,
    )

    edge_items = [item for item in items if item.get("kind") == "edge"]

    def collect(kind_suffix: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        signs = []
        for edge in edge_items:
            if not _base_name(edge).endswith(kind_suffix):
                continue
            ref_names, ref_signs = _edge_refs(edge)
            rows.append([name_to_idx[name] for name in ref_names])
            signs.append(ref_signs)
        if not rows:
            return torch.zeros((0, 3), dtype=torch.long), torch.zeros((0, 3), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(signs, dtype=torch.float32)

    cycles, cycle_signs = collect("cycle")
    walks, walk_signs_full = collect("walk")
    walk_signs = torch.prod(walk_signs_full, dim=1).long() if walks.numel() else torch.zeros((0,), dtype=torch.long)

    edge_rows = []
    edge_signs = []
    sequence_edges = []
    for edge in edge_items:
        base = _base_name(edge)
        ref_names, ref_signs = _edge_refs(edge)
        if base.endswith("link"):
            edge_rows.append([name_to_idx[name] for name in ref_names[:2]])
            edge_signs.append(int(np.prod(ref_signs[:2])))
        elif base.endswith("sequence"):
            sequence_edges.append([name_to_idx[name] for name in ref_names[:2]])
    edges = torch.tensor(edge_rows, dtype=torch.long)
    edge_signs_t = torch.tensor(edge_signs, dtype=torch.long)

    order = [0]
    for src, dst in sequence_edges:
        if src == order[-1]:
            order.append(dst)
    if len(order) != len(names):
        order = list(range(len(names)))

    return ToyGraph(
        names=names,
        features=features,
        tier_of=tier_of,
        cycles=cycles,
        cycle_signs=cycle_signs,
        walks=walks,
        walk_signs=walk_signs,
        edges=edges,
        edge_signs=edge_signs_t,
        sequence_order=order,
    )


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _sparse_walk_membership(walks: torch.Tensor, n_nodes: int) -> torch.Tensor:
    rows = walks.reshape(-1)
    cols = torch.arange(walks.shape[0]).repeat_interleave(walks.shape[1])
    indices = torch.stack([rows, cols], dim=0)
    values = torch.full((rows.shape[0],), 1.0 / walks.shape[1])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            (n_nodes, walks.shape[0]),
            check_invariants=True,
        ).coalesce()


def run_hsikan(graph: ToyGraph) -> dict[str, Any]:
    torch.manual_seed(10)
    model = MiddleHSiKAN(
        n_nodes=len(graph.names),
        d_in=graph.features.shape[1],
        d_layer=8,
        cycle_k=graph.cycles.shape[1],
    )
    out = model(graph.features, graph.cycles, graph.cycle_signs)
    return {
        "shape": list(out.shape),
        "sum": float(out.sum().item()),
        "std": float(out.std().item()),
        "n_params": _count_params(model),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def run_gomb(graph: ToyGraph) -> dict[str, Any]:
    torch.manual_seed(11)
    cfg = GombConfig(
        n_nodes=len(graph.names),
        d_embed=8,
        d_outer=4,
        M_outer=2,
        d_middle=8,
        d_core=8,
        n_tiers=int(graph.tier_of.max().item()) + 1,
        cycle_k=graph.cycles.shape[1],
    )
    model = HymeKoGomb(cfg)
    scores = model(graph.cycles, graph.cycle_signs, graph.tier_of, graph.edges)
    return {
        "shape": list(scores.shape),
        "scores": [float(x) for x in scores.detach().tolist()],
        "mean": float(scores.mean().item()),
        "n_params": model.n_params(),
        "finite": bool(torch.isfinite(scores).all().item()),
    }


def run_gomb_soma(graph: ToyGraph) -> dict[str, Any]:
    torch.manual_seed(12)
    cfg = HypergraphConvConfig(
        in_features=graph.features.shape[1],
        out_features=8,
        k_arity=graph.walks.shape[1],
    )
    model = WalkConvLayer(cfg)
    membership = _sparse_walk_membership(graph.walks, len(graph.names))
    out = model(graph.features, graph.walks, graph.walk_signs, membership)
    return {
        "shape": list(out.shape),
        "sum": float(out.sum().item()),
        "std": float(out.std().item()),
        "n_params": _count_params(model),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def run_fsr(graph: ToyGraph) -> dict[str, Any]:
    torch.manual_seed(13)
    n_blocks = 2
    model = FiberSpikeRotorMixer(
        n_blocks=n_blocks,
        max_seq_len=len(graph.sequence_order),
        gate_rank=4,
        spike_k=2,
    )
    seq = graph.features[graph.sequence_order]
    h = torch.cat([seq, seq[:, :2]], dim=1).unsqueeze(0)
    out = model(h)
    return {
        "shape": list(out.shape),
        "sum": float(out.sum().item()),
        "std": float(out.std().item()),
        "n_params": _count_params(model),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def prepare_toy_suite(path: Path = DEFAULT_SOURCE, threads: int | None = 1) -> ToyModelSuite:
    """Parse once and instantiate all toy adapters once for hot-path runs."""
    if threads is not None:
        torch.set_num_threads(threads)
    graph = load_toy_graph(path)

    torch.manual_seed(10)
    hsikan = MiddleHSiKAN(
        n_nodes=len(graph.names),
        d_in=graph.features.shape[1],
        d_layer=8,
        cycle_k=graph.cycles.shape[1],
    ).eval()
    fixed_hsikan = FixedTopologyHSIKAN(hsikan, graph).eval()

    torch.manual_seed(11)
    gomb = HymeKoGomb(
        GombConfig(
            n_nodes=len(graph.names),
            d_embed=8,
            d_outer=4,
            M_outer=2,
            d_middle=8,
            d_core=8,
            n_tiers=int(graph.tier_of.max().item()) + 1,
            cycle_k=graph.cycles.shape[1],
        )
    ).eval()
    fixed_gomb = FixedTopologyGomb(gomb, graph).eval()

    torch.manual_seed(14)
    sa_hsikan = FixedHolonomyHSIKAN(graph, hidden=8, walk_len=2).eval()

    torch.manual_seed(12)
    gomb_soma = WalkConvLayer(
        HypergraphConvConfig(
            in_features=graph.features.shape[1],
            out_features=8,
            k_arity=graph.walks.shape[1],
        )
    ).eval()
    walk_membership = _sparse_walk_membership(graph.walks, len(graph.names))

    torch.manual_seed(13)
    fsr = FiberSpikeRotorMixer(
        n_blocks=2,
        max_seq_len=len(graph.sequence_order),
        gate_rank=4,
        spike_k=2,
    ).eval()
    seq = graph.features[graph.sequence_order]
    fsr_input = torch.cat([seq, seq[:, :2]], dim=1).unsqueeze(0)

    return ToyModelSuite(
        source=path,
        graph=graph,
        hsikan=hsikan,
        fixed_hsikan=fixed_hsikan,
        gomb=gomb,
        fixed_gomb=fixed_gomb,
        sa_hsikan=sa_hsikan,
        gomb_soma=gomb_soma,
        fsr=fsr,
        walk_membership=walk_membership,
        fsr_input=fsr_input,
    )


def trace_toy_suite(suite: ToyModelSuite) -> ToyModelSuite:
    """Trace cached modules for this fixed toy graph shape."""
    graph = suite.graph
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        hsikan = torch.jit.trace(
            suite.hsikan,
            (graph.features, graph.cycles, graph.cycle_signs),
            check_trace=False,
        )
        fixed_hsikan = torch.jit.trace(
            suite.fixed_hsikan,
            (graph.features,),
            check_trace=False,
        )
        gomb = torch.jit.trace(
            suite.gomb,
            (graph.cycles, graph.cycle_signs, graph.tier_of, graph.edges),
            check_trace=False,
        )
        fixed_gomb = torch.jit.trace(suite.fixed_gomb, (), check_trace=False)
        sa_hsikan = torch.jit.trace(
            suite.sa_hsikan,
            (graph.features,),
            check_trace=False,
        )
        gomb_soma = torch.jit.trace(
            suite.gomb_soma,
            (graph.features, graph.walks, graph.walk_signs, suite.walk_membership),
            check_trace=False,
        )
        fsr = torch.jit.trace(
            suite.fsr,
            (suite.fsr_input,),
            check_trace=False,
        )
    return ToyModelSuite(
        source=suite.source,
        graph=graph,
        hsikan=hsikan,
        fixed_hsikan=fixed_hsikan,
        gomb=gomb,
        fixed_gomb=fixed_gomb,
        sa_hsikan=sa_hsikan,
        gomb_soma=gomb_soma,
        fsr=fsr,
        walk_membership=suite.walk_membership,
        fsr_input=suite.fsr_input,
    )


def _summary_tensor(out: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(out.shape),
        "sum": float(out.sum().item()),
        "std": float(out.std().item()),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def _signed_degree(graph: ToyGraph) -> torch.Tensor:
    deg = torch.zeros(len(graph.names), dtype=torch.float32)
    for (src, dst), sign in zip(graph.edges.tolist(), graph.edge_signs.tolist()):
        deg[src] += float(sign)
        deg[dst] += float(sign)
    return deg


def _regression_target(graph: ToyGraph) -> torch.Tensor:
    weights = torch.tensor([0.45, -0.25, 0.35, 0.15], dtype=torch.float32)
    return graph.features @ weights + 0.20 * graph.tier_of.float() + 0.10 * _signed_degree(graph)


def _with_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=1)


def _ridge_weights(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    xb = _with_bias(x)
    eye = torch.eye(xb.shape[1], dtype=xb.dtype, device=xb.device)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(xb.T @ xb + ridge * eye, xb.T @ y)


def _ridge_predict(x_train: torch.Tensor, y_train: torch.Tensor, x_eval: torch.Tensor) -> torch.Tensor:
    return _with_bias(x_eval) @ _ridge_weights(x_train, y_train)


def _classification_metrics(x: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    y = torch.nn.functional.one_hot(labels, num_classes=int(labels.max().item()) + 1).float()
    logits = _ridge_predict(x, y, x)
    pred = logits.argmax(dim=1)
    loo_pred = []
    for i in range(x.shape[0]):
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        mask[i] = False
        loo_logits = _ridge_predict(x[mask], y[mask], x[i : i + 1])
        loo_pred.append(int(loo_logits.argmax(dim=1).item()))
    loo = torch.tensor(loo_pred, dtype=torch.long)
    return {
        "labels": labels.tolist(),
        "pred": pred.tolist(),
        "loo_pred": loo.tolist(),
        "train_acc": float((pred == labels).float().mean().item()),
        "loo_acc": float((loo == labels).float().mean().item()),
    }


def _classification_fit_eval(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> dict[str, Any]:
    y = torch.nn.functional.one_hot(y_train, num_classes=int(y_train.max().item()) + 1).float()
    logits = _ridge_predict(x_train, y, x_eval)
    pred = logits.argmax(dim=1)
    return {
        "acc": float((pred == y_eval).float().mean().item()),
        "pred": pred.tolist(),
        "entropy_mean": float(_softmax_entropy(logits).mean().item()),
    }


def _regression_metrics(x: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred = _ridge_predict(x, target.unsqueeze(1), x).squeeze(1)
    loo_pred = []
    for i in range(x.shape[0]):
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        mask[i] = False
        loo_pred.append(float(_ridge_predict(x[mask], target[mask].unsqueeze(1), x[i : i + 1]).item()))
    loo = torch.tensor(loo_pred, dtype=target.dtype)
    return {
        "target": [float(v) for v in target.tolist()],
        "pred": [float(v) for v in pred.tolist()],
        "loo_pred": [float(v) for v in loo.tolist()],
        "train_rmse": float(torch.sqrt(torch.mean((pred - target) ** 2)).item()),
        "loo_rmse": float(torch.sqrt(torch.mean((loo - target) ** 2)).item()),
    }


def _regression_fit_eval(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> dict[str, Any]:
    pred = _ridge_predict(x_train, y_train.unsqueeze(1), x_eval).squeeze(1)
    return {
        "rmse": float(torch.sqrt(torch.mean((pred - y_eval) ** 2)).item()),
        "pred": [float(v) for v in pred.tolist()],
    }


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    denom = torch.log(torch.tensor(float(logits.shape[-1]), dtype=logits.dtype, device=logits.device))
    return -(probs * log_probs).sum(dim=-1) / denom.clamp_min(1e-6)


def _entropy_feedback_features(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.nn.functional.one_hot(y_train, num_classes=int(y_train.max().item()) + 1).float()
    train_logits = _ridge_predict(x_train, y, x_train)
    eval_logits = _ridge_predict(x_train, y, x_eval)
    train_entropy = _softmax_entropy(train_logits).unsqueeze(1)
    eval_entropy = _softmax_entropy(eval_logits).unsqueeze(1)
    return torch.cat([x_train, train_entropy], dim=1), torch.cat([x_eval, eval_entropy], dim=1)


def _global_pool(nodes: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        nodes.mean(dim=1),
        nodes.std(dim=1, unbiased=False),
        nodes.amax(dim=1),
    ], dim=1)


def _node_representations(suite: ToyModelSuite) -> dict[str, torch.Tensor]:
    graph = suite.graph
    with torch.inference_mode():
        fsr_nodes = suite.fsr(suite.fsr_input).squeeze(0)
        return {
            "raw_features": graph.features,
            "hsikan": suite.hsikan(graph.features, graph.cycles, graph.cycle_signs),
            "fixed_hsikan": suite.fixed_hsikan(graph.features),
            "sa_hsikan": suite.sa_hsikan(graph.features),
            "gomb_soma": suite.gomb_soma(
                graph.features,
                graph.walks,
                graph.walk_signs,
                suite.walk_membership,
            ),
            "fsr": fsr_nodes,
    }


def _general_problem_features(graph: ToyGraph, n_samples: int) -> torch.Tensor:
    base = graph.features
    samples = []
    for i in range(n_samples):
        phase = float(i + 1)
        scale = 0.75 + 0.035 * phase
        tier_bias = (graph.tier_of.float().unsqueeze(1) - 1.0) * (0.03 * np.sin(phase))
        wave = torch.sin(base * (0.7 + 0.11 * phase) + phase * 0.17)
        cross = torch.cos(torch.flip(base, dims=[1]) * (0.9 + 0.03 * phase))
        samples.append(scale * base + 0.11 * wave + 0.07 * cross + tier_bias)
    return torch.stack(samples, dim=0)


def _general_problem_targets(graph: ToyGraph, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    signed_degree = _signed_degree(graph)
    tier = graph.tier_of.float()
    latent = (
        0.40 * xs[:, :, 0].mean(dim=1)
        - 0.25 * xs[:, :, 1].mean(dim=1)
        + 0.35 * (xs[:, :, 2] * (tier + 1.0)).mean(dim=1)
        + 0.18 * (xs[:, :, 3] * signed_degree).mean(dim=1)
        + 0.05 * xs[:, :, 0].amax(dim=1)
    )
    q1, q2 = torch.quantile(latent, torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
    labels = torch.bucketize(latent, torch.tensor([q1, q2]))
    return labels.long(), latent.float()


def _encode_general_representations(suite: ToyModelSuite, xs: torch.Tensor) -> dict[str, torch.Tensor]:
    graph = suite.graph
    reps: dict[str, list[torch.Tensor]] = {
        "raw_features": [],
        "hsikan": [],
        "fixed_hsikan": [],
        "sa_hsikan": [],
        "gomb_soma": [],
        "fsr": [],
    }
    with torch.inference_mode():
        seq_batch = xs[:, graph.sequence_order, :]
        fsr_batch_input = torch.cat([seq_batch, seq_batch[:, :, :2]], dim=2)
        fsr_batch = suite.fsr(fsr_batch_input)
        for features in xs:
            reps["raw_features"].append(features)
            reps["hsikan"].append(suite.hsikan(features, graph.cycles, graph.cycle_signs))
            reps["fixed_hsikan"].append(suite.fixed_hsikan(features))
            reps["sa_hsikan"].append(suite.sa_hsikan(features))
            reps["gomb_soma"].append(
                suite.gomb_soma(
                    features,
                    graph.walks,
                    graph.walk_signs,
                    suite.walk_membership,
                )
            )
        reps["fsr"] = [row for row in fsr_batch]
    return {name: torch.stack(rows, dim=0) for name, rows in reps.items()}


def _forward_dataset_timing_us(
    suite: ToyModelSuite,
    xs: torch.Tensor,
    repeats: int,
) -> dict[str, dict[str, float]]:
    graph = suite.graph
    funcs: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "raw_features": lambda features: features,
        "hsikan": lambda features: suite.hsikan(features, graph.cycles, graph.cycle_signs),
        "fixed_hsikan": lambda features: suite.fixed_hsikan(features),
        "sa_hsikan": lambda features: suite.sa_hsikan(features),
        "gomb_soma": lambda features: suite.gomb_soma(
            features,
            graph.walks,
            graph.walk_signs,
            suite.walk_membership,
        ),
        "fsr": lambda features: suite.fsr(
            torch.cat([
                features[graph.sequence_order],
                features[graph.sequence_order][:, :2],
            ], dim=1).unsqueeze(0)
        ).squeeze(0),
    }
    out = {}
    with torch.inference_mode():
        for name, fn in funcs.items():
            for features in xs[: min(4, xs.shape[0])]:
                _global_pool(fn(features).unsqueeze(0))
            times = []
            for i in range(repeats):
                features = xs[i % xs.shape[0]]
                start = time.perf_counter_ns()
                _global_pool(fn(features).unsqueeze(0))
                times.append((time.perf_counter_ns() - start) / 1000.0)
            out[name] = {
                "mean_us": statistics.mean(times),
                "median_us": statistics.median(times),
                "min_us": min(times),
                "max_us": max(times),
            }
        seq_batch = xs[:, graph.sequence_order, :]
        fsr_batch_input = torch.cat([seq_batch, seq_batch[:, :, :2]], dim=2)
        for _ in range(10):
            _global_pool(suite.fsr(fsr_batch_input))
        batch_times = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            _global_pool(suite.fsr(fsr_batch_input))
            batch_times.append((time.perf_counter_ns() - start) / 1000.0 / xs.shape[0])
        out["fsr_batched_amortized"] = {
            "mean_us": statistics.mean(batch_times),
            "median_us": statistics.median(batch_times),
            "min_us": min(batch_times),
            "max_us": max(batch_times),
        }
    return out


def _stratified_indices(labels: torch.Tensor, train_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels, sorted=True)
    n_total = int(labels.shape[0])
    train_parts = []
    test_parts = []
    remaining_train = train_samples
    remaining_classes = int(classes.numel())
    for cls in classes.tolist():
        idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        quota = int(round(train_samples * idx.numel() / n_total))
        quota = max(1, min(int(idx.numel()) - 1, quota))
        if remaining_classes == 1:
            quota = remaining_train
        quota = max(1, min(int(idx.numel()) - 1, quota))
        train_parts.append(idx[:quota])
        test_parts.append(idx[quota:])
        remaining_train -= quota
        remaining_classes -= 1
    train = torch.cat(train_parts)
    test = torch.cat(test_parts)
    return train, test


def _structure_search_levers() -> dict[str, Any]:
    """P-graph style structure-search levers for speed, selection, and generation."""
    return {
        "msg": {
            "name": "Maximal Structure Generation",
            "uses": [
                "generate the maximal admissible structural candidate set",
                "precompute all feasible feature/topology branches before learning",
                "define the superset for alpha-mixing and branch pruning",
            ],
        },
        "ssg": {
            "name": "Solution Structure Generation",
            "uses": [
                "enumerate feasible subsets of MSG candidates",
                "act as discrete feature selection over structural generators",
                "emit alternative HSIKAN/Gomb/FSR wiring candidates for small searches",
            ],
        },
        "abb": {
            "name": "Accelerated Branch-and-Bound",
            "uses": [
                "prune SSG search when candidate sets are large",
                "optimize multi-objective structure scores such as accuracy, latency, parameters, and entropy",
                "select inference-time branches when learned alpha weights collapse onto a sparse support",
            ],
        },
        "targets": [
            "cycle/walk arity selection",
            "structural feature enrichment selection",
            "pgraph-generated topology branches",
            "alpha-mixing branch pruning",
            "cached sparse-incidence generation for fixed-topology speedups",
        ],
        "existing_repo_hooks": [
            "hymeko_neuro/experiments/hsikan_pgraph_mapping.py",
            "hymeko_neuro/experiments/gomb_pgraph_mapping.py",
            "hymeko_pgraph::msg / ssg / abb_solve via hymeko_pgraph_dump",
        ],
    }


def _closed_form_feedback_summary(models: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, row in models.items():
        cls_plain = row["classification"]["plain"]["acc"]
        cls_fb = row["classification"]["entropy_feedback"]["acc"]
        reg_plain = row["regression"]["plain"]["rmse"]
        reg_fb = row["regression"]["entropy_feedback"]["rmse"]
        rows[name] = {
            "classification_acc_delta": float(cls_fb - cls_plain),
            "regression_rmse_delta": float(reg_fb - reg_plain),
            "entropy_mean_delta": float(
                row["classification"]["entropy_feedback"]["entropy_mean"]
                - row["classification"]["plain"]["entropy_mean"]
            ),
        }
    return {
        "mechanism": "two closed-form ridge solves; append normalized predictive entropy from pass 1 to pooled features for pass 2",
        "training_loop": "none",
        "ridge_equation": "W = (X^T X + lambda I)^-1 X^T Y",
        "per_model_delta": rows,
    }


def _fsr_clifford_diagnostics(
    suite: ToyModelSuite,
    xs: torch.Tensor,
    timing: dict[str, dict[str, float]],
) -> dict[str, Any]:
    graph = suite.graph
    seq_batch = xs[:, graph.sequence_order, :]
    fsr_input = torch.cat([seq_batch, seq_batch[:, :, :2]], dim=2)
    with torch.inference_mode():
        out = suite.fsr(fsr_input)
        t = fsr_input.shape[1]
        n_blocks = suite.fsr.n_blocks
        rotor = cayley_to_unit_quat(suite.fsr.offset_bivec[:t])
        signs = torch.tanh(suite.fsr.offset_sign[:t])
        q_norm = rotor.norm(dim=-1)
        input_block_norm = fsr_input.view(fsr_input.shape[0], t, n_blocks, 3).norm(dim=-1)
        output_block_norm = out.view(out.shape[0], t, n_blocks, 3).norm(dim=-1)
    batch = int(fsr_input.shape[0])
    dense_slots = batch * t * t
    causal_slots = batch * t * (t + 1) // 2
    sparse_k = min(int(suite.fsr.spike_k or t), t)
    sparse_slots = batch * t * sparse_k
    return {
        "algebra": "unit quaternion rotor, even Clifford subalgebra Cl(0,2)+",
        "hidden_shape": list(fsr_input.shape),
        "n_blocks": int(n_blocks),
        "relative_offsets": int(t),
        "spike_k": int(sparse_k),
        "transport_slots": {
            "dense_all_pairs": int(dense_slots),
            "causal_pairs": int(causal_slots),
            "sparse_topk_slots": int(sparse_slots),
            "sparse_vs_dense_ratio": float(sparse_slots / dense_slots),
            "sparse_vs_causal_ratio": float(sparse_slots / causal_slots),
        },
        "rotor": {
            "unit_norm_max_abs_error": float((q_norm - 1.0).abs().max().item()),
            "bivector_norm_max": float(suite.fsr.offset_bivec[:t].norm(dim=-1).max().item()),
            "bivector_norm_mean": float(suite.fsr.offset_bivec[:t].norm(dim=-1).mean().item()),
        },
        "sign": {
            "soft_sign_min": float(signs.min().item()),
            "soft_sign_max": float(signs.max().item()),
            "soft_sign_mean": float(signs.mean().item()),
        },
        "energy": {
            "input_block_norm_mean": float(input_block_norm.mean().item()),
            "output_block_norm_mean": float(output_block_norm.mean().item()),
            "output_block_norm_std": float(output_block_norm.std().item()),
        },
        "timing_us": {
            "per_graph_forward_pool_median": timing["fsr"]["median_us"],
            "batched_amortized_forward_pool_median": timing["fsr_batched_amortized"]["median_us"],
        },
    }


def run_general_problem_suite(
    path: Path = DEFAULT_SOURCE,
    n_samples: int = 36,
    train_samples: int = 24,
    timing_repeats: int = 300,
    trace: bool = False,
) -> dict[str, Any]:
    """Graph-level classification/regression with global pooling and entropy feedback."""
    torch.manual_seed(20)
    np.random.seed(20)
    diagnostic_suite = prepare_toy_suite(path, threads=1)
    suite = diagnostic_suite
    if trace:
        suite = trace_toy_suite(suite)
    graph = suite.graph
    xs = _general_problem_features(graph, n_samples)
    labels, target = _general_problem_targets(graph, xs)
    reps = _encode_general_representations(suite, xs)
    timing = _forward_dataset_timing_us(suite, xs, timing_repeats)

    train, test = _stratified_indices(labels, train_samples)
    models = {}
    for name, nodes in reps.items():
        pooled = _global_pool(nodes)
        x_train = pooled[train].float()
        x_test = pooled[test].float()
        y_train = labels[train]
        y_test = labels[test]
        r_train = target[train]
        r_test = target[test]

        cls_plain = _classification_fit_eval(x_train, y_train, x_test, y_test)
        reg_plain = _regression_fit_eval(x_train, r_train, x_test, r_test)
        x_train_fb, x_test_fb = _entropy_feedback_features(x_train, y_train, x_test)
        cls_fb = _classification_fit_eval(x_train_fb, y_train, x_test_fb, y_test)
        reg_fb = _regression_fit_eval(x_train_fb, r_train, x_test_fb, r_test)
        models[name] = {
            "pooled_dim": int(pooled.shape[1]),
            "classification": {
                "plain": cls_plain,
                "entropy_feedback": cls_fb,
            },
            "regression": {
                "plain": reg_plain,
                "entropy_feedback": reg_fb,
            },
            "forward_pool_us": timing[name],
        }

    feedback = _closed_form_feedback_summary(models)
    fsr_clifford = _fsr_clifford_diagnostics(diagnostic_suite, xs, timing)
    return {
        "source": str(path),
        "n_samples": n_samples,
        "train_samples": int(train.numel()),
        "test_samples": int(test.numel()),
        "train_indices": train.tolist(),
        "test_indices": test.tolist(),
        "global_pool": "concat(mean, std, max) over node representations",
        "entropy_feedback": "append normalized predictive entropy from first-pass classifier to pooled graph features",
        "structure_search": _structure_search_levers(),
        "mode": "traced" if trace else "eager",
        "classification_labels": labels.tolist(),
        "regression_target": [float(v) for v in target.tolist()],
        "optimized_forward_pool_us": {
            "fsr_batched_amortized": timing["fsr_batched_amortized"],
        },
        "closed_form_entropy_feedback": feedback,
        "fsr_clifford": fsr_clifford,
        "models": models,
    }


def run_supervised_toy_suite(path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Run deterministic node classification/regression probes on toy representations."""
    suite = prepare_toy_suite(path, threads=1)
    graph = suite.graph
    reps = _node_representations(suite)
    cls = {
        name: _classification_metrics(rep.float(), graph.tier_of)
        for name, rep in reps.items()
    }
    target = _regression_target(graph)
    reg = {
        name: _regression_metrics(rep.float(), target)
        for name, rep in reps.items()
    }
    return {
        "source": str(path),
        "n_nodes": len(graph.names),
        "classification": {
            "task": "predict parsed node tier {0,1,2}",
            "models": cls,
        },
        "regression": {
            "task": "predict synthetic signed-topology scalar from parsed features, tier, and link signs",
            "models": reg,
        },
    }


def run_prepared_toy_suite(suite: ToyModelSuite) -> dict[str, Any]:
    """Run cached modules once; this is the toy-model hot path."""
    graph = suite.graph
    with torch.inference_mode():
        hsikan = suite.hsikan(graph.features, graph.cycles, graph.cycle_signs)
        fixed_hsikan = suite.fixed_hsikan(graph.features)
        sa_hsikan = suite.sa_hsikan(graph.features)
        gomb = suite.gomb(graph.cycles, graph.cycle_signs, graph.tier_of, graph.edges)
        fixed_gomb = suite.fixed_gomb()
        gomb_soma = suite.gomb_soma(
            graph.features,
            graph.walks,
            graph.walk_signs,
            suite.walk_membership,
        )
        fsr = suite.fsr(suite.fsr_input)

    hsikan_result = _summary_tensor(hsikan)
    hsikan_result["n_params"] = _count_params(suite.hsikan)
    fixed_hsikan_result = _summary_tensor(fixed_hsikan)
    fixed_hsikan_result["n_params"] = _count_params(suite.hsikan)
    fixed_hsikan_result["max_abs_delta_vs_hsikan"] = float(
        (fixed_hsikan - hsikan).abs().max().item()
    )
    sa_hsikan_result = _summary_tensor(sa_hsikan)
    sa_hsikan_result["n_params"] = _count_params(suite.sa_hsikan)
    gomb_result = {
        "shape": list(gomb.shape),
        "scores": [float(x) for x in gomb.detach().tolist()],
        "mean": float(gomb.mean().item()),
        "n_params": suite.gomb.n_params(),
        "finite": bool(torch.isfinite(gomb).all().item()),
    }
    fixed_gomb_result = {
        "shape": list(fixed_gomb.shape),
        "scores": [float(x) for x in fixed_gomb.detach().tolist()],
        "mean": float(fixed_gomb.mean().item()),
        "max_abs_delta_vs_gomb": float((fixed_gomb - gomb).abs().max().item()),
        "n_params": suite.gomb.n_params() if hasattr(suite.gomb, "n_params") else 0,
        "finite": bool(torch.isfinite(fixed_gomb).all().item()),
    }
    gomb_soma_result = _summary_tensor(gomb_soma)
    gomb_soma_result["n_params"] = _count_params(suite.gomb_soma)
    fsr_result = _summary_tensor(fsr)
    fsr_result["n_params"] = _count_params(suite.fsr)
    return {
        "source": str(suite.source),
        "n_nodes": len(graph.names),
        "n_cycles": int(graph.cycles.shape[0]),
        "n_walks": int(graph.walks.shape[0]),
        "n_edges": int(graph.edges.shape[0]),
        "sequence_len": len(graph.sequence_order),
        "hsikan": hsikan_result,
        "fixed_hsikan": fixed_hsikan_result,
        "sa_hsikan": sa_hsikan_result,
        "gomb": gomb_result,
        "fixed_gomb": fixed_gomb_result,
        "gomb_soma": gomb_soma_result,
        "fsr": fsr_result,
    }


def _time_us(fn: Callable[[], Any], repeats: int) -> dict[str, float]:
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn()
            times.append((time.perf_counter_ns() - start) / 1000.0)
    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_prepared_suite(
    path: Path = DEFAULT_SOURCE,
    repeats: int = 1000,
    warmup: int = 50,
    threads: int = 1,
    trace: bool = False,
) -> dict[str, Any]:
    """Benchmark cached CPU forwards in microseconds."""
    suite = prepare_toy_suite(path, threads=threads)
    if trace:
        suite = trace_toy_suite(suite)
    graph = suite.graph
    cases = {
        "hsikan": lambda: suite.hsikan(graph.features, graph.cycles, graph.cycle_signs),
        "fixed_hsikan": lambda: suite.fixed_hsikan(graph.features),
        "sa_hsikan": lambda: suite.sa_hsikan(graph.features),
        "gomb": lambda: suite.gomb(
            graph.cycles,
            graph.cycle_signs,
            graph.tier_of,
            graph.edges,
        ),
        "fixed_gomb": lambda: suite.fixed_gomb(),
        "gomb_soma": lambda: suite.gomb_soma(
            graph.features,
            graph.walks,
            graph.walk_signs,
            suite.walk_membership,
        ),
        "fsr": lambda: suite.fsr(suite.fsr_input),
    }

    with torch.inference_mode():
        for _ in range(warmup):
            for fn in cases.values():
                fn()

    def all_forwards() -> None:
        for fn in cases.values():
            fn()

    return {
        "source": str(path),
        "threads": threads,
        "warmup": warmup,
        "repeats": repeats,
        "mode": "traced" if trace else "eager",
        "hot_all_forwards": _time_us(all_forwards, repeats),
        "per_adapter": {name: _time_us(fn, repeats) for name, fn in cases.items()},
    }


def run_toy_suite(path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Run all four toy model adapters and return JSON-serializable metrics."""
    torch.manual_seed(0)
    np.random.seed(0)
    suite = prepare_toy_suite(path, threads=None)
    return run_prepared_toy_suite(suite)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--general-problem", action="store_true")
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--timing-repeats", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.general_problem:
        result = run_general_problem_suite(
            args.source,
            n_samples=args.samples,
            train_samples=args.train_samples,
            timing_repeats=args.timing_repeats,
            trace=args.trace,
        )
    elif args.supervised:
        result = run_supervised_toy_suite(args.source)
    elif args.benchmark:
        result = benchmark_prepared_suite(
            args.source,
            repeats=args.repeats,
            warmup=args.warmup,
            threads=args.threads,
            trace=args.trace,
        )
    else:
        result = run_toy_suite(args.source)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
