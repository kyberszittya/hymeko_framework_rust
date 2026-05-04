"""Phase 11 — Graph-level HSiKAN tasks on synthetic kinematic mechanisms.

Tasks
-----
  T1. Mechanism family classification  (4-class softmax)
  T2. Mobility (DOF) regression        (continuous scalar)
  T3. End-effector position regression (continuous XYZ — STUB; needs forward-kin sim)
  T4. Loop-closure solvability         (binary, given target pose — STUB; needs IK sim)
  T5. Failure-mode prediction          (per-edge, given load — STUB; needs dynamics sim)

T1 & T2 work on the bare graph + cycles. T3-T5 need continuous context
beyond the graph; we expose the architecture hook (graph + extra
features → head) but defer experiments to data adapters.

Variable-arity handling: each mechanism has ONE dominant cycle arity:
  - four_bar:    k=4 only
  - stewart:     k=6 only
  - delta_3rrr:  k=6 only
  - serial_N:    no cycles (filtered out)
We train ONE model per arity (per dominant cycle length); each model
sees only the mechanisms whose cycles match.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .datasets import SignedGraph
from .hyperedges import construct
from .kinematic_fixtures import _serial_arm_urdf, write_fixture
from .kinematic_graph import urdf_to_signed_graph
from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples)
from .n_tuples import construct_k
from .signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)


FAMILY_LABEL_BY_NAME = {"four_bar": 0, "stewart": 1,
                          "delta_3rrr": 2, "serial": 3}


@dataclass
class MechInst:
    g: SignedGraph
    family: str
    family_label: int
    dof: int
    name: str


def build_random_mechanism(rng: random.Random) -> MechInst:
    family = rng.choices(
        ["four_bar", "stewart", "delta_3rrr", "serial"],
        weights=[3, 2, 2, 4],
    )[0]
    if family == "serial":
        n_links = rng.randint(2, 9)
        import tempfile
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_serial_{n_links}.urdf", delete=False)
        f.write(_serial_arm_urdf(n_links)); f.close()
        path = Path(f.name); dof = n_links
    else:
        path = write_fixture(family)
        dof = {"four_bar": 1, "stewart": 6, "delta_3rrr": 3}[family]
    try:
        g, _, _ = urdf_to_signed_graph(path)
    finally:
        path.unlink(missing_ok=True)
    return MechInst(g=g, family=family,
                     family_label=FAMILY_LABEL_BY_NAME[family],
                     dof=dof,
                     name=f"{family}_{rng.randint(0, 1<<30):x}")


def detect_dominant_arity(g: SignedGraph, candidates=(3, 4, 5, 6, 7, 8)) -> int | None:
    """Return the smallest arity for which ≥1 cycle exists."""
    for k in candidates:
        try:
            if k == 3:
                cycles = construct(g)
            else:
                cycles = construct_k(g, k=k, max_cycles=10)
            if len(cycles) > 0:
                return k
        except Exception:
            continue
    return None


def _build_per_arity_input(g: SignedGraph, arity: int,
                             max_per: int, device, seed: int,
                             n_nodes_pad: int | None = None):
    """``n_nodes_pad``: if set, pad M_vt to this many rows so a single
    HSiKAN model with ``n_nodes_max`` embedding can consume mechanisms
    of varying size. Vertex IDs in triad_v map directly into
    ``[0, n_nodes_pad)`` (this requires that mechanism vertex IDs
    don't exceed n_nodes_pad)."""
    if arity == 3:
        t_k = construct(g)
    else:
        t_k = construct_k(g, k=arity, max_cycles=max_per, seed=seed)
    if not t_k:
        return None
    if len(t_k) > max_per:
        t_k = subsample_tuples(t_k, max_per, seed=seed)
    triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
    triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
    triad_v = torch.from_numpy(triad_v_np).to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
    n_v = n_nodes_pad if n_nodes_pad is not None else g.n_nodes
    M_vt = build_vertex_triad_incidence(triad_v_np, n_v, device, mode="sum")
    edge_to_tuples = build_edge_to_tuples(t_k)
    e_self = g.edges
    rows, cols, vals = [], [], []
    for ei, e in enumerate(e_self):
        key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
        ids = edge_to_tuples.get(key, [])
        if not ids: continue
        w = 1.0 / float(len(ids))
        for t in ids:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    if rows:
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e = torch.sparse_coo_tensor(idx, v, (e_self.shape[0], len(t_k))).coalesce()
    else:
        M_e = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
            (e_self.shape[0], len(t_k))).to(device)
    return [(triad_v, triad_sigma, M_vt, M_e)]


class GraphLevelHSiKAN(nn.Module):
    def __init__(self, n_nodes_max: int, arity: int,
                  hidden: int = 16, n_layers: int = 2, grid: int = 5,
                  n_classes: int = 4):
        super().__init__()
        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=n_nodes_max, n_layers=n_layers,
                hidden_dim=hidden, grid=grid, k=3,
                spline_kinds=["catmull_rom"] * n_layers,
                init_scale=0.05, pool_mode="sum", jk_mode="concat",
                layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none", use_residual=True,
            ),
            arities=(arity,), init_arity_logits=(0.0,),
        )
        self.backbone = MixedAritySignedKAN(cfg)
        d_jk = hidden * n_layers
        self.cls_head = nn.Linear(d_jk, n_classes)
        self.reg_head = nn.Linear(d_jk, 1)

    def forward(self, per_arity_inputs):
        graph_emb = self.backbone.encode_graph(per_arity_inputs)
        return self.cls_head(graph_emb), self.reg_head(graph_emb).squeeze(-1)


def train_per_arity(insts: list[MechInst], arity: int, n_epochs: int,
                      device, seed: int, hidden: int = 16):
    """Train a single-arity GraphLevelHSiKAN on the mechanisms whose
    dominant arity matches ``arity``. Returns (model, train_inputs)."""
    # First pass: figure out n_nodes_max so we can build M_vt with
    # consistent padding across all mechanisms.
    candidates = []
    for inst in insts:
        if detect_dominant_arity(inst.g) == arity:
            candidates.append(inst)
    if not candidates:
        return None, []
    n_nodes_max = max(c.g.n_nodes for c in candidates)
    matched = []
    for inst in candidates:
        inputs = _build_per_arity_input(inst.g, arity, 30_000, device, seed,
                                          n_nodes_pad=n_nodes_max)
        if inputs is None: continue
        matched.append((inst, inputs))
    if not matched:
        return None, []
    model = GraphLevelHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                hidden=hidden, n_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_epochs))
    y_cls = torch.tensor([m[0].family_label for m in matched],
                          dtype=torch.long, device=device)
    y_reg = torch.tensor([float(m[0].dof) for m in matched],
                          dtype=torch.float32, device=device)
    inputs_list = [m[1] for m in matched]
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(inputs_list))
        loss_total = 0.0
        for i in perm:
            cls_logits, reg_pred = model(inputs_list[i])
            l_cls = F.cross_entropy(cls_logits.unsqueeze(0), y_cls[i:i+1])
            l_reg = F.mse_loss(reg_pred, y_reg[i])
            loss = l_cls + 0.05 * l_reg
            opt.zero_grad(); loss.backward(); opt.step()
            loss_total += loss.item()
        sched.step()
    return model, matched


def evaluate_per_arity(model, insts: list[MechInst], arity: int,
                          device, seed: int):
    if model is None:
        return None
    n_nodes_pad = model.backbone.cfg.base.n_nodes
    matched = []
    for inst in insts:
        dom = detect_dominant_arity(inst.g)
        if dom != arity: continue
        if inst.g.n_nodes > n_nodes_pad:
            continue   # test mechanism larger than any training mech
        inputs = _build_per_arity_input(inst.g, arity, 30_000, device, seed,
                                          n_nodes_pad=n_nodes_pad)
        if inputs is None: continue
        matched.append((inst, inputs))
    if not matched:
        return None
    model.eval()
    cls_preds, reg_preds, y_true_cls, y_true_reg = [], [], [], []
    with torch.no_grad():
        for inst, inputs in matched:
            cls_logits, reg_pred = model(inputs)
            cls_preds.append(int(cls_logits.argmax().item()))
            reg_preds.append(float(reg_pred.item()))
            y_true_cls.append(inst.family_label)
            y_true_reg.append(float(inst.dof))
    return dict(
        n=len(matched),
        acc=accuracy_score(y_true_cls, cls_preds),
        f1m=f1_score(y_true_cls, cls_preds, average="macro", zero_division=0),
        dof_mae=float(np.mean(np.abs(np.array(reg_preds) - np.array(y_true_reg)))),
        alpha=model.backbone.alpha().detach().cpu().tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--n_epochs", type=int, default=80)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    print("=== Phase 11 — Graph-level HSiKAN on synthetic kinematic mechanisms ===",
          flush=True)
    print(f"Tasks: family classification (4-class) + DOF regression",
          flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    aggregate = []
    for seed in args.seeds:
        rng = random.Random(seed)
        torch.manual_seed(seed); np.random.seed(seed)
        train = [build_random_mechanism(rng) for _ in range(args.n_train)]
        test = [build_random_mechanism(rng) for _ in range(args.n_test)]
        family_dist = defaultdict(int)
        for m in train: family_dist[m.family] += 1
        # Dominant-arity distribution
        arity_dist = defaultdict(int)
        for m in train:
            dom = detect_dominant_arity(m.g)
            arity_dist[dom] += 1
        if seed == args.seeds[0]:
            print(f"\nseed={seed} train family: {dict(family_dist)}", flush=True)
            print(f"           train arity:  {dict(arity_dist)}", flush=True)

        # Train one model per arity that has any cycles.
        results_per_arity = {}
        for arity in (4, 6):
            t0 = time.time()
            model, matched = train_per_arity(train, arity, args.n_epochs,
                                                device, seed)
            if model is None:
                results_per_arity[arity] = None
                continue
            res = evaluate_per_arity(model, test, arity, device, seed)
            if res is None:
                results_per_arity[arity] = None
                continue
            res["train_n"] = len(matched)
            res["time"] = time.time() - t0
            results_per_arity[arity] = res
            print(f"  seed={seed} arity={arity}  "
                  f"n_train={len(matched)} n_test={res['n']}  "
                  f"acc={res['acc']:.3f}  f1m={res['f1m']:.3f}  "
                  f"dof_mae={res['dof_mae']:.2f}  "
                  f"{res['time']:.0f}s", flush=True)
        aggregate.append(results_per_arity)

    print("\n=== Aggregated (median across seeds) ===", flush=True)
    for arity in (4, 6):
        accs, f1ms, maes = [], [], []
        for r in aggregate:
            if r.get(arity) is None: continue
            accs.append(r[arity]["acc"])
            f1ms.append(r[arity]["f1m"])
            maes.append(r[arity]["dof_mae"])
        if not accs: continue
        print(f"arity={arity}  acc_med={statistics.median(accs):.3f}  "
              f"f1m_med={statistics.median(f1ms):.3f}  "
              f"dof_mae_med={statistics.median(maes):.2f}  n_seeds={len(accs)}",
              flush=True)


if __name__ == "__main__":
    main()
