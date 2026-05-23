"""Phase 12 — Position regression on synthetic kinematic mechanisms.

Task: predict per-link 3D position (xyz) from the kinematic graph
structure. Uses HSiKAN's cycle-pool features pooled to vertices via
M_vt, with a per-vertex regression head.

Synthetic dataset: random mechanisms (4-bar / Stewart / delta /
serial), each with random but deterministic per-link origins (so the
model has a learnable target — not just noise). We use the same fixture
URDFs as phase 11 but parse the `<origin xyz=...>` attributes from
each `<link>`'s child elements; for fixtures without origin info, we
synthesize positions from the mechanism's geometry (e.g., 4-bar
positions on a unit circle).

Architecture: ``MixedAritySignedKAN`` backbone + per-vertex head:
``h_v_final = node_embed.weight + Σ_a α_a · M_vt_a · h_t_a^L``
``xyz_pred = Linear(h_v_final, 3)``

Loss: MSE over per-vertex xyz targets.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signedkan_wip.src.datasets import SignedGraph
from signedkan_wip.src.kinematic import _serial_arm_urdf, write_fixture
from signedkan_wip.src.kinematic import urdf_to_signed_graph
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples)
from signedkan_wip.src.core.n_tuples import construct_k
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)


def synth_positions(family: str, n_links: int, rng: random.Random) -> np.ndarray:
    """Generate plausible per-link XYZ positions for each fixture family.
    All in unit-meter scale; deterministic given rng."""
    if family == "four_bar":
        # 4 links on a unit circle, randomly rotated.
        offset = rng.random() * 2 * math.pi
        return np.array([
            [math.cos(offset + i * math.pi / 2),
             math.sin(offset + i * math.pi / 2),
             0.0]
            for i in range(4)
        ])
    elif family == "stewart":
        # Base circle, ee circle, struts between. 14 links total.
        z_base = 0.0; z_ee = 1.0
        r_base = 1.0; r_ee = 0.5
        positions = [[0, 0, z_base], [0, 0, z_ee]]   # base, ee
        for i in range(6):
            theta = i * 2 * math.pi / 6
            # leg_lower at base
            positions.append([r_base * math.cos(theta),
                                r_base * math.sin(theta), z_base + 0.1])
            # leg_upper at ee
            positions.append([r_ee * math.cos(theta + math.pi / 6),
                                r_ee * math.sin(theta + math.pi / 6),
                                z_ee - 0.1])
        return np.array(positions)
    elif family == "delta_3rrr":
        # Base at 3 corners + ee at center, plus 6 mid-links.
        positions = [[0, 0, 0], [0, 0, 1.0]]   # base, ee
        for i in range(3):
            theta = i * 2 * math.pi / 3
            positions.append([0.5 * math.cos(theta),
                                0.5 * math.sin(theta), 0.5])  # upper
            positions.append([0.3 * math.cos(theta),
                                0.3 * math.sin(theta), 0.7])  # lower
        return np.array(positions)
    elif family == "serial":
        # Straight chain along z, links stacked.
        return np.array([[0, 0, i * 0.2] for i in range(n_links + 1)])
    else:
        return rng.random((n_links, 3))


def build_mechanism_with_positions(rng: random.Random):
    family = rng.choices(
        ["four_bar", "stewart", "delta_3rrr", "serial"],
        weights=[3, 2, 2, 4],
    )[0]
    if family == "serial":
        n_links = rng.randint(3, 8)
        import tempfile
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_serial_{n_links}.urdf", delete=False)
        f.write(_serial_arm_urdf(n_links)); f.close()
        path = Path(f.name)
    else:
        n_links = {"four_bar": 4, "stewart": 14, "delta_3rrr": 8}[family]
        path = write_fixture(family)
    try:
        g, _, _ = urdf_to_signed_graph(path)
    finally:
        path.unlink(missing_ok=True)
    pos = synth_positions(family, n_links, rng)
    # Pad / truncate to match g.n_nodes (URDF parser may return fewer
    # if fixed joints are skipped, etc).
    if pos.shape[0] < g.n_nodes:
        pos = np.vstack([pos, np.zeros((g.n_nodes - pos.shape[0], 3))])
    elif pos.shape[0] > g.n_nodes:
        pos = pos[:g.n_nodes]
    return g, pos.astype(np.float32), family


class PositionRegHSiKAN(nn.Module):
    """Per-vertex position regression on cycle-pool vertex embeddings."""
    def __init__(self, n_nodes_max: int, arity: int,
                  hidden: int = 16, n_layers: int = 2, grid: int = 5):
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
        # Per-vertex head from h_v: hidden_dim → 3 (xyz)
        self.pos_head = nn.Linear(hidden, 3)

    def forward(self, per_arity_input):
        # Run encode_edges once to set up vertex updates (side-effect on
        # node_embed via the mix). For per-vertex output we just read
        # node_embed.weight after one forward "warm" pass.
        # Simpler approach: pool cycle-pool vertex embeddings via M_vt.
        triad_v, triad_sigma, M_vt, _ = per_arity_input[0]
        h_v = self.backbone.node_embed.weight   # (V, hidden)
        # One layer of cycle-pool aggregation.
        h_t = self.backbone.base.shared_layer(h_v, triad_v, triad_sigma)
        h_v_step = torch.sparse.mm(M_vt, h_t)
        h_v_new = h_v + h_v_step
        return self.pos_head(h_v_new)            # (V, 3)


def _build_input(g: SignedGraph, arity: int, max_per: int, device,
                   seed: int, n_nodes_pad: int):
    if arity == 3:
        t_k = construct(g)
    else:
        t_k = construct_k(g, k=arity, max_cycles=max_per, seed=seed)
    if not t_k: return None
    if len(t_k) > max_per:
        t_k = subsample_tuples(t_k, max_per, seed=seed)
    triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
    triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
    triad_v = torch.from_numpy(triad_v_np).to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
    M_vt = build_vertex_triad_incidence(triad_v_np, n_nodes_pad, device, mode="sum")
    edge_to_tuples = build_edge_to_tuples(t_k)
    rows, cols, vals = [], [], []
    for ei, e in enumerate(g.edges):
        key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
        ids = edge_to_tuples.get(key, [])
        if not ids: continue
        w = 1.0 / float(len(ids))
        for t in ids:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    if rows:
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e = torch.sparse_coo_tensor(idx, v, (g.edges.shape[0], len(t_k))).coalesce()
    else:
        M_e = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
            (g.edges.shape[0], len(t_k))).to(device)
    return [(triad_v, triad_sigma, M_vt, M_e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_epochs", type=int, default=80)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    print("=== Phase 12 — Per-vertex position regression on kinematic graphs ===",
          flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregate_mae = {4: [], 6: []}
    for seed in args.seeds:
        rng = random.Random(seed)
        torch.manual_seed(seed); np.random.seed(seed)
        train, test = [], []
        for _ in range(args.n_train):
            train.append(build_mechanism_with_positions(rng))
        for _ in range(args.n_test):
            test.append(build_mechanism_with_positions(rng))

        for arity in (4, 6):
            from .run_phase11_kinematic_tasks import detect_dominant_arity
            cands_tr = [(g, p, fam) for g, p, fam in train
                          if detect_dominant_arity(g) == arity]
            cands_te = [(g, p, fam) for g, p, fam in test
                          if detect_dominant_arity(g) == arity]
            if not cands_tr or not cands_te:
                continue
            n_nodes_max = max(g.n_nodes for g, p, _ in cands_tr + cands_te)
            train_inputs = []
            train_targets = []
            for g, pos, fam in cands_tr:
                inp = _build_input(g, arity, 30_000, device, seed, n_nodes_max)
                if inp is None: continue
                # Pad position target to n_nodes_max.
                pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
                pos_pad[:pos.shape[0]] = pos
                # Mask so only real vertices contribute to loss.
                mask = np.zeros(n_nodes_max, dtype=np.float32)
                mask[:g.n_nodes] = 1.0
                train_inputs.append((inp, torch.from_numpy(pos_pad).to(device),
                                       torch.from_numpy(mask).to(device)))
            test_inputs = []
            for g, pos, fam in cands_te:
                inp = _build_input(g, arity, 30_000, device, seed, n_nodes_max)
                if inp is None: continue
                pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
                pos_pad[:pos.shape[0]] = pos
                mask = np.zeros(n_nodes_max, dtype=np.float32)
                mask[:g.n_nodes] = 1.0
                test_inputs.append((inp, torch.from_numpy(pos_pad).to(device),
                                      torch.from_numpy(mask).to(device)))
            if not train_inputs or not test_inputs: continue

            model = PositionRegHSiKAN(n_nodes_max=n_nodes_max,
                                          arity=arity, hidden=16,
                                          n_layers=2, grid=5).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=5e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)
            for ep in range(args.n_epochs):
                model.train()
                perm = torch.randperm(len(train_inputs))
                loss_total = 0.0
                for i in perm:
                    inp, tgt, mask = train_inputs[i]
                    pred = model(inp)             # (V, 3)
                    err = (pred - tgt) * mask.unsqueeze(-1)
                    loss = (err.pow(2).sum(-1) * mask).sum() / max(1, int(mask.sum()))
                    opt.zero_grad(); loss.backward(); opt.step()
                    loss_total += loss.item()
                sched.step()
            model.eval()
            test_mse_sum = 0.0; n_real = 0
            with torch.no_grad():
                for inp, tgt, mask in test_inputs:
                    pred = model(inp)
                    err = (pred - tgt) * mask.unsqueeze(-1)
                    test_mse_sum += float(err.pow(2).sum().item())
                    n_real += int(mask.sum().item())
            test_rmse = math.sqrt(test_mse_sum / max(1, n_real * 3))
            aggregate_mae[arity].append(test_rmse)
            print(f"  seed={seed} arity={arity}  n_train={len(train_inputs)} "
                  f"n_test={len(test_inputs)}  test_rmse={test_rmse:.3f} m",
                  flush=True)

    print("\n=== Aggregated (median across seeds) ===", flush=True)
    for arity in (4, 6):
        if not aggregate_mae[arity]: continue
        med = statistics.median(aggregate_mae[arity])
        print(f"arity={arity}  rmse_med={med:.3f} m  n_seeds={len(aggregate_mae[arity])}",
              flush=True)


if __name__ == "__main__":
    main()
