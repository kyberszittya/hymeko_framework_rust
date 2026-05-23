"""Multi-seed sweep — SGT vs HSiKAN vs SGCN across Bitcoin Alpha /
OTC / SBM / Epinions for the SMC-paper extension.

Reuses existing model + dataset machinery; SGT is the new baseline
defined in `signedkan_wip.src.baselines.sgt`.

Output: signedkan_wip/experiments/results/sgt_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def time_per_call(fn, n_warmup=10, n_repeats=20):
    cuda = torch.cuda.is_available()
    for _ in range(n_warmup):
        fn()
        if cuda: torch.cuda.synchronize()
    import statistics, time
    samples = []
    for _ in range(n_repeats):
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda: torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def run_sgt(g, e_tr, s_tr, e_te, s_te, hidden, n_layers, n_epochs,
             device, lr=5e-3):
    from signedkan_wip.src.baselines.sgt import SGT, build_signed_neighbours
    nbrs, sgns = build_signed_neighbours(e_tr, s_tr, g.n_nodes)
    model = SGT(n_nodes=g.n_nodes, hidden_dim=hidden, n_heads=4,
                 n_layers=n_layers).to(device)
    y_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    n_pos = int(y_tr.sum().item()); n_neg = int((1 - y_tr).sum().item())
    pw = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                       device=device)
    e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
    e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for _ in range(n_epochs):
        z = model.encode_nodes(nbrs, sgns)
        logits = model.edge_logits(z, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                   pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0

    model.eval()
    def fwd():
        with torch.no_grad():
            z = model.encode_nodes(nbrs, sgns)
            return model.edge_logits(z, e_te_t)
    with torch.no_grad():
        probs = torch.sigmoid(fwd()).cpu().numpy()
    y_te = (s_te == 1).astype(int)
    auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else float("nan")
    f1m = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
    lat = time_per_call(fwd)
    return dict(auc=float(auc), f1m=float(f1m), train_s=train_time,
                fwd_per_call_ms=float(lat),
                n_params=model.num_parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc", "sbm_n200"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/sgt_sweep.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")  # truncate

    from signedkan_wip.src.datasets import load, split
    from signedkan_wip.src.datasets import sbm_signed

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if dataset.startswith("sbm_n"):
                n_nodes = int(dataset.split("_n")[1])
                g, _ = sbm_signed(n_nodes=n_nodes, n_communities=4,
                                    seed=seed)
                n_ep = 200
            else:
                g = load(dataset)
                n_ep = args.n_epochs
            tr_idx, _, te_idx = split(g, seed=seed)
            e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
            e_te, s_te = g.edges[te_idx], g.signs[te_idx]

            try:
                t0 = time.time()
                res = run_sgt(g, e_tr, s_tr, e_te, s_te,
                                args.hidden, args.n_layers, n_ep, device)
                wall = time.time() - t0
                row = dict(dataset=dataset, model="SGT", hidden=args.hidden,
                              n_layers=args.n_layers, seed=seed,
                              wall_s=round(wall, 1), **res)
                print(f"  seed={seed}  AUC={res['auc']:.4f}  "
                      f"F1m={res['f1m']:.4f}  lat={res['fwd_per_call_ms']:.1f}ms  "
                      f"({wall:.1f}s wall)")
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                print(f"  seed={seed} FAILED: {e}")
                with out_path.open("a") as f:
                    f.write(json.dumps(dict(
                        dataset=dataset, model="SGT", seed=seed,
                        error=str(e))) + "\n")

    # Pretty per-dataset summary
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l]
    print("\n── SGT 3-seed summary ──")
    from collections import defaultdict
    import statistics
    ds_rows = defaultdict(list)
    for r in rows:
        if "auc" in r:
            ds_rows[r["dataset"]].append(r)
    for ds in sorted(ds_rows):
        aucs = [r["auc"] for r in ds_rows[ds]]
        f1s  = [r["f1m"] for r in ds_rows[ds]]
        m_auc = statistics.mean(aucs)
        s_auc = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        m_f1  = statistics.mean(f1s)
        s_f1  = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"  {ds:>16s}  AUC={m_auc:.4f} ± {s_auc:.4f}   "
              f"F1m={m_f1:.4f} ± {s_f1:.4f}   ({len(aucs)} seeds)")


if __name__ == "__main__":
    main()
