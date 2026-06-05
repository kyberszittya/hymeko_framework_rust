"""IMDB binary-sentiment benchmark: AC-HSiKAN vs Transformer.

Sub-samples the IMDB train/test split (normally 25k+25k) to make this
runnable as a smoke in ~10-15 min. Switch ``--device cuda`` to fire the
Triton-fused pool-scatter kernel, ``--pool-scatter`` to enable the
v1.6 dual-path (synaptic ⊕ rhythm), and ``--rotor`` to add the
entropy Hamilton rotor on the scatter direction. ``--telemetry-out``
streams an evolvens JSONL trace from the backward pass.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from signedkan_wip.src.ac_hsikan import AcHsikanClassifier, AcHsikanConfig
from signedkan_wip.src.sequence.imdb_dataset import load_imdb
from signedkan_wip.src.sequence.iso_param_transformer import (
    IMDBTransformerBaseline,
)


def _to_dataloader(ids, mask, labels, batch_size: int, shuffle: bool):
    ds = TensorDataset(
        torch.from_numpy(ids).long(),
        torch.from_numpy(mask).bool(),
        torch.from_numpy(labels).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one(model: nn.Module, train_dl, val_dl, n_epochs: int, lr: float,
              device: str, want_mask: bool = True,
              weight_decay: float = 1e-4, grad_clip: float = 0.0,
              lr_schedule: str = "constant",
              lr_2d_mult: float = 1.0) -> dict:
    model.to(device)
    # Split parameter groups: the 2D-CR coef tensors (when present) get
    # a separate (lower) learning rate so the V-axis deviations don't
    # outpace the main backbone -- mitigates the d>16 collapse seen
    # when the 2D coefs train at the full lr.
    if lr_2d_mult != 1.0:
        coef_2d, main = [], []
        for n, p in model.named_parameters():
            (coef_2d if ("coef_pos_2d" in n or "coef_neg_2d" in n)
             else main).append(p)
        param_groups = [
            {"params": main, "lr": lr},
            {"params": coef_2d, "lr": lr * lr_2d_mult},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=lr * 0.1,
        )
    best_val_acc = 0.0
    epoch_history: list[float] = []
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for ids, mask, labels in train_dl:
            ids = ids.to(device); mask = mask.to(device); labels = labels.to(device)
            logits = model(ids, mask=mask) if want_mask else model(ids)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if sched is not None:
            sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for ids, mask, labels in val_dl:
                ids = ids.to(device); mask = mask.to(device); labels = labels.to(device)
                pred_logits = model(ids, mask=mask) if want_mask else model(ids)
                pred = pred_logits.argmax(dim=1)
                correct += (pred == labels).sum().item(); total += labels.numel()
        acc = correct / total
        epoch_history.append(acc)
        if acc > best_val_acc:
            best_val_acc = acc
    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"val_acc": best_val_acc, "wall_s": round(wall, 1),
            "n_params": n_params, "epoch_acc": epoch_history}


def subsample(ids, mask, labels, n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ids), generator=g)[:n].numpy()
    return ids[perm], mask[perm], labels[perm]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--device", default="cpu")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--l-max", type=int, default=128)
    p.add_argument("--out", default="/tmp/ac_hsikan_imdb_smoke.json")
    p.add_argument("--use-value-projection", action="store_true",
                   help="enable AcHsikanConfig.use_value_projection")
    p.add_argument("--ffn-hidden", type=int, default=0,
                   help="post-attention FFN hidden size (0 = disabled)")
    p.add_argument("--use-magnitude-weight", action="store_true",
                   help="enable softmax(|score|) weighting alongside sign")
    p.add_argument("--clifford-fir", action="store_true",
                   help="enable Clifford-FIR context pre-processing")
    p.add_argument("--fused-walk", action="store_true",
                   help="enable fused chain/cycle walk (v1.6 #2)")
    p.add_argument("--sparse-sign-head", action="store_true",
                   help="enable sparse multi-head sign computation")
    p.add_argument("--pool-scatter", action="store_true",
                   help="enable v1.6 dual-path pool-scatter (synaptic primitive)")
    p.add_argument("--rotor", action="store_true",
                   help="enable entropy Hamilton rotor on scatter direction "
                        "(requires --pool-scatter and --device cuda)")
    p.add_argument("--telemetry-out", default=None,
                   help="path to write evolvens-telemetry JSONL trace "
                        "(only meaningful with --rotor)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the AC classifier with fullgraph=False "
                        "(graph-break at the Triton autograd Function)")
    p.add_argument("--d-model", type=int, default=16,
                   help="model dim for AC + transformer (both scale identically)")
    p.add_argument("--n-layers", type=int, default=2,
                   help="number of blocks in AC + transformer")
    p.add_argument("--sign-head-hidden", type=int, default=None,
                   help="AC sign-attention bottleneck dim. None = match d_model "
                        "(scales with model). Default config value is 16, "
                        "i.e. fixed 16-dim bottleneck regardless of d_model.")
    p.add_argument("--dynamic-topk", action="store_true",
                   help="AC: per-anchor top-K candidate selection by "
                        "batch-mean |sign-magnitude| (ABB-per-vertex analogue). "
                        "Replaces the static positional `local` with content-"
                        "adaptive neighbours every forward.")
    p.add_argument("--long-cycle-prior", type=float, default=0.0,
                   help="AC: distance-weighting exponent α on the dynamic-topk "
                        "score. α=0 disables (pure magnitude); α>0 biases the "
                        "top-K toward positionally distant candidates (longer "
                        "cycles through walk-op composition). The sequence-"
                        "native replacement for HSiKAN's DAG-axiom pruning.")
    p.add_argument("--dyn-topk-static-k", type=int, default=0,
                   help="AC: K_static slots from static positional neighbours, "
                        "remaining (K - K_static) from dynamic top-K of the "
                        "non-static positions. Combines proven local prior "
                        "with adaptive long-range signal.")
    p.add_argument("--top-k", type=int, default=8,
                   help="AC: total top_k_per_position. Default 8.")
    p.add_argument("--use-2d-cr", action="store_true",
                   help="AC: 2D Catmull-Rom patch surface over (R, V_c) "
                        "instead of separable 1D CR + multiplicative V gate. "
                        "Learned non-separable bivariate activation. "
                        "Uses PyTorch reference path (no Triton acceleration).")
    p.add_argument("--walk-kind", choices=["star", "chain", "cycle"],
                   default="star",
                   help="AC walk-op variant. star=product of first k signs; "
                        "chain=chain product; cycle=chain + closing edge "
                        "(real signed cycle).")
    p.add_argument("--lr-2d-mult", type=float, default=1.0,
                   help="LR multiplier for the 2D CR coef params. <1 slows "
                        "the V-axis deviations from flat-init; useful when "
                        "scaling beyond d=16.")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"],
                   default="constant",
                   help="LR schedule. cosine: cosine-annealing from lr to lr*0.1.")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Gradient norm clipping threshold. 0 = off.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay. Default 1e-4.")
    p.add_argument("--arity-bias", type=str, default=None,
                   help="AC: comma-separated α_init for the per-arity walk-op "
                        "weights. Default uniform 0.25 each. Pass e.g. "
                        "'0.10,0.15,0.30,0.45' to bias toward longer cycles.")
    p.add_argument("--pre-norm", action="store_true",
                   help="AC: use pre-norm (LayerNorm at block entry) instead "
                        "of post-norm. Recommended for n_layers >= 4.")
    p.add_argument("--layer-scale", action="store_true",
                   help="AC: per-channel learnable scale on the residual "
                        "branch (Touvron 2021). Pairs with --pre-norm "
                        "for deep stacks.")
    args = p.parse_args(argv)

    print("== loading IMDB (cached) ==")
    train, test, vocab = load_imdb(vocab_size=args.vocab_size, L_max=args.l_max)
    print(f"   train={len(train)} test={len(test)} vocab={len(vocab)}")

    results = {"ac_hsikan": [], "transformer": []}
    for seed in args.seeds:
        torch.manual_seed(seed)
        ids_tr, mask_tr, lab_tr = subsample(train.ids, train.mask, train.labels,
                                             args.n_train, seed)
        ids_va, mask_va, lab_va = subsample(test.ids, test.mask, test.labels,
                                             args.n_val, seed + 10_000)
        train_dl = _to_dataloader(ids_tr, mask_tr, lab_tr, args.batch_size, True)
        val_dl = _to_dataloader(ids_va, mask_va, lab_va, args.batch_size, False)

        # AC-HSiKAN
        # Scale pool-scatter h with d_model to keep the synaptic primitive
        # proportional. h must be 4·n_quat in v1; pick n_quat = d_model // 8.
        ps_n_quat = max(2, args.d_model // 8)
        ps_h = 4 * ps_n_quat
        # Sign-attention bottleneck: default 16 doesn't scale with d_model,
        # so at d=32 it's a 50% compression. Pass --sign-head-hidden or
        # default to d_model when scaling up.
        sh_hidden = (args.sign_head_hidden if args.sign_head_hidden is not None
                     else args.d_model)
        cfg = AcHsikanConfig(
            d_model=args.d_model, n_positions=args.l_max,
            arities=(2, 3, 4, 5), top_k_per_position=args.top_k,
            walk_kind=args.walk_kind,
            n_layers=args.n_layers, n_classes=2, dropout=0.1, pool="mean",
            sign_head_hidden=sh_hidden,
            use_value_projection=args.use_value_projection,
            ffn_hidden=args.ffn_hidden,
            use_magnitude_weight=args.use_magnitude_weight,
            use_clifford_fir_context=args.clifford_fir,
            use_fused_walk=args.fused_walk,
            sparse_sign_head=args.sparse_sign_head,
            use_pool_scatter=args.pool_scatter,
            use_pool_scatter_rotor=args.rotor,
            pool_scatter_h=ps_h,
            pool_scatter_n_quat=ps_n_quat,
            use_pre_norm=args.pre_norm,
            use_layer_scale=args.layer_scale,
            use_dynamic_topk=args.dynamic_topk,
            long_cycle_prior_alpha=args.long_cycle_prior,
            dynamic_topk_static_k=args.dyn_topk_static_k,
            use_2d_cr=args.use_2d_cr,
            **({"alpha_init": tuple(float(v) for v in args.arity_bias.split(","))}
               if args.arity_bias else {}),
        )
        if args.rotor and not args.pool_scatter:
            raise SystemExit("--rotor requires --pool-scatter")
        if args.rotor and args.device != "cuda":
            raise SystemExit("--rotor needs --device cuda (Triton kernel)")
        torch.manual_seed(seed)
        ac = AcHsikanClassifier(vocab_size=len(vocab), cfg=cfg)
        if args.compile:
            # Compile with fullgraph=False so the autograd-wrapped Triton
            # forward (which Inductor can't trace) is treated as opaque;
            # everything around it (embedding, sign-attn, walk-op, FFN,
            # layer-norm, classifier head) is fused. Measured ~1.48× wall
            # speedup on the full classifier with parity to 1e-8 on
            # forward outputs and bit-equal loss.
            ac = torch.compile(ac, mode="default", fullgraph=False)
        # Optional evolvens-telemetry context wraps the AC training only;
        # the rotor lives inside AC's FusedPoolScatter, so transformer
        # baseline never enters this sink.
        if args.rotor and args.telemetry_out:
            from signedkan_wip.src.ac_hsikan import EvolventTelemetry
            telem_path = f"{args.telemetry_out}.seed{seed}.jsonl"
            with EvolventTelemetry(out_path=telem_path) as tel:
                r_ac = train_one(ac, train_dl, val_dl, args.n_epochs,
                                  args.lr, args.device, want_mask=False,
                                  weight_decay=args.weight_decay,
                                  grad_clip=args.grad_clip,
                                  lr_schedule=args.lr_schedule,
                                  lr_2d_mult=args.lr_2d_mult)
            r_ac["telemetry_path"] = telem_path
            r_ac["telemetry_summary"] = tel.summary()
        else:
            r_ac = train_one(ac, train_dl, val_dl, args.n_epochs, args.lr,
                              args.device, want_mask=False,
                              weight_decay=args.weight_decay,
                              grad_clip=args.grad_clip,
                              lr_schedule=args.lr_schedule,
                                  lr_2d_mult=args.lr_2d_mult)
        r_ac["seed"] = seed
        r_ac["alpha"] = [a.tolist() for a in ac.alpha()]
        results["ac_hsikan"].append(r_ac)
        print(f"  seed={seed} AC-HSiKAN  val_acc={r_ac['val_acc']:.4f} "
              f"wall={r_ac['wall_s']}s n_params={r_ac['n_params']:,} "
              f"epochs={['%.3f' % e for e in r_ac['epoch_acc']]}")

        # Transformer baseline -- scale dim_ff proportionally (4·d_model)
        # so both models scale equivalently with --d-model.
        torch.manual_seed(seed)
        tr = IMDBTransformerBaseline(
            vocab_size=len(vocab), d_model=args.d_model,
            n_heads=max(2, args.d_model // 8), dim_ff=4 * args.d_model,
            n_layers=args.n_layers,
        )
        r_tr = train_one(tr, train_dl, val_dl, args.n_epochs, args.lr,
                          args.device, want_mask=True,
                          weight_decay=args.weight_decay,
                          grad_clip=args.grad_clip,
                          lr_schedule=args.lr_schedule,
                                  lr_2d_mult=args.lr_2d_mult)
        r_tr["seed"] = seed
        results["transformer"].append(r_tr)
        print(f"  seed={seed} Transform  val_acc={r_tr['val_acc']:.4f} "
              f"wall={r_tr['wall_s']}s n_params={r_tr['n_params']:,} "
              f"epochs={['%.3f' % e for e in r_tr['epoch_acc']]}")

    print()
    print("=== paired summary ===")
    for name in ("ac_hsikan", "transformer"):
        accs = [r["val_acc"] for r in results[name]]
        params = results[name][0]["n_params"]
        walls = [r["wall_s"] for r in results[name]]
        m = statistics.mean(accs)
        s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {name:<12} acc={m:.4f} ± {s:.4f}  "
              f"n_params={params:,}  wall_total={sum(walls):.1f}s")
    ac_m = statistics.mean([r["val_acc"] for r in results["ac_hsikan"]])
    tr_m = statistics.mean([r["val_acc"] for r in results["transformer"]])
    print(f"  delta (AC - TR): {ac_m - tr_m:+.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
