"""Stage D — HyMeYOLO Stage C architecture on PASCAL VOC2007.

Companion to ``train_voc.py``: where ``train_voc.py`` sweeps the
5 legacy configs at the default tiny backbone, this script runs
\\emph{one} config — the Stage C architecture
(``backbone='resnet'`` + ``fpn='2level'``) — that landed the
0.8955 5-seed mAP_50 on Cluttered MNIST. Stage D = does it
transfer to real natural images.

Architecture (per plan ``docs/plans/2026-05-17-hymeyolo-stage-d-pascal-voc``):
    RicciHyMeYOLOMulti(
        n_box_queries=12, n_circle_queries=0,
        n_classes=20, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone='resnet', fpn='2level',
    )

Falsifier: 5-seed mean test mAP_50 < 0.20 → architecture does
not transfer from synthetic to natural images at this scale.

Usage:
    # Unit smoke (100 images, 1 epoch, CPU OK):
    python -m signedkan_wip.src.vision.train_voc_stagec \\
        --n-images 100 --epochs 1 --input-size 224 --seed 0

    # Production-scale smoke (1 seed, full trainval, 30 epochs):
    python -m signedkan_wip.src.vision.train_voc_stagec \\
        --image-set trainval --epochs 30 --input-size 224 --seed 0 \\
        --save-checkpoint signedkan_wip/experiments/results/stage_d_voc/

    # 5-seed launch (use the orchestrator script, not this directly).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .hymeyolo_circles_ricci import RicciHyMeYOLOMulti
from .train_circles_ricci import train_one_config
from .voc_dataset import VOC_CLASSES, load_voc_hungarian


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="2007", choices=["2007", "2012"])
    p.add_argument("--image-set", default="trainval",
                    choices=["train", "val", "trainval", "test"])
    p.add_argument("--n-images", type=int, default=None,
                    help="Subset size; None = full split. Use small "
                         "value (~100) for the unit smoke.")
    p.add_argument("--input-size", type=int, default=224,
                    help="Square resize H=W. Plan default 224 fits "
                         "Stage C in <= 5 GB GPU at batch 8.")
    p.add_argument("--max-objects", type=int, default=12,
                    help="Pad slot count. VOC max objects per image "
                         "is around 10-15.")
    p.add_argument("--n-box-queries", type=int, default=12,
                    help="Hungarian query count. Plan: 12 (VOC avg "
                         "2.4 objects/image; max 12-ish).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=8,
                    help="Plan: 8 at input_size=224 keeps GPU "
                         "memory <= 5 GB on RTX 2070 SUPER.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="data/torchvision")
    p.add_argument("--jsonl-out", default=None)
    p.add_argument("--save-checkpoint", default=None,
                    help="If set, save the trained model under <dir>/"
                         "stage_c_seed<seed>.pt for the Tk demo.")
    p.add_argument("--ricci-scale", type=float, default=1.0)
    p.add_argument("--schedule", default="cosine",
                    choices=["constant", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--backbone",
                    choices=["resnet", "resnet18_imagenet", "hsikan"],
                    default="resnet",
                    help="Backbone name; 'resnet' = Stage D (from-scratch "
                         "ResNet-tiny); 'resnet18_imagenet' = Stage D-1 "
                         "(ImageNet-pretrained ResNet18 truncated to layer2); "
                         "'hsikan' = HSiKAN-CR (Catmull-Rom basis activations) "
                         "for the family-paper purity comparison.")
    p.add_argument("--lam-no-obj", type=float, default=0.5,
                    help="Weight on the no-object CE term. Default 0.5 = "
                         "Stage D / D-1 baseline. Stage D-2b raises to "
                         "2.0–5.0 to pressure queries into suppressing.")
    p.add_argument("--query-head-kind",
                    choices=["hungarian", "nodelet"],
                    default="hungarian",
                    help="Query head type. 'hungarian' = legacy "
                         "(Stage D/D-1/D-2). 'nodelet' = Stage D-3 "
                         "(explicit per-query objectness gate).")
    p.add_argument("--lam-gate-neg", type=float, default=None,
                    help="Stage D-3-bis: weight on the unmatched-gate "
                         "BCE term in the nodelet loss. None (default) "
                         "= D-3 auto-balance (N_pos/N_neg ≈ 0.18). "
                         "Set to 1.0 to give the suppression-side "
                         "gradient ~5× more aggregate pressure. "
                         "Ignored when --query-head-kind=hungarian.")
    p.add_argument("--lam-gate-match-cost", type=float, default=None,
                    help="Stage D-3-tris: weight on the (1 - gate) term "
                         "in the Hungarian matcher cost matrix. None "
                         "(default) = nodelet head's internal default "
                         "(1.0). Raise to 2.0–5.0 to make the matcher "
                         "more reluctant to assign GTs to low-gate "
                         "queries, recovering matched-cls accuracy. "
                         "Ignored when --query-head-kind=hungarian.")
    p.add_argument("--gate-loss-kind",
                    choices=["bce", "focal"], default="bce",
                    help="Stage D-3-tris: unmatched-gate loss kind. "
                         "'bce' (default) = BCE -log(1-g). 'focal' = "
                         "g^γ · -log(1-g) — easy-to-suppress queries "
                         "get less gradient, borderline ones get more. "
                         "Ignored when --query-head-kind=hungarian.")
    p.add_argument("--gate-focal-gamma", type=float, default=2.0,
                    help="Focal gamma for --gate-loss-kind=focal. "
                         "Default 2.0 (standard focal-loss value).")
    p.add_argument("--backbone-checkpoint", action="store_true",
                    help="Stage D-3-quinquies: enable PyTorch activation "
                         "checkpointing on the backbone (currently only "
                         "honoured by --backbone hsikan). Trades ~30%% wall "
                         "for ~70%% backbone-activation memory, "
                         "unblocking the 7.6 GiB consumer-GPU regime that "
                         "OOMed Stage D-3c.")
    p.add_argument("--warmstart-mode",
                    choices=["off", "saliency", "quadtree"], default="off",
                    help="Stage 2026-05-21 lever: initialise query corners "
                         "from a data-aware prior before training. 'off' "
                         "(default) = current behaviour. 'saliency' uses "
                         "mean-of-absolute-pixels FPS (proven on Cluttered "
                         "MNIST, +0.124 paired Δ). 'quadtree' uses the "
                         "Rust-backed adaptive-quadtree leaf centres "
                         "(variance + Forman κ), the curvature-aware "
                         "natural-image generalisation.")
    p.add_argument("--warmstart-bootstrap-n", type=int, default=128,
                    help="Bootstrap batch size for --warmstart-mode "
                         "{saliency,quadtree}. 0 = use the full train set.")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    n_classes = len(VOC_CLASSES)

    print(f"\n=== Stage D — HyMeYOLO Stage C on VOC{args.year} ===")
    print(f"  image_set={args.image_set}  input={args.input_size}  "
          f"epochs={args.epochs}  batch={args.batch_size}  "
          f"n_box_queries={args.n_box_queries}  "
          f"n_classes={n_classes}  device={device}")
    print(f"  arch: backbone={args.backbone!r}  fpn='2level'  "
          f"ricci_modulation=True  ricci_scale={args.ricci_scale}")

    t0 = time.perf_counter()
    Xn, boxes_n, classes_n, counts_n, names = load_voc_hungarian(
        year=args.year, image_set=args.image_set,
        input_size=args.input_size, max_objects=args.max_objects,
        root=args.data_root, subset_n=args.n_images, download=True,
    )
    load_wall = time.perf_counter() - t0
    print(f"  loaded in {load_wall:.1f}s — X={Xn.shape}  "
          f"counts.mean={counts_n.mean():.2f}  "
          f"counts.max={counts_n.max()}  n_classes={n_classes}")

    # ─── Lazy GPU transfer (2026-05-22 Phase 7 refactor) ─────────────
    # Phase 6 diagnosed that the upstream pre-load `X = ...to(device)`
    # OOMs at input>=320 px on the 8 GiB GPU (5011·3·320²·4 ≈ 6.16 GiB
    # before model + activations).  Keep X on CPU; transfer each
    # mini-batch on demand in train_one_config / compute_detection_metrics.
    # Boxes / classes / counts are small (~MB) and stay on the GPU.
    X = torch.from_numpy(Xn)  # CPU tensor
    boxes = torch.from_numpy(boxes_n).to(device)
    classes = torch.from_numpy(classes_n).to(device)
    # Padding sentinel -1 must not index into class output; counts mask
    # ensures these never reach the loss, but we still map to a safe
    # in-range value (0) so indexing-based ops do not OOB.
    classes_safe = torch.where(classes < 0, torch.zeros_like(classes),
                                 classes)
    counts = torch.from_numpy(counts_n).to(device)

    label = (
        "stage_c_voc" if args.backbone == "resnet" else "stage_d1_voc"
    )
    model = RicciHyMeYOLOMulti(
        n_box_queries=args.n_box_queries,
        n_circle_queries=0,
        n_classes=n_classes,
        d_hidden=32,
        ricci_modulation=True,
        ricci_scale=args.ricci_scale,
        use_layernorm=False,
        backbone=args.backbone,
        fpn="2level",
        query_head_kind=args.query_head_kind,
        backbone_use_checkpoint=args.backbone_checkpoint,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    # ─── Warmstart query corners (Stage 2026-05-21 lever) ────────────
    if args.warmstart_mode != "off":
        n_boot = args.warmstart_bootstrap_n if args.warmstart_bootstrap_n > 0 else X.shape[0]
        X_boot = X[: n_boot]
        if args.warmstart_mode == "saliency":
            from signedkan_wip.src.vision.hymeyolo_warmstart import (
                warmstart_query_corners,
            )
            warmstart_query_corners(model, X_boot, seed=args.seed)
            print(f"  [warmstart] saliency / FPS over {X_boot.shape[0]} images")
        elif args.warmstart_mode == "quadtree":
            from signedkan_wip.src.vision.hymeyolo_warmstart_quadtree import (
                warmstart_query_corners_quadtree,
            )
            # Pick quadtree patch_size_initial as input_size // 4 (a
            # generous root cell), patch_size_min as input_size // 32.
            ps_init = max(1, args.input_size // 4)
            ps_min  = max(1, args.input_size // 32)
            # input_size must be divisible by ps_init; nudge if not.
            if args.input_size % ps_init != 0:
                ps_init = args.input_size // (args.input_size // ps_init)
            warmstart_query_corners_quadtree(
                model, X_boot,
                patch_size_initial=ps_init,
                patch_size_min=ps_min,
                variance_weight=1.0,
                curvature_weight=1.0,
                seed=args.seed,
            )
            print(
                f"  [warmstart] quadtree (ps_init={ps_init}, ps_min={ps_min}) "
                f"over {X_boot.shape[0]} images"
            )

    torch.manual_seed(args.seed)
    result = train_one_config(
        label, model, X, boxes, classes_safe, counts,
        epochs=args.epochs, lr=args.lr, device=device,
        batch_size=args.batch_size,
        lam_no_obj=args.lam_no_obj,
        lam_gate_neg_override=args.lam_gate_neg,
        lam_gate_match_cost_override=args.lam_gate_match_cost,
        gate_loss_kind=args.gate_loss_kind,
        gate_focal_gamma=args.gate_focal_gamma,
    )

    det = result.get("det_metrics", {})
    loss_drop_pct = (result["losses"][0] - result["losses"][-1]) \
                    / max(1e-9, result["losses"][0]) * 100.0

    print(f"\n=== Stage D result ===")
    print(f"  loss_start={result['losses'][0]:.4f}  "
          f"loss_end={result['losses'][-1]:.4f}  "
          f"drop={loss_drop_pct:.1f}%")
    print(f"  box_cls_acc={result['accs']['box_cls_acc']:.4f}")
    print(f"  mAP_50={det.get('mAP_50')}  mAP_50_95={det.get('mAP_50_95')}")
    print(f"  wall_s={result['wall']:.1f}")

    if args.save_checkpoint:
        ckpt_dir = Path(args.save_checkpoint)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{label}_seed{args.seed}.pt"
        torch.save({
            "label": label,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "ricci_scale": float(args.ricci_scale),
            "warm_start": False,
            "schedule": args.schedule,
            "warmup_epochs": int(args.warmup_epochs),
            "use_layernorm": False,
            "weight_decay": 0.0,
            "cls_loss": "ce",
            "box_loss": "giou",
            "backbone": args.backbone,
            "fpn": "2level",
            "n_box_queries": int(args.n_box_queries),
            "n_classes": n_classes,
            "input_size": int(args.input_size),
            "state_dict": model.state_dict(),
            "model_class": "RicciHyMeYOLOMulti",
            "dataset": f"voc{args.year}_{args.image_set}",
        }, ckpt_path)
        print(f"  [checkpoint] saved → {ckpt_path}")

    if args.jsonl_out:
        Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl_out, "w") as fh:
            rec = {
                "dataset":     f"voc{args.year}_{args.image_set}",
                "label":       label,
                "n_images":    Xn.shape[0],
                "input_size":  args.input_size,
                "max_objects": args.max_objects,
                "n_box_queries": args.n_box_queries,
                "n_classes":   n_classes,
                "epochs":      args.epochs,
                "lr":          args.lr,
                "seed":        args.seed,
                "n_params":    n_params,
                "backbone":    args.backbone,
                "fpn":         "2level",
                "wall_s":      result["wall"],
                "loss_start":  result["losses"][0],
                "loss_end":    result["losses"][-1],
                "loss_drop_pct": loss_drop_pct,
                "box_cls_acc": result["accs"]["box_cls_acc"],
                "mAP_50":      det.get("mAP_50"),
                "mAP_50_95":   det.get("mAP_50_95"),
                "mean_iou_matched": det.get("mean_iou_matched"),
                "n_preds_used":     det.get("n_preds_used"),
                "n_gts_total":      det.get("n_gts_total"),
                "losses_per_epoch": result["losses"],
            }
            fh.write(json.dumps(rec) + "\n")
        print(f"  Wrote {args.jsonl_out}")

    print(f"\n  Total wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
