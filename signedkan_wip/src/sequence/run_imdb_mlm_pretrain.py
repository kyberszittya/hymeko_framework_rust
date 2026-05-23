"""Entry point for MLM pretraining either architecture on IMDB unsup.

Usage::

    python -m signedkan_wip.src.sequence.run_imdb_mlm_pretrain \\
        --arch hsikan --epochs 30 \\
        --state-dict-out checkpoints/pretrain/hsikan_unsup.pt

    python -m signedkan_wip.src.sequence.run_imdb_mlm_pretrain \\
        --arch transformer --epochs 30 \\
        --state-dict-out checkpoints/pretrain/transformer_unsup.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .imdb_dataset import build_imdb_vocab, download_imdb
from .imdb_pretrain import materialise_unsup, pretrain_mlm
from .iso_param_transformer import IMDBTransformerBaseline
from .text_classifier import IMDBClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["hsikan", "transformer"], required=True)
    p.add_argument("--data-root", default="data/imdb")
    p.add_argument("--vocab-size", type=int, default=20_000)
    p.add_argument("--L-max", type=int, default=200)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--mask-prob", type=float, default=0.15)
    # HSiKAN-only
    p.add_argument("--enc-depth", type=int, default=3)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-channels", type=int, default=4)
    # Transformer-only
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--dim-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--state-dict-out", required=True,
                    help="Path to save the pretrained model's state_dict.")
    p.add_argument("--jsonl-out", default=None,
                    help="Optional JSONL with loss curve + wall.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.perf_counter()

    print(f"\n=== IMDB MLM pretrain — arch={args.arch} ===")
    print(f"  vocab_size={args.vocab_size}  L_max={args.L_max}  "
          f"epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  "
          f"device={device}  seed={args.seed}", flush=True)

    aclimdb = download_imdb(Path(args.data_root))
    vocab = build_imdb_vocab(aclimdb, vocab_size=args.vocab_size)
    print(f"  vocab size: {len(vocab)} (UNK_ID doubles as MLM mask sentinel)",
          flush=True)
    unsup_ids_np, unsup_mask_np = materialise_unsup(
        aclimdb, vocab, L_max=args.L_max,
    )
    print(f"  unsup split: {unsup_ids_np.shape[0]} docs × L_max={args.L_max} "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)
    unsup_ids = torch.from_numpy(unsup_ids_np).to(device)
    unsup_mask = torch.from_numpy(unsup_mask_np).to(device)

    if args.arch == "hsikan":
        model = IMDBClassifier(
            vocab_size=len(vocab),
            enc_depth=args.enc_depth, K=args.K,
            n_channels=args.n_channels, max_len=args.L_max,
            n_classes=2,  # head exists but isn't pretrained on
        ).to(device)
        embedding = model.encoder.embed.embed  # nn.Embedding under TokenMultivectorEmbedding
        is_hsikan = True
    else:
        model = IMDBTransformerBaseline(
            vocab_size=len(vocab), d_model=args.d_model,
            n_heads=args.n_heads, dim_ff=args.dim_ff,
            n_layers=args.n_layers, max_len=args.L_max,
            n_classes=2, dropout=args.dropout,
        ).to(device)
        embedding = model.embed
        is_hsikan = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    result = pretrain_mlm(
        model=model,
        embedding=embedding,
        unsup_ids=unsup_ids, unsup_mask=unsup_mask,
        vocab_size=len(vocab),
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        device=device, mask_prob=args.mask_prob,
        seed=args.seed, is_hsikan=is_hsikan,
    )
    wall = time.perf_counter() - t0
    print(f"\n  pretrain done in {wall:.0f}s  "
          f"final loss={result['losses_per_epoch'][-1]:.4f}", flush=True)

    out_path = Path(args.state_dict_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"  saved state_dict → {out_path}", flush=True)

    if args.jsonl_out:
        Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl_out, "w") as f:
            f.write(json.dumps({
                "phase": "mlm_pretrain",
                "arch": args.arch,
                "n_params": n_params,
                "vocab_size": len(vocab),
                "n_unsup_docs": int(unsup_ids_np.shape[0]),
                "L_max": args.L_max,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "mask_prob": args.mask_prob,
                "losses_per_epoch": result["losses_per_epoch"],
                "wall_s": wall,
                "state_dict_path": str(out_path),
                "seed": args.seed,
            }) + "\n")
        print(f"  Wrote {args.jsonl_out}", flush=True)


if __name__ == "__main__":
    main()
