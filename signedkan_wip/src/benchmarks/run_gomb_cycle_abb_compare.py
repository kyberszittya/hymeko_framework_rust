"""Paired Gömb smoke: compare cycle ABB modes on the same architecture.

Spawns ``run_gomb_smoke`` once per mode (default: ``none``, ``start_local``),
parses the trailing JSON summary line, prints a Markdown table to stdout, and
optionally appends one JSON object per run to ``--jsonl-out``.

Example::

    PYTHONPATH=$PWD python -m signedkan_wip.src.benchmarks.run_gomb_cycle_abb_compare \\
        --dataset bitcoin_otc --edge-split 80_10_10 --device cpu \\
        --n-epochs 8 --topk 48 \\
        --d-embed 24 --d-outer 12 --M-outer 4 --d-middle 24 --d-core 24 \\
        --modes none start_local global_min
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_last_gomb_json_line(stdout: str) -> dict:
    row = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            row = json.loads(line)
            break
    if row is None:
        raise ValueError("no trailing JSON summary line in run_gomb_smoke stdout")
    return row


def _smoke_cmd(
    *,
    py: str,
    dataset: str,
    edge_split: str,
    device: str,
    seed: int,
    n_epochs: int,
    topk: int,
    d_embed: int,
    d_outer: int,
    m_outer: int,
    d_middle: int,
    d_core: int,
    k: int,
    abb_mode: str | None,
    abb_fullness_gate: float,
) -> list[str]:
    cmd: list[str] = [
        py,
        "-m",
        "signedkan_wip.experiments.runs.run_gomb_smoke",
        "--dataset",
        dataset,
        "--edge-split",
        edge_split,
        "--device",
        device,
        "--seed",
        str(seed),
        "--n-epochs",
        str(n_epochs),
        "--topk",
        str(topk),
        "--d-embed",
        str(d_embed),
        "--d-outer",
        str(d_outer),
        "--M-outer",
        str(m_outer),
        "--d-middle",
        str(d_middle),
        "--d-core",
        str(d_core),
        "--k",
        str(k),
        "--cycle-abb-fullness-gate",
        str(abb_fullness_gate),
    ]
    if abb_mode and abb_mode != "none":
        cmd.extend(["--cycle-abb-mode", abb_mode])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--edge-split", default="80_10_10")
    ap.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=8)
    ap.add_argument("--topk", type=int, default=48)
    ap.add_argument("--d-embed", type=int, default=24)
    ap.add_argument("--d-outer", type=int, default=12)
    ap.add_argument("--M-outer", type=int, default=4)
    ap.add_argument("--d-middle", type=int, default=24)
    ap.add_argument("--d-core", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["none", "start_local"],
        help="Cycle ABB modes to compare (repeat 'none' if desired).",
    )
    ap.add_argument("--cycle-abb-fullness-gate", type=float, default=0.25)
    ap.add_argument(
        "--jsonl-out",
        default=None,
        help="Append one JSON record per mode (creates parent dirs).",
    )
    ap.add_argument("--timeout-s", type=int, default=3600)
    args = ap.parse_args()
    device = args.device or _default_device()

    repo = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    py = sys.executable

    rows: list[dict] = []
    for mode in args.modes:
        cmd = _smoke_cmd(
            py=py,
            dataset=args.dataset,
            edge_split=args.edge_split,
            device=device,
            seed=args.seed,
            n_epochs=args.n_epochs,
            topk=args.topk,
            d_embed=args.d_embed,
            d_outer=args.d_outer,
            m_outer=args.M_outer,
            d_middle=args.d_middle,
            d_core=args.d_core,
            k=args.k,
            abb_mode=mode,
            abb_fullness_gate=float(args.cycle_abb_fullness_gate),
        )
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo),
            env=env,
            timeout=int(args.timeout_s),
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2500:]
            print(f"[fail] mode={mode!r} rc={proc.returncode}\n{tail}", file=sys.stderr)
            raise SystemExit(1)
        row = parse_last_gomb_json_line(proc.stdout)
        row["_compare_mode"] = mode
        rows.append(row)
        if args.jsonl_out:
            outp = Path(args.jsonl_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    base = rows[0]
    keys = [
        "wall_s",
        "n_cycles",
        "n_params",
        "val_auc_best",
        "val_auroc",
        "test_auroc",
        "infer_wall_s",
        "infer_edges_per_s",
    ]
    print()
    hdr = "| mode | " + " | ".join(keys) + " |"
    sep = "| --- | " + " | ".join(["---" for _ in keys]) + " |"
    print(hdr)
    print(sep)
    for r in rows:
        cells = [str(r.get("_compare_mode"))]
        for k in keys:
            v = r.get(k)
            if isinstance(v, float):
                cells.append(f"{v:.6g}" if abs(v) < 1e4 else f"{v:.4g}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")

    if len(rows) > 1:
        print()
        print("Deltas vs first row (`" + str(base.get("_compare_mode")) + "`):")
        for r in rows[1:]:
            print(f"  mode={r.get('_compare_mode')!r}")
            for k in keys:
                vb, va = base.get(k), r.get(k)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    print(
                        f"    {k}: {float(vb):.6g} -> {float(va):.6g}  "
                        f"(Δ {float(va) - float(vb):+.6g})"
                    )


if __name__ == "__main__":
    main()
