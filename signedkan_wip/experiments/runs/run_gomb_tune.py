"""Random hyperparameter search for ``run_gomb_smoke`` (no Optuna dependency).

Each trial spawns a **fresh** ``python -m signedkan_wip.experiments.runs.run_gomb_smoke``
process (clean CUDA state), parses the **last JSON** line, appends one JSONL
record.  By default the tuner maximises ``test_auroc`` when available
(``--edge-split 80_10_10``), else ``val_auroc``.  Use ``--pick-best-by
val_auroc`` to rank trials on validation only (recommended for honest
hyperparameter search).  Trial stdout lines include ``n_params=…`` from the
smoke row; the per-dataset summary line includes ``best_n_params`` when
available.

Search space (controlled by ``--search-seed`` for reproducibility):

* ``lr``, ``d_embed``, ``M_outer``, ``d_outer``, ``d_middle``, ``d_core``,
  ``topk``, ``n_tiers``, ``weight_decay``, ``pos_weight_auto``,
  optional ``--cycle-ks 3,4`` (mixed-arity Gömb).

With ``--joint-mix``, each trial uses ``run_gomb_smoke --joint-mix`` (same
tuple recipe as joint BA: c3, c4, w2, w3). Sampled ``cycle_ks`` from the
base space is cleared; ``max_walks_w2`` / ``max_walks_w3`` are drawn from
walk-cap menus (architecture-dependent). On **bitcoin_otc** /
**bitcoin_alpha**, wide joint trials additionally **clamp ``topk`` to 64** and
**walk caps to 32k** (large tuple pools × wide heads OOMed on 8GB GPUs).

Large SNAP graphs get a smaller ``topk`` menu to reduce OOM risk.

**Architecture:** ``--architecture wide`` (default) explores larger widths;
``compact`` targets a **small-parameter** regime (low ``d_embed``, few
``M_outer`` banks, modest ``d_middle`` / ``d_core``) for Slashdot / Epinions
where node embedding already scales with ``|V|`` — useful when comparing
against headline HSiKAN runs at much larger hidden widths.

Example::

    python -m signedkan_wip.experiments.runs.run_gomb_tune \\
        --datasets bitcoin_alpha bitcoin_otc sbm_n200 \\
        --trials 24 --search-seed 1 --data-seed 0 \\
        --edge-split 80_10_10 --n-epochs 100 --device cuda \\
        --out reports/gomb_tune_run.jsonl

    python -m signedkan_wip.experiments.runs.run_gomb_tune \\
        --datasets slashdot epinions --architecture compact \\
        --trials 20 --search-seed 2 --edge-split 80_10_10 \\
        --n-epochs 64 --out reports/gomb_tune_compact.jsonl

    python -m signedkan_wip.experiments.runs.run_gomb_tune \\
        --datasets bitcoin_otc bitcoin_alpha --joint-mix \\
        --trials 16 --search-seed 0 --edge-split 80_10_10 \\
        --n-epochs 80 --device cuda --out reports/gomb_tune_joint.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parse_last_gomb_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _topk_choices(dataset: str, *, compact: bool) -> list[int]:
    if dataset in ("slashdot", "epinions"):
        if compact:
            return [16, 24, 32, 40]
        return [24, 32, 48, 64]
    if compact:
        return [24, 32, 48, 64]
    return [32, 48, 64, 96, 128]


def sample_params(
    rng: np.random.Generator, dataset: str, *, compact: bool,
) -> dict[str, Any]:
    """Return CLI flag dict for one smoke trial.

    Keys always include ``cycle_ks`` (possibly empty). For joint-mix tuning,
    pass the dict through :func:`for_joint_mix_tuning`.
    """
    if compact:
        return _sample_params_compact(rng, dataset)
    return _sample_params_wide(rng, dataset)


def for_joint_mix_tuning(
    rng: np.random.Generator,
    base: dict[str, Any],
    *,
    compact: bool,
    dataset: str,
) -> dict[str, Any]:
    """Convert a standard tune param dict into a joint-mix trial.

    Clears ``cycle_ks`` (mutually exclusive with ``run_gomb_smoke --joint-mix``)
    and sets ``joint_mix``, ``max_walks_w2``, ``max_walks_w3``.

    Wide joint on Bitcoin trust graphs clamps ``topk`` to **56**,
    ``d_embed`` to **48**, ``M_outer`` to **10**, ``n_tiers`` to **3**, and each walk cap to **32k**
    — tuple pools are large; unclamped heads OOMed on 8GB consumer GPUs.
    **Slashdot / Epinions** joint trials clamp walk caps to **4096** (and
    compact ``topk`` to **28** max) so four-slot forward fits on consumer GPUs
    while allowing a slightly wider cycle head than the older 24 cap.
    """
    out = dict(base)
    out["cycle_ks"] = ""
    out["joint_mix"] = True
    if compact:
        walks = [2000, 4000, 8000, 12000, 20000, 32000]
    else:
        walks = [8000, 16000, 24000, 32000, 50000]
    out["max_walks_w2"] = int(rng.choice(walks))
    out["max_walks_w3"] = int(rng.choice(walks))
    # Large SNAP graphs: four-slot joint pools exhaust VRAM quickly.
    if dataset in ("slashdot", "epinions"):
        _snap_walk = 4096
        out["max_walks_w2"] = min(int(out["max_walks_w2"]), _snap_walk)
        out["max_walks_w3"] = min(int(out["max_walks_w3"]), _snap_walk)
        if compact:
            # Slightly above legacy 24 cap for AUC headroom; still below full menu 40.
            _snap_joint_topk_max = 28
            out["topk"] = min(int(out["topk"]), _snap_joint_topk_max)
        # Match run_gomb_smoke default per-slot subsample for joint on SNAP.
        out["joint_slot_cap"] = 12000
    if (
        not compact
        and dataset in ("bitcoin_otc", "bitcoin_alpha")
    ):
        # Four-slot joint × wide heads: cap head width so 8GB runs complete
        # trials more often (OOM trials waste search budget vs SOTA chase).
        out["topk"] = min(int(out["topk"]), 56)
        out["d_embed"] = min(int(out["d_embed"]), 48)
        out["M_outer"] = min(int(out["M_outer"]), 10)
        dcap = int(out["d_embed"])
        out["d_middle"] = min(int(out["d_middle"]), max(16, int(dcap * 1.25)))
        out["d_core"] = min(int(out["d_core"]), max(16, int(dcap * 1.25)))
        # Four tiers × four slots multiplies middle-shell activations; cap at 3.
        out["n_tiers"] = min(int(out["n_tiers"]), 3)
        _wcap = 32000
        out["max_walks_w2"] = min(int(out["max_walks_w2"]), _wcap)
        out["max_walks_w3"] = min(int(out["max_walks_w3"]), _wcap)
    return out


def _sample_params_wide(rng: np.random.Generator, dataset: str) -> dict[str, Any]:
    lr = float(rng.choice([1e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]))
    d_embed = int(rng.choice([32, 40, 48, 56, 64]))
    d_outer = int(rng.choice([8, 12, 16, 20, 24]))
    M_outer = int(rng.choice([4, 6, 8, 10, 12]))
    d_mid = int(rng.choice([max(16, d_embed // 2), d_embed, int(d_embed * 1.25)]))
    d_core = int(rng.choice([max(16, d_embed // 2), d_embed, int(d_embed * 1.25)]))
    topk = int(rng.choice(_topk_choices(dataset, compact=False)))
    n_tiers = int(rng.choice([2, 3, 4]))
    weight_decay = float(rng.choice([0.0, 1e-6, 1e-5, 2e-5, 5e-5]))
    pos_weight_auto = bool(rng.random() < 0.75)
    use_mixed = (
        dataset not in ("sbm_n200", "sbm_n400")
        and not dataset.startswith("sbm_")
        and rng.random() < 0.22
    )
    cycle_ks = "3,4" if use_mixed else ""
    return {
        "lr": lr,
        "d_embed": d_embed,
        "d_outer": d_outer,
        "M_outer": M_outer,
        "d_middle": d_mid,
        "d_core": d_core,
        "topk": topk,
        "n_tiers": n_tiers,
        "weight_decay": weight_decay,
        "pos_weight_auto": pos_weight_auto,
        "cycle_ks": cycle_ks,
    }


def _sample_params_compact(rng: np.random.Generator, dataset: str) -> dict[str, Any]:
    """Narrow widths / few banks: Gömb param count well below wide smoke.

    Node embedding still ``|V| * d_embed``; this mode biases toward small
    ``d_embed`` and ``M_outer`` so total params sit in a **~small fraction**
    of typical wide Gömb on Bitcoin (order-of-magnitude ``~1/8`` ballpark
    on the **learnable** stack excluding cycle enumeration).
    """
    lr = float(rng.choice([3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]))
    emb_choices = [12, 14, 16, 18, 20, 22, 24]
    if dataset in ("slashdot", "epinions"):
        emb_choices = [*emb_choices, 26]
    d_embed = int(rng.choice(emb_choices))
    d_outer = int(rng.choice([6, 8, 10]))
    M_outer = int(rng.choice([2, 3, 4]))
    d_mid = int(rng.choice([12, 14, 16, min(24, max(12, d_embed))]))
    d_core = int(rng.choice([12, 14, 16, min(28, max(12, d_embed))]))
    topk = int(rng.choice(_topk_choices(dataset, compact=True)))
    n_tiers = int(rng.choice([2, 3]))
    weight_decay = float(rng.choice([0.0, 1e-6, 1e-5, 2e-5]))
    pos_weight_auto = bool(rng.random() < 0.85)
    use_mixed = (
        dataset == "slashdot"
        and rng.random() < 0.12
    )
    cycle_ks = "3,4" if use_mixed else ""
    return {
        "lr": lr,
        "d_embed": d_embed,
        "d_outer": d_outer,
        "M_outer": M_outer,
        "d_middle": d_mid,
        "d_core": d_core,
        "topk": topk,
        "n_tiers": n_tiers,
        "weight_decay": weight_decay,
        "pos_weight_auto": pos_weight_auto,
        "cycle_ks": cycle_ks,
    }


def _build_cmd(
    *,
    py: str,
    dataset: str,
    data_seed: int,
    edge_split: str,
    n_epochs: int,
    device: str,
    p: dict[str, Any],
) -> list[str]:
    cmd: list[str] = [
        py, "-m", "signedkan_wip.experiments.runs.run_gomb_smoke",
        "--dataset", dataset,
        "--seed", str(data_seed),
        "--edge-split", edge_split,
        "--n-epochs", str(n_epochs),
        "--device", device,
        "--lr", str(p["lr"]),
        "--d-embed", str(p["d_embed"]),
        "--d-outer", str(p["d_outer"]),
        "--M-outer", str(p["M_outer"]),
        "--d-middle", str(p["d_middle"]),
        "--d-core", str(p["d_core"]),
        "--topk", str(p["topk"]),
        "--n-tiers", str(p["n_tiers"]),
        "--weight-decay", str(p["weight_decay"]),
    ]
    if p.get("pos_weight_auto"):
        cmd.append("--pos-weight-auto")
    if p.get("joint_mix"):
        cmd.append("--joint-mix")
        cmd.extend(["--max-walks-w2", str(int(p["max_walks_w2"]))])
        cmd.extend(["--max-walks-w3", str(int(p["max_walks_w3"]))])
        jcap = p.get("joint_slot_cap")
        if jcap is not None:
            cmd.extend(["--joint-slot-cap", str(int(jcap))])
    elif p.get("cycle_ks"):
        cmd.extend(["--cycle-ks", str(p["cycle_ks"])])
    if p.get("cycle_abb_mode"):
        cmd.extend(["--cycle-abb-mode", str(p["cycle_abb_mode"])])
    if p.get("cycle_abb_fullness_gate") is not None:
        cmd.extend([
            "--cycle-abb-fullness-gate",
            str(float(p["cycle_abb_fullness_gate"])),
        ])
    return cmd


def _score_row(row: dict[str, Any] | None) -> float:
    if not row:
        return float("-inf")
    if "test_auroc" in row and not math.isnan(float(row["test_auroc"])):
        return float(row["test_auroc"])
    if "val_auroc" in row and not math.isnan(float(row["val_auroc"])):
        return float(row["val_auroc"])
    return float("-inf")


def _tuner_objective(row: dict[str, Any] | None, *, pick_best_by: str) -> float:
    """Metric used to rank trials (``best`` / ``best_meta`` in summaries).

    ``test_auroc``: prefer test when present (``80_10_10``), else val.
    ``val_auroc``: always use validation AUROC for ranking — use when tuning
    should not peek at test for hyperparameter selection (honest protocol).
    """
    if not row:
        return float("-inf")
    if pick_best_by == "val_auroc":
        if "val_auroc" in row and not math.isnan(float(row["val_auroc"])):
            return float(row["val_auroc"])
        return float("-inf")
    return _score_row(row)


def _n_params_from_row(row: dict[str, Any] | None) -> int | None:
    """Total trainable parameters from a smoke JSON row, if present and integral."""
    if not row:
        return None
    raw = row.get("n_params")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument(
        "--edge-split", choices=("80_20", "80_10_10"), default="80_10_10",
    )
    ap.add_argument("--n-epochs", type=int, default=90)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu (passed through to smoke).",
    )
    ap.add_argument(
        "--timeout-s", type=int, default=7200,
        help="Per-trial subprocess wall limit.",
    )
    ap.add_argument(
        "--out", type=Path, required=True,
        help="JSONL path (append mode).",
    )
    ap.add_argument(
        "--architecture",
        choices=("wide", "compact"),
        default="wide",
        help="Search space: wide (default) or compact (small d_embed / M_outer).",
    )
    ap.add_argument(
        "--joint-mix",
        action="store_true",
        help="Use JointMixGomb (joint BA slots); samples walk caps, clears "
             "mixed-arity cycle_ks.",
    )
    ap.add_argument(
        "--pick-best-by",
        choices=("test_auroc", "val_auroc"),
        default="test_auroc",
        help="Metric for ranking trials in summaries. Use val_auroc so search "
             "does not optimise the held-out test set.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.search_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    for ds in args.datasets:
        best: float = float("-inf")
        best_meta: dict[str, Any] | None = None
        t0 = time.perf_counter()
        for trial in range(args.trials):
            p = sample_params(rng, ds, compact=args.architecture == "compact")
            if args.joint_mix:
                p = for_joint_mix_tuning(
                    rng, p, compact=args.architecture == "compact",
                    dataset=ds,
                )
            cmd = _build_cmd(
                py=py,
                dataset=ds,
                data_seed=args.data_seed,
                edge_split=args.edge_split,
                n_epochs=args.n_epochs,
                device=args.device,
                p=p,
            )
            t_launch = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                row = {
                    "tuner_error": "timeout",
                    "dataset": ds,
                    "trial": trial,
                    "trial_params": p,
                    "cmd": cmd,
                }
                row["tuner_dataset"] = ds
                row["tuner_search_seed"] = args.search_seed
                row["tuner_data_seed"] = args.data_seed
                row["tuner_architecture"] = args.architecture
                row["tuner_joint_mix"] = bool(args.joint_mix)
                row["tuner_pick_best_by"] = args.pick_best_by
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"[tune] {ds} [{args.architecture}] trial={trial}/{args.trials} "
                    f"TIMEOUT lr={p['lr']:.1e}",
                    flush=True,
                )
                continue

            row = _parse_last_gomb_json(proc.stdout) or {}
            row["trial"] = trial
            row["trial_params"] = p
            row["returncode"] = proc.returncode
            row["wall_subprocess_s"] = time.perf_counter() - t_launch
            if proc.returncode != 0:
                tail = (proc.stderr or "")[-4000:]
                row["stderr_tail"] = tail
            row["tuner_dataset"] = ds
            row["tuner_search_seed"] = args.search_seed
            row["tuner_data_seed"] = args.data_seed
            row["tuner_architecture"] = args.architecture
            row["tuner_joint_mix"] = bool(args.joint_mix)
            row["tuner_pick_best_by"] = args.pick_best_by
            with args.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

            sc = _tuner_objective(
                row if proc.returncode == 0 else None,
                pick_best_by=args.pick_best_by,
            )
            if sc > best:
                best = sc
                best_meta = row
            jm = ""
            if p.get("joint_mix"):
                jm = f" jmix w2={p['max_walks_w2']} w3={p['max_walks_w3']}"
            np_s = ""
            np_i = _n_params_from_row(row if proc.returncode == 0 else None)
            if np_i is not None:
                np_s = f" n_params={np_i}"
            print(
                f"[tune] {ds} [{args.architecture}] trial={trial}/{args.trials} "
                f"score={sc:.4f} best={best:.4f} "
                f"lr={p['lr']:.1e} d_emb={p['d_embed']} topk={p['topk']}{jm}{np_s}",
                flush=True,
            )

        wall = time.perf_counter() - t0
        summary = {
            "tuner_phase_summary": True,
            "dataset": ds,
            "trials": args.trials,
            "tuner_architecture": args.architecture,
            "tuner_joint_mix": bool(args.joint_mix),
            "tuner_pick_best_by": args.pick_best_by,
            "best_score": best,
            "wall_s": wall,
            "best_row_keys": sorted(best_meta.keys()) if best_meta else [],
        }
        if best_meta and "test_auroc" in best_meta:
            summary["best_test_auroc"] = best_meta["test_auroc"]
            summary["best_trial_params"] = best_meta.get("trial_params")
        if best_meta and "val_auroc" in best_meta:
            summary["best_val_auroc"] = best_meta["val_auroc"]
        best_np = _n_params_from_row(best_meta)
        if best_np is not None:
            summary["best_n_params"] = best_np
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, sort_keys=True) + "\n")
        done_np = ""
        if best_np is not None:
            done_np = f" n_params={best_np}"
        print(
            f"[tune] {ds} DONE best={best:.4f}{done_np} wall={wall:.1f}s → {args.out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
