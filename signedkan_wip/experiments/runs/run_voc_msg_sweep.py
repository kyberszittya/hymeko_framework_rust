"""HyMeYOLO Pascal VOC 2007 sweep driver — Phase 15 (2026-05-20).

Parallel structure to the HSIKAN / Gömb / cortical sweep drivers.
Drives `data/hsikan/sweep_msg_voc.hymeko` via the Rust
``hymeko_pgraph_dump`` binary, lifts ABB-selected operating units
onto ``train_voc_stagec`` kwargs via
:mod:`signedkan_wip.src.voc_pgraph_mapping`, and (with ``--train``)
runs a VOC training cell.

Dry-run by default. Use ``--train`` to actually invoke the trainer;
expect ~60 s per cell at the smoke defaults (100 images, 3 epochs,
input_size=128) or ~15 min at production scale.

Mirrors the field-by-field interface of
``run_cortical_msg_sweep.py``: emits a JSONL row per (selection,
seed) with the Friedler certificate stamped on, ready for
downstream filtering via the recipe in
``docs/book/src/recipes/filter-by-friedler-certificate.md``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PGRAPH = REPO_ROOT / "data" / "hsikan" / "sweep_msg_voc.hymeko"


def _find_dump_binary(repo: Path) -> Path:
    rel = repo / "target" / "release" / "hymeko_pgraph_dump"
    dbg = repo / "target" / "debug" / "hymeko_pgraph_dump"
    if rel.exists():
        return rel
    if dbg.exists():
        return dbg
    subprocess.run(
        ["cargo", "build", "-p", "hymeko_pgraph",
         "--bin", "hymeko_pgraph_dump"],
        cwd=str(repo), check=True,
    )
    return rel if rel.exists() else dbg


def run_pgraph_dump(
    repo: Path,
    pgraph_path: Path,
    algorithm: str,
    relaxed_msg: bool = False,
    weights: str | None = None,
) -> dict[str, Any]:
    bin_path = _find_dump_binary(repo)
    cmd = [str(bin_path), str(pgraph_path), "--algorithm", algorithm]
    if relaxed_msg:
        cmd.append("--relaxed-msg")
    if weights:
        cmd.extend(["--weights", weights])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo))
    if proc.returncode not in (0, 2):
        raise RuntimeError(
            f"hymeko_pgraph_dump failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if not proc.stdout.strip():
        raise RuntimeError("hymeko_pgraph_dump produced empty stdout")
    return json.loads(proc.stdout)


def _selected_unit_sets(analysis: dict[str, Any], algorithm: str) -> list[list[str]]:
    if algorithm == "abb":
        abb = analysis.get("abb")
        if not abb or not abb.get("units"):
            return []
        return [list(abb["units"])]
    if algorithm == "ssg":
        return list(analysis.get("ssg_structures") or [])
    return []


def _format_cert(cert: dict[str, Any] | None, label: str) -> str:
    if cert is None:
        return f"  {label}: n/a"
    status = cert.get("status", "?")
    if status == "PASS":
        return f"  {label}: PASS"
    tags = ",".join(cert.get("violation_tags", []))
    return f"  {label}: FAIL [{tags}]"


def _run_one_voc_seed(*, seed: int, kw: dict[str, Any]) -> dict[str, Any]:
    """Drive the VOC trainer for one seed under the supplied
    (mapped) kwargs. The trainer only emits JSON when
    `--jsonl-out PATH` is given; we use a tempfile and parse it.
    """
    import tempfile
    import time
    fd, tmp_path = tempfile.mkstemp(prefix="voc_pgraph_", suffix=".jsonl")
    import os
    os.close(fd)
    jsonl_tmp = Path(tmp_path)
    cmd = [
        sys.executable, "-m", "signedkan_wip.src.vision.train_voc_stagec",
        "--year",       kw["year"],
        "--image-set",  kw["image_set"],
        "--input-size", str(kw["input_size"]),
        "--max-objects", str(kw["max_objects"]),
        "--n-box-queries", str(kw["n_box_queries"]),
        "--epochs",     str(kw["epochs"]),
        "--lr",         str(kw["lr"]),
        "--batch-size", str(kw["batch_size"]),
        "--seed",       str(seed),
        "--backbone",   kw["backbone"],
        "--lam-no-obj", str(kw["lam_no_obj"]),
        "--query-head-kind", kw["query_head_kind"],
        "--schedule",   kw["schedule"],
        "--warmup-epochs", str(kw["warmup_epochs"]),
        "--ricci-scale", str(kw["ricci_scale"]),
        "--jsonl-out",  str(jsonl_tmp),
    ]
    if kw.get("n_images") is not None:
        cmd.extend(["--n-images", str(kw["n_images"])])
    if kw.get("device"):
        cmd.extend(["--device", str(kw["device"])])
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    wall = time.time() - t0
    if proc.returncode != 0:
        try:
            jsonl_tmp.unlink()
        except OSError:
            pass
        return {
            "error": "trainer rc != 0",
            "rc": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-2000:],
            "wall_s_outer": round(wall, 1),
        }
    out: dict[str, Any] | None = None
    try:
        with jsonl_tmp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out = json.loads(line)  # last valid line wins
                except Exception:  # noqa: BLE001
                    continue
    except FileNotFoundError:
        pass
    finally:
        try:
            jsonl_tmp.unlink()
        except OSError:
            pass
    if out is None:
        return {"error": "trainer produced no JSONL output",
                "stdout_tail": proc.stdout[-1500:],
                "wall_s_outer": round(wall, 1)}
    out["wall_s_outer"] = round(wall, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgraph", type=Path, default=DEFAULT_PGRAPH)
    ap.add_argument("--algorithm", choices=("msg", "ssg", "abb"), default="abb")
    ap.add_argument("--relaxed-msg", action="store_true")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true",
                    help="Actually run the VOC trainer on the selected "
                         "architecture(s). Otherwise dry-run.")
    # Smoke defaults: tiny so the wiring runs in ~60 s.
    ap.add_argument("--smoke", action="store_true",
                    help="Apply tiny smoke defaults: 100 images, 3 epochs, "
                         "input_size=128. Otherwise use the trainer's "
                         "production defaults (5011 images, 30 epochs, "
                         "input_size=224 — ~15 min/seed).")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    analysis = run_pgraph_dump(
        REPO_ROOT, args.pgraph, args.algorithm,
        relaxed_msg=args.relaxed_msg, weights=args.weights,
    )
    print(json.dumps({"phase": "pgraph_analysis", **analysis},
                     indent=2, default=str))
    print(_format_cert(analysis.get("canonical_full"), "canonical full"),
          file=sys.stderr)
    print(_format_cert(analysis.get("extension_full"), "extension full"),
          file=sys.stderr)
    print(_format_cert(analysis.get("canonical_abb_subschema"),
                       "canonical ABB"), file=sys.stderr)
    print(_format_cert(analysis.get("extension_abb_subschema"),
                       "extension ABB"), file=sys.stderr)

    if not analysis.get("ok"):
        sys.exit(2)
    selections = _selected_unit_sets(analysis, args.algorithm)
    if not selections:
        print(f"\nno concrete selection for algorithm={args.algorithm}",
              file=sys.stderr)
        return

    from signedkan_wip.src.voc_pgraph_mapping import (
        merge_structure_knobs, train_voc_kwargs,
    )
    base: dict[str, Any] = {}
    if args.smoke:
        base.update({
            "n_images": 100,
            "epochs": 3,
            "input_size": 128,
            "batch_size": 4,
            "warmup_epochs": 1,
        })
    rows: list[dict[str, Any]] = []
    for sel in selections:
        merged = merge_structure_knobs(sel)
        kw = train_voc_kwargs(seed=args.seed, structure=merged, base=base)
        row: dict[str, Any] = {
            "pgraph_units": sel,
            "merged_structure": merged,
            "train_voc_kwargs": kw,
            "canonical_abb_status": (
                analysis.get("canonical_abb_subschema", {}).get("status")
                if analysis.get("canonical_abb_subschema") else None
            ),
            "extension_abb_status": (
                analysis.get("extension_abb_subschema", {}).get("status")
                if analysis.get("extension_abb_subschema") else None
            ),
            "strict_no_excess": analysis.get("strict_no_excess", True),
        }
        if args.train:
            print(f"\n[training] units={sel}", file=sys.stderr)
            r = _run_one_voc_seed(seed=args.seed, kw=kw)
            row["train_result"] = r
            mAP = r.get("mAP_50") if isinstance(r, dict) else None
            print(f"  → mAP_50={mAP}  wall={r.get('wall_s_outer')}s",
                  file=sys.stderr)
        rows.append(row)
        print(f"\nselection {sel}: kwargs={kw}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        print(f"\nwrote {len(rows)} rows → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
