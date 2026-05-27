"""GömbSoma cortical-benchmark sweep driver — Phase 12 (2026-05-20).

Parses ``data/hsikan/sweep_msg_cortical.hymeko`` via the Rust
``hymeko_pgraph_dump`` binary, lifts the ABB-selected operating
units onto :class:`CorticalBenchmarkExperiment.run_seed` kwargs via
:mod:`signedkan_wip.src.cortical_pgraph_mapping`, and optionally
runs the Slice-1 cortical benchmark on each selection.

Use::

    PYTHONPATH=$PWD python -m \\
        signedkan_wip.experiments.runs.run_cortical_msg_sweep \\
        --pgraph data/hsikan/sweep_msg_cortical.hymeko \\
        --algorithm abb

By default the driver does **not** run the benchmark: it emits the
P-graph analysis + the mapped kwargs + the canonical / extension
Friedler certificate. Pass ``--train`` to actually run the
cortical pipeline (Slice 1's
:class:`CorticalBenchmarkExperiment`).

Parallel in shape to ``run_hsikan_msg_sweep.py`` /
``run_gomb_msg_sweep.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PGRAPH = REPO_ROOT / "data" / "hsikan" / "sweep_msg_cortical.hymeko"


def _find_dump_binary(repo: Path) -> Path:
    dbg = repo / "target" / "debug" / "hymeko_pgraph_dump"
    rel = repo / "target" / "release" / "hymeko_pgraph_dump"
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
    # No-excess regime is the deliberate default for the cortical sweeps
    # (see run_gomb_msg_sweep). The engine defaults to canonical since
    # 2026-05-27, so request it explicitly unless --relaxed-msg.
    if relaxed_msg:
        cmd.append("--relaxed-msg")
    else:
        cmd.append("--strict-no-excess")
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


def _resolve_binning(name: str):
    """Translate the structure-derived binning flag into a concrete
    :class:`BinningConfig`."""
    from signedkan_wip.src.cortical import BinningConfig
    if name == "shallow":
        return BinningConfig(bins_per_depth={0: (2, 2)})
    return BinningConfig(bins_per_depth={0: (2, 2), 1: (4, 4), 2: (8, 8)})


def _build_backbone(kind: str, *, image_size: int, in_channels: int,
                    d_hidden: int, binning):
    """Return a feature-extractor module for the chosen backbone.

    * `kind == "resnet"`: Slice-1 baseline `ResNetTinyCortical`.
    * `kind == "gomb"`: hypergraph-machine `RicciStimBackbone`
      wrapped in `CorticalFeatureExtractor` (Phase 12.5
      hypergraph-CV branch).
    """
    from signedkan_wip.src.cortical import (
        CorticalFeatureExtractor, ResNetTinyCortical,
    )
    if kind == "resnet":
        return ResNetTinyCortical(
            image_h=image_size, image_w=image_size,
            in_channels=in_channels, d_hidden=d_hidden,
            binning_config=binning,
        )
    if kind == "gomb":
        from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_backbone import (
            RicciStimBackbone,
        )
        rb = RicciStimBackbone(
            image_h=image_size, image_w=image_size,
            in_channels=in_channels, d_hidden=d_hidden,
            max_depth=2, max_anchors=256,
        )
        return CorticalFeatureExtractor(
            backbone=rb, image_h=image_size, image_w=image_size,
            d_hidden=d_hidden, binning_config=binning,
        )
    raise ValueError(f"unknown backbone: {kind!r}")


def _run_one_benchmark_seed(*, seed: int, kw: dict[str, Any]) -> dict[str, Any]:
    """Drive the Slice-1 cortical benchmark for a single seed under
    the supplied (mapped) kwargs. The `backbone` kwarg selects
    `resnet` (Slice 1 baseline) or `gomb` (Phase-12.5 hypergraph
    machine).
    """
    import torch
    from signedkan_wip.src.cortical import (
        BrainScorer, make_synthetic_cichy_like, score_all_rois,
    )
    binning = _resolve_binning(kw["binning"])
    torch.manual_seed(seed)
    ds = make_synthetic_cichy_like(
        n_images=kw["n_images"], n_subjects=kw["n_subjects"],
        image_size=kw["image_size"], in_channels=kw["in_channels"],
        snr=kw["snr"], seed=seed,
    )
    model = _build_backbone(
        kw.get("backbone", "resnet"),
        image_size=kw["image_size"], in_channels=kw["in_channels"],
        d_hidden=kw["d_hidden"], binning=binning,
    )
    model.eval()
    with torch.no_grad():
        features = model.extract_batch(ds.images)
    scorer = BrainScorer(
        n_pls_components=kw["n_pls_components"],
        n_cv_folds=kw["n_cv_folds"], seed=seed,
    )
    scores = score_all_rois(scorer, features, ds.roi_signals)
    return {
        "seed": seed,
        "backbone": kw.get("backbone", "resnet"),
        "d_hidden": kw["d_hidden"],
        "binning": kw["binning"],
        "n_pls_components": kw["n_pls_components"],
        **{f"{roi}_r2": s.r_squared for roi, s in scores.items()},
        **{f"{roi}_corrected": s.noise_ceiling_corrected
           for roi, s in scores.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgraph", type=Path, default=DEFAULT_PGRAPH)
    ap.add_argument("--algorithm", choices=("msg", "ssg", "abb"), default="abb")
    ap.add_argument("--relaxed-msg", action="store_true")
    ap.add_argument("--weights", type=str, default=None,
                    help="Multi-cost weights (when fixture carries them)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true",
                    help="Actually run the cortical benchmark for the "
                         "selected architecture(s)")
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

    from signedkan_wip.src.cortical_pgraph_mapping import (
        merge_structure_knobs, benchmark_kwargs,
    )
    rows: list[dict[str, Any]] = []
    for sel in selections:
        merged = merge_structure_knobs(sel)
        kw = benchmark_kwargs(seed=args.seed, structure=merged, base={})
        row: dict[str, Any] = {
            "pgraph_units": sel,
            "merged_structure": merged,
            "benchmark_kwargs": kw,
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
            try:
                import time
                t0 = time.time()
                bench = _run_one_benchmark_seed(seed=args.seed, kw=kw)
                row["benchmark_result"] = bench
                row["benchmark_wall_s"] = round(time.time() - t0, 2)
                print(f"\n[trained] units={sel} bench={bench}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                row["benchmark_error"] = repr(exc)
                print(f"\n[error] units={sel} {exc!r}", file=sys.stderr)
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
