"""Analyze a cProfile .prof file with views tuned for finding the
missing 100 min in the Epinions production run.

Usage:
    python -m signedkan_wip.src.analyze_cprofile /path/to/cprofile.prof
"""
from __future__ import annotations
import pstats
import sys
from collections import defaultdict


def by_file(stats_obj: pstats.Stats, min_tot: float = 1.0):
    """Aggregate cumulative time by source file (path containing
    'signedkan_wip' or top-level libraries we care about)."""
    by_file_cum: dict[str, float] = defaultdict(float)
    by_file_tot: dict[str, float] = defaultdict(float)
    for func, (cc, nc, tt, ct, callers) in stats_obj.stats.items():
        path, _line, _name = func
        if path == "~":
            key = "<built-in>"
        else:
            # Bucket by short module path
            for needle in ("signedkan_wip/", "torch/", "triton/", "numpy/",
                            "pytorch", "rayon", "site-packages/"):
                if needle in path:
                    key = needle.rstrip("/")
                    if needle == "signedkan_wip/":
                        key = path.split("signedkan_wip/")[-1]
                    elif needle == "site-packages/":
                        # Bucket per-package within site-packages.
                        rest = path.split("site-packages/")[-1]
                        key = "pkg:" + rest.split("/")[0]
                    break
            else:
                key = path
        by_file_cum[key] += ct
        by_file_tot[key] += tt
    rows = sorted(by_file_tot.items(), key=lambda kv: -kv[1])
    print(f"\n{'─'*70}\nBy source file/module — sorted by tottime (>{min_tot}s)\n{'─'*70}")
    print(f"{'tottime':>10s}  {'cumtime':>10s}  module")
    for k, v in rows:
        if v < min_tot:
            continue
        print(f"{v:10.2f}  {by_file_cum[k]:10.2f}  {k}")


def main(prof_path: str):
    st = pstats.Stats(prof_path)
    st_strip = pstats.Stats(prof_path).strip_dirs()
    print(f"# Profile: {prof_path}")
    print(f"# Total functions: {len(st.stats)}")
    print(f"# Total time     : {st.total_tt:.2f}s")
    print(f"# Total calls    : {st.total_calls}")

    print(f"\n{'─'*70}\nTop 40 by cumulative time (whole-tree)\n{'─'*70}")
    st_strip.sort_stats("cumulative").print_stats(40)

    print(f"\n{'─'*70}\nTop 40 by tottime (self-time, dominant inner workers)\n{'─'*70}")
    st_strip.sort_stats("tottime").print_stats(40)

    by_file(st, min_tot=1.0)

    # Specific functions of interest from run_final_cell preprocessing:
    print(f"\n{'─'*70}\nrun_final_cell.cell_signed_graph hot-spots (filtered)\n{'─'*70}")
    st_strip.sort_stats("tottime").print_stats(
        "run_final_cell|build_me|encode_edges|construct|forward|backward")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/tmp/profile_cprofile_2026_05_10/cprofile.prof")
