"""cProfile wrapper around run_final_cell.main() that adds live stage
markers via monkey-patched wrappers, so we get progress signal during
the ~115 min Epinions production-config run.

Usage (matches run_adaptive_mv_5seed_2026_05_10.sh "fixed" variant):
    python -m signedkan_wip.src.profile_full_run

Outputs:
    /tmp/profile_cprofile_2026_05_10/cprofile.prof   - cProfile stats
    /tmp/profile_cprofile_2026_05_10/run.log         - stdout+stderr (stage markers)

Env vars are read from os.environ as the underlying run_final_cell
expects. Set them in the launching shell or before calling main().
"""
from __future__ import annotations
import cProfile
import pstats
import time
import sys
import gc
import resource
import os


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _mark(label: str) -> None:
    print(f"[STAGE {time.strftime('%H:%M:%S')}  rss={_rss_gb():.2f}GB]  {label}",
          flush=True)


def _wrap(module, name: str, label: str, key_fields: tuple = ()):
    """Wrap module.name with stage prints. key_fields is a list of
    kwarg names whose values get appended to the label (works whether
    the caller passes them positionally or by keyword)."""
    orig = getattr(module, name)
    import inspect
    sig = inspect.signature(orig)
    param_names = list(sig.parameters.keys())

    def wrapper(*args, **kw):
        suffix = []
        for fld in key_fields:
            if fld in kw:
                suffix.append(f"{fld}={kw[fld]}")
            else:
                try:
                    idx = param_names.index(fld)
                    if idx < len(args):
                        suffix.append(f"{fld}={args[idx]}")
                except ValueError:
                    pass
        full = label + (f" ({', '.join(suffix)})" if suffix else "")
        _mark(f">>> ENTER {full}")
        gc.collect()
        t0 = time.perf_counter()
        out = orig(*args, **kw)
        dt = time.perf_counter() - t0
        n = len(out) if hasattr(out, "__len__") else "n/a"
        _mark(f"<<< EXIT  {full}  dt={dt:.2f}s  n={n}")
        return out

    wrapper.__wrapped__ = orig
    setattr(module, name, wrapper)


def install_stage_markers():
    """Patch the enumeration entry points with stage markers."""
    from . import n_tuples, walks, hyperedges
    _wrap(n_tuples, "construct_2", "construct_2 (k=2)")
    _wrap(n_tuples, "construct_k", "construct_k", ("k", "max_cycles", "seed"))
    _wrap(walks, "construct_walks", "construct_walks", ("walk_len", "max_walks", "seed"))
    _wrap(hyperedges, "construct", "hyperedges.construct (k=3 Python)")


def run():
    _mark("=== profile_full_run starting ===")
    _mark(f"argv={sys.argv}")
    install_stage_markers()

    # Import after patching so run_final_cell uses the wrapped functions.
    from . import run_final_cell
    _mark(">>> calling run_final_cell.main()")
    t0 = time.perf_counter()
    run_final_cell.main()
    _mark(f"<<< run_final_cell.main() returned, total={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    PROFOUT = os.environ.get("CPROFILE_OUT",
                             "/tmp/profile_cprofile_2026_05_10/cprofile.prof")
    os.makedirs(os.path.dirname(PROFOUT), exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run()
    finally:
        profiler.disable()
        profiler.dump_stats(PROFOUT)
        _mark(f"=== cProfile stats written to {PROFOUT} ===")

        # Print top 50 cumulative-time entries inline.
        st = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
        _mark("=== top 50 cumulative ===")
        st.print_stats(50)
        _mark("=== top 50 tottime ===")
        st.sort_stats("tottime").print_stats(50)
