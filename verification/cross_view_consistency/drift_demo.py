"""Drift-prevention demonstration: why cross-view consistency *by construction* matters.

Two regimes, one semantic edit (add a requirement realised by a component):

  (1) HyMeKo single-source. The edit is applied to the ONE .hymeko source; both views are re-emitted from the
      recompiled IR. The cross-view square still holds: X_sysml(eps_sysml(H')) == X_dot(eps_dot(H')), and the new
      requirement appears in BOTH views. Drift introduced: 0 (impossible by Proposition cross-view).

  (2) Pairwise / multi-file maintenance (the status quo without a canonical IR). The two views are independently
      maintained files; a human applies the edit to ONE of them (the SysML model) and --- as happens --- does not
      mirror it into the other (the DOT graph). The views now DISAGREE: the requirement is present in one view and
      absent from the other. Drift introduced: >= 1, with no mechanism to catch it.

The script asserts (1) drift == 0 and (2) drift >= 1, so the demonstration is real, not a strawman: the same edit
is benign under single-source emission and silently divergent under multi-file maintenance.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_witness import PAPER, TraceInvariant, _cli, extract_dot, extract_sysml  # noqa: E402

SRC = PAPER / "traceability_smc.hymeko"
META = PAPER / "meta_sysml_trace.hymeko"
NEW_REQ = "R_estop_latency"


def _emit_from_text(src_text: str, fmt: str) -> str:
    """Compile an arbitrary .hymeko source text (with its import available) and emit format f via the real CLI."""
    cli = _cli()
    if cli is None:
        raise RuntimeError("no hymeko CLI binary built")
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        shutil.copy(META, dd / "meta_sysml_trace.hymeko")     # make the relative import resolve
        f = dd / "variant.hymeko"
        f.write_text(src_text, encoding="utf-8")
        out = subprocess.run([str(cli), "emit", str(f), "--format", fmt, "--name", "pick_place_cell"],
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError(f"emit {fmt} failed: {out.stderr[:200]}")
        return out.stdout


def _add_requirement_to_source(src_text: str) -> str:
    """Semantic edit at the SOURCE: a new requirement realised by SafetyController (inserted before the body's end)."""
    inject = (f'    {NEW_REQ}: meta_sysml_trace.sysml_trace.elements.requirement '
              f'{{ text "E-stop latency is logged."; }}\n'
              f'    @sat_estop: meta_sysml_trace.sysml_trace.satisfies '
              f'{{ (+ SafetyController, - {NEW_REQ}); }}\n')
    idx = src_text.rstrip().rfind("}")
    return src_text[:idx] + inject + src_text[idx:]


def _add_requirement_to_sysml_text(sysml_text: str) -> str:
    """The SAME edit applied to ONE emitted file only (the multi-file-maintenance mistake)."""
    inject = f"    requirement def {NEW_REQ} {{\n        doc /* E-stop latency is logged. */\n    }}\n"
    idx = sysml_text.rstrip().rfind("}")
    return sysml_text[:idx] + inject + sysml_text[idx:]


def _drift(a: TraceInvariant, b: TraceInvariant) -> int:
    """Number of cross-view discrepancies between two views (symmetric differences over entities + relations)."""
    return sum(len(set(x) ^ set(y)) for x, y in (
        (a.requirements, b.requirements), (a.blocks, b.blocks),
        (a.satisfy, b.satisfy), (a.allocate, b.allocate), (a.derive, b.derive)))


def run() -> tuple[int, int, bool]:
    base_src = SRC.read_text(encoding="utf-8")

    # Baseline sanity: the unedited model is cross-view consistent.
    s0 = extract_sysml(_emit_from_text(base_src, "requirements_sysml"))
    d0 = extract_dot(_emit_from_text(base_src, "requirements_dot"))
    assert _drift(s0, d0) == 0, "unedited model is already inconsistent"

    # (1) HyMeKo: edit the SOURCE, re-emit BOTH views from the recompiled IR.
    edited_src = _add_requirement_to_source(base_src)
    s1 = extract_sysml(_emit_from_text(edited_src, "requirements_sysml"))
    d1 = extract_dot(_emit_from_text(edited_src, "requirements_dot"))
    hymeko_drift = _drift(s1, d1)
    new_in_both = NEW_REQ in s1.requirements and NEW_REQ in d1.requirements

    # (2) Pairwise: apply the SAME edit to ONE emitted file only; the other is untouched.
    s2 = extract_sysml(_add_requirement_to_sysml_text(_emit_from_text(base_src, "requirements_sysml")))
    d2 = d0  # the DOT file was not mirrored
    pairwise_drift = _drift(s2, d2)

    print("Drift-prevention demonstration (semantic edit: add one requirement + its satisfies edge):\n")
    print(f"  (1) HyMeKo single-source : re-emit both views from one IR -> drift = {hymeko_drift} "
          f"(new requirement in both views: {new_in_both})")
    print(f"  (2) Pairwise multi-file  : edit the SysML file only       -> drift = {pairwise_drift} "
          f"(requirement present in SysML, absent from DOT)")
    ok = hymeko_drift == 0 and new_in_both and pairwise_drift >= 1
    print("\n  => the same edit is benign under single-source emission (0 drift, propagates to all views) and")
    print(f"     silently divergent under multi-file maintenance ({pairwise_drift} drift). [{ok}]")
    return hymeko_drift, pairwise_drift, ok


if __name__ == "__main__":
    sys.exit(0 if run()[2] else 1)
