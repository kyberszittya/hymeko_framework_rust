"""R11.7A U5 — duplicate-ownership guard.

The model builder ``compose_planar_scene`` and the object-injection literal must not proliferate. These static
checks fail if an experiment/module reintroduces its own model construction outside the sanctioned set, so the
unification cannot silently erode. Physical removal of the two allowlisted legacy builders is deferred to
post-U6 (user decision: nothing legacy is deleted while variants are still validated against it) — until then
they are pinned here and may not grow.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]          # .../hymeko_coin_r9_wt
_PKG = _ROOT / "hymeko_rl"

# Sanctioned callers of the model builder. `planar_grasp_env.py` is THE canonical builder (PlanarGraspEnv +
# the def). The other two are pre-R11.7A direct builders, allowlisted and flagged for post-U6 consolidation —
# this set MAY NOT GROW.
_SANCTIONED_BUILDERS = {
    "hymeko_rl/env/planar_grasp_env.py",
    "hymeko_rl/coin_delivery/coin_robot_variant.py",           # legacy: direct MjModel for a robot-variant probe
    "hymeko_rl/coin_delivery/scenarios/kinematic_variant.py",  # legacy: direct MjModel for a kinematic scenario
}


def _calls(py: Path, fn_name: str) -> bool:
    """True iff the module makes a real *call* to ``fn_name`` (AST — docstring/comment mentions do not count)."""
    with warnings.catch_warnings():                       # some scanned modules carry pre-existing SyntaxWarnings
        warnings.simplefilter("ignore")                   # (e.g. invalid escapes) — not this guard's concern
        tree = ast.parse(py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == fn_name:
                return True
    return False


def _modules() -> list[Path]:
    return [p for p in _PKG.rglob("*.py") if not p.name.startswith("test_") and "/tests/" not in p.as_posix()]


def test_compose_planar_scene_has_no_new_direct_callers() -> None:
    offenders = [str(p.relative_to(_ROOT)) for p in _modules()
                 if _calls(p, "compose_planar_scene") and str(p.relative_to(_ROOT)) not in _SANCTIONED_BUILDERS]
    assert not offenders, (
        f"new direct compose_planar_scene call(s) outside the sanctioned builders: {offenders}. "
        "Object variants must go through build_manipulation_rig / PlanarGraspEnv, not a fresh model builder.")


def test_sanctioned_builder_allowlist_is_current() -> None:
    # The allowlist must not carry stale entries (a builder that no longer calls compose_planar_scene should be
    # dropped from the list, not left pinning a non-existent risk).
    stale = [rel for rel in _SANCTIONED_BUILDERS
             if not (_ROOT / rel).is_file() or not _calls(_ROOT / rel, "compose_planar_scene")]
    assert not stale, f"allowlist has stale entries (no longer call compose_planar_scene): {stale}"


def test_bv_make_env_object_is_scene_sourced_not_hardcoded() -> None:
    # The R11.6C acquisition must read the manipuland from the HyMeKo scene, not a `coin_shape="cylinder", hx=…`
    # literal (the unpinned seam). Guards against a regression that re-hardcodes the object.
    text = (_PKG / "experiments" / "bv_identification_benchmark.py").read_text(encoding="utf-8")
    assert "EnvSpec.from_hymeko(_PLANAR_ENV).object" in text, "bv _make_env no longer sources the object from the scene"
    assert 'hx=0.020' not in text, "the hardcoded coin literal reappeared in bv _make_env"
