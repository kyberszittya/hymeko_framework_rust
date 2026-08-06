"""Gate D — the ``core/`` isolation boundary (see docs/plans/2026-07-04-hymeko-neuro-merge/).

``hymeko_neuro.core`` is the validated pairwise signed-GCN core, kept *pristine*: nothing under
``core/`` may import a research subpackage (``models``, ``graph``, ``hyperedge``, ``eval``,
``experiments``, ``data``, ``baselines``, ``paperkit``, ``kernels``, ``runtime``, ``hymeko``, ...).
It may depend only on the standard library, third-party libs (torch/numpy/...), and its own
``core`` submodules. This is a static AST audit, so a violating ``import`` fails the build even if
that code path is never executed at runtime.

The decoupling was bought by commit ec98095 and must survive the merge by the *boundary*, not by luck.
"""
from __future__ import annotations

import ast
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "core"


def _core_modules() -> list[Path]:
    return [p for p in _CORE.rglob("*.py") if "__pycache__" not in p.parts and "tests" not in p.parts]


def _forbidden_import(node: ast.AST) -> str | None:
    """Return the offending target string if this import escapes ``hymeko_neuro.core``, else None.

    # Preconditions ``node`` is an ``ast.Import`` or ``ast.ImportFrom``.
    # Postconditions a non-None result names a research subpackage the core must not depend on.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("hymeko_neuro.") and not alias.name.startswith("hymeko_neuro.core"):
                return alias.name
    elif isinstance(node, ast.ImportFrom):
        # relative: level 1 == within core (fine); level >= 2 escapes core into hymeko_neuro.<research>
        if node.level >= 2:
            return f"{'.' * node.level}{node.module or ''}"
        mod = node.module or ""
        if node.level == 0 and mod.startswith("hymeko_neuro.") and not mod.startswith("hymeko_neuro.core"):
            return mod
    return None


def test_core_has_modules() -> None:
    """Guard the audit itself: if the glob finds nothing the boundary test would vacuously pass."""
    assert len(_core_modules()) >= 5, "core/ audit found too few modules — wrong path?"


def test_core_does_not_import_research_subpackages() -> None:
    violations: list[str] = []
    for f in _core_modules():
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            target = _forbidden_import(node)
            if target is not None:
                violations.append(f"{f.relative_to(_CORE.parent)} -> {target}")
    assert not violations, "core/ must stay pristine (no research-subpackage imports):\n  " + "\n  ".join(violations)
