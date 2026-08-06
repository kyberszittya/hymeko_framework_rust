"""Reusable CIP-0 conformance battery.

The core profile self-tests these against a toy adapter (see
``tests/test_cip0_conformance.py``); each SCENARIO reuses the same battery
against its own model + adapter, so the ten conformance guarantees are checked
identically everywhere and never re-implemented per scenario (repo rule 6.1).

The battery is split into:

* pure schema checks (:func:`assert_schema_accepts` / :func:`assert_schema_rejects`);
* a positive-lifecycle driver (:func:`run_positive_lifecycle`) that a scenario
  adapter must survive without any contract violation;
* a static import-isolation scan (:func:`import_isolation_violations`).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..cip.protocol import CIP0Adapter
from ..cip.runtime import CIP0Runtime, TickRecord
from ..language.ir import ControlModel
from ..language.validator import ValidationError, validate

#: Top-level module names a shared-core file must never import.
FORBIDDEN_CORE_IMPORTS: tuple[str, ...] = (
    "torch",
    "hymeko_rl",
    "coin",
    "coin_delivery",
    "theta_option",
    "galambos",
    "scenarios",
)


def assert_schema_accepts(raw: Mapping[str, Any]) -> ControlModel:
    """Validate a good spec; raise ``AssertionError`` if it is rejected."""
    try:
        return validate(raw)
    except ValidationError as err:  # pragma: no cover - failure path
        raise AssertionError(f"expected schema to accept spec, got: {err}") from err


def assert_schema_rejects(raw: Mapping[str, Any], reason: str) -> None:
    """Assert a malformed spec is rejected with :class:`ValidationError`."""
    try:
        validate(raw)
    except ValidationError:
        return
    raise AssertionError(f"expected schema to REJECT ({reason}), but it was accepted")


def run_positive_lifecycle(
    model: ControlModel,
    adapter: CIP0Adapter,
    max_ticks: int = 16,
    task: Any = None,
) -> list[TickRecord]:
    """Drive ``adapter`` through the runtime on a clean (converging) episode.

    # Postconditions
    Returns a non-empty record list. Any CIP-0 contract violation (causality,
    mode legality, intent bounds, authority provenance, decode determinism,
    option provenance) surfaces as the corresponding runtime exception -- that
    is the point of the check. Also verifies the runtime did not mutate the
    adapter's states (no hidden state modification): every record's intent is
    still bounded and its trace still references its option.
    """
    runtime = CIP0Runtime(model=model, adapter=adapter)
    records = runtime.run(max_ticks=max_ticks, task=task)
    assert records, "lifecycle produced no ticks"
    for rec in records:
        assert rec.intent.is_bounded(), "intent lost boundedness after a tick"
        assert rec.trace.references(rec.option), "trace/option provenance broke"
        assert rec.mode in model.mode_names(), "tick mode not in model"
    return records


def _iter_python_files(package_dir: Path) -> Iterable[Path]:
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _top_level_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def import_isolation_violations(
    package_dir: Path,
    forbidden: Sequence[str] = FORBIDDEN_CORE_IMPORTS,
    *,
    ignore_lazy: bool = True,
) -> list[str]:
    """Return a list of ``file: module`` import-isolation violations (empty = clean).

    Scans every ``.py`` under ``package_dir`` and flags any import of a forbidden
    top-level module. When ``ignore_lazy`` is true, imports nested inside a
    function/method body are ignored (a lazy optional import such as ``yaml`` is
    allowed); only module-level imports count against isolation.
    """
    forbidden_set = set(forbidden)
    violations: list[str] = []
    for path in _iter_python_files(package_dir):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = _module_level_nodes(tree) if ignore_lazy else list(ast.walk(tree))
        for node in nodes:
            for mod in _import_targets(node):
                if mod in forbidden_set:
                    violations.append(f"{path.name}: import {mod}")
    return violations


def _module_level_nodes(tree: ast.AST) -> list[ast.AST]:
    """Import nodes that live at module top level (not inside a def/class body)."""
    out: list[ast.AST] = []
    body = getattr(tree, "body", [])
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(node)
    return out


def _import_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return {node.module.split(".")[0]}
    return set()
