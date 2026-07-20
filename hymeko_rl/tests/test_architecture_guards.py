"""Architecture-regression guards (Phase 10.8) — codify the canonicalization invariants so violations cannot GROW.

The repository-wide canonicalization is a large, incremental consolidation; these guards ratchet the debt down (a
count that may only shrink) and hard-fail NEW violations of the load-bearing rules: production library code must not
import experiment entry points, and there must be one canonical owner for the generic RL responsibilities.
"""
from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # hymeko_rl/
_LIB_EXCLUDE = ("experiments", "tests")                 # experiment/test code MAY import experiments


def _lib_modules() -> list[Path]:
    return [p for p in _ROOT.rglob("*.py")
            if not any(part in _LIB_EXCLUDE for part in p.relative_to(_ROOT).parts)]


def _imports_experiments(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hymeko_rl.experiments"):
            return True
        if isinstance(node, ast.Import) and any(a.name.startswith("hymeko_rl.experiments") for a in node.names):
            return True
    return False


# Ratchet baseline measured 2026-07-21 (Phase 10 survey). Production→experiments imports may only DECREASE — a new
# violation fails here, and every migration should lower this number. Dominant cluster: galambos_demo generic helpers
# (PhasePushController / _ik_action / _extract_arms) pulled into ~13 agents/env modules → migrate to a production home.
_DEPENDENCY_DIRECTION_BASELINE = 45


def test_production_does_not_import_experiments_ratchet() -> None:
    offenders = [str(p.relative_to(_ROOT)) for p in _lib_modules() if _imports_experiments(p)]
    assert len(offenders) <= _DEPENDENCY_DIRECTION_BASELINE, (
        f"production→experiments imports GREW to {len(offenders)} (baseline {_DEPENDENCY_DIRECTION_BASELINE}); "
        f"new offenders must import from a production home instead:\n" + "\n".join(sorted(offenders)))


def test_paired_stats_is_the_canonical_bootstrap_owner() -> None:
    """The canonical percentile-bootstrap owner exists and is importable (the home the ~20 scattered copies migrate to)."""
    from hymeko_rl.eval.paired_stats import boot_ci, paired_stats
    assert callable(boot_ci) and callable(paired_stats)


def test_contact_actor_bank_selector_is_injectable() -> None:
    """The generic bank has ONE owner and a task-injectable selector (no per-task bank fork) — §13.3 transfer seam."""
    import inspect

    from hymeko_rl.train.sac import ContactActorBank, build_sac
    assert "selector" in inspect.signature(ContactActorBank.__init__).parameters
    assert "bank_selector" in inspect.signature(build_sac).parameters
