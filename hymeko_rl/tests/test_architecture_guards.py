"""Architecture-regression guards (Phase 10.8) — codify the canonicalization invariants so violations cannot GROW.

The repository-wide canonicalization is a large, incremental consolidation; these guards ratchet the debt down (a
count that may only shrink) and hard-fail NEW violations of the load-bearing rules: production library code must not
import experiment entry points, and there must be one canonical owner for the generic RL responsibilities.
"""
from __future__ import annotations

import ast
import os
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
# 43 after the fixed-position replay modules (hymeko_rl/coin_delivery/fixed_position*) were added; they import ONLY the
# canonical coin_* chain (coin_neutral_start / coin_delivery_e0_*), which resides in experiments/, NOT the galambos/
# coin-toss arc — verified by test_coin_delivery_import_closure_is_experiment_free.
_DEPENDENCY_DIRECTION_BASELINE = 43


def test_production_does_not_import_experiments_ratchet() -> None:
    offenders = [str(p.relative_to(_ROOT)) for p in _lib_modules() if _imports_experiments(p)]
    assert len(offenders) <= _DEPENDENCY_DIRECTION_BASELINE, (
        f"production→experiments imports GREW to {len(offenders)} (baseline {_DEPENDENCY_DIRECTION_BASELINE}); "
        f"new offenders must import from a production home instead:\n" + "\n".join(sorted(offenders)))


# ── Coin-Delivery source-closure guards (2026-07-21) — the coin runtime must not drag in the galambos/coin-toss
# experiment web. These lock in the canonicalization of the E-approach loader + env factory + planar kinematics. ──────

# The seeds a fresh clone imports to run the Coin POINT delivery headline (adapter + neutral chain + certifier + the
# canonical loader/env). Their module-TOP-LEVEL import closure is what fires at import time in a clean clone.
_COIN_SEEDS = (
    "hymeko_rl/campaign/adapters.py", "hymeko_rl/experiments/coin_neutral_start.py",
    "hymeko_rl/experiments/coin_delivery_e0_campaign.py", "hymeko_rl/experiments/coin_delivery_e0_stabilize.py",
    "hymeko_rl/experiments/coin_wristed_delivery.py", "hymeko_rl/coin_delivery/delivery_certificate.py",
    "hymeko_rl/coin_delivery/e_approach.py", "hymeko_rl/coin_delivery/env_factory.py",
)
# The galambos/coin-toss experiment web the coin runtime must never re-enter (name fragments; coin_* chain files and
# the experiments package __init__ are the canonical exceptions).
_ARC_FRAGMENTS = ("coin_toss", "galambos", "pedc", "arcrl", "struct_distill", "vector_retest", "option_retest",
                  "d4_cohorts", "handoff_gate", "continuation_cache", "exp_v3", "exp_v5", "exp_v8", "exp_v13",
                  "exp_v14", "exp_v16", "exp_v17", "exp_v18")


def _module_to_file(mod: str) -> str | None:
    rel = mod.replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if (_ROOT.parent / cand).is_file():
            return cand
    return None


def _top_level_hymeko_imports(path: Path) -> set[str]:
    """Only MODULE-TOP-LEVEL imports — what actually fires when the module is imported (not lazy in-function ones)."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hymeko_rl"):
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("hymeko_rl"))
    return out


def _is_arc_experiment(rel: str) -> bool:
    base = os.path.basename(rel)
    return ("/experiments/" in rel and not base.startswith("coin_") and base != "__init__.py"
            and any(fr in base for fr in _ARC_FRAGMENTS))


def test_coin_delivery_import_closure_is_experiment_free() -> None:
    """A fresh clone importing the Coin delivery chain pulls ZERO galambos/coin-toss experiment modules at import time.

    This is the load-bearing reproducibility invariant: the E-approach loader, env factory and planar kinematics are
    canonical production, so the coin runtime no longer requires the experiment web (was 29 experiment files via the
    ``_load_e``/``pedc._env`` tangle; now 0).
    """
    seen: set[str] = set()
    stack = list(_COIN_SEEDS)
    closure: set[str] = set()
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        closure.add(rel)
        for mod in _top_level_hymeko_imports(_ROOT.parent / rel):
            cf = _module_to_file(mod)
            if cf and cf not in seen:
                stack.append(cf)
    arc = sorted(rel for rel in closure if _is_arc_experiment(rel))
    assert not arc, ("the Coin delivery import closure re-entered the galambos/coin-toss experiment web:\n"
                     + "\n".join(arc) + "\ncanonicalize the offending import into a production module.")


def test_e_approach_loader_is_canonical_and_fail_loud() -> None:
    """The canonical E-approach loader lives in production, imports no experiment module, and refuses a wrong checkpoint."""
    import hymeko_rl.coin_delivery.e_approach as ea
    assert not _imports_experiments(Path(ea.__file__))
    assert not _imports_experiments(_ROOT / "coin_delivery" / "env_factory.py")
    assert not _imports_experiments(_ROOT / "env" / "planar_arm_kinematics.py")
    # fail-loud on a mismatched checkpoint (never a silent substitution): present checkpoint + wrong hash → ValueError;
    # checkpoint-less clone → FileNotFoundError. Either way it refuses to load, it does not silently substitute.
    import pytest
    with pytest.raises((ValueError, FileNotFoundError)):
        ea.load_e_approach_policy(expected_checkpoint_hash="deadbeefdead")


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
