"""Stage signedkan-wip-organize Slice B: shell ABC for the future
``ExperimentBase`` refactor (Slice H, not yet executed).

The 101 ``run_*.py`` scripts under this directory currently each
re-implement:

  - argparse parsing
  - dataset loading + train/val/test split
  - per-seed training loop
  - per-epoch eval + best-model selection
  - JSONL result emission
  - random-seed pinning
  - device-placement (CPU vs CUDA)

CLAUDE.md §6.5 #3 names this as an explicit anti-pattern. The fix
is to extract a single :class:`ExperimentBase` with observer hooks
that every concrete ``run_*.py`` subclasses, reducing each script
to ~20 lines of config + a model constructor.

This module ships an *empty shell* of the target ABC so the
directory-layout commits the architectural shape without
prematurely migrating 101 files. Concrete migration is deferred to
Slice H per the operating contract's "one phase per session" rule
(``feedback_one_phase_per_session.md``).

Object-oriented, observer-pattern target shape (sketch only):
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


# ─── Event payload types — frozen for the observer protocol ─────────


@dataclass(frozen=True)
class EpochEvent:
    """Emitted at the start AND end of every training epoch."""

    seed: int
    epoch: int
    total_epochs: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedEvent:
    """Emitted at the start AND end of every seed."""

    seed: int
    final_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunEvent:
    """Emitted at the start AND end of an entire experiment run
    (all seeds, all epochs)."""

    label: str
    seeds: list[int]
    summary: dict[str, Any] = field(default_factory=dict)


# ─── Observer protocol ──────────────────────────────────────────────


class ExperimentObserver(abc.ABC):
    """Observer interface every concrete observer subclasses.

    Each ``on_*`` method is called exactly twice per scope: once at
    start (``*_start``) and once at end (``*_end``). All methods
    have default no-op implementations so observers only override
    what they care about.
    """

    def on_run_start(self, ev: RunEvent) -> None:
        """Called once at the start of the experiment run."""

    def on_run_end(self, ev: RunEvent) -> None:
        """Called once at the end of the experiment run."""

    def on_seed_start(self, ev: SeedEvent) -> None:
        """Called at the start of each seed."""

    def on_seed_end(self, ev: SeedEvent) -> None:
        """Called at the end of each seed."""

    def on_epoch_start(self, ev: EpochEvent) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, ev: EpochEvent) -> None:
        """Called at the end of each epoch."""


# ─── ExperimentBase ABC (Slice H target shape) ──────────────────────


class ExperimentBase(abc.ABC):
    """Abstract base for every ``run_*.py`` script.

    The concrete-subclass contract is the four ``build_*`` methods
    plus the metric definition. Everything else (the training loop,
    seed iteration, JSONL emission, observer dispatch) lives in the
    base.

    .. note::
       This is the **shell** of the future Slice H refactor. The
       ``run`` method is not yet implemented; subclasses cannot
       currently use it. The shape is committed so the layout
       reflects the architectural intent without breaking the
       101 existing scripts. Migrate one script at a time when
       Slice H lands.
    """

    def __init__(self) -> None:
        self._observers: list[ExperimentObserver] = []

    # ─── Subclass contract (concrete in Slice H) ────────────────

    @abc.abstractmethod
    def build_dataset(self, seed: int) -> tuple[Any, Any, Any]:
        """Return ``(train, val, test)`` splits, seeded deterministically."""

    @abc.abstractmethod
    def build_model(self, seed: int) -> Any:
        """Construct the model for this seed."""

    @abc.abstractmethod
    def build_optimizer(self, model: Any) -> Any:
        """Construct the optimizer for this seed."""

    @abc.abstractmethod
    def train_step(self, model: Any, optimizer: Any, batch: Any) -> dict[str, float]:
        """One training step; return metrics dict."""

    @abc.abstractmethod
    def eval_step(self, model: Any, val_loader: Iterable[Any]) -> dict[str, float]:
        """One epoch's evaluation; return metrics dict."""

    # ─── Observer registration ──────────────────────────────────

    def add_observer(self, obs: ExperimentObserver) -> None:
        """Register an observer for run/seed/epoch hooks."""
        self._observers.append(obs)

    def _emit(self, name: str, ev: Any) -> None:
        for obs in self._observers:
            getattr(obs, name)(ev)

    # ─── Entry point (NOT YET IMPLEMENTED in this shell) ───────

    def run(
        self,
        label: str,
        seeds: Iterable[int],
        epochs: int,
        **base_config: Any,
    ) -> dict[str, Any]:
        """Run the experiment across seeds, emitting observer hooks.

        .. warning::
           Slice B (today) only ships the *shape*. Implementation
           lands in Slice H, when individual ``run_*.py`` scripts
           are migrated one at a time.
        """
        raise NotImplementedError(
            "ExperimentBase.run is the Slice H target; "
            "see docs/plans/2026-05-19-signedkan-wip-organize/ for the "
            "phased migration plan."
        )


# ─── Concrete observers shipped with the shell ──────────────────────


class StdoutObserver(ExperimentObserver):
    """Minimal observer that prints epoch/seed boundaries to stdout."""

    def on_seed_start(self, ev: SeedEvent) -> None:
        print(f"[seed {ev.seed}] start")

    def on_seed_end(self, ev: SeedEvent) -> None:
        print(f"[seed {ev.seed}] end: {ev.final_metrics}")

    def on_epoch_end(self, ev: EpochEvent) -> None:
        if ev.metrics:
            print(f"  seed {ev.seed} ep {ev.epoch}/{ev.total_epochs}: {ev.metrics}")


class JsonlObserver(ExperimentObserver):
    """Append one JSONL line per ``on_seed_end`` event. Mirrors what
    every current ``run_*.py`` writes by hand.

    ``mode="a"`` (default): pure append — pre-existing file content is
    preserved; useful for resuming a partial run.

    ``mode="w"``: truncate the file at the start of the run
    (``on_run_start``) and append per seed thereafter. The naive
    ``open(path, "w")`` on every seed would lose all but the last seed
    line — this observer avoids that pitfall.
    """

    def __init__(self, path: str, mode: str = "a") -> None:
        if mode not in ("a", "w"):
            raise ValueError(f"JsonlObserver mode must be 'a' or 'w', got {mode!r}")
        self._path = path
        self._mode = mode

    def on_run_start(self, ev: RunEvent) -> None:
        # In write mode, truncate the file at run start so subsequent
        # appends produce a single coherent JSONL for this run.
        if self._mode == "w":
            with open(self._path, "w"):
                pass

    def on_seed_end(self, ev: SeedEvent) -> None:
        import json

        with open(self._path, "a") as f:
            f.write(
                json.dumps({"seed": ev.seed, **ev.final_metrics}) + "\n"
            )


class CallbackObserver(ExperimentObserver):
    """Observer that fires user-supplied callbacks. Useful for
    lightweight one-shot hooks (e.g.\\ from tests) without needing a
    bespoke subclass."""

    def __init__(
        self,
        *,
        on_seed_end: Callable[[SeedEvent], None] | None = None,
        on_epoch_end: Callable[[EpochEvent], None] | None = None,
    ) -> None:
        self._on_seed_end = on_seed_end
        self._on_epoch_end = on_epoch_end

    def on_seed_end(self, ev: SeedEvent) -> None:  # type: ignore[override]
        if self._on_seed_end is not None:
            self._on_seed_end(ev)

    def on_epoch_end(self, ev: EpochEvent) -> None:  # type: ignore[override]
        if self._on_epoch_end is not None:
            self._on_epoch_end(ev)


# ─── SimpleExperiment — pilot-ready adapter (Slice H pilot, 2026-05-19) ─


class SimpleExperiment:
    """A thinner adapter than :class:`ExperimentBase` for scripts that
    already have a `run_one_seed(seed, **cfg) -> dict` style entry point
    and don't want to refactor into the full build_*/train_step/eval_step
    contract.

    Subclasses implement :meth:`run_seed`; the base orchestrates the
    seed loop and observer dispatch, including writing a final
    aggregated summary. This is the lightweight migration path for the
    101 ``run_*.py`` scripts in this directory — adopt the observer
    pattern in $\\sim 20$ lines per script.

    For new code or scripts that ARE willing to refactor, prefer
    :class:`ExperimentBase` (proper build_*/train_step/eval_step
    contract; orchestrates the inner training loop too).
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._observers: list[ExperimentObserver] = []

    def add_observer(self, obs: ExperimentObserver) -> "SimpleExperiment":
        """Register an observer. Chainable for fluent setup."""
        self._observers.append(obs)
        return self

    def _emit(self, name: str, ev: Any) -> None:
        for obs in self._observers:
            getattr(obs, name)(ev)

    # ─── Subclass contract: a single method ───────────────────────

    def run_seed(self, seed: int, **cfg: Any) -> dict[str, Any]:
        """Run one seed end-to-end; return a result dict.

        Subclasses override this with the existing script body.
        Returning a flat ``dict[str, Any]`` lets the base write a
        clean JSONL output and dispatch observer events.
        """
        raise NotImplementedError(
            "SimpleExperiment subclasses must override run_seed."
        )

    # ─── Orchestration ────────────────────────────────────────────

    def run(
        self,
        seeds: Iterable[int],
        **cfg: Any,
    ) -> list[dict[str, Any]]:
        """Run :meth:`run_seed` across all `seeds`; emit observer events.

        Returns the list of per-seed result dicts (also accessible via
        any observer that consumed them).
        """
        seed_list = list(seeds)
        run_ev = RunEvent(label=self.label, seeds=seed_list)
        self._emit("on_run_start", run_ev)

        results: list[dict[str, Any]] = []
        for seed in seed_list:
            start_ev = SeedEvent(seed=seed, final_metrics={})
            self._emit("on_seed_start", start_ev)
            result = self.run_seed(seed, **cfg)
            results.append(result)
            # Only float-castable values get into the SeedEvent's
            # metrics dict; everything else is preserved in the
            # returned list of dicts.
            float_metrics = {
                k: float(v)
                for k, v in result.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            end_ev = SeedEvent(seed=seed, final_metrics=float_metrics)
            self._emit("on_seed_end", end_ev)

        # Aggregate summary stats for the run-end event.
        summary: dict[str, Any] = {"n_seeds": len(seed_list)}
        if results:
            # Best-effort numeric aggregation per key.
            from statistics import fmean, pstdev
            keys = set().union(*(r.keys() for r in results))
            for k in keys:
                vals = [r[k] for r in results
                        if isinstance(r.get(k), (int, float))
                        and not isinstance(r.get(k), bool)]
                if len(vals) >= 2:
                    summary[f"{k}_mean"] = fmean(vals)
                    summary[f"{k}_std"] = pstdev(vals)
                elif len(vals) == 1:
                    summary[f"{k}_mean"] = vals[0]
        end_run = RunEvent(label=self.label, seeds=seed_list, summary=summary)
        self._emit("on_run_end", end_run)
        return results


__all__ = [
    "EpochEvent",
    "SeedEvent",
    "RunEvent",
    "ExperimentObserver",
    "ExperimentBase",
    "SimpleExperiment",
    "StdoutObserver",
    "JsonlObserver",
    "CallbackObserver",
]
