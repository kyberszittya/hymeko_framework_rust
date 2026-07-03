"""Experiment class registry: maps YAML ``experiment_class`` names
to concrete Experiment subclasses.

The registry is auto-populated when concrete Experiment subclasses
register themselves; see runner.py's ``__init_subclass__`` hook.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .runner import Experiment


class ExperimentRegistry:
    """Singleton-ish registry of {name: Experiment subclass}."""
    _classes: dict[str, Type["Experiment"]] = {}

    @classmethod
    def register(cls, name: str, klass: Type["Experiment"]) -> None:
        if name in cls._classes and cls._classes[name] is not klass:
            raise ValueError(f"registry: name '{name}' already bound "
                             f"to {cls._classes[name]}, not {klass}")
        cls._classes[name] = klass

    @classmethod
    def lookup(cls, name: str) -> Type["Experiment"]:
        if name not in cls._classes:
            raise KeyError(
                f"no experiment class registered as '{name}'; known: "
                f"{sorted(cls._classes)}"
            )
        return cls._classes[name]

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._classes)
