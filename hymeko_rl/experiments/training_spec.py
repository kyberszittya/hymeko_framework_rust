"""Declarative training-strategy loader — the training ALGORITHM becomes data (like reward/scenario/controller).

An `experiment_spec`'s `@budget` block declares `algorithm "bc" | "td3_bc" | "dagger"`, and a strategy-specific
knob block (`@offpolicy` for TD3+BC, `@dagger` for DAgger) carries its hyperparameters. `TrainingSpec.from_hymeko`
parses it via the generic profile shim (`hymeko_rl.env._profile`), and a family entrypoint dispatches on
`.algorithm`. The `.hymeko` grammar is generic (no per-tag rules), so this needs **no Rust/CORE change** — only a
new declared field + this Python parse. Launching a different strategy is pointing at a different `.hymeko`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALGORITHMS = ("bc", "td3_bc", "dagger")
_STRATEGY_KINDS = ("dagger", "offpolicy")   # blocks whose numeric fields are trainer knobs, not budget
_INT_FIELDS = ("total_steps", "n_demos", "bc_epochs", "bc_batch",
               "dagger_iters", "rollouts_per_iter", "n_eval")


@dataclass(frozen=True)
class TrainingSpec:
    """A parsed training strategy: the chosen `algorithm`, the shared `budget` (demos/epochs/seeds/…), and the
    strategy-specific `strategy` knobs (DAgger or off-policy). `profile` is the declaring `.hymeko` (provenance).

    # Preconditions the profile declares exactly one `experiment_spec` bundle (generic grammar).
    # Postconditions `algorithm in ALGORITHMS`; int budget/strategy fields are coerced to `int`; `seeds` a tuple.
    """

    algorithm: str
    budget: "dict[str, Any]"
    strategy: "dict[str, float]"
    profile: str

    @classmethod
    def from_hymeko(cls, profile: "str | Path") -> "TrainingSpec":
        """Read the declared strategy from `profile`'s `experiment_spec` bundle.

        # Errors `FileNotFoundError`; `ValueError` (no/multiple `experiment_spec`, or an unknown `algorithm`).
        """
        from hymeko_rl.env._profile import parse_fields, read_bundle

        budget: "dict[str, Any]" = {}
        strategy: "dict[str, float]" = {}
        algorithm: "str | None" = None
        for _name, kind, body, _w in read_bundle(profile, "experiment_spec"):
            fields = parse_fields(body)
            if kind in _STRATEGY_KINDS:                       # @dagger / @offpolicy → numeric trainer knobs
                strategy.update({k: v for k, v in fields.items() if isinstance(v, (int, float))})
            else:                                             # @budget / @arms / any config block
                if "algorithm" in fields:
                    algorithm = str(fields.pop("algorithm"))
                budget.update(fields)

        algorithm = algorithm if algorithm is not None else "bc"
        if algorithm not in ALGORITHMS:
            raise ValueError(f"{profile}: algorithm {algorithm!r} not in {ALGORITHMS}")

        for field_map in (budget, strategy):
            for k in _INT_FIELDS:
                if k in field_map and isinstance(field_map[k], float):
                    field_map[k] = int(field_map[k])
        if "seeds" in budget:
            budget["seeds"] = tuple(int(s) for s in budget["seeds"])
        return cls(algorithm=algorithm, budget=budget, strategy=strategy, profile=str(profile))
