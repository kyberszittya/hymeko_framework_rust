"""CLI entry: ``python -m hymeko_neuro.experiments.run --config <yaml>``

Replaces the historical ~221 launcher scripts in this directory.
New experiments = new YAML config in ``configs/``, never a new
.py/.sh script (CLAUDE.md §6.5 #3 + #13).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # Importing here so --list does not need pyyaml.
    from hymeko_neuro.experiments.lib import ExperimentConfig
    from hymeko_neuro.experiments.lib.registry import ExperimentRegistry
    from hymeko_neuro.experiments.lib.runner import run_from_config

    # Trigger subclass registration (the runner module's import
    # populates the registry via __init_subclass__).
    import hymeko_neuro.experiments.lib.runner  # noqa: F401

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   help="YAML config path "
                        "(hymeko_neuro/experiments/configs/<name>.yaml)")
    p.add_argument("--list", action="store_true",
                   help="list registered Experiment classes")
    p.add_argument("--list-configs", action="store_true",
                   help="list available YAML configs")
    p.add_argument("--explain", action="store_true",
                   help="print the parsed ExperimentConfig and exit "
                        "(no run)")
    args = p.parse_args(argv)

    if args.list:
        print("registered Experiment classes:")
        for n in ExperimentRegistry.list_all():
            print(f"  - {n}")
        return 0
    if args.list_configs:
        configs_dir = Path(__file__).parent / "configs"
        if not configs_dir.exists():
            print(f"(no configs dir at {configs_dir})")
            return 0
        for f in sorted(configs_dir.glob("*.yaml")):
            print(f"  - {f.name}")
        return 0
    if args.config is None:
        p.error("--config required (or use --list / --list-configs)")

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.explain:
        print(f"name:        {cfg.name}")
        print(f"description: {cfg.description}")
        print(f"experiment_class: {cfg.experiment_class}")
        print(f"n_cells:     {len(cfg.cells)}")
        print(f"monitors:    {list(cfg.monitor_names)}")
        print(f"output jsonl: {cfg.resolved_output_jsonl()}")
        print(f"cells:")
        for c in cfg.cells:
            print(f"  - dataset={c.dataset:<14} mode={c.mode:<8} seed={c.seed}")
        return 0

    result = run_from_config(cfg)
    # Result is SweepSummary / ExperimentResult / argv list; print
    # a short final summary line either way.
    print(f"[run] DONE config={cfg.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
