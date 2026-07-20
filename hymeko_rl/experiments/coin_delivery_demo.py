"""CLI: run ONE coin-delivery scenario × actor × seed and print/save the full per-episode state.

    python -m hymeko_rl.experiments.coin_delivery_demo --scenario c0k0 --actor symmetric_push --seed 0
    python -m hymeko_rl.experiments.coin_delivery_demo --list-scenarios --list-actors

Thin wrapper around :class:`~hymeko_rl.coin_delivery.runners.demo.DemoRunner` — no logic beyond argument parsing.
"""
from __future__ import annotations

import argparse

from hymeko_rl.coin_delivery.runners.demo import DemoRunner
from hymeko_rl.coin_delivery.scenarios.registry import ScenarioRegistry


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one coin-delivery scenario demonstration.")
    p.add_argument("--scenario", default="c0k0", help="scenario id or short alias (c0k0/c1k0/c0k1/c1k1)")
    p.add_argument("--actor", default="symmetric_push", help="actor name (see --list-actors)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--deterministic", action="store_true", help="record the deterministic flag (rollout is seed-exact)")
    p.add_argument("--save-metrics", default=None, help="path to write the summary JSON")
    p.add_argument("--save-trajectory", default=None, help="path to write the per-step trajectory JSON")
    p.add_argument("--render", action="store_true", help="invoke the (headless-safe) render hook")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--list-actors", action="store_true")
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list_scenarios:
        for sid in ScenarioRegistry.list_scenarios():
            print(f"  {sid}", flush=True)
        for short, full in ScenarioRegistry.aliases().items():
            print(f"  {short:6s} -> {full}", flush=True)
    if args.list_actors:
        for name in ScenarioRegistry.list_actors():
            print(f"  {name}", flush=True)
    if args.list_scenarios or args.list_actors:
        return 0
    runner = DemoRunner()
    result = runner.run(args.scenario, args.actor, args.seed, max_steps=args.max_steps,
                        deterministic=args.deterministic, save_trajectory=args.save_trajectory is not None)
    runner.print_report(result)
    if args.render or args.record_video:
        runner.render(result, record_video=args.record_video)
    runner.save(result, metrics_path=args.save_metrics, trajectory_path=args.save_trajectory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
