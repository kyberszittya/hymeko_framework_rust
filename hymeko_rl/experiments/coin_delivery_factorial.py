"""CLI: batch factorial evaluation over coin-delivery scenarios × actors × paired seeds.

    python -m hymeko_rl.experiments.coin_delivery_factorial \
        --scenarios c0k0,c1k0,c0k1,c1k1 --actors symmetric_push,v_plow --paired-seeds 0,1,2,3,4

Thin wrapper around :class:`~hymeko_rl.coin_delivery.runners.experiment.ExperimentRunner`. Writes the report
JSON if ``--out`` is given. This runs NO RL and produces no campaign write-up (the caller owns the reports).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hymeko_rl.coin_delivery.runners.experiment import ExperimentRunner


def _csv(text: str) -> list[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _int_csv(text: str) -> list[int]:
    return [int(tok) for tok in _csv(text)]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Factorial coin-delivery scenario evaluation (no RL).")
    p.add_argument("--scenarios", default="c0k0,c1k0,c0k1,c1k1")
    p.add_argument("--actors", default="symmetric_push,v_plow")
    p.add_argument("--paired-seeds", default="0,1,2,3,4")
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--no-scramble", action="store_true", help="skip the correct-vs-scramble control")
    p.add_argument("--out", default=None, help="path to write the report JSON")
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    report = ExperimentRunner().run(_csv(args.scenarios), _csv(args.actors), _int_csv(args.paired_seeds),
                                    scramble=not args.no_scramble, max_steps=args.max_steps)
    print(f"[factorial] cross-scenario: {report.cross_scenario}", flush=True)
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2, default=float))
        print(f"[factorial] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
