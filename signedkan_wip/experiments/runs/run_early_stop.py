"""Step-1 of the SGCN-gap-closing plan: validation-based early
stopping. Run the same configurations as the comparison sweep but
with val-AUC checkpointing; report the test AUC of the best-val
checkpoint.

Free-lunch hypothesis: stopping at the validation-AUC peak should
recover ~0.02-0.03 test AUC over the fixed-100-epoch protocol,
because the saturation curve we measured in run_saturation showed
SignedKAN AUC peaks near epoch 50 and decays thereafter.

Run:
  python -m signedkan_wip.experiments.runs.run_early_stop

Migrated to the :class:`SimpleExperiment` pattern on 2026-05-19
(Slice H pilot of the signedkan_wip reorganisation). The script
body is now ~30 LOC of config + a 4-line ``run_seed``; the
previous 60-LOC version's argparse, triple-nested loop, JSON
emission, and printing all happen in :class:`SimpleExperiment` and
its registered observers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ._experiment_base import (
    JsonlObserver,
    SeedEvent,
    SimpleExperiment,
    StdoutObserver,
)
from .run_compare import run_one


class EarlyStopExperiment(SimpleExperiment):
    """Sweeps two models × N datasets × M seeds with val-AUC early
    stopping. Subclass contract is satisfied by :meth:`run_seed`;
    the (model, dataset) pair is folded into the seed loop by
    overriding :meth:`run` to iterate combinations.
    """

    def __init__(self, datasets: list[str], hidden: int, lr: float,
                 n_epochs: int, val_every: int) -> None:
        super().__init__(label="early_stop")
        self.datasets = datasets
        self.hidden = hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.val_every = val_every

    def run_seed(self, seed: int, **cfg) -> dict:
        model = cfg["model"]
        dataset = cfg["dataset"]
        return run_one(
            model, dataset, self.hidden, seed, self.n_epochs,
            lr=self.lr, early_stopping=True, val_every=self.val_every,
        )

    def run_grid(self, seeds: list[int]) -> list[dict]:
        """The grid sweep: (model, dataset, seed) combinations.
        Reuses the base's observer dispatch by calling ``run`` once
        per (model, dataset)."""
        all_results: list[dict] = []
        for dataset in self.datasets:
            for model in ("signedkan", "vanillakan"):
                results = self.run(seeds, model=model, dataset=dataset)
                # Annotate each result so the consumer can demux.
                for r in results:
                    r["dataset"] = dataset
                    r["model"] = model
                all_results.extend(results)
        return all_results


class _PrintObserver(StdoutObserver):
    """Format the per-seed end-of-experiment line like the original
    script did."""

    def __init__(self, hidden: int) -> None:
        self.hidden = hidden

    def on_seed_end(self, ev: SeedEvent) -> None:
        m = ev.final_metrics
        if not m:
            return
        be = int(m.get("best_epoch", 0))
        print(f"  seed={ev.seed}  "
              f"best_ep={be:3d}  "
              f"val_auc={m.get('best_val_auc', 0):.4f}  "
              f"test_auc={m.get('test_auc', 0):.4f}  "
              f"F1_mac={m.get('test_f1_macro', 0):.4f}  "
              f"{m.get('elapsed_s', 0):.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/early_stop.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exp = EarlyStopExperiment(
        datasets=args.datasets,
        hidden=args.hidden,
        lr=args.lr,
        n_epochs=args.n_epochs,
        val_every=args.val_every,
    )
    exp.add_observer(_PrintObserver(args.hidden))
    exp.add_observer(JsonlObserver(str(out_path.with_suffix(".jsonl"))))

    all_results = exp.run_grid(args.seeds)

    # Final aggregate JSON (compatible with the pre-pilot output format).
    import json
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}  ({len(all_results)} runs)")


if __name__ == "__main__":
    main()
