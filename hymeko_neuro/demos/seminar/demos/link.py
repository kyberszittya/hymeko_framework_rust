"""Demo 3 — signed-graph link-prediction inference (the numbers are real).

Loads a pretrained HSiKAN checkpoint, predicts held-out edge signs, prints
AUC / F1 / n_params / forward-time, and writes ROC + αₖ-regime figures. Pure
reuse of ``src/demo/{inference,registry,plotting}`` — no model or metric logic
lives here (CLAUDE.md §6.5 #2).

Acceptance (SEMINAR_DEMOS §3): reproduces the checkpoint's committed AUC within
±0.002 on the fixed split. The runner enforces seed/device/RSS.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..base import REPO_ROOT, DemoContext, DemoResult

if TYPE_CHECKING:  # avoid importing torch-backed modules at load time.
    from hymeko_neuro.experiments.demo.inference import ModelBundle, PredictionResult

# Acceptance tolerance against the registry's committed AUC.
AUC_TOLERANCE = 0.002


class LinkDemo:
    name = "link"
    help = "Signed-graph link prediction: load checkpoint, AUC/F1, ROC + αₖ."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset", default="bitcoin_otc",
            help="dataset name (used to pick a checkpoint from the registry "
                 "when --checkpoint is omitted).",
        )
        parser.add_argument(
            "--checkpoint", default=None,
            help="explicit checkpoint path; overrides registry lookup.",
        )
        parser.add_argument(
            "--no-figures", action="store_true",
            help="skip ROC / αₖ PNG rendering (table only).",
        )

    def _resolve_checkpoint(self, args: argparse.Namespace) -> tuple[Path, float | None]:
        """Return (checkpoint_path, committed_auc). Committed AUC is the
        registry's recorded value, used only as the acceptance reference.

        The registry (and its yaml dependency) is imported lazily so the
        explicit ``--checkpoint`` path needs only torch + scikit-learn."""
        if args.checkpoint is not None:
            return Path(args.checkpoint), None
        from hymeko_neuro.experiments.demo.registry import load_registry, ModelEntry

        entries = load_registry()
        candidates: list[ModelEntry] = [
            e for e in entries
            if e.dataset == args.dataset and e.framework == "HSiKAN" and e.available
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no available HSiKAN checkpoint for dataset {args.dataset!r} "
                f"in the registry; pass --checkpoint explicitly."
            )
        chosen = candidates[0]
        return chosen.path, chosen.auc

    def run(self, args: argparse.Namespace, ctx: DemoContext) -> DemoResult:
        # Headless figure backend before any pyplot import (via plotting).
        os.environ.setdefault("MPLBACKEND", "Agg")
        from ..compat import register_legacy_checkpoint_aliases
        from hymeko_neuro.experiments.demo.inference import load_bundle, predict_test_edges

        register_legacy_checkpoint_aliases()
        ckpt, committed_auc = self._resolve_checkpoint(args)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        ckpt_rel = os.path.relpath(ckpt, REPO_ROOT)

        bundle = load_bundle(ckpt, device=ctx.device)
        pred = predict_test_edges(bundle)
        fwd_ms = self._time_forward(bundle, reps=3 if ctx.quick else 7)

        result = DemoResult(name=self.name)
        result.metrics.update({
            "dataset": bundle.dataset,
            "n_test": int(len(pred.true_signs)),
            "AUC": round(pred.auc, 4),
            "F1_macro": round(pred.f1_macro, 4),
            "accuracy": round(pred.accuracy, 4),
            "n_params": bundle.n_params,
            "fwd_ms_median": round(fwd_ms, 2),
        })
        for key in ("AUC", "F1_macro", "accuracy", "n_params"):
            result.provenance[key] = ckpt_rel
        result.provenance["fwd_ms_median"] = f"{ckpt_rel} (single-config diagnostic timing)"

        # Acceptance reference: the checkpoint's OWN recorded test AUC
        # (deterministic on the fixed split). The registry value is a 10-seed
        # *mean* (std ~0.002), so it is reported as context, not the gate.
        ref_auc = bundle.meta.test_auc
        ref_src = "checkpoint meta.test_auc"
        if ref_auc is None:
            ref_auc, ref_src = committed_auc, "registry 10-seed mean"
        if ref_auc is not None:
            delta = abs(pred.auc - ref_auc)
            verdict = "PASS" if delta <= AUC_TOLERANCE else "FAIL"
            result.metrics["reference_AUC"] = round(float(ref_auc), 4)
            result.metrics["AUC_delta"] = round(delta, 4)
            result.notes.append(
                f"acceptance: |measured - {ref_src} {ref_auc:.4f}| = {delta:.4f} "
                f"(tol {AUC_TOLERANCE}) -> {verdict}"
            )
        if committed_auc is not None and committed_auc != ref_auc:
            result.metrics["registry_mean_AUC"] = committed_auc
        result.notes.append(
            "checkpoints are optuna_best (transductive convention); "
            "the strict protocol is the architectural baseline."
        )

        if not args.no_figures:
            result.artifacts.extend(self._render(bundle, pred, ctx.out_dir))
        return result

    @staticmethod
    def _time_forward(bundle: "ModelBundle", *, reps: int) -> float:
        """Median wall-ms of one test-set forward pass. Diagnostic only — the
        reportable benchmark is the `latency` demo (CLAUDE.md §10)."""
        import numpy as np
        import torch

        ib = bundle.inference_bundle
        if ib is None:
            return float("nan")
        q_te = torch.as_tensor(ib.query_edges, dtype=torch.long, device=bundle.device)
        clf = bundle.classifier

        def _once() -> None:
            # encode_edges / classifier are nn.Module attrs mypy resolves to
            # Tensor; calling them is correct (mirrors predict_test_edges).
            with torch.no_grad():
                emb = bundle.model.encode_edges(  # type: ignore[operator]
                    ib.per_arity_te, query_edges=q_te)
                if clf is not None:
                    clf(emb)
                else:
                    bundle.model.classifier(emb)  # type: ignore[operator]

        _once()  # warm-up
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _once()
            times.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(times))

    @staticmethod
    def _render(bundle: "ModelBundle", pred: "PredictionResult", out_dir: Path) -> list[Path]:
        from hymeko_neuro.experiments.demo.plotting import alpha_figure, roc_figure

        roc = roc_figure(pred, title=f"ROC — {bundle.dataset}")
        roc_path = out_dir / f"ROC_{bundle.dataset}.png"
        roc.savefig(roc_path, dpi=140, bbox_inches="tight")

        alpha = alpha_figure(
            bundle.alpha_vector(), bundle.tuple_labels(),
            title=f"Learned αₖ — {bundle.dataset}",
        )
        alpha_path = out_dir / f"alpha_k_{bundle.dataset}.png"
        alpha.savefig(alpha_path, dpi=140, bbox_inches="tight")

        import matplotlib.pyplot as plt

        plt.close(roc)
        plt.close(alpha)
        return [roc_path, alpha_path]


__all__ = ["LinkDemo"]
