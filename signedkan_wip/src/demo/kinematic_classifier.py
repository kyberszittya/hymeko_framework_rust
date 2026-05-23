"""Kinematic family classifier for the demo (v0.5).

A small ``GraphLevelHSiKAN`` per dominant arity, trained on synthetic
fixtures (4-bar, Stewart, delta, serial) to predict mechanism family
from cycle-pool features.

The classifier is the αₖ side of the story we already tell on the
signed-link demo: HSiKAN's learned arity weight is the kinematic
fingerprint. Pre-training is a one-shot job (~30 s/arity on CPU);
checkpoints land in ``checkpoints/kinematic/``.

Inference path at GUI time:

  1. user picks a URDF → ``load_urdf_bundle`` → ``KinematicBundle``
  2. ``detect_dominant_arity(bundle.graph)`` selects a classifier
  3. if no cycles: rule-based "serial" or "tree" via topology_signature
  4. else: build per-arity input + forward through the matching
     ``GraphLevelHSiKAN`` checkpoint → softmax → ``ClassificationResult``

The checkpoint format is intentionally a separate dict from the
signed-link demo's ``demo.checkpoint.save_checkpoint`` — there's no
``state_dict + cfg + inference_bundle + classifier_module`` quartet
here, just ``state_dict + arity + n_nodes_max + family_names``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..kinematic import urdf_to_signed_graph
from signedkan_wip.experiments.runs.run_phase11_kinematic_tasks import (
    FAMILY_LABEL_BY_NAME, GraphLevelHSiKAN, MechInst,
    _build_per_arity_input, build_random_mechanism, detect_dominant_arity,
)
from .kinematic import KinematicBundle, REPO_ROOT


FAMILY_NAMES: list[str] = ["four_bar", "stewart", "delta_3rrr", "serial"]
PRETRAINED_DIR = REPO_ROOT / "checkpoints" / "kinematic"
CLASSIFIER_FORMAT_VERSION = 1


@dataclass
class ClassificationResult:
    """One prediction with α_κ provenance."""

    predicted_label: int
    predicted_family: str
    confidence: float
    probs: np.ndarray         # (n_families,) softmax
    arity_used: int | None    # which classifier was queried (None = rule-based)
    alpha_kappa: float | None = None   # softmax weight on this arity
    rule_based: bool = False  # True if no cycles → topology heuristic only

    @property
    def class_labels(self) -> list[str]:
        return list(FAMILY_NAMES)


def _filter_by_arity(insts: list[MechInst], arity: int
                      ) -> list[tuple[MechInst, int]]:
    """Return (inst, dominant_arity) pairs whose arity matches."""
    out = []
    for inst in insts:
        a = detect_dominant_arity(inst.g)
        if a == arity:
            out.append((inst, a))
    return out


def train_classifier(
    arity: int,
    n_train: int = 80,
    n_epochs: int = 60,
    hidden: int = 16,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> tuple[GraphLevelHSiKAN, int, float]:
    """Train a ``GraphLevelHSiKAN`` to classify mechanism family.

    Returns ``(model, n_nodes_max, train_accuracy)``.

    Pulls random mechanisms via ``build_random_mechanism`` until enough
    samples whose dominant arity matches ``arity`` have been collected.
    On CPU this is < 60 s for arity=4 and < 90 s for arity=6 at the
    default ``n_train``.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # Oversample so we definitely get n_train mechanisms with the right
    # arity (some sampled instances will have other dominant arities).
    matched: list[MechInst] = []
    n_samples = 0
    while len(matched) < n_train and n_samples < n_train * 6:
        inst = build_random_mechanism(rng)
        n_samples += 1
        if detect_dominant_arity(inst.g) == arity:
            matched.append(inst)
    if len(matched) < 4:
        raise RuntimeError(
            f"Could not collect enough k={arity} mechanisms after "
            f"{n_samples} draws (got {len(matched)}). Try a different "
            f"arity or raise the upper sample cap."
        )
    if verbose:
        print(f"[kinematic_classifier] arity=k{arity}: collected "
              f"{len(matched)} mechanisms after {n_samples} draws.")
    n_nodes_max = max(m.g.n_nodes for m in matched)
    inputs = []
    labels = []
    for inst in matched:
        inp = _build_per_arity_input(inst.g, arity, 30_000, dev, seed,
                                       n_nodes_pad=n_nodes_max)
        if inp is None:
            continue
        inputs.append(inp)
        labels.append(inst.family_label)
    if not inputs:
        raise RuntimeError(
            f"All k={arity} mechanism inputs failed to build; "
            f"check _build_per_arity_input."
        )
    y_cls = torch.tensor(labels, dtype=torch.long, device=dev)

    model = GraphLevelHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                hidden=hidden, n_classes=len(FAMILY_NAMES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_epochs))

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(inputs))
        ep_loss = 0.0
        for i in perm:
            cls_logits, _ = model(inputs[i])
            loss = F.cross_entropy(
                cls_logits.unsqueeze(0), y_cls[i:i + 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:>3d}  mean_loss={ep_loss/len(inputs):.4f}")

    # Final training accuracy.
    model.eval()
    correct = 0
    with torch.no_grad():
        for inp, y in zip(inputs, labels):
            cls_logits, _ = model(inp)
            pred = int(cls_logits.argmax().item())
            if pred == y:
                correct += 1
    train_acc = correct / len(inputs)
    if verbose:
        print(f"[kinematic_classifier] arity=k{arity}: train_acc={train_acc:.3f}")
    return model, n_nodes_max, train_acc


def save_classifier(path: str | Path, model: GraphLevelHSiKAN,
                      arity: int, n_nodes_max: int,
                      train_acc: float | None = None,
                      hidden: int = 16) -> Path:
    """Persist a trained classifier to ``path`` (creating dirs as needed)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CLASSIFIER_FORMAT_VERSION,
        "kind": "kinematic_family_classifier",
        "arity": int(arity),
        "n_nodes_max": int(n_nodes_max),
        "hidden": int(hidden),
        "family_names": list(FAMILY_NAMES),
        "family_labels": dict(FAMILY_LABEL_BY_NAME),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "train_accuracy": float(train_acc) if train_acc is not None else None,
    }
    torch.save(payload, p)
    return p


def load_classifier(
    path: str | Path,
    device: str = "cpu",
) -> tuple[GraphLevelHSiKAN, int, int]:
    """Returns ``(model, arity, n_nodes_max)`` ready for inference."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"classifier not found: {p}")
    payload = torch.load(p, map_location=device, weights_only=False)
    if payload.get("kind") != "kinematic_family_classifier":
        raise ValueError(
            f"{p} is not a kinematic_family_classifier "
            f"(kind={payload.get('kind')!r})"
        )
    arity = int(payload["arity"])
    n_nodes_max = int(payload["n_nodes_max"])
    hidden = int(payload.get("hidden", 16))
    model = GraphLevelHSiKAN(
        n_nodes_max=n_nodes_max, arity=arity,
        hidden=hidden, n_classes=len(payload["family_names"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, arity, n_nodes_max


def _pretrained_path_for_arity(arity: int) -> Path:
    return PRETRAINED_DIR / f"family_classifier_k{arity}.pt"


def predict_family(
    bundle: KinematicBundle,
    classifiers_dir: str | Path | None = None,
    device: str = "cpu",
) -> ClassificationResult:
    """Classify a URDF's kinematic family.

    Decision tree:
      - bundle has no cycles → rule-based "serial" (or "tree" → reported
        as "serial" since that's the matching FAMILY_NAMES entry)
      - bundle has cycles at arity k → try loading pretrained classifier
        for that k; if missing, fall back to the topology heuristic
    """
    classifiers_dir = (Path(classifiers_dir) if classifiers_dir
                        else PRETRAINED_DIR)
    n_cycles = sum(bundle.cycle_counts.values())
    n_families = len(FAMILY_NAMES)

    if n_cycles == 0:
        # Rule-based: any open chain / tree → "serial".
        probs = np.zeros(n_families, dtype=np.float64)
        probs[FAMILY_LABEL_BY_NAME["serial"]] = 1.0
        return ClassificationResult(
            predicted_label=FAMILY_LABEL_BY_NAME["serial"],
            predicted_family="serial",
            confidence=1.0,
            probs=probs,
            arity_used=None,
            alpha_kappa=None,
            rule_based=True,
        )

    # Find the dominant arity in the bundle's cycle counts.
    arity = next((k for k in sorted(bundle.cycle_counts)
                  if bundle.cycle_counts[k] > 0), None)
    if arity is None:
        probs = np.zeros(n_families, dtype=np.float64)
        probs[FAMILY_LABEL_BY_NAME["serial"]] = 1.0
        return ClassificationResult(
            predicted_label=FAMILY_LABEL_BY_NAME["serial"],
            predicted_family="serial",
            confidence=1.0, probs=probs, arity_used=None,
            alpha_kappa=None, rule_based=True,
        )

    ckpt_path = classifiers_dir / f"family_classifier_k{arity}.pt"
    if not ckpt_path.is_file():
        # No classifier for this arity — degrade to topology heuristic.
        sig = "stewart" if arity == 6 else "four_bar" if arity == 4 else "serial"
        probs = np.zeros(n_families, dtype=np.float64)
        probs[FAMILY_LABEL_BY_NAME[sig]] = 1.0
        return ClassificationResult(
            predicted_label=FAMILY_LABEL_BY_NAME[sig],
            predicted_family=sig,
            confidence=1.0, probs=probs, arity_used=arity,
            alpha_kappa=None, rule_based=True,
        )

    model, _, n_nodes_max = load_classifier(ckpt_path, device=device)
    # If the test URDF is bigger than what the model was padded to, the
    # vertex IDs won't fit. Refuse rather than truncate silently.
    if bundle.n_links > n_nodes_max:
        raise ValueError(
            f"URDF has {bundle.n_links} links, classifier was trained "
            f"with n_nodes_max={n_nodes_max}. Re-train with --n-train "
            f"large enough to include a mechanism of this size."
        )
    dev = torch.device(device)
    inp = _build_per_arity_input(bundle.graph, arity, 30_000, dev,
                                    seed=0, n_nodes_pad=n_nodes_max)
    if inp is None:
        # Cycles existed but input build failed — fall back.
        sig = "stewart" if arity == 6 else "four_bar"
        probs = np.zeros(n_families, dtype=np.float64)
        probs[FAMILY_LABEL_BY_NAME[sig]] = 1.0
        return ClassificationResult(
            predicted_label=FAMILY_LABEL_BY_NAME[sig],
            predicted_family=sig,
            confidence=1.0, probs=probs, arity_used=arity,
            alpha_kappa=None, rule_based=True,
        )
    with torch.no_grad():
        cls_logits, _ = model(inp)
        probs_t = F.softmax(cls_logits, dim=-1)
    probs = probs_t.detach().cpu().numpy()
    pred_label = int(probs.argmax())
    pred_family = FAMILY_NAMES[pred_label]
    confidence = float(probs[pred_label])
    # α_κ is single-arity here so it's just 1.0; preserved for parity
    # with the multi-arity α_κ in the signed-link demo.
    return ClassificationResult(
        predicted_label=pred_label,
        predicted_family=pred_family,
        confidence=confidence,
        probs=probs,
        arity_used=arity,
        alpha_kappa=1.0,
        rule_based=False,
    )


def pretrain_all(
    output_dir: str | Path | None = None,
    arities: tuple[int, ...] = (4, 6),
    n_train: int = 80,
    n_epochs: int = 60,
    device: str = "cpu",
    seed: int = 0,
) -> dict[int, Path]:
    """Pre-train one classifier per arity in ``arities`` and save."""
    out_dir = Path(output_dir) if output_dir else PRETRAINED_DIR
    saved: dict[int, Path] = {}
    for arity in arities:
        print(f"\n=== pretrain classifier — arity k={arity} ===")
        model, n_nodes_max, acc = train_classifier(
            arity=arity, n_train=n_train, n_epochs=n_epochs,
            device=device, seed=seed,
        )
        path = save_classifier(out_dir / f"family_classifier_k{arity}.pt",
                                  model, arity=arity,
                                  n_nodes_max=n_nodes_max, train_acc=acc)
        print(f"  saved → {path}")
        saved[arity] = path
    return saved


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Pre-train kinematic-family classifiers for the demo."
    )
    ap.add_argument("--arities", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    pretrain_all(
        output_dir=args.output_dir,
        arities=tuple(args.arities),
        n_train=args.n_train,
        n_epochs=args.n_epochs,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CLASSIFIER_FORMAT_VERSION",
    "ClassificationResult",
    "FAMILY_NAMES",
    "PRETRAINED_DIR",
    "load_classifier",
    "predict_family",
    "pretrain_all",
    "save_classifier",
    "train_classifier",
]
