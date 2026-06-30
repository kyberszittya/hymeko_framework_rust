"""Phase-0 smoke: train the FSR LM on the lag-copy toy; assert it learns to route.

One entry, ``mode`` selects scale (§6.5#13: no ``_v2`` files). This is the de-risk run, not
the Phase-1 go/no-go A/B vs a matched transformer (that is a separate, multi-seed experiment).

    uv run python -m hymeko_lm.smoke --mode smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from hymeko_lm.config import FSRConfig
from hymeko_lm.data import make_lag_copy_batch
from hymeko_lm.model import FSRLanguageModel

_MODES: dict[str, dict[str, int]] = {
    "smoke": {"vocab": 32, "n_blocks": 8, "n_layers": 3, "seq_len": 32, "lag": 4,
              "batch": 32, "steps": 200},
    "full": {"vocab": 64, "n_blocks": 16, "n_layers": 4, "seq_len": 64, "lag": 8,
             "batch": 64, "steps": 800},
}


def run_smoke(mode: str = "smoke", *, device: str | None = None, seed: int = 0) -> dict[str, object]:
    """Train the toy and return provenance + the learning curve endpoints.

    # Postconditions returns a dict with ``initial_loss``, ``final_loss``, ``uniform_loss``
    (``ln vocab``), ``wall_s``, ``peak_mb``, ``n_params``; ``final_loss < uniform_loss`` is the
    smoke gate (routing learned).
    """
    cfg_d = _MODES[mode]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    cfg = FSRConfig(vocab_size=cfg_d["vocab"], n_blocks=cfg_d["n_blocks"],
                    n_layers=cfg_d["n_layers"], max_seq_len=cfg_d["seq_len"])
    model = FSRLanguageModel(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(cfg_d["steps"]):
        ids, tgt = make_lag_copy_batch(cfg_d["batch"], cfg_d["seq_len"], cfg_d["vocab"],
                                       cfg_d["lag"], gen)
        loss = model.loss(ids.to(dev), tgt.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub leaves Tensor.backward untyped
        opt.step()
        losses.append(float(loss.detach()))
    wall = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated(dev) / 1e6) if dev.type == "cuda" else float("nan")

    return {
        "mode": mode, "device": dev.type, "n_params": model.n_parameters(),
        "initial_loss": round(sum(losses[:5]) / 5, 4), "final_loss": round(sum(losses[-5:]) / 5, 4),
        "uniform_loss": round(math.log(cfg_d["vocab"]), 4), "wall_s": round(wall, 2),
        "peak_mb": round(peak_mb, 1), "steps": cfg_d["steps"], "lag": cfg_d["lag"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(_MODES), default="smoke")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    report = run_smoke(a.mode, device=a.device, seed=a.seed)
    print(json.dumps(report, indent=2))
    if not report["final_loss"] < report["uniform_loss"]:  # type: ignore[operator]
        print("SMOKE FAIL: final loss did not beat the uniform (no-routing) baseline")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
