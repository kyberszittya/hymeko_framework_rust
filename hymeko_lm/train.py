"""Train a single FSR language model to maturity — checkpointing, periodic eval, sample generation.

Distinct from ``phase1_ab`` (which compares FSR vs a transformer): this trains ONE model with
checkpoint/resume and qualitative samples, for scale-up. One entry, ``--preset`` selects size (§6.5#13).
Default model is the Phase-1 winner: pre-norm residual + hard top-k spike gate (``spike_k=16``).

    uv run python -m hymeko_lm.train --preset base --corpus <text> --out runs/base
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from hymeko_lm.checkpoint import load_checkpoint, save_checkpoint
from hymeko_lm.config import FSRConfig, ResidualMode
from hymeko_lm.model import FSRLanguageModel
from hymeko_lm.text_data import BYTE_VOCAB, ByteCorpus

_PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"n_blocks": 16, "n_layers": 3, "seq": 64, "batch": 16, "steps": 300},
    "base": {"n_blocks": 32, "n_layers": 4, "seq": 128, "batch": 16, "steps": 1500},
    "scale": {"n_blocks": 48, "n_layers": 6, "seq": 256, "batch": 16, "steps": 3000},
}


def _lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.05) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * step / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


@torch.no_grad()
def _val_bpb(model: FSRLanguageModel, corpus: ByteCorpus, *, seq: int, batch: int, n: int,
             device: torch.device) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(12345)
    nats = [float(model.loss(*(t.to(device) for t in corpus.batch(batch, seq, "val", gen))))
            for _ in range(n)]
    return sum(nats) / len(nats) / math.log(2.0)


def _sample(model: FSRLanguageModel, device: torch.device, *, n_tokens: int = 200) -> str:
    gen = torch.Generator(device=device).manual_seed(0)
    seed = torch.randint(0, BYTE_VOCAB, (1, 1), generator=torch.Generator().manual_seed(0)).to(device)
    out = model.generate(seed, n_tokens, temperature=0.8, top_k=40, generator=gen)[0].tolist()
    return bytes(b for b in out if 9 <= b < 127).decode("ascii", errors="replace")


def train(preset: str, corpus_path: str, out_dir: str, *, device: str | None = None,
          seed: int = 0, resume: str | None = None, eval_every: int = 250) -> dict[str, object]:
    """Train, checkpointing to ``out_dir`` every ``eval_every`` steps. Returns the run summary."""
    p = _PRESETS[preset]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    corpus = ByteCorpus(corpus_path)
    torch.manual_seed(seed)
    if resume:
        model, start, _ = load_checkpoint(resume, map_location=str(dev))
        model = model.to(dev)
    else:
        cfg = FSRConfig(vocab_size=BYTE_VOCAB, n_blocks=p["n_blocks"], n_layers=p["n_layers"],
                        max_seq_len=p["seq"], gate_rank=32, residual_mode=ResidualMode.PRENORM, spike_k=16)
        model, start = FSRLanguageModel(cfg).to(dev), 0
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    out = Path(out_dir)
    gen = torch.Generator().manual_seed(1000 + seed)
    curve: list[tuple[int, float]] = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for step in range(start, p["steps"]):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, p["steps"], 3e-3)
        ids, tgt = (t.to(dev) for t in corpus.batch(p["batch"], p["seq"], "train", gen))
        loss = model.loss(ids, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub: Tensor.backward untyped
        opt.step()
        if (step + 1) % eval_every == 0 or step + 1 == p["steps"]:
            bpb = _val_bpb(model, corpus, seq=p["seq"], batch=p["batch"], n=40, device=dev)
            curve.append((step + 1, round(bpb, 4)))
            save_checkpoint(out / "ckpt.pt", model, step + 1, {"val_bpb": bpb, "preset": preset})
            print(f"step {step + 1}/{p['steps']}  val_bpb {bpb:.4f}", flush=True)
    wall = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated(dev) / 1e6) if dev.type == "cuda" else float("nan")
    sample = _sample(model, dev)
    summary = {"preset": preset, "device": dev.type, "n_params": model.n_parameters(),
               "final_val_bpb": curve[-1][1] if curve else None, "curve": curve,
               "wall_s": round(wall, 1), "peak_mb": round(peak_mb, 1), "sample": sample}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="base")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="runs/fsr")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args(argv)
    summary = train(a.preset, a.corpus, a.out, device=a.device, seed=a.seed, resume=a.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, indent=2))
    print("--- sample ---\n" + str(summary["sample"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
