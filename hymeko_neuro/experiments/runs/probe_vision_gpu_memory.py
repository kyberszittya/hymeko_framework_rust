"""GPU memory probe for the 4 fast vision models (closes the open
'less GPU memory than CNN' claim from the 2026-05-29 retraction report).

For each of cnn / mlp / hgnn / hsikan: build the model at h=32 on 28x28x1
input, run one forward+backward at batch=128, print
  torch.cuda.max_memory_allocated()  (peak GPU bytes for activations+params+grads)
  torch.cuda.max_memory_reserved()   (peak GPU bytes incl. caching allocator slack)
  n_params
The probe is cuda.reset_peak_memory_stats()'d between models so the
numbers are per-model, not cumulative.

Usage:
  PYTHONPATH=$PWD python -m hymeko_neuro.experiments.runs.probe_vision_gpu_memory
"""
from __future__ import annotations

import json

import torch

from hymeko_neuro.experiments.vision.vision_bench_cell import build_model

MODELS = ("cnn", "mlp", "hgnn", "hsikan")
B = 128
H, W = 28, 28


def probe(name: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model = build_model(name, h=H, w=W, n_classes=10, hidden=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(B, 1, H, W, device=device)
    y = torch.randint(0, 10, (B,), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak_alloc_b = torch.cuda.max_memory_allocated()
    peak_reserved_b = torch.cuda.max_memory_reserved()
    return {
        "model": name,
        "n_params": int(n_params),
        "peak_alloc_mib": round(peak_alloc_b / 2**20, 1),
        "peak_reserved_mib": round(peak_reserved_b / 2**20, 1),
    }


if __name__ == "__main__":
    out = [probe(m) for m in MODELS]
    print(json.dumps(out, indent=2))
    print()
    print(f"{'model':10s} {'params':>10s} {'peak_alloc_MiB':>16s} {'peak_reserved_MiB':>20s}")
    for r in out:
        print(f"  {r['model']:8s} {r['n_params']:>10d} {r['peak_alloc_mib']:>16.1f} {r['peak_reserved_mib']:>20.1f}")
