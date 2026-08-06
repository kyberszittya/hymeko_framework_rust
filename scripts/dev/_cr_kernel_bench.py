"""Parity + benchmark for the fused Triton CR activation vs eager CR and fixed activations."""
from __future__ import annotations

import time

import torch

from hymeko_rl.agents.cr_kernel import fused_catmull_rom
from hymeko_rl.agents.policy import _catmull_rom

DEV = "cuda"
C, G = 64, 5


def parity() -> None:
    torch.manual_seed(0)
    coef = torch.randn(C, G, device=DEV, requires_grad=True)
    x = (torch.randn(256, 6, C, device=DEV) * 1.5).requires_grad_(True)
    out_f = fused_catmull_rom(x, coef)
    gx_f, gc_f = torch.autograd.grad(out_f.sum(), [x, coef])
    coef2 = coef.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    out_e = _catmull_rom(coef2, x2, G)
    gx_e, gc_e = torch.autograd.grad(out_e.sum(), [x2, coef2])
    print("PARITY (fused Triton vs eager):")
    print("  fwd       allclose=%s  maxdiff=%.2e" % (torch.allclose(out_f, out_e, atol=1e-4),
                                                     (out_f - out_e).abs().max().item()))
    print("  grad_x    allclose=%s  maxdiff=%.2e" % (torch.allclose(gx_f, gx_e, atol=1e-4),
                                                     (gx_f - gx_e).abs().max().item()))
    print("  grad_coef allclose=%s  maxdiff=%.2e" % (torch.allclose(gc_f, gc_e, atol=1e-3),
                                                     (gc_f - gc_e).abs().max().item()))


def _bench(fn, x, coef, *, bwd: bool, iters: int = 500) -> float:
    for _ in range(30):
        if bwd:
            torch.autograd.grad(fn(x, coef).sum(), x)
        else:
            with torch.no_grad():
                fn(x, coef)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        if bwd:
            torch.autograd.grad(fn(x, coef).sum(), x)
        else:
            with torch.no_grad():
                fn(x, coef)
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000.0


def benchmark() -> None:
    coef = torch.randn(C, G, device=DEV)
    acts = {
        "relu": lambda x, c: torch.relu(x),
        "tanh": lambda x, c: torch.tanh(x),
        "gelu": lambda x, c: torch.nn.functional.gelu(x),
        "CR eager": lambda x, c: _catmull_rom(c, x, G),
        "CR compile": torch.compile(lambda x, c: _catmull_rom(c, x, G)),
        "CR fused": fused_catmull_rom,
    }
    print("\nBENCHMARK (GPU, 256x6x64, ms/call):")
    print("  %-12s %8s %8s" % ("activation", "fwd", "fwd+bwd"))
    for name, fn in acts.items():
        x = (torch.randn(256, 6, C, device=DEV)).requires_grad_(True)
        try:
            f = _bench(fn, x, coef, bwd=False)
            fb = _bench(fn, x, coef, bwd=True)
            print("  %-12s %8.4f %8.4f" % (name, f, fb))
        except Exception as e:  # noqa: BLE001
            print("  %-12s FAILED: %s" % (name, str(e)[:60]))


if __name__ == "__main__":
    print("cuda:", torch.cuda.is_available())
    parity()
    benchmark()
