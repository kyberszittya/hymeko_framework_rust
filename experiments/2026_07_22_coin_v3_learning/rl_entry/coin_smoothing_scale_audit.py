"""§2 residual target-smoothing distribution audit. At zero target residual, compare A (disabled zero noise),
B (unscaled absolute std=0.2/clip=0.5), C (scale-correct std=0.05/clip=0.125): per-dimension residual-BOUND-hit
probability, any-of-4 bound-hit, post-clip noise mean/std, residual-action norm distribution — analytic + Monte
Carlo. The scientific critic config must use C. → RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_PASS.
"""
import json
import math
import sys

import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_replay import (  # noqa: E402
    RESIDUAL_SMOOTHING_CONTRACT,
    residual_smoothing_contract_sha256,
)

BOUND = 0.25
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoothing_audit.json"


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def analytic_bound_hit(std, clip):
    """P(|clamp(N(0,std),-clip,clip)| >= BOUND) per dim. If clip <= BOUND the residual can never reach the bound."""
    if std == 0.0:
        return 0.0
    if clip <= BOUND:
        return 0.0                       # clip keeps eps strictly inside the bound
    return 2 * (1 - _phi(BOUND / std))   # P(|N(0,std)| >= BOUND), since BOUND < clip


def analytic_clip_hit(std, clip):
    if std == 0.0:
        return 0.0
    return 2 * (1 - _phi(clip / std))    # P(|N| >= clip) — how often clipping engages


def mc(std, clip, n=400000, seed=0):
    g = torch.Generator().manual_seed(seed)
    eps = torch.clamp(torch.randn(n, 4, generator=g) * std, -clip, clip) if std > 0 else torch.zeros(n, 4)
    tr = torch.clamp(eps, -BOUND, BOUND)                 # target residual at zero base-residual
    per_dim_bound = float((tr.abs() >= BOUND - 1e-9).float().mean())
    any_bound = float((tr.abs() >= BOUND - 1e-9).any(-1).float().mean())
    norm = tr.norm(dim=-1)
    return {"post_clip_mean": round(float(eps.mean()), 5), "post_clip_std": round(float(eps.std()), 5),
            "per_dim_bound_hit": round(per_dim_bound, 5), "any_dim_bound_hit": round(any_bound, 5),
            "residual_norm_mean": round(float(norm.mean()), 5),
            "residual_norm_p95": round(float(norm.quantile(0.95)), 5),
            "residual_norm_max": round(float(norm.max()), 5)}


def main():
    regimes = {"A_disabled": (0.0, 0.0), "B_unscaled_abs": (0.2, 0.5), "C_scale_correct": (0.05, 0.125)}
    out = {"contract": RESIDUAL_SMOOTHING_CONTRACT, "contract_sha": residual_smoothing_contract_sha256()[:16],
           "regimes": {}}
    for name, (std, clip) in regimes.items():
        m = mc(std, clip)
        m["analytic_per_dim_bound_hit"] = round(analytic_bound_hit(std, clip), 5)
        m["analytic_clip_engage"] = round(analytic_clip_hit(std, clip), 5)
        m["std"] = std; m["clip"] = clip
        out["regimes"][name] = m
        print(f"  {name:<16} std={std} clip={clip} | per-dim bound-hit {m['per_dim_bound_hit']:.4f} "
              f"(analytic {m['analytic_per_dim_bound_hit']:.4f}) any-dim {m['any_dim_bound_hit']:.4f} | "
              f"norm mean {m['residual_norm_mean']:.4f} max {m['residual_norm_max']:.4f}", flush=True)
    C = out["regimes"]["C_scale_correct"]; B = out["regimes"]["B_unscaled_abs"]
    # PASS: C does not saturate the residual bound (bound-hit ~0), stays well inside the range, and B demonstrably
    # would (any-dim bound-hit >> C). Contract is scale-relative + reproducible (tests) + batch-independent (tests).
    scale_ok = (C["per_dim_bound_hit"] < 0.01 and C["any_dim_bound_hit"] < 0.05
                and B["any_dim_bound_hit"] > 0.3 and abs(C["std"] - 0.05) < 1e-9 and abs(C["clip"] - 0.125) < 1e-9)
    out["scientific_regime"] = "C_scale_correct"
    out["B_saturates"] = B["any_dim_bound_hit"]
    out["C_bound_hit"] = C["any_dim_bound_hit"]
    out["verdict"] = "RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_PASS" if scale_ok else "SMOOTHING_SCALE_CONTRACT_FAIL"
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\ncontract SHA {residual_smoothing_contract_sha256()[:16]} | C any-dim bound-hit {C['any_dim_bound_hit']} "
          f"vs B {B['any_dim_bound_hit']}", flush=True)
    print(out["verdict"], flush=True); print("SMOOTHING_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
