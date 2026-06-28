"""Structural health map of an HSiKAN policy — *where* it broke, not just *that* it broke.

An MLP that diverges is a flat blown-up weight vector: you can see it failed, not where. HSiKAN's parameters are
**named by structure** (the self-loop ``W_self``, the down-/up-chain signed aggregations ``W+``/``W-``, the
Catmull-Rom edge activations, the highway gate, the heads), so a divergence localises to a *role*. This dumps,
per structural component: magnitude (max/mean ``|.|``) and a NaN / inf / denormal / blow-up flag — and, given a
sample observation, the per-vertex activation health on the kinematic hypergraph. The graphical half (§9) is a
structural health bar chart.

This is the structurally-accountable-AI thesis made operational: the failure is legible.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_DENORMAL_F32 = 1.1754944e-38   # smallest normal float32; below this (and > 0) is a subnormal (slow + a blow-up tell)
_BLOWUP = 1.0e4                  # |w| above this is treated as a numerical blow-up


def _role(name: str) -> str:
    """Map a parameter name to its HSiKAN structural role (the readable localisation)."""
    n = name.lower()
    layer = next((p.split(".")[0] for p in n.split("layers.")[1:]), "") if "layers." in n else ""
    tag = f" L{layer}" if layer else ""
    if "w_self" in n:
        return f"self-loop W_self{tag}"
    if "w_pos" in n or "a_pos" in n:
        return f"down-chain signed-agg W+{tag}"
    if "w_neg" in n or "a_neg" in n:
        return f"up-chain signed-agg W-{tag}"
    if "arc" in n or "incidence" in n:
        return f"signed incidence / arc weights{tag}"
    if any(k in n for k in ("coef", "spline", "cr", "act", "knot")):
        return f"Catmull-Rom edge activation{tag}"
    if any(k in n for k in ("skip", "highway", "gate")):
        return f"highway gate{tag}"
    if "actor_mean" in n or "actor.mean" in n:
        return "actor head"
    if "critic" in n:
        return "critic head"
    if "log_std" in n:
        return "action-noise log_std"
    return f"other{tag}"


@dataclass
class ParamHealth:
    name: str
    role: str
    numel: int
    max_abs: float
    mean_abs: float
    n_nan: int
    n_inf: int
    n_denormal: int

    @property
    def bad(self) -> bool:
        return bool(self.n_nan or self.n_inf or self.n_denormal or self.max_abs > _BLOWUP)


@dataclass
class Diagnosis:
    params: list[ParamHealth]
    vertex_activation_max: list[float] = field(default_factory=list)  # per-vertex |activation| (if a sample obs given)
    vertex_bad: list[int] = field(default_factory=list)              # vertices whose activation is NaN/inf
    forward_error: str | None = None        # set if the forward itself crashed — a fully-diverged policy can't run

    @property
    def bad(self) -> list[ParamHealth]:
        return [p for p in self.params if p.bad]

    @property
    def healthy(self) -> bool:
        return not self.bad and not self.vertex_bad and self.forward_error is None


def diagnose(model: torch.nn.Module, *, sample_obs: torch.Tensor | None = None) -> Diagnosis:
    """Per-parameter (and optionally per-vertex) structural health of ``model``.

    # Preconditions ``model`` has named parameters; for vertex health it exposes ``node_activations`` (HSiKAN).
    # Postconditions a :class:`Diagnosis`; ``healthy`` is False iff any component is NaN/inf/denormal/blown-up.
    """
    params: list[ParamHealth] = []
    for name, p in model.named_parameters():
        a = p.detach()
        absa = a.abs()
        params.append(ParamHealth(
            name=name, role=_role(name), numel=int(a.numel()),
            max_abs=float(absa.max()) if a.numel() else 0.0,
            mean_abs=float(absa.mean()) if a.numel() else 0.0,
            n_nan=int(torch.isnan(a).sum()), n_inf=int(torch.isinf(a).sum()),
            n_denormal=int(((absa > 0) & (absa < _DENORMAL_F32)).sum())))
    diag = Diagnosis(params=params)
    # the per-vertex map lives on the backbone; fall back to ``actor_backbone`` for an ActorCritic.
    na = getattr(model, "node_activations", None) or getattr(
        getattr(model, "actor_backbone", None), "node_activations", None)
    if sample_obs is not None and callable(na):
        try:
            with torch.no_grad():
                h = na(sample_obs)                   # (B, N, feat)
            per_v = h.abs().amax(dim=(0, 2))         # (N,) worst |activation| per vertex
            diag.vertex_activation_max = [float(x) for x in per_v]
            diag.vertex_bad = [int(i) for i in torch.nonzero(torch.isnan(h).any(dim=(0, 2))
                                                             | torch.isinf(h).any(dim=(0, 2))).flatten()]
        except Exception as exc:   # noqa: BLE001 — a diverged policy can crash its own forward; that IS the signal
            diag.forward_error = f"{type(exc).__name__}: {exc}"
    return diag


def format_diagnosis(diag: Diagnosis) -> str:
    """A readable structural health map (the textual output)."""
    lines = [f"HSiKAN structural health: {'HEALTHY' if diag.healthy else 'DIVERGED'}",
             f"  {'role':36} {'max|.|':>11} {'mean|.|':>10}  flags"]
    for p in sorted(diag.params, key=lambda q: q.max_abs, reverse=True):
        flags = " ".join(f for f, c in (("NaN", p.n_nan), ("Inf", p.n_inf), ("denorm", p.n_denormal)) if c)
        if p.max_abs > _BLOWUP and "Inf" not in flags:
            flags = (flags + " BLOWUP").strip()
        lines.append(f"  {p.role:36} {p.max_abs:11.3e} {p.mean_abs:10.3e}  {flags or 'ok'}")
    if diag.vertex_bad:
        lines.append(f"  diverged vertices (graph nodes): {diag.vertex_bad}")
    if diag.forward_error:
        lines.append(f"  forward UNRUNNABLE (severe divergence): {diag.forward_error}")
    return "\n".join(lines)


def plot_diagnosis(diag: Diagnosis, out_path: str | Path) -> Path:
    """Structural health bar chart: per-component max|.| (log), bad components highlighted (the §9 plotted output)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import math
    ps = sorted(diag.params, key=lambda q: q.max_abs if math.isfinite(q.max_abs) else math.inf, reverse=True)
    roles = [p.role for p in ps]
    # cap non-finite (inf/NaN) to a large finite so the log bar renders cleanly; the crimson colour marks it bad.
    vals = [1e30 if not math.isfinite(p.max_abs) else max(p.max_abs, 1e-30) for p in ps]
    colors = ["crimson" if p.bad else "steelblue" for p in ps]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(ps))))
    ax.barh(range(len(ps)), vals, color=colors)
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels(roles, fontsize=7)
    ax.set_xscale("log")
    ax.axvline(_BLOWUP, color="crimson", ls="--", lw=1, label=f"blow-up ({_BLOWUP:.0e})")
    ax.invert_yaxis()
    ax.set_xlabel("max |weight|  (log)")
    ax.set_title(f"HSiKAN structural health — {'HEALTHY' if diag.healthy else 'DIVERGED'}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _load_galambos_policy(ckpt: str, *, difficulty: float = 0.3) -> tuple[Any, torch.Tensor]:
    """Build the galambos HSiKAN policy matching ``ckpt`` and load it; return ``(policy, a sample obs)``."""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    from hymeko_rl.policy import build_policy
    env = PlanarGraspEnv(robot=None, max_steps=160, difficulty=difficulty)
    feat = int(env.observation_space.shape[1])   # type: ignore[index]
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg, hidden=64)
    ac.load_state_dict(torch.load(ckpt, map_location="cpu"))
    obs, _ = env.reset(seed=0)
    return ac, torch.as_tensor(obs[None], dtype=torch.float32)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="galambos HSiKAN policy .pt (an ActorCritic state_dict)")
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--plot", default=None, help="write the structural health bar chart here")
    a = ap.parse_args(argv)
    ac, obs = _load_galambos_policy(a.ckpt, difficulty=a.difficulty)
    diag = diagnose(ac, sample_obs=obs)
    print(format_diagnosis(diag))
    if a.plot:
        print(f"plot: {plot_diagnosis(diag, a.plot)}")
    print(json.dumps({"healthy": diag.healthy, "n_bad": len(diag.bad),
                      "bad_roles": [p.role for p in diag.bad]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
