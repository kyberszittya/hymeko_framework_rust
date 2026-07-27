"""CIP-PNP-01 gate runner: drive the CIP-0 lifecycle and evaluate PNP-0..4.

Usage (from the worktree root, with the venv + hymeko CLI available)::

    PYTHONPATH=. python -m scenarios.pick_place.run_pnp --seed 1

Writes gate JSON + a trajectory plot (and attempts a GIF) under
``reports/2026-07-27-cip-pick-place/``. No RL: the option is realized by the
scripted expert. PNP-4 requires ONE complete, externally certified trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hymeko_control.cip.protocol import CIP0Adapter
from hymeko_control.cip.runtime import CIP0Runtime

from . import load_model
from .adapter import CHAIN, PickPlaceCIPAdapter

_OUT = Path("reports/2026-07-27-cip-pick-place")


def _episode_summary(records: list) -> dict[str, Any]:
    visited: list[str] = []
    both_contact_ever = False
    max_lifted = 0.0
    dropped = False
    any_death = False
    carry_ok = True
    placed_final = False
    all_steps: list[dict] = []
    for rec in records:
        visited.append(rec.mode)
        prov = rec.trace.provenance
        max_lifted = max(max_lifted, float(prov.get("max_lifted", 0.0)))
        dropped = dropped or bool(prov.get("dropped"))
        any_death = any_death or bool(prov.get("any_death"))
        if rec.mode == "CARRY":
            carry_ok = carry_ok and bool(prov.get("carry_contact_ok", True))
        for s in rec.trace.signals:
            all_steps.append(s)
            if s.get("both_contact", 0.0) >= 1.0:
                both_contact_ever = True
    if records:
        placed_final = bool(records[-1].trace.provenance.get("placed_stable"))
    reached_chain_max = max((CHAIN.index(m) for m in visited), default=0)
    return {
        "visited_modes": visited,
        "final_mode": records[-1].next_mode if records else None,
        "final_certificate_passed": bool(records[-1].certificate.passed) if records else False,
        "both_contact_ever": both_contact_ever,
        "max_lifted": max_lifted,
        "dropped": dropped,
        "any_death": any_death,
        "carry_contact_ok": carry_ok,
        "placed_stable_final": placed_final,
        "reached_chain_max": reached_chain_max,
        "n_ticks": len(records),
        "n_steps": len(all_steps),
        "_all_steps": all_steps,
    }


def _gates(model, adapter: PickPlaceCIPAdapter, summ: dict, lift_thresh: float) -> dict:
    pnp0 = isinstance(adapter, CIP0Adapter) and model.name.startswith("cip_pnp_01")
    pnp1 = summ["both_contact_ever"] and not summ["any_death"]
    pnp2 = summ["both_contact_ever"] and summ["max_lifted"] >= lift_thresh
    pnp3 = (
        summ["reached_chain_max"] >= CHAIN.index("CARRY")
        and summ["carry_contact_ok"]
        and not summ["dropped"]
    )
    pnp4 = (
        summ["placed_stable_final"]
        and not summ["dropped"]
        and not summ["any_death"]
        and summ["final_mode"] == "SETTLE"
        and summ["final_certificate_passed"]
    )
    gates = {"PNP-0": pnp0, "PNP-1": pnp1, "PNP-2": pnp2, "PNP-3": pnp3, "PNP-4": pnp4}
    highest = "none"
    for name in ("PNP-0", "PNP-1", "PNP-2", "PNP-3", "PNP-4"):
        if gates[name]:
            highest = name
        else:
            break
    return {"gates": gates, "highest_passed": highest}


def _plot(steps: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not steps:
        return False
    xs = range(len(steps))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(xs, [s["lifted"] for s in steps], label="lifted (m)")
    ax[0].plot(xs, [s["obj_to_target"] for s in steps], label="obj_to_target (m)")
    ax[0].axhline(0.035, ls="--", c="gray", lw=0.8, label="lift_thresh")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("m")
    ax[0].set_title("CIP-PNP-01 certified trajectory")
    ax[1].plot(xs, [s["both_contact"] for s in steps], label="both_contact")
    ax[1].plot(xs, [s["placed_stable"] for s in steps], label="placed_stable")
    ax[1].plot(xs, [s["settled"] for s in steps], label="settled")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("bool")
    ax[1].set_xlabel("env step")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _gif(seed: int, expert_version: int, path: Path) -> str | None:
    """Best-effort GIF of a fresh certified rollout. Returns path or a reason string."""
    try:
        import imageio.v2 as imageio
        import mujoco
        import numpy as np
        from hymeko_rl.viz.render_pick_place import fanuc_pick_env, pick_camera
        env = fanuc_pick_env(require_settle=True, target_bin=True, expert_version=expert_version)
        env.reset(seed=seed)
        # default offscreen framebuffer is 640x480; enlarge for slide-grade frames
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.model, height=720, width=960)
        cam = pick_camera()
        frames = []
        for k in range(getattr(env, "max_steps", 620)):
            _o, _r, term, trunc, _i = env.step(np.asarray(env.expert_action, dtype=np.float64))
            if k % 8 == 0:
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render())
            if term or trunc:
                break
        imageio.mimsave(path, frames, fps=15)
        return str(path)
    except Exception as exc:  # rendering is fragile headless; never fake it
        return f"gif_unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expert-version", type=int, default=3)
    ap.add_argument("--gif", action="store_true", help="attempt a GIF render")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    adapter = PickPlaceCIPAdapter(model=model, seed=args.seed,
                                  expert_version=args.expert_version)
    runtime = CIP0Runtime(model=model, adapter=adapter)
    records = runtime.run(max_ticks=40)

    summ = _episode_summary(records)
    steps = summ.pop("_all_steps")
    result = _gates(model, adapter, summ, adapter.lift_thresh)
    result["seed"] = args.seed
    result["expert_version"] = args.expert_version
    result["summary"] = summ
    plotted = _plot(steps, _OUT / "pnp_trajectory.png")
    result["plot"] = "pnp_trajectory.png" if plotted else None
    if args.gif:
        result["gif"] = _gif(args.seed, args.expert_version, _OUT / "pnp_trajectory.gif")

    (_OUT / "pnp_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "gates": result["gates"],
                      "highest_passed": result["highest_passed"],
                      "final_mode": summ["final_mode"],
                      "placed_stable_final": summ["placed_stable_final"],
                      "max_lifted": round(summ["max_lifted"], 4),
                      "visited": summ["visited_modes"]}, indent=2))


if __name__ == "__main__":
    main()
