"""Run and visualise a cart-pole policy — including one loaded back from its ``.hymeko`` storage.

Closes the loop: a policy stored as a HyMeKo hypergraph (``policy_store``) is loaded, run in the cart-pole
MuJoCo sim, and rendered to a GIF (offscreen) plus a trajectory plot. The architecture is *inferred from the
stored tensor shapes*, so the artifact is self-sufficient — no separate config.

    python -m hymeko_rl.render_inverted_pendulum --policy data/nn/cartpole_hsikan_policy.hymeko --out reports/gifs/cartpole
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.evaluate import render_episode_gif
from hymeko_rl.policy import build_policy
from hymeko_rl.policy_store import hymeko_to_policy, read_provenance
from hymeko_rl.train_inverted_pendulum import GreedyPolicy


def _hsikan_dims(sd: "dict[str, torch.Tensor]", prefix: str) -> tuple[int, int, int]:
    """Infer ``(hidden, per-vertex feat, n_layers)`` of an hsikan backbone from its stored tensor shapes."""
    w0 = sd[f"{prefix}.layers.0.w_self.weight"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith(f"{prefix}.layers."))
    return int(w0.shape[0]), int(w0.shape[1]), n_layers


def _require_hsikan(sd: "dict[str, torch.Tensor]", prefix: str, env: InvertedPendulumEnv) -> None:
    if f"{prefix}.a_pos" not in sd:
        raise ValueError("render reconstructs hsikan-family policies only; this is an mlp policy")
    if int(sd[f"{prefix}.a_pos"].shape[0]) != env.hg.n_vertices:
        raise ValueError("stored policy's vertex count does not match the cart-pole env")


def load_policy_from_hymeko(path: str | Path, env: InvertedPendulumEnv) -> GreedyPolicy:
    """Reconstruct a greedy policy from a stored ``.hymeko`` (or ``.pt``) — architecture inferred from the
    tensor shapes, weights loaded bit-exact. Dispatches on the stored keys to the right actor:
    ``actor_mean`` → PPO ``ActorCritic``; ``mu``/``log_std`` → SAC ``SquashedGaussianActor``; ``head`` →
    DDPG/TD3 ``DeterministicActor``. (hsikan-family only; mlp raises a clear error.)

    # Postconditions a policy exposing ``action_mean`` whose weights equal the stored ones.
    """
    path = Path(path)
    sd = (hymeko_to_policy(path) if path.suffix == ".hymeko"
          else {k: v.float() for k, v in torch.load(path, weights_only=True).items()})
    action_dim = int(env.action_space.shape[0])   # type: ignore[index]
    scale = float(env.force_mag)
    ac: GreedyPolicy
    if "actor_mean.weight" in sd:                  # PPO ActorCritic
        _require_hsikan(sd, "actor_backbone", env)
        hidden, feat, n_layers = _hsikan_dims(sd, "actor_backbone")
        ac = build_policy("hsikan", obs_dim=feat, action_dim=action_dim, hg_state=env.hg,
                          hidden=hidden, n_layers=n_layers)
    elif "mu.weight" in sd:                        # SAC SquashedGaussianActor
        from hymeko_rl.sac import build_sac
        _require_hsikan(sd, "backbone", env)
        hidden, feat, n_layers = _hsikan_dims(sd, "backbone")
        ac = build_sac("hsikan", obs_dim=feat, flat_dim=env.hg.n_vertices * feat, action_dim=action_dim,
                       action_scale=scale, n_critics=1, hidden=hidden, hg_state=env.hg, n_layers=n_layers)[0]
    elif "head.weight" in sd:                      # DDPG/TD3 DeterministicActor
        from hymeko_rl.ddpg import build_offpolicy
        _require_hsikan(sd, "backbone", env)
        hidden, feat, n_layers = _hsikan_dims(sd, "backbone")
        ac = build_offpolicy("hsikan", obs_dim=feat, flat_dim=env.hg.n_vertices * feat,
                             action_dim=action_dim, action_scale=scale, n_critics=1, hidden=hidden,
                             hg_state=env.hg, n_layers=n_layers)[0]
    else:
        raise ValueError("unrecognized stored policy (no actor_mean / mu / head keys)")
    ac.load_state_dict(sd)
    ac.eval()
    return ac


def greedy_action_fn(ac: GreedyPolicy) -> "Any":
    """An ``action_fn(env, obs) -> np.ndarray`` that takes the deterministic policy mean (for eval/render)."""
    @torch.no_grad()
    def act(_env: Any, obs: np.ndarray) -> np.ndarray:
        mean = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
        return mean.squeeze(0).numpy()
    return act


def side_camera() -> "Any":
    """A side view framing the X--Z plane (cart slides on X, pole swings in X--Z)."""
    import mujoco
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 2.4
    cam.lookat[:] = [0.0, 0.0, 0.4]
    return cam


def _trajectory(env: InvertedPendulumEnv, ac: GreedyPolicy, *, seed: int) -> dict[str, list[float]]:
    """Roll one greedy episode, recording cart position and pole angle per step (no render)."""
    act = greedy_action_fn(ac)
    obs, _ = env.reset(seed=seed)
    cart, pole = [], []
    for _ in range(env.max_steps):
        obs, _r, term, trunc, info = env.step(act(env, obs))
        cart.append(float(info["cart_x"]))
        pole.append(float(info["pole_angle"]))
        if term or trunc:
            break
    return {"cart_x": cart, "pole_angle": pole}


def _plot_trajectory(traj: dict[str, list[float]], out: Path, *, angle_limit: float, cart_limit: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(traj["pole_angle"])
    fig, (axp, axc) = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True)
    axp.plot(traj["pole_angle"], color="#db2777")
    for y in (angle_limit, -angle_limit):
        axp.axhline(y, ls="--", c="gray", lw=1)
    axp.set_ylabel("pole angle (rad)")
    axp.set_title(f"balanced {n} steps")
    axc.plot(traj["cart_x"], color="#2563eb")
    for y in (cart_limit, -cart_limit):
        axc.axhline(y, ls="--", c="gray", lw=1)
    axc.set_ylabel("cart x (m)")
    axc.set_xlabel("step")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _hud(algo: str = "", backbone: str = "") -> Callable[[dict[str, Any]], list[str]]:
    """Build the cart-pole HUD callback. The title line names the algorithm + backbone (architecture);
    the per-frame lines are step, cumulative return (= steps balanced), pole angle, cart x, applied force."""
    title = " · ".join(p for p in (algo.upper(), backbone) if p)

    def hud(ctx: dict[str, Any]) -> list[str]:
        deg = float(ctx.get("pole_angle", 0.0)) * 180.0 / np.pi
        status = ("FELL" if ctx.get("fell") else
                  "OUT OF BOUNDS" if ctx.get("out_of_bounds") else "BALANCING")
        lines = [title] if title else []
        lines += [f"step {ctx['step']}", f"return {ctx['return']:.0f}",
                  f"pole {deg:+.1f} deg", f"cart {float(ctx.get('cart_x', 0.0)):+.2f} m"]
        if ctx.get("action") is not None:
            lines.append(f"force {float(np.asarray(ctx['action']).reshape(-1)[0]):+.1f} N")
        lines.append(status)
        return lines
    return hud


def render_run(ac: GreedyPolicy, env: InvertedPendulumEnv, out_dir: str | Path, *,
               seed: int = 0, fps: int = 30, algo: str = "", backbone: str = "") -> dict[str, Path]:
    """Render one greedy episode to a GIF (with a HUD naming algo+backbone) plus a trajectory PNG.

    The GIF/PNG stems encode the algorithm + backbone (e.g. ``cartpole_td3_hsikan.gif``).
    # Postconditions writes ``<out_dir>/<stem>.gif`` and ``<out_dir>/<stem>_trajectory.png``.
    """
    out_dir = Path(out_dir)
    stem = "_".join(p for p in ("cartpole", algo, backbone) if p)
    gif = render_episode_gif(env, greedy_action_fn(ac), out_dir / stem,
                             seed=seed, width=480, height=360, fps=fps, camera=side_camera(),
                             overlay=_hud(algo, backbone))
    png = _plot_trajectory(_trajectory(env, ac, seed=seed), out_dir / f"{stem}_trajectory.png",
                           angle_limit=env.angle_limit, cart_limit=env.cart_limit)
    return {"gif": gif, "trajectory": png}


def _infer_backbone(path: str | Path) -> str:
    """Best-effort backbone label from a stored ``.hymeko`` (signed adjacency ⇒ hsikan-family, else mlp).
    signedkan vs hsikan is not recoverable from weights, so the provenance block / ``--backbone`` is exact."""
    if Path(path).suffix != ".hymeko":
        return ""
    return "hsikan" if ".a_pos" in Path(path).read_text(encoding="utf-8") else "mlp"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="data/nn/cartpole_hsikan_policy.hymeko",
                    help=".hymeko (stored) or .pt checkpoint")
    ap.add_argument("--out", default="reports/gifs")
    ap.add_argument("--algo", default="", help="algorithm label (default: the .hymeko provenance block)")
    ap.add_argument("--backbone", default="", help="architecture label (default: provenance, else inferred)")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)
    env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    ac = load_policy_from_hymeko(a.policy, env)
    # Auto-label from the stored provenance (algo/backbone), overridable by the flags.
    prov = read_provenance(a.policy) if Path(a.policy).suffix == ".hymeko" else {}
    algo = a.algo or prov.get("algo", "")
    backbone = a.backbone or prov.get("backbone", "") or _infer_backbone(a.policy)
    paths = render_run(ac, env, a.out, seed=a.seed, fps=a.fps, algo=algo, backbone=backbone)
    print(f"wrote {paths['gif']} and {paths['trajectory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
