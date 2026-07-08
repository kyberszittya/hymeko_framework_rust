"""Render MetaWorld coffee-push episodes to GIFs (presentation artifacts for the Ito+Kato scenario).

Rolls the scripted ``SawyerCoffeePushV3Policy`` (+ per-episode action noise — the same protocol the CIP runs use)
and writes animated GIFs via the shared :func:`hymeko_rl.eval.evaluate._write_gif` / :func:`compare_gif` (§6.1 —
reuse the canonical GIF writer + stamping, do not re-implement). Low noise → a clean success; high noise → a
failure; a side-by-side compare GIF shows both. MetaWorld renders itself (``render_mode="rgb_array"``), so this
does **not** go through the HyMeKo-env ``render_episode_frames`` path.

Requires the ``metaworld`` package. No training; read-only rollouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def render_coffee_push_gif(seed: int, noise: float, out_path: "str | Path", *, max_steps: int = 150, fps: int = 20,
                           stride: int = 2, downsample: int = 2) -> "tuple[Path, int, list[np.ndarray]]":
    """Roll one coffee-push episode (scripted policy + ``noise``) → (gif path, mw_success, kept frames).

    ``stride`` keeps every ``stride``-th step and ``downsample`` shrinks each frame by that factor, so the GIF is
    a few MB (slide/email-friendly) rather than tens of MB. # Preconditions ``metaworld`` importable.
    """
    import warnings

    import metaworld.policies as mp

    from hymeko_rl.eval.cip.metaworld_cip import _REAL_TASKS
    from hymeko_rl.eval.evaluate import _write_gif
    env_key, policy_name = _REAL_TASKS["coffee_push"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no py.typed
        env: Any = ENVS[env_key](render_mode="rgb_array")
        policy = getattr(mp, policy_name)()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        frames: list[np.ndarray] = []
        success = 0
        for step in range(max_steps):
            if step % stride == 0:
                frames.append(np.asarray(env.render(), dtype=np.uint8)[::downsample, ::downsample])
            act = np.clip(np.asarray(policy.get_action(obs), np.float32)
                          + rng.normal(0, noise, 4).astype(np.float32), -1.0, 1.0)
            obs, _r, terminated, truncated, info = env.step(act)
            success = max(success, int(info.get("success", 0)))
            if terminated or truncated:
                frames.append(np.asarray(env.render(), dtype=np.uint8)[::downsample, ::downsample])
                break
    path = _write_gif(frames, out_path, fps=fps, stamp=f"coffee-push  noise={noise:.2f}  success={success}")
    print(f"[cip-gif] noise={noise:.2f} success={success} frames={len(frames)} -> {path}", flush=True)
    return path, success, frames


def make_coffee_push_gifs(out_dir: Path, seed: int = 0) -> "dict[str, Any]":
    """Produce the success / failure / side-by-side compare GIFs for the coffee-push scenario."""
    from hymeko_rl.eval.evaluate import compare_gif
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_path, ok_success, ok_frames = render_coffee_push_gif(seed, 0.0, out_dir / "coffee_push_success.gif")
    bad_path, bad_success, bad_frames = render_coffee_push_gif(seed + 7, 0.9, out_dir / "coffee_push_failure.gif")
    compare = compare_gif([ok_frames, bad_frames], out_dir / "coffee_push_compare.gif", fps=30,
                          stamp="left: clean (noise 0.0)   |   right: noisy (noise 0.9)")
    result = {"success_gif": str(ok_path), "failure_gif": str(bad_path), "compare_gif": str(compare),
              "success_reached": {"clean": ok_success, "noisy": bad_success}}
    print(f"[cip-gif] wrote 3 GIFs to {out_dir}", flush=True)
    return result


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="Render MetaWorld coffee-push GIFs (success / failure / compare)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="reports/gifs/metaworld_coffee_push")
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out.startswith("reports") else experiment_dir(args.out, "cip_metaworld_gifs")
    make_coffee_push_gifs(out_dir, int(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
