"""R11.7A U4 — the O0 parity gate (production smoke).

Proves the unification is bit-exact for the reference coin: on the SAME reconstructed demo state, the NEW
generated path (``build_manipulation_rig(object_spec=COIN_OBJECT, robot=BALLTIP)``) produces an environment
identical to the OLD direct call (``reconstruct_handoff(coin_shape="cylinder", disk_radius_override=0.020,
geom="POINT", arm_mjcf_transform=_ball_tf)`` — unchanged, so it reproduces the pre-refactor behavior). We
compare the compiled MuJoCo model (manipuland geometry + mass), the FULL collision contract
(contype/conaffinity over every geom), the topology (nq/nv/nu/ngeom/nbody), and a deterministic rollout
trace (same seed + same actions → identical qpos/qvel). No O1–O4 result is valid until this passes.

Run:  python -m hymeko_rl.experiments.r11_7a_o0_parity
"""
from __future__ import annotations

import sys

import numpy as np

from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff
from hymeko_rl.coin_delivery.manipulation_rig import build_manipulation_rig
from hymeko_rl.env.object_spec import COIN_OBJECT
from hymeko_rl.experiments.video_coin_variants import BALLTIP_ROBOT, FAMS, _R, _ball_tf, _setup


def _model_fingerprint(model) -> dict:
    """Bit-comparable model summary: manipuland geom+mass, full collision arrays, topology."""
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "ngeom": int(model.ngeom), "nbody": int(model.nbody),
        "disk_geom_type": int(model.geom_type[gid]),
        "disk_geom_size": model.geom_size[gid].copy(),
        "disk_mass": float(model.body_mass[bid]),
        "contype": model.geom_contype.copy(),          # the collision contract (ARM_LEGALITY)
        "conaffinity": model.geom_conaffinity.copy(),
    }


def _rollout_trace(rl, *, seed: int, steps: int = 40) -> np.ndarray:
    """Deterministic trace: reset at ``seed`` then apply a FIXED action sequence; return the qpos+qvel path."""
    rng = np.random.default_rng(20260806)
    acts = rng.uniform(-2.0, 2.0, size=(steps, 4)).astype(np.float32)
    rl.reset(seed)
    d = rl.inner.data
    trace = []
    for a in acts:
        rl.step(a)
        trace.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
    return np.asarray(trace, dtype=np.float64)


def _compare_models(fa: dict, fb: dict) -> list[str]:
    diffs = []
    for k in ("nq", "nv", "nu", "ngeom", "nbody", "disk_geom_type", "disk_mass"):
        if fa[k] != fb[k]:
            diffs.append(f"{k}: A={fa[k]} B={fb[k]}")
    for k in ("disk_geom_size", "contype", "conaffinity"):
        if not np.array_equal(fa[k], fb[k]):
            diffs.append(f"{k}: A={fa[k].tolist()} B={fb[k].tolist()}")
    return diffs


def main() -> int:
    pi0, base, forbidden = _setup()
    demos, comp, _hist = build_boundary_panel(
        pi0, range(20000, 20240), forbidden, want=3, families=FAMS,
        strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    if not demos:
        print("PARITY INCONCLUSIVE: no reconstructable demo in the probe range", flush=True)
        return 2
    print(f"probed demos: {len(demos)} (families {comp})", flush=True)

    all_ok = True
    for i, demo in enumerate(demos):
        # OLD direct call (unchanged code path = pre-refactor behavior)
        rl_a = reconstruct_handoff(pi0, demo, geom="POINT", arm_mjcf_transform=_ball_tf,
                                   coin_shape="cylinder", disk_radius_override=_R)[0]
        # NEW generated path (object from the typed HyMeKo-sourced spec)
        rl_b = build_manipulation_rig(pi0, demo, object_spec=COIN_OBJECT, robot=BALLTIP_ROBOT).rl

        model_diffs = _compare_models(_model_fingerprint(rl_a.inner.model),
                                      _model_fingerprint(rl_b.inner.model))
        ta, tb = _rollout_trace(rl_a, seed=demo.seed), _rollout_trace(rl_b, seed=demo.seed)
        roll_ok = ta.shape == tb.shape and np.array_equal(ta, tb)
        max_dev = float(np.max(np.abs(ta - tb))) if ta.shape == tb.shape else float("nan")

        ok = not model_diffs and roll_ok
        all_ok = all_ok and ok
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] demo{i} seed={demo.seed} prefix={demo.prefix_steps}: "
              f"model_diffs={model_diffs or 'none'} rollout_bit_exact={roll_ok} max_dev={max_dev:.2e}",
              flush=True)

    verdict = "O0_PARITY_PASS" if all_ok else "O0_PARITY_FAIL"
    print(f"\n{verdict}: generated path {'==' if all_ok else '!='} frozen R11.6C on the reference coin", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
