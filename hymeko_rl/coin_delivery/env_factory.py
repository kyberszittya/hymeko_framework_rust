"""Canonical typed Coin-Delivery environment construction — production, experiment-free.

Extracted verbatim (behaviour-preserving) from ``experiments.exp_galambos_coord_ab.make_env`` so the neutral-delivery
chain no longer imports the galambos/coin-toss experiment web to build its env. Uses ONLY production modules
(:mod:`hymeko_rl.env.planar_grasp_env`, :mod:`hymeko_rl.env.reward`, :mod:`hymeko_rl.coin_delivery.scenarios`). The
reward is loaded from its ``.hymeko`` source (no in-memory term surgery); the fingertip-geometry axis and every
arm/coin/target/strict-predicate semantic are unchanged.
"""
from __future__ import annotations

from typing import Any

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv, with_fingertip_clamp, with_fingertip_shape

# canonical constants (were experiment module-level; behaviour identical)
COIN_MAX_STEPS = 300
FLAT_PAD_SIZE = "0.004 0.016 0.02"          # box half-extents: thin in the contact-normal x, wide in y/z
DELIVER_V2B = "data/robotics/galambos_task_deliver_v2b.hymeko"   # the sound coin-toss DELIVERY reward (v2b)
COORD_HYMEKO = "data/robotics/galambos_task_coord.hymeko"
VALID_EMBODIMENTS = ("POINT", "FLAT_PAD", "CONCAVE_CLAMP", "E1_WRIST", "E2_CLOSURE", "E3_WRIST_CLOSURE")


def _fingertip_kwargs(embodiment: str) -> dict:
    """Map an embodiment name to the PlanarGraspEnv fingertip/arm-transform kwargs (only geometry/distal DoF change)."""
    from hymeko_rl.coin_delivery.scenarios.kinematic_variant import with_distal_pad_orientation, with_pad_closure
    if embodiment == "POINT":
        return {}
    if embodiment == "FLAT_PAD":
        return {"fingertip_shape": "box", "fingertip_size": FLAT_PAD_SIZE}
    if embodiment == "CONCAVE_CLAMP":
        return {"arm_mjcf_transform": with_fingertip_clamp}

    def _box(mj: str) -> str:
        return with_fingertip_shape(mj, "box", FLAT_PAD_SIZE)
    tf = {"E1_WRIST": lambda mj: with_distal_pad_orientation(_box(mj)),
          "E2_CLOSURE": lambda mj: with_pad_closure(_box(mj)),
          "E3_WRIST_CLOSURE": lambda mj: with_pad_closure(with_distal_pad_orientation(_box(mj)))}[embodiment]
    return {"arm_mjcf_transform": tf}


def make_coin_env(*, embodiment: str = "POINT", difficulty: float = 0.3, deliver: bool = True, coord: bool = False,
                  max_steps: int = COIN_MAX_STEPS, treatment_hymeko: str = COORD_HYMEKO,
                  arm_mjcf_transform=None, coin_shape: str = "cylinder", disk_radius_override=None,
                  disk_radius_y_override=None) -> PlanarGraspEnv:
    """Build the canonical planar Coin env. ``embodiment`` selects the fingertip geometry (POINT golden, …). ``deliver``
    loads the v2b delivery reward; ``coord`` (wins if both set) loads the coordination reward; else the PlanarGraspEnv
    default. Reward-in-hymeko. # Preconditions ``embodiment`` in :data:`VALID_EMBODIMENTS`. # Postconditions
    ``env.reward_file`` names the active reward source. # Errors ``ValueError`` on an unknown embodiment (fail loud)."""
    if embodiment not in VALID_EMBODIMENTS:
        raise ValueError(f"embodiment must be one of {VALID_EMBODIMENTS}; got {embodiment!r}")
    kw = _fingertip_kwargs(embodiment)
    if arm_mjcf_transform is not None:                                 # a variant supplies its whole arm MJCF (overrides
        kw["arm_mjcf_transform"] = arm_mjcf_transform                  # the embodiment's default transform; used by the
                                                                       # BALLTIP robot-variant matched-panel regression)
    env = PlanarGraspEnv(robot_source="hymeko_spec", scene_source="hymeko_spec", max_steps=max_steps,
                         difficulty=difficulty, coin_shape=coin_shape, disk_radius_override=disk_radius_override,
                         disk_radius_y_override=disk_radius_y_override, **kw)
    if coord or deliver:
        from hymeko_rl.env.reward import RewardSpec
        src = treatment_hymeko if coord else DELIVER_V2B
        env.reward_spec = RewardSpec.from_hymeko(src)
        env.reward_file = src
    return env


# canonical contact-formation bank / contract / config (were re-exported by the experiment `pedc_selection` from the
# coin-toss `exp_coin_toss_arcrl`/`struct_distill` web; extracted here verbatim so the coin chain imports no experiment)
C1_HORIZON = 60                             # the C1 contact-formation suffix horizon
CONTACT_BANK_DIR = "experiments/2026_07_18_arcrl"       # where the held-seed pre-contact banks live (read-only data)


def c1_contact_config():
    """C1 one-finger-recovery contact-formation config (identical to the old `c1_config()`)."""
    from hymeko_rl.env.contact_formation_env import ContactFormationConfig
    return ContactFormationConfig(w_streak=0.4, w_flicker=1.0, w_linger=0.4, w_lost_after_both=1.5, hold_k=4)


def coin_monitor_contract(*, embodiment: str = "POINT"):
    """The canonical Coin :class:`MonitorContract` — production `MonitorContract.from_env(make_coin_env())` (verified
    field-identical to the old experiment `_ctx()['contract']`)."""
    from hymeko_rl.eval.task_monitor.contract import MonitorContract
    return MonitorContract.from_env(make_coin_env(embodiment=embodiment))


def load_contact_bank(path: str = "c1_heldseed_bank.pkl", *, holdout: bool = False) -> list:
    """Load a pre-contact snapshot bank pickle (``{'items': [...]}``) from :data:`CONTACT_BANK_DIR`, taking the
    train (``holdout=False``) or holdout (``i%4==0``) partition — identical to the old `_load_pkl_bank`.
    # Errors ``FileNotFoundError`` if the bank is absent (fail loud; no fabricated bank)."""
    import os
    import pickle
    p = os.path.join(CONTACT_BANK_DIR, path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"contact bank missing: {p} (place the read-only bank or fix CONTACT_BANK_DIR)")
    with open(p, "rb") as fh:
        items = pickle.load(fh)["items"]
    return [it for i, it in enumerate(items) if (i % 4 == 0) == holdout]


def make_coin_contact_env(embodiment: str = "POINT", *, holdout: bool = False, seed: int = 0):
    """The canonical C1 contact-formation env over the Coin env — production-only (was the experiment
    ``pedc_selection._env``, identical construction: plain :class:`ContactFormationEnv` on the pre-contact bank,
    the POINT monitor contract, :data:`C1_HORIZON`, :func:`c1_contact_config`). Non-wristed; for the wristed
    variant with joint limits see :func:`experiments.coin_wristed_delivery.build_wristed_contact_env`.

    # Preconditions ``embodiment`` in :data:`VALID_EMBODIMENTS`; the ``c1_heldseed_bank.pkl`` bank exists.
    # Errors ``FileNotFoundError`` if the bank is absent; ``ValueError`` on an unknown embodiment.
    """
    from hymeko_rl.env.contact_formation_env import ContactFormationEnv
    return ContactFormationEnv(make_coin_env(embodiment=embodiment),
                               load_contact_bank("c1_heldseed_bank.pkl", holdout=holdout),
                               coin_monitor_contract(), horizon=C1_HORIZON, cfg=c1_contact_config(), seed=seed)


def coin_env_fingerprint(env: Any) -> dict:
    """A deterministic fingerprint of a built env (for reproducibility gates)."""
    import hashlib

    import numpy as np
    return {"obs_space": list(env.observation_space.shape), "action_space": list(env.action_space.shape),
            "model_nq": int(env.model.nq), "model_nu": int(env.model.nu),
            "geom_fp": hashlib.sha256(np.ascontiguousarray(env.model.geom_size).tobytes()).hexdigest()[:12],
            "reward_file": getattr(env, "reward_file", None)}
