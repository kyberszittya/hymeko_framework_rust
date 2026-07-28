"""Home-posture contracts for the two-arm planar gripper (single source of truth).

Two *distinct* home contracts, deliberately kept separate because they pose two different problems:

``HOME_STATE_V1_GENERIC`` — both arms retracted to a fixed generic pose ``[-0.9, -1.4, 0.0, 2.7]`` (committed
``6d62b53e``). Both fingertips sit on the *same* (lower) side of the coin, so reaching the pre-capture straddle requires
routing the left tip ~168° around the coin — a topology-changing / IK-branch stress test for the geometric approach
layer. Retained verbatim; this file documents it, it does not redefine it.

``HOME_STATE_V2_READY`` — the wide-open *reference* of the task-ready set: fingertips on their assigned left/right sides
~75 mm from the coin, no contact, qvel = 0, prev_tau = 0. It is a ready-set target *region*, **not** a hard pose and
**not** a replacement for V1; it grants no momentum, no torque preload, no contact, no H_dyn membership. Its joint
vector is the analytic-IK solution of the two 75 mm assigned-side tip targets (confirmed by the round-trip test). Note
its left tip sits at full extension (elbow ~0, r_base = 0.300 = the left-arm singularity), so V2 is a posture one *ends
near*, not one to transit *through*: the geometric-approach transit hands off at the ~55 mm **pre-capture shell**
(``CoinStraddleTargets``, tips on the *same* assigned sides with a 15 mm surface margin, non-singular), which is the
closest clean staging point for the downstream dynamic capture. Both live in the READY set (assigned-side, no-contact,
coin-clear); they differ only in how wide open the arms are.

Both are start-state definitions (like the cradle itself), never mid-rollout state edits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc


@dataclass(frozen=True)
class HomeState:
    """A named collision-free home posture: the arm joint vector plus its contract metadata.

    # Invariants: ``q`` is length-4 (the 4 actuated arm joints); qvel and prev_tau are zero at the home (built by
    ``build_home_snapshot``); the coin is placed at the canonical cradle, never edited mid-rollout.
    """

    name: str
    q: np.ndarray
    mode: str
    description: str

    def __post_init__(self) -> None:
        assert self.q.shape == (4,), f"home q must be length-4, got {self.q.shape}"


HOME_STATE_V1_GENERIC = HomeState(
    name="HOME_STATE_V1_GENERIC",
    q=np.array([-0.9, -1.4, 0.0, 2.7]),
    mode="HOME_GENERIC",
    description="Generic retracted home; both tips on the lower side of the coin -> left tip must route ~168 deg "
    "around the coin (topology / IK-branch stress test). Committed 6d62b53e; retained verbatim.",
)

HOME_STATE_V2_READY = HomeState(
    name="HOME_STATE_V2_READY",
    q=np.array([-0.69, 0.0, -0.167, 2.191]),
    mode="HOME_READY",
    description="Wide-open task-ready home; tips already on their assigned L/R sides ~75 mm from the coin, no contact. "
    "Analytic-IK solution of the 75 mm assigned-side targets; a ready-set target, not a replacement for V1.",
)


def build_home_snapshot(cradle: Any, home: HomeState = HOME_STATE_V1_GENERIC) -> Any:
    """Build a branchable ``TransportSnapshot`` at ``home``: coin at the canonical cradle, arms at ``home.q``,
    qvel = 0, prev_tau = 0, no contact.

    # Preconditions: ``cradle`` is a branchable snapshot exposing ``.branch()`` and ``.stack``.
    # Postconditions: returns a snapshot duck-typing the rollout interface; ``branch()`` is deterministic; no straddle
    #   is assumed (built via ``from_live``); the coin is unperturbed at the cradle position.
    """
    rl = cradle.branch()
    data, model = rl.inner.data, rl.inner.model
    coin = _coin_xy(cradle.branch())
    data.qpos[:4] = home.q
    data.qpos[4:6] = coin
    data.qpos[6] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return kc.TransportSnapshot.from_live(rl, cradle.stack, np.zeros(4))
