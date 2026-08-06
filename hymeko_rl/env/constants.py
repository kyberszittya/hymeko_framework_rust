"""Physical and semantic constants for the RL environments, in an ontological hierarchy.

Constants are grouped by *what kind of quantity* they are (a taxonomy) — ``Physics``, ``Collision``,
``Geometry`` (phase 2) — not by which env module happens to use them. Namespaced classes give the
hierarchy; an :class:`enum.IntEnum` gives the collision scheme a checked, self-documenting type
instead of a bare integer whose meaning previously lived only in a comment.

Behaviour-preserving: every value here equals the literal it replaced across the MJCF builders
(``arm_world``, ``gripper_world``, ``planar_grasp_env``, ``pick_place_env``).
"""
from __future__ import annotations

from enum import IntEnum


class Physics:
    """MuJoCo ``<option>`` defaults shared by every hand-authored world.

    These were duplicated verbatim across four MJCF builders; this is their single home.
    """

    TIMESTEP = 2e-3                    # s — sim step emitted into the MJCF <option>
    STABLE_DT = 5e-4                   # s — max stable sub-step for contact-rich grasping
    GRAVITY = (0.0, 0.0, -9.81)        # m/s^2
    INTEGRATOR = "implicitfast"

    @classmethod
    def gravity_attr(cls) -> str:
        """The ``gravity="x y z"`` MJCF attribute value (space-joined, ``:g``-formatted).

        # Postconditions equals the string literal ``"0 0 -9.81"`` for the default gravity.
        """
        return " ".join(f"{g:g}" for g in cls.GRAVITY)

    @classmethod
    def option_attrs(cls) -> str:
        """The full ``timestep=… integrator=… gravity=…`` attribute run for an ``<option>`` tag."""
        return f'timestep="{cls.TIMESTEP:g}" integrator="{cls.INTEGRATOR}" gravity="{cls.gravity_attr()}"'


class Collision:
    """The MuJoCo contact-bitmask scheme for the grasping scenes.

    MuJoCo collides geoms A,B iff ``(contype_A & conaffinity_B) | (contype_B & conaffinity_A) != 0``.
    The coin is isolated on **bit 2** (``COIN`` channel, ``2/2``), so arm links — which use the MuJoCo
    default ``1/1`` — cannot touch it: only the fingertips (``FINGERTIP``, affinity bit 2 set) and the
    floor move the coin. Finger-on-finger contact is **not** masked out here; it is detected separately
    as a crash (``fingers_self_contact`` in ``planar_grasp_env``). Verified against MuJoCo semantics,
    not assumed.
    """

    class Type(IntEnum):
        """``contype`` — which contact group(s) a geom *belongs to*."""

        VISUAL = 0        # belongs to nothing → collides with nothing (rendering-only geom)
        DEFAULT = 1       # MuJoCo default: arm links, floor, fingertips
        COIN = 2          # the manipuland, isolated on its own bit

    class Affinity(IntEnum):
        """``conaffinity`` — which contact group(s) a geom *collides with* (a bitmask)."""

        NONE = 0
        DEFAULT = 1       # collides with DEFAULT(1) only
        COIN = 2          # collides with COIN(2) only
        ANY = 3           # bits 1|2: collides with DEFAULT and COIN (fingertip, floor)

    # (contype, conaffinity) channels actually emitted, named by the geom's role.
    FINGERTIP = (Type.DEFAULT, Affinity.ANY)     # 1 / 3 — touches coin(2) and floor
    COIN = (Type.COIN, Affinity.COIN)            # 2 / 2 — only affinity-2 geoms move it
    FLOOR = (Type.DEFAULT, Affinity.ANY)         # 1 / 3
    ARM_DEFAULT = (Type.DEFAULT, Affinity.DEFAULT)  # 1 / 1 — v1 prototype: coin passes through arm links
    ARM_LEGALITY = (Type.DEFAULT, Affinity.ANY)  # 1 / 3 — v2: arm links physically collide with coin(2). The mask
                                                 # only *enables* the contact; whether a given arm–coin contact is
                                                 # legal (fingertip) or forbidden (any other arm geom) is decided in
                                                 # software by geom id in ``compute_planar_metrics`` — not by the mask.
    VISUAL = (Type.VISUAL, Affinity.NONE)        # 0 / 0 — non-colliding decoration

    @staticmethod
    def collide(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """MuJoCo's contact predicate for two ``(contype, conaffinity)`` channels.

        # Preconditions each arg is a ``(contype, conaffinity)`` pair of non-negative ints.
        # Postconditions ``True`` iff MuJoCo would generate contacts between the two geoms.
        """
        (ct_a, ca_a), (ct_b, ca_b) = a, b
        return bool((ct_a & ca_b) | (ct_b & ca_a))

    @staticmethod
    def attr(channel: tuple[int, int]) -> str:
        """The ``contype="c" conaffinity="a"`` MJCF attribute fragment for a channel."""
        return f'contype="{int(channel[0])}" conaffinity="{int(channel[1])}"'
