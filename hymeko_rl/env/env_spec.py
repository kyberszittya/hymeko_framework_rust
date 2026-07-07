"""The environment scene/task geometry as a declarative spec — the in-memory form of the
``meta_env.hymeko`` vocabulary (the environment half of an agent description).

An :class:`EnvSpec` is read from a ``.hymeko`` scene profile's ``env_spec`` bundle (target zone,
coin spawn region, workspace bounds, success criterion), so the env's geometry is described in
HyMeKo rather than hardcoded in Python — the same pattern as :class:`hymeko_rl.env.reward.RewardSpec`
and :class:`hymeko_rl.env.observation.ObservationSpec`. Each config term carries scalar fields only
(no arcs), so the shared narrow reader (:func:`hymeko_rl.env._profile.read_bundle`) parses it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hymeko_rl.env._profile import read_bundle

# A 4-tuple sampling box: (x_lo, x_hi, y_lo, y_hi).
Box = tuple[float, float, float, float]


def _num(body: str, name: str, default: float) -> float:
    """Read a scalar field ``name <value>;`` from a config-term body (``default`` if absent)."""
    m = re.search(rf"\b{name}\s+(-?[\d.]+)", body)
    return float(m.group(1)) if m else default


@dataclass(frozen=True)
class EnvSpec:
    """Declarative environment geometry. Defaults match the harder Galambos task.

    # Preconditions (``from_hymeko``) the profile declares one ``env_spec`` bundling
    ``target_zone`` / ``coin_spawn`` / ``workspace`` / ``success`` config terms.
    # Postconditions every field is finite; ``success_steps >= 1``; regions are (lo, hi, lo, hi).
    """

    zone_x: float = 0.0
    zone_y: float = 0.16
    zone_half: float = 0.04
    zone_region: Box = (-0.05, 0.05, 0.10, 0.18)
    randomize_zone: bool = True
    coin_region: Box = (-0.20, 0.20, 0.05, 0.23)
    coin_clearance: float = 0.03
    out_bound: float = 0.40
    y_min: float = -0.08
    y_max: float = 0.45
    success_steps: int = 5
    disk_radius: float = 0.02
    # Contact-quality contract (v2 physics; declared by an optional `contact` config term). Off by default
    # → v1 abstract fingertip-only prototype (the coin passes through the arm bodies). On → arm bodies
    # physically collide with the coin; a non-fingertip arm↔coin contact is a non-preferred contact, tracked
    # and reward-penalised, and used to grade a delivery (clean/assisted/exploit). `contact_mode`: "graded"
    # (default — never voids the episode) or "strict" (voids/terminates, for clean-paper validation). The
    # role split (fingertip vs arm-body) is built once into a ContactLegalitySpec; `valid_contact_prefix` is
    # the fingertip geom-name convention.
    contact_legality: bool = False
    contact_mode: str = "graded"
    valid_contact_prefix: str = "fingertip"

    @classmethod
    def from_hymeko(cls, env_profile: str | Path) -> "EnvSpec":
        """Build the spec from a ``.hymeko`` scene profile's ``env_spec`` bundle.

        # Errors ``FileNotFoundError``; ``ValueError`` (no/multiple ``env_spec``, missing term kind).
        """
        terms = {kind: body for _name, kind, body, _w in read_bundle(env_profile, "env_spec")}
        required = {"target_zone", "coin_spawn", "workspace", "success", "disk"}
        missing = required - terms.keys()
        if missing:
            raise ValueError(f"{env_profile}: env_spec missing config term(s) {sorted(missing)}")
        z, s, w, su = (terms["target_zone"], terms["coin_spawn"],
                       terms["workspace"], terms["success"])
        c = terms.get("contact", "")          # optional: declares the v2 contact-legality contract
        return cls(
            disk_radius=_num(terms["disk"], "radius", 0.02),
            zone_half=_num(z, "half", 0.04),
            zone_region=(_num(z, "rx_lo", -0.05), _num(z, "rx_hi", 0.05),
                         _num(z, "ry_lo", 0.10), _num(z, "ry_hi", 0.18)),
            randomize_zone=_num(z, "randomize", 1.0) > 0.5,
            coin_region=(_num(s, "rx_lo", -0.20), _num(s, "rx_hi", 0.20),
                         _num(s, "ry_lo", 0.05), _num(s, "ry_hi", 0.23)),
            coin_clearance=_num(s, "clearance", 0.03),
            out_bound=_num(w, "x_bound", 0.40),
            y_min=_num(w, "y_min", -0.08),
            y_max=_num(w, "y_max", 0.45),
            success_steps=int(_num(su, "steps", 5.0)),
            contact_legality=_num(c, "legality", 0.0) > 0.5,
            contact_mode="strict" if _num(c, "strict", 0.0) > 0.5 else "graded",
        )


# The default Galambos environment (matches galambos_env.hymeko) for direct construction.
DEFAULT_ENV = EnvSpec()
