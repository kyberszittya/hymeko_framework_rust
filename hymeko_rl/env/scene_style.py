"""Render-time scene beautification for emitted MJCF arms.

The ``hymeko emit -f mjcf`` output is physically correct but visually bare: flat-coloured geoms,
the default headlight, no floor, no skybox, no shadows. :func:`beautify_mjcf` decorates such an
MJCF *string* with visual assets only --- a skybox gradient, a textured checker floor, a small
light rig with shadows, a ``<visual>`` quality block, and polished link materials --- so a render
looks like a studio shot rather than a wireframe.

It is a **render-time** decoration: it adds no body and no joint, so the kinematic tree, the
hypergraph (:meth:`HypergraphState.from_mjcf`), and the actuators are bit-for-bit the same as the
bare input. A floor *geom* is added to ``worldbody`` (body 0) --- a collision surface, not a body
--- so physics gains a ground plane but the vertex set does not change. The training env never
sees this; only the renderer applies it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneStyle:
    """Visual parameters for :func:`beautify_mjcf`. Pure config; no behaviour.

    Fields (all with studio-neutral defaults):
      * ``sky_top`` / ``sky_bottom`` -- skybox gradient RGB (top, horizon), each in ``[0,1]^3``.
      * ``floor_rgb1`` / ``floor_rgb2`` -- the two checker tints of the floor, ``[0,1]^3``.
      * ``floor_size`` -- half-extent of the floor plane (m); ``floor_reflectance`` in ``[0,1]``.
      * ``floor_texrepeat`` -- checker tiles across the plane.
      * ``mat_specular`` / ``mat_shininess`` / ``mat_reflectance`` -- link-material polish, ``[0,1]``.
      * ``lights`` -- a rig of ``(pos, dir, diffuse, castshadow)`` spotlights.
    """

    sky_top: tuple[float, float, float] = (0.45, 0.62, 0.85)
    sky_bottom: tuple[float, float, float] = (0.90, 0.94, 0.98)
    floor_rgb1: tuple[float, float, float] = (0.24, 0.26, 0.30)
    floor_rgb2: tuple[float, float, float] = (0.31, 0.34, 0.39)
    floor_size: float = 2.0
    floor_reflectance: float = 0.18
    floor_texrepeat: int = 8
    mat_specular: float = 0.5
    mat_shininess: float = 0.4
    mat_reflectance: float = 0.12
    lights: tuple[tuple[tuple[float, float, float], tuple[float, float, float],
                        tuple[float, float, float], bool], ...] = field(default=(
        # (pos, dir, diffuse, castshadow): a key light (shadowed) + a softer fill.
        ((0.7, -0.7, 1.6), (-0.4, 0.4, -1.0), (0.7, 0.7, 0.68), True),
        ((-0.9, 0.5, 1.2), (0.5, -0.25, -1.0), (0.32, 0.32, 0.36), False),
    ))


def _fmt(rgb: tuple[float, ...]) -> str:
    return " ".join(f"{c:g}" for c in rgb)


def _asset_extra(style: SceneStyle) -> str:
    """Skybox + checker-floor texture + floor material XML (goes inside ``<asset>``)."""
    return (
        f'<texture name="hk_sky" type="skybox" builtin="gradient" '
        f'rgb1="{_fmt(style.sky_top)}" rgb2="{_fmt(style.sky_bottom)}" '
        f'width="256" height="256"/>'
        f'<texture name="hk_grid" type="2d" builtin="checker" '
        f'rgb1="{_fmt(style.floor_rgb1)}" rgb2="{_fmt(style.floor_rgb2)}" '
        f'width="512" height="512"/>'
        f'<material name="hk_grid_mat" texture="hk_grid" '
        f'texrepeat="{style.floor_texrepeat} {style.floor_texrepeat}" '
        f'reflectance="{style.floor_reflectance:g}" specular="0.3" shininess="0.3"/>'
    )


def _visual_block() -> str:
    """A ``<visual>`` quality block: tuned headlight, haze, shadows."""
    return (
        '<visual>'
        '<headlight ambient="0.35 0.35 0.38" diffuse="0.55 0.55 0.55" '
        'specular="0.2 0.2 0.2"/>'
        '<rgba haze="0.86 0.91 0.96 1"/>'
        '<quality shadowsize="4096" offsamples="8"/>'
        '<map shadowclip="2" shadowscale="0.7"/>'
        '</visual>'
    )


def _worldbody_extra(style: SceneStyle) -> str:
    """Floor plane geom + the light rig (goes right after ``<worldbody>``)."""
    floor = (f'<geom name="hk_floor" type="plane" '
             f'size="{style.floor_size:g} {style.floor_size:g} 0.1" material="hk_grid_mat"/>')
    lights = "".join(
        f'<light name="hk_light{i}" pos="{_fmt(pos)}" dir="{_fmt(direction)}" '
        f'directional="false" diffuse="{_fmt(diff)}" specular="0.25 0.25 0.25" '
        f'castshadow="{"true" if shadow else "false"}"/>'
        for i, (pos, direction, diff, shadow) in enumerate(style.lights))
    return floor + lights


def _polish_materials(mjcf: str, style: SceneStyle) -> str:
    """Add specular / shininess / reflectance to the emitted ``<material .../>`` link entries
    that lack them (matched by the emitter's ``mat_`` name prefix)."""
    polish = (f' specular="{style.mat_specular:g}" shininess="{style.mat_shininess:g}" '
              f'reflectance="{style.mat_reflectance:g}"')

    def add(m: re.Match[str]) -> str:
        body = m.group(1)
        if "specular=" in body:        # already polished — leave it
            return m.group(0)
        return f"<material {body}{polish}/>"

    return re.sub(r'<material (name="mat_[^"]*"[^/]*?)\s*/>', add, mjcf)


def beautify_mjcf(mjcf: str, style: SceneStyle | None = None) -> str:
    """Return ``mjcf`` decorated with skybox, textured floor, lights, shadows, and polished
    link materials --- visual assets only, no body / joint / actuator change.

    # Preconditions ``mjcf`` is a well-formed MuJoCo XML string containing ``<worldbody>``
      (with or without an ``<asset>`` block).
    # Postconditions Returns a well-formed MJCF that loads in MuJoCo and whose bodies, joints,
      and actuators equal the input's; a skybox texture, a checker floor (texture+material+geom),
      a ``<visual>`` block, and ``>= 1`` light are present.
    # Errors ``ValueError`` if ``<worldbody>`` is absent (nothing to decorate).
    """
    style = style or SceneStyle()
    if "<worldbody>" not in mjcf:
        raise ValueError("beautify_mjcf: MJCF has no <worldbody> to decorate")

    out = _polish_materials(mjcf, style)

    # Assets: extend an existing <asset> block, or create one before <worldbody>.
    if "</asset>" in out:
        out = out.replace("</asset>", _asset_extra(style) + "</asset>", 1)
    else:
        out = out.replace("<worldbody>", f"<asset>{_asset_extra(style)}</asset>\n  <worldbody>", 1)

    # Visual quality block: insert just before <worldbody>.
    out = out.replace("<worldbody>", _visual_block() + "\n  <worldbody>", 1)

    # Floor geom + lights: insert just inside <worldbody>.
    out = out.replace("<worldbody>", "<worldbody>\n    " + _worldbody_extra(style), 1)
    return out
