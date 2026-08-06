"""R11.7B — flagship video: the first non-circular (box) teacher-free exact-zero strict-K6 delivery.

Reproduces the exact flagship handoff (dev scenario `bank_c2_+0.015_+0.015`, seed 3) via the canonical exact-zero
reach→capture, then renders the delivery under the frozen certificate's θ (a k3-weighted retrieval blend — an
interpolation, per the R11.7B audit). The clip is the flagship *delivery* (the box transported into the zone + the
strict-K6 dwell), the money shot the `R11_7A_FIRST_NONCIRCULAR_EXACT_ZERO_K6_CERTIFICATE` records. Viz only — selects
nothing, changes no state; the K6 verdict is the frozen monitor's.

Run:  python -m hymeko_rl.experiments.r11_7b_flagship_video
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import reach_capture_descriptor
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_7a_u6b_box_pilot import _VARIANT
from hymeko_rl.experiments.video_r11_6c_composition import _Filmer, _cam, _trim_settled
from hymeko_rl.viz.rollout_overlay import encode_clip

_OUT = Path("reports/2026-08-06-r11-7a-u6b-box-pilot")
_CERT = _OUT / "flagship_certificate.json"


def main() -> int:
    cert = json.loads(_CERT.read_text())
    sid, seed = cert["scenario_id"], int(cert["seed"])
    theta = np.asarray(cert["selected_theta"], np.float64)
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)

    h = reach_capture_descriptor(rig, scenario_by_id(sid), seed, cfg, conf, obj)
    if h.record is not None:                                     # must reproduce the certified handoff
        print(f"FLAGSHIP VIDEO ABORT: handoff did not reproduce ({h.record.outcome_class})", flush=True)
        return 1
    film = _Filmer(_cam())
    m = rollout_primitive(h.snap, clip_theta(theta), CLOSED_LOOP_CFG, frame_hook=film)
    film.close()
    k6 = bool(delivery_success(m, CLOSED_LOOP_CFG))
    final_dtz = round(float(m["dtz_end"]) * 1000, 2)
    frames, _dtz = _trim_settled(film.frames, film.dtz_mm, k6)
    if not k6:
        print(f"FLAGSHIP VIDEO WARN: reproduced delivery is NOT K6 (dtz {final_dtz} mm) — not encoding", flush=True)
        return 1

    _OUT.mkdir(parents=True, exist_ok=True)
    mp4 = str(_OUT / "flagship_box_exact_zero_k6.mp4")
    gif = str(_OUT / "flagship_box_exact_zero_k6.gif")
    pil = [Image.fromarray(np.asarray(f, np.uint8)) for f in frames]   # encode_clip expects PIL images
    encode_clip(pil, mp4, fps=30)
    encode_clip(pil, gif, fps=20)
    manifest = {"certificate": cert["certificate"], "scenario_id": sid, "seed": seed, "variant": _VARIANT,
                "k6": k6, "final_dtz_mm": final_dtz, "n_frames": len(frames),
                "theta_provenance": cert["theta_provenance"], "mp4": mp4, "gif": gif}
    (_OUT / "flagship_video_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"FLAGSHIP VIDEO OK: box exact-zero K6 (dtz {final_dtz} mm), {len(frames)} frames → {mp4}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
