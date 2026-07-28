"""R10.2 Stage 2 — pre-training coordinate identity gates for the structured-option torque-path action.

This is NOT a training module and imports no trainer. It proves, before any RL, that the corrected action coordinate is
a transparent glass tube: a genuinely-zero structured-option actor changes *nothing* and still delivers strict K6. Only
after this may exploration / TD3 begin (later boundaries).

Three gates (the task's names):

  * ``TORQUE_PATH_ZERO_DELTA_IDENTITY`` — a real 15-D zero-init actor emits ``theta = 0`` exactly, and the composed
    per-step action equals the scaffold's increment bit-exact (zero delta): structured params, the executable torque
    path, and the physical action trace all match a pure-scaffold reference.
  * ``TORQUE_PATH_BIT_EXACT_SCAFFOLD`` — the whole zero-theta rollout is bit-exact to ``PhaseShapeCapture.roll(pi0)``
    (q, qvel, prev_tau, contacts) and produces an identical downstream input, event/kind trace, and strict-K6 result
    (identity test, training, and deployment share the *same* ``TorquePathCaptureRoll.rollout`` code path).
  * ``OPTION_ZERO_POLICY_K6`` — the zero-theta option delivers strict K6 through the frozen downstream.

Saturation masks (slew-limited / action-clipped / torque-clamped) are logged even for the zero policy, so a later
terminal-offset audit can distinguish requested / slew-limited / clamped / physically-executed corrections.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_structured_option_gates``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

OUT = Path("reports/2026-07-28-r10-structured-option-torque-path-td3")


def _fingertip_count(rl: Any) -> int:
    from hymeko_rl.coin_delivery.forward_displacement import primary_fingertip_contacts
    con = primary_fingertip_contacts(rl)
    return int(con["left"] is not None) + int(con["right"] is not None)


def _reference_scaffold_trace(cap: mp.PhaseShapeCapture, pi0: mp.CaptureParams, ref: mp.HandoffReference) -> dict:
    """A pure-scaffold per-step trace via the frozen ``scaffold_action``/``apply_step`` (no offset) — the independent
    anchor for the zero-delta identity. Identical calls to ``PhaseShapeCapture._track``; records what ``roll`` discards."""
    coeffs = mp.quintic_coeffs(cap.q0, cap.v0, ref.precursor(pi0.n), pi0.s * ref.qvel_star, pi0.steps * ref.control_dt)
    knot_t = np.linspace(0, len(pi0.residual) - 1, pi0.steps)
    rl = cap.ready.branch()
    prev = cap.prev0.copy()
    prevs, acts, cons = [], [], []
    mujoco.set_mjcb_control(cap._governor())
    try:
        for i in range(pi0.steps):
            a = cap.scaffold_action(rl, prev, i, coeffs, pi0, knot_t)
            prev = cap.apply_step(rl, prev, a)
            prevs.append(prev.copy())
            acts.append(a.astype(np.float32))
            cons.append(_fingertip_count(rl))
    finally:
        mujoco.set_mjcb_control(None)
    snap = kc.TransportSnapshot.from_live(copy.deepcopy(rl), cap.stack, prev.copy())
    return {"prev_seq": prevs, "act_seq": acts, "contacts": cons, "snapshot": snap, "prev": prev}


def _all_equal(seq_a: list, seq_b: list) -> bool:
    return len(seq_a) == len(seq_b) and all(np.array_equal(a, b) for a, b in zip(seq_a, seq_b))


def _params_bit_exact(p: mp.CaptureParams, q: mp.CaptureParams) -> bool:
    return bool(p.n == q.n and p.s == q.s and p.preload_start == q.preload_start and p.bmax == q.bmax
                and p.steps == q.steps and p.kp == q.kp and p.kd == q.kd
                and np.array_equal(np.asarray(p.residual), np.asarray(q.residual)))


def _same_delivery(zero: tuple, scaf: tuple) -> bool:
    """Downstream deliveries agree: strict-K6, min_dtz, and safety all identical."""
    (k6_0, md_0, safe_0, _), (k6_s, md_s, safe_s, _) = zero, scaf
    return bool(k6_0) == bool(k6_s) and md_0 == md_s and bool(safe_0) == bool(safe_s)


def _bit_exact_fields(res0: dict, scaffold: Any, reference: dict, pi0: mp.CaptureParams,
                      zero: tuple, scaf: tuple) -> dict:
    """Per-field bit-exact evidence: the zero-theta roll vs the canonical scaffold roll AND a pure-scaffold reference."""
    sd, rd = scaffold.snapshot.branch().inner.data, res0["snapshot"].branch().inner.data
    same_qpos = bool(np.array_equal(sd.qpos, rd.qpos))
    return {
        "structured_params": _params_bit_exact(res0["params"], pi0),
        "physical_action_trace": _all_equal(res0["acts"], reference["act_seq"]),
        "executable_torque_path": _all_equal(res0["desired_path"], reference["prev_seq"]),
        "contact_event_trace": res0["contacts"] == reference["contacts"],
        "qpos_vs_scaffold": same_qpos,
        "qvel_vs_scaffold": bool(np.array_equal(sd.qvel, rd.qvel)),
        "prev_tau_vs_scaffold": bool(np.array_equal(np.asarray(scaffold.snapshot.prev_tau), np.asarray(res0["prev"]))),
        "prev_tau_vs_reference": bool(np.array_equal(np.asarray(reference["prev"]), np.asarray(res0["prev"]))),
        "downstream_input_qpos": same_qpos,
        "downstream_kind_trace": zero[3] == scaf[3],
        "termination_step": len(res0["acts"]) == pi0.steps == len(reference["act_seq"]),
        "strict_k6_result": _same_delivery(zero, scaf),
    }


def _identity_verdicts(actor_exactly_zero: bool, be: dict, zero_k6: tuple) -> dict:
    """The three coordinate identity verdicts from the bit-exact evidence (``all([...])`` keeps complexity flat)."""
    k6_0, safe_0, md_0 = zero_k6
    zero_delta = all([actor_exactly_zero, be["structured_params"], be["physical_action_trace"],
                      be["executable_torque_path"], be["prev_tau_vs_reference"]])
    bit_exact_scaffold = all([be["qpos_vs_scaffold"], be["qvel_vs_scaffold"], be["prev_tau_vs_scaffold"],
                              be["contact_event_trace"], be["downstream_kind_trace"], be["strict_k6_result"]])
    return {"TORQUE_PATH_ZERO_DELTA_IDENTITY_PASS": bool(zero_delta),
            "TORQUE_PATH_BIT_EXACT_SCAFFOLD_PASS": bool(bit_exact_scaffold),
            "OPTION_ZERO_POLICY_K6_PASS": bool(k6_0 and safe_0 and md_0 < 10.0)}


def _mask_summary(masks: list) -> dict:
    """Per-joint saturation counts across steps (logged even for the zero policy)."""
    return {"steps": len(masks),
            "slew_limited_step_joint": int(sum(int(np.any(m.slew_limited)) for m in masks)),
            "action_clipped_any": bool(any(np.any(m.action_clipped) for m in masks)),
            "torque_clamped_step_joint": int(sum(int(np.any(m.torque_clamped)) for m in masks))}


def run(out: Path = OUT) -> dict:
    rig = _rig()
    ready, ref, stack, pi0, coin, down = rig["ready"], rig["ref"], rig["stack"], rig["pi0"], rig["coin"], rig["down"]

    roller = tpo.TorquePathCaptureRoll(ready, ref, stack, pi0, coin)
    cap = mp.PhaseShapeCapture(ready, ref, stack)

    # Independent anchors: the canonical scaffold roll + a pure-scaffold per-step reference trace.
    scaffold = cap.roll(pi0)
    reference = _reference_scaffold_trace(cap, pi0, ref)

    # The SHARED code path at theta = 0 (same rollout used for training + deployment).
    z0 = np.zeros(tpo.THETA_DIM, dtype=np.float32)
    res0 = roller.rollout(z0)

    # A real 15-D zero-init actor must emit theta = 0 EXACTLY (not a mocked zero vector).
    actor = crl.make_zero_actor(1, act_dim=tpo.THETA_DIM)
    actor_fn = crl.policy_residual(actor)
    actor_out = np.array([actor_fn(o) for o in res0["obs"]])
    actor_exactly_zero = bool(np.all(actor_out == 0.0))

    # Downstream on both inputs (event/kind trace + strict-K6 result).
    k6_0, md_0, safe_0, kinds_0 = down.deliver_with_trace(res0["snapshot"])
    k6_s, md_s, safe_s, kinds_s = down.deliver_with_trace(scaffold.snapshot)

    zero, scaf = (k6_0, md_0, safe_0, kinds_0), (k6_s, md_s, safe_s, kinds_s)
    bit_exact = _bit_exact_fields(res0, scaffold, reference, pi0, zero, scaf)
    non_bit_exact = sorted(k for k, v in bit_exact.items() if not v)

    # theta = 0 terminal-offset sanity (the full TERMINAL_OFFSET_TRACKING gate at non-zero dtau_T is the NEXT boundary).
    tube = tpo.record_phase_tube(roller)
    toff = tpo.terminal_offset_report(roller, z0, tube)
    mask_summary = _mask_summary(res0["masks"])
    verdicts = _identity_verdicts(actor_exactly_zero, bit_exact, (k6_0, safe_0, md_0))

    summary = {
        "contract": "STRUCTURED_OPTION_TORQUE_PATH_IDENTITY_V1",
        "boundary": "coordinate identity gates (pre-training) — NO training, NO exploration, NO SAC/PPO",
        "immutable_scaffold_source": "d6974b95", "parent": "4d983d4f", "theta_dim": tpo.THETA_DIM,
        "medoid": {"s": pi0.s, "preload_start": pi0.preload_start, "bmax": pi0.bmax,
                   "residual_norm": round(float(np.linalg.norm(pi0.residual)), 4)},
        "actor_output_exactly_zero": actor_exactly_zero,
        "bit_exact_fields": {k: bool(v) for k, v in bit_exact.items()},
        "non_bit_exact_fields": non_bit_exact,
        "zero_policy_k6": {"k6": bool(k6_0), "min_dtz_mm": round(md_0, 3), "safe": bool(safe_0)},
        "scaffold_reference_k6": {"k6": bool(k6_s), "min_dtz_mm": round(md_s, 3), "safe": bool(safe_s)},
        "terminal_offset_zero_theta": {"err_norm": round(toff["err_norm"], 12),
                                       "requested": [round(float(x), 9) for x in toff["requested"]],
                                       "executed": [round(float(x), 9) for x in toff["executed"]]},
        "saturation_masks_zero_theta": mask_summary,
        "verdicts": verdicts,
        "non_claims": ["NO reward-driven policy trained", "NO structured exploration run",
                       "TERMINAL_OFFSET_TRACKING at non-zero dtau_T + exploration admissibility are the NEXT boundary",
                       "SAC/PPO deferred until this coordinate is frozen"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity_gates.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run()
    print(f"actor theta==0 exactly: {r['actor_output_exactly_zero']}")
    print(f"zero-policy K6: {r['zero_policy_k6']} | scaffold ref: {r['scaffold_reference_k6']}")
    print(f"terminal-offset(theta=0) err_norm: {r['terminal_offset_zero_theta']['err_norm']}")
    print(f"masks(theta=0): {r['saturation_masks_zero_theta']}")
    if r["non_bit_exact_fields"]:
        print(f"NON-bit-exact fields: {r['non_bit_exact_fields']}")
    for k, v in r["verdicts"].items():
        print(f"  {'PASS' if v else '----'} {k}")
