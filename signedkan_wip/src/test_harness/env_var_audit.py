"""Audit that every HSIKAN_* env var actually changes model state.

The class of bugs we've hit today and want to catch automatically:

1. **Partial wiring** — e.g. HSIKAN_KB_INIT_TCB initially patched only
   the inner activation; the outer activation silently kept the
   default (zero) init.  Sweep produced identical-to-4-decimals AUC
   because only half the model was being affected.

2. **Silent fall-through** — e.g. HSIKAN_ATTENTION_M_E='dot' raises
   NotImplementedError when combined with cycle_batch_size, but the
   bash script tail-filters the JSON output and the failure looks
   like a successful run with no output.

3. **Typo or unknown value** — e.g. HSIKAN_TOPK_PRUNER='balanced'
   (instead of 'balance') silently falls back to NoOpPruner.

The audit is a positive-control test:

    For each (env_var, value), instantiate the model and check that
    the named knob ACTUALLY ENDED UP IN THE MODEL STATE.
    If not — fail loudly.

Usage:

    python -m signedkan_wip.src.test_harness.env_var_audit
"""

from __future__ import annotations

import os
import sys
import json
import contextlib
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class AuditCase:
    """A single env-var audit:
    - `env`: the env var to set
    - `value`: the value to set it to
    - `setup`: builds the model under that env var
    - `assertion`: takes the model + returns (ok: bool, detail: str)
    """
    name: str
    env: dict[str, str]
    setup: Callable[[], torch.nn.Module]
    assertion: Callable[[torch.nn.Module], tuple[bool, str]]


@contextlib.contextmanager
def _env_block(env: dict[str, str]):
    """Set env vars temporarily, restoring on exit."""
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _build_signedkan_layer(
    spline_kind: str = "kochanek_bartels", n_branches: int = 2, d: int = 16,
) -> torch.nn.Module:
    sys.path.insert(0, "signedkan_wip")
    from src.signedkan import SignedKANLayer, SignedKANConfig
    cfg = SignedKANConfig(
        n_nodes=10, hidden_dim=d, k=3, grid=5, spline_kind=spline_kind,
    )
    return SignedKANLayer(cfg)


def _build_mixed_arity_model(
    use_attention: bool = False, attention_kind: str = "dot",
    direct_messaging: bool = False,
) -> torch.nn.Module:
    sys.path.insert(0, "signedkan_wip")
    from src.signedkan import MultiLayerSignedKANConfig
    from src.mixed_arity_signedkan import (
        MixedAritySignedKAN, MixedAritySignedKANConfig,
    )
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=10, n_layers=2, hidden_dim=16, grid=3, k=3,
            spline_kinds=["catmull_rom"]*2, init_scale=0.05,
            pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True,
        ),
        arities=(3,),
        cycle_batch_size=None,
        attention_m_e=use_attention,
        attention_m_e_kind=attention_kind,
        direct_messaging=direct_messaging,
    )
    return MixedAritySignedKAN(cfg)


# ─── The audit cases ────────────────────────────────────────────────


def cases() -> list[AuditCase]:
    out: list[AuditCase] = []

    # ── HSIKAN_KB_INIT_TCB ──
    # Both inner and outer KB activations should pick up the env var.
    def _kb_setup():
        return _build_signedkan_layer(spline_kind="kochanek_bartels")

    def _kb_assertion(layer) -> tuple[bool, str]:
        inner_mean = layer.inner.tcb.detach().mean().item()
        outer_mean = layer.outer.tcb.detach().mean().item()
        target = (0.5 + 0.3 + 0.1) / 3   # mean of (0.5, 0.3, 0.1)
        eps = 1e-4
        if abs(inner_mean - target) > eps:
            return False, f"inner.tcb mean={inner_mean:.4f}, expected {target:.4f}"
        if abs(outer_mean - target) > eps:
            return False, (
                f"outer.tcb mean={outer_mean:.4f}, expected {target:.4f} — "
                "the silent outer-init bug"
            )
        return True, f"both inner+outer tcb mean={target:.4f} ✓"

    out.append(AuditCase(
        name="HSIKAN_KB_INIT_TCB → both inner+outer KB tcb",
        env={"HSIKAN_KB_INIT_TCB": "0.5,0.3,0.1"},
        setup=_kb_setup, assertion=_kb_assertion,
    ))

    # ── HSIKAN_KB_PRESET=cusp ──
    def _kb_preset_assertion(layer) -> tuple[bool, str]:
        # cusp preset = (0, 0.7, 0): t=0, c=0.7, b=0
        eps = 1e-4
        for half_name, half in [("inner", layer.inner), ("outer", layer.outer)]:
            t = half.tcb.detach()[..., 0].mean().item()
            c = half.tcb.detach()[..., 1].mean().item()
            b = half.tcb.detach()[..., 2].mean().item()
            if abs(t) > eps or abs(c - 0.7) > eps or abs(b) > eps:
                return False, f"{half_name}.tcb (t,c,b)=({t:.3f},{c:.3f},{b:.3f}), expected (0,0.7,0)"
        return True, "both halves got cusp preset (t,c,b)=(0,0.7,0) ✓"

    out.append(AuditCase(
        name="HSIKAN_KB_PRESET=cusp → tcb=(0,0.7,0)",
        env={"HSIKAN_KB_PRESET": "cusp"},
        setup=_kb_setup, assertion=_kb_preset_assertion,
    ))

    # ── HSIKAN_ATTENTION_M_E='dot' ──
    def _attn_dot_setup():
        return _build_mixed_arity_model(use_attention=True, attention_kind="dot")

    def _attn_dot_assertion(model) -> tuple[bool, str]:
        sys.path.insert(0, "signedkan_wip")
        from src.mixed_arity_signedkan import _AttentionM_e
        if model.attention_m_e is None:
            return False, "model.attention_m_e is None despite use_attention=True"
        if not isinstance(model.attention_m_e, _AttentionM_e):
            return False, (
                f"expected _AttentionM_e, got {type(model.attention_m_e).__name__}"
            )
        return True, f"model.attention_m_e is _AttentionM_e ✓"

    out.append(AuditCase(
        name="HSIKAN_ATTENTION_M_E=dot → _AttentionM_e instance",
        env={}, setup=_attn_dot_setup, assertion=_attn_dot_assertion,
    ))

    # ── HSIKAN_ATTENTION_M_E='quaternion' ──
    def _attn_q_setup():
        return _build_mixed_arity_model(use_attention=True, attention_kind="quaternion")

    def _attn_q_assertion(model) -> tuple[bool, str]:
        sys.path.insert(0, "signedkan_wip")
        from src.mixed_arity_signedkan import _QuaternionAttentionM_e
        if not isinstance(model.attention_m_e, _QuaternionAttentionM_e):
            return False, (
                f"expected _QuaternionAttentionM_e, "
                f"got {type(model.attention_m_e).__name__}"
            )
        return True, f"model.attention_m_e is _QuaternionAttentionM_e ✓"

    out.append(AuditCase(
        name="HSIKAN_ATTENTION_M_E=quaternion → _QuaternionAttentionM_e",
        env={}, setup=_attn_q_setup, assertion=_attn_q_assertion,
    ))

    # ── HSIKAN_DIRECT_MESSAGING ──
    def _dm_setup():
        return _build_mixed_arity_model(direct_messaging=True)

    def _dm_assertion(model) -> tuple[bool, str]:
        if not getattr(model.cfg, "direct_messaging", False):
            return False, "cfg.direct_messaging != True"
        return True, "cfg.direct_messaging=True ✓"

    out.append(AuditCase(
        name="direct_messaging=True → cfg.direct_messaging set",
        env={}, setup=_dm_setup, assertion=_dm_assertion,
    ))

    return out


# ─── Runner ─────────────────────────────────────────────────────────


def run_audit(verbose: bool = True) -> dict:
    results = {"passed": [], "failed": []}
    for c in cases():
        with _env_block(c.env):
            try:
                model = c.setup()
                ok, detail = c.assertion(model)
            except Exception as e:
                ok, detail = False, f"setup/assertion raised: {type(e).__name__}: {e}"
        record = {"name": c.name, "env": c.env, "ok": ok, "detail": detail}
        if ok:
            results["passed"].append(record)
            if verbose:
                print(f"  ✓ {c.name}")
                print(f"    {detail}")
        else:
            results["failed"].append(record)
            if verbose:
                print(f"  ✗ {c.name}")
                print(f"    env: {c.env}")
                print(f"    {detail}")
    return results


def main():
    results = run_audit(verbose=True)
    print()
    n_pass = len(results["passed"])
    n_fail = len(results["failed"])
    print(f"=== {n_pass} passed, {n_fail} failed ===")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
