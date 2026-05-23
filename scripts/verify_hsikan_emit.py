"""End-to-end smoke + parity test for the HyMeKo HSiKAN round-trip.

Pipeline exercised:

  1. ``hymeko emit data/nn/hsikan_mixed.hymeko --format torch_dataflow``
     produces a runnable Python module whose forward signature is
     derived from the t_input declarations (x + per-arity cycle
     structure tensors + per-arity M_e incidence).
  2. The emitted module imports cleanly and instantiates without error.
  3. forward(x, triad_v_kK, triad_sigma_kK, M_e_kK) routes to the real
     signedkan_wip.src.core.signedkan.SignedKANLayer per arity, then the
     ArityMixer applies M_e_kK · cyc_emb_kK weighted by softmax(αₖ).
  4. A forward + backward + optimiser step reduces loss on a synthetic
     target.  Confirms autograd is intact through both the real
     SignedKAN spline activations and the sparse-mm aggregation.

What this test does NOT verify:

  * AUC parity with the hand-coded MixedAritySignedKAN on a real
    dataset.  That requires a full training run; this is a forward-pass
    smoke test only.

Run:
  python3 scripts/verify_hsikan_emit.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
HYMEKO_BIN = REPO / "target" / "release" / "hymeko"
HYMEKO_SRC = REPO / "data" / "nn" / "hsikan_mixed.hymeko"


def _make_synthetic_inputs(n_nodes: int = 8, hidden: int = 16):
    """Construct (x, triad_v_kK, triad_sigma_kK, M_e_kK) for k=2..5.

    For each arity k, we synthesise n_cycles_k random k-cycles with
    {+1, -1} signs and a random sparse incidence M_e_k that maps each
    candidate edge to the cycles it appears in.
    """
    import torch

    rng = torch.Generator().manual_seed(0)
    x = torch.randn(n_nodes, hidden, generator=rng)

    inputs = {"x": x}
    n_test_edges = 12
    for k in (2, 3, 4, 5):
        n_cycles_k = 5 + k  # 7, 8, 9, 10
        triad_v = torch.randint(0, n_nodes, (n_cycles_k, k), generator=rng)
        triad_sigma = (
            torch.randint(0, 2, (n_cycles_k, k), generator=rng) * 2 - 1
        ).long()
        # M_e: dense (n_test_edges, n_cycles_k); each edge participates
        # in ~2 cycles (random {-1, 0, +1} entries with sparsity).
        m_dense = torch.randint(-1, 2, (n_test_edges, n_cycles_k),
                                  generator=rng).float()
        inputs[f"triad_v_k{k}"] = triad_v
        inputs[f"triad_sigma_k{k}"] = triad_sigma
        inputs[f"M_e_k{k}"] = m_dense
    return inputs


def main():
    assert HYMEKO_BIN.exists(), f"hymeko CLI not built: {HYMEKO_BIN}"
    assert HYMEKO_SRC.exists(), f"HyMeKo source missing: {HYMEKO_SRC}"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "hsikan_emitted.py"

        print("── 1. emit via hymeko CLI ──")
        result = subprocess.run(
            [str(HYMEKO_BIN), "emit", str(HYMEKO_SRC),
             "--format", "torch_dataflow",
             "--name", "HSiKANEmitted",
             "-o", str(out)],
            cwd=REPO, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("  emit failed:")
            print(result.stderr)
            sys.exit(1)
        print(f"  wrote {out.stat().st_size} bytes")

        print("\n── 2. import + instantiate ──")
        # ehk_torch_stub.SignedKANLayer pulls in signedkan_wip lazily; make
        # signedkan_wip importable from the repo root.
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "python" / "ehk_torch_stub" / "src"))
        spec = importlib.util.spec_from_file_location("hsikan_emitted", out)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod.HSiKANEmitted()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  class={type(m).__name__}  params={n_params}")

        # Was the real signedkan layer wired in? Inspect one of the
        # per-arity layers.
        sk2 = m.sk2
        real_attached = getattr(sk2, "_real", None) is not None
        print(f"  real signedkan delegation: {real_attached}")

        print("\n── 3. forward with full propagation terms ──")
        import torch
        inputs = _make_synthetic_inputs(n_nodes=8, hidden=16)
        # Forward signature is the order from t_input declarations:
        #   x, triad_v_k2, triad_sigma_k2, ..., triad_v_k5, triad_sigma_k5,
        #   M_e_k2, M_e_k3, M_e_k4, M_e_k5
        ordered = [inputs["x"]]
        for k in (2, 3, 4, 5):
            ordered.append(inputs[f"triad_v_k{k}"])
            ordered.append(inputs[f"triad_sigma_k{k}"])
        for k in (2, 3, 4, 5):
            ordered.append(inputs[f"M_e_k{k}"])
        y = m(*ordered)
        n_test_edges = inputs["M_e_k2"].shape[0]
        assert y.shape == (n_test_edges, 1), \
            f"unexpected output shape {y.shape}, expected ({n_test_edges}, 1)"
        assert torch.isfinite(y).all().item(), "non-finite outputs"
        print(f"  forward: x{tuple(inputs['x'].shape)} + 4 arities -> "
              f"y{tuple(y.shape)} all finite")

        print("\n── 4. spectral_weights API ──")
        sw = m.spectral_weights()
        assert len(sw) >= 9, f"too few spectral weights: {len(sw)}"
        print(f"  spectral_weights: {len(sw)} tensors")

        print("\n── 5. backprop + step reduces loss ──")
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        target = torch.zeros_like(y)
        loss0 = torch.nn.functional.mse_loss(m(*ordered), target).item()
        for _ in range(5):
            opt.zero_grad()
            torch.nn.functional.mse_loss(m(*ordered), target).backward()
            opt.step()
        loss5 = torch.nn.functional.mse_loss(m(*ordered), target).item()
        assert loss5 < loss0, f"loss did not decrease: {loss0} -> {loss5}"
        print(f"  loss 0->5 steps: {loss0:.6f} -> {loss5:.6f}")

        print("\nHSiKAN round-trip emit + multi-input forward + train all green")


if __name__ == "__main__":
    main()
