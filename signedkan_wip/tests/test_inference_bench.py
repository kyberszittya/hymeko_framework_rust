"""Tests for the forward-latency bench (build-item 4) and its PNG renderer.

Pure tests (``_summarize``, ``bars_from_rows``, ``render``) run everywhere.
The HSiKAN-variant integration + performance tests load a real dataset and the
model stack, so they skip cleanly when torch is unavailable. Run with::

    uv run --group ml --group dev pytest -p no:randomly \
        signedkan_wip/tests/test_inference_bench.py
"""
from __future__ import annotations

import json

import pytest

from signedkan_wip.experiments.runs.bench_to_png import (
    MODEL_ORDER, bars_from_rows, render,
)
from signedkan_wip.experiments.runs.run_inference_bench import (
    HSIKAN_VARIANTS, _summarize,
)

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment-dependent
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# _summarize — pure latency statistics                                        #
# --------------------------------------------------------------------------- #
def test_summarize_median_iqr_worst():
    # samples in seconds: 1..5 ms; median 3 ms, IQR = 4-2 = 2 ms, worst 5 ms.
    samples = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    s = _summarize(samples, n_te=10)
    assert s["fwd_med_ms"] == pytest.approx(3.0)
    assert s["fwd_iqr_ms"] == pytest.approx(2.0)
    assert s["fwd_worst_ms"] == pytest.approx(5.0)
    # per-query = median (3 ms = 3000 us) / 10 queries.
    assert s["per_query_us"] == pytest.approx(300.0)


def test_summarize_worst_ge_median_invariant():
    s = _summarize([2e-3, 9e-3, 1e-3, 4e-3, 4e-3], n_te=5)
    assert s["fwd_worst_ms"] >= s["fwd_med_ms"]
    assert s["fwd_iqr_ms"] >= 0.0


def test_summarize_rejects_empty():
    with pytest.raises(AssertionError):
        _summarize([], n_te=1)


def test_summarize_rejects_nonpositive_n_te():
    with pytest.raises(AssertionError):
        _summarize([1e-3], n_te=0)


# --------------------------------------------------------------------------- #
# bars_from_rows — pure data shaping for the slide-18 chart                    #
# --------------------------------------------------------------------------- #
def _synthetic_rows():
    out = []
    for dataset in ("bitcoin_alpha", "bitcoin_otc"):
        for dev in ("cpu", "cuda"):
            for model, med in (("SGCN", 4.0), ("HSiKAN-lean", 30.5),
                               ("HSiKAN-joint", 342.0)):
                out.append(dict(dataset=dataset, device=dev, model=model,
                                fwd_med_ms=med, fwd_iqr_ms=0.5,
                                fwd_worst_ms=med * 1.1))
    return out


def test_bars_filters_device_and_orders_models():
    bars = bars_from_rows(_synthetic_rows(), device="cpu")
    # 2 datasets × 3 models present in the fixture, none cuda.
    assert len(bars) == 6
    # within bitcoin_alpha, the present models follow MODEL_ORDER's relative order.
    alpha = [b["model"] for b in bars if b["dataset"] == "bitcoin_alpha"]
    expected = [m for m in MODEL_ORDER if m in alpha]
    assert alpha == expected


def test_bars_skip_unknown_model_not_invented():
    rows = _synthetic_rows() + [dict(dataset="bitcoin_alpha", device="cpu",
                                     model="MysteryNet", fwd_med_ms=1.0)]
    bars = bars_from_rows(rows, device="cpu")
    assert all(b["model"] in MODEL_ORDER for b in bars)


def test_bars_empty_for_missing_device():
    assert bars_from_rows(_synthetic_rows(), device="rocm") == []


def test_render_writes_png(tmp_path):
    bars = bars_from_rows(_synthetic_rows(), device="cpu")
    out = render(bars, "cpu", tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_rejects_empty():
    with pytest.raises(AssertionError):
        render([], "cpu", "unused.png")


def test_main_renders_from_json(tmp_path):
    from signedkan_wip.experiments.runs import bench_to_png
    jp = tmp_path / "bench.json"
    jp.write_text(json.dumps(_synthetic_rows()))
    out = bench_to_png.main(json_path=jp, device="cpu",
                            out_path=tmp_path / "out.png")
    assert out.exists()


# --------------------------------------------------------------------------- #
# HSiKAN width variants — integration + performance (skip without torch)       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch/model stack unavailable")
def test_baseline_registry_builds_and_runs():
    """Every registered baseline builds, times, and returns a logits tensor."""
    from signedkan_wip.experiments.runs.run_inference_bench import (
        BASELINES, CONFIGS,
    )
    import torch as _t

    cfg = CONFIGS["bitcoin_alpha"]
    for name, builder in BASELINES.items():
        unit = builder("bitcoin_alpha", cfg, seed=0, device=_t.device("cpu"))
        assert unit.n_params > 0, name
        out = unit.forward()
        assert out.shape[0] == unit.n_te, name


@pytest.mark.skipif(not _HAS_TORCH, reason="torch/model stack unavailable")
def test_lean_has_fewer_params_than_joint():
    from signedkan_wip.experiments.runs.run_inference_bench import (
        CONFIGS, build_hsikan_data, build_hsikan_model,
    )
    import torch as _t

    cfg = dict(CONFIGS["bitcoin_alpha"])
    cfg["max_k"] = dict(k3=0, k4=200, k5=0)  # tiny cap → fast, deterministic
    _data, _n_te, _setup, n_nodes, arities = build_hsikan_data(
        "bitcoin_alpha", cfg, seed=0, device=_t.device("cpu"))
    lean = build_hsikan_model(arities, n_nodes, HSIKAN_VARIANTS["lean"],
                              _t.device("cpu"))
    joint = build_hsikan_model(arities, n_nodes, HSIKAN_VARIANTS["joint"],
                               _t.device("cpu"))
    assert lean.num_parameters() < joint.num_parameters()


@pytest.mark.skipif(not _HAS_TORCH, reason="torch/model stack unavailable")
def test_lean_forward_latency_budget():
    """Perf gate: ≥5 repeats after warmup; median under a generous CPU bound."""
    from signedkan_wip.experiments.runs.run_inference_bench import (
        CONFIGS, _time_block, build_hsikan_data, build_hsikan_model,
        hsikan_forward,
    )
    import torch as _t

    cfg = dict(CONFIGS["bitcoin_alpha"])
    cfg["max_k"] = dict(k3=0, k4=200, k5=0)
    data, n_te, _setup, n_nodes, arities = build_hsikan_data(
        "bitcoin_alpha", cfg, seed=0, device=_t.device("cpu"))
    model = build_hsikan_model(arities, n_nodes, HSIKAN_VARIANTS["lean"],
                               _t.device("cpu"))
    samples = _time_block(lambda: hsikan_forward(model, data),
                          n_repeats=5, warmup=2)
    assert len(samples) == 5
    stats = _summarize(samples, n_te)
    assert stats["fwd_worst_ms"] >= stats["fwd_med_ms"]
    # Generous guard: a tiny-cap lean forward must be well under 2 s on CPU.
    assert stats["fwd_med_ms"] < 2000.0
