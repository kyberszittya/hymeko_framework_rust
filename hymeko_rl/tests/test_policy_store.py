"""Tests for storing a trained policy AS a HyMeKo hypergraph (the storage-thesis artifact).

Layers (§3): unit (the matrix↔star-expansion identity; positive/negative number formatting), integration
(a state_dict round-trips bit-exact through a written .hymeko; the file is valid HyMeKo when the CLI is built).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from hymeko_rl.policy_store import (
    _decode_b64, _encode_b64, _fnum, hymeko_to_policy, hypergraph_to_weight, policy_to_hymeko,
    read_provenance, weight_to_hypergraph,
)

_REPO = Path(__file__).resolve().parents[2]


# ── unit: the matrix ↔ star-expansion identity ────────────────────────────────
def test_weight_hypergraph_roundtrip_exact() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((5, 3)).astype(np.float32)
    h = weight_to_hypergraph(w)
    assert h.n_vertices == 5 and h.n_hyperedges == 3
    assert np.array_equal(hypergraph_to_weight(h), w)
    assert len(h.star_edges) == 15 and h.star_edges[0] == (0, 0, float(w[0, 0]))


def test_weight_to_hypergraph_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        weight_to_hypergraph(np.zeros(4))


# ── unit: number formatting (valid HyMeKo, round-trippable) ───────────────────
@pytest.mark.parametrize("x", [0.0, -0.0, 1.5, -1.23456, 1e-5, -3.2e-7, 12345.678])
def test_fnum_no_exponent_and_roundtrips_float32(x: float) -> None:
    s = _fnum(x)
    assert "e" not in s and "E" not in s, f"{s} has scientific notation (HyMeKo rejects it)"
    assert np.float32(float(s)) == np.float32(x)


# ── integration: state_dict ⇄ .hymeko ─────────────────────────────────────────
def _toy_state_dict() -> "dict[str, torch.Tensor]":
    torch.manual_seed(0)
    return {
        "log_std": torch.randn(1),
        "backbone.a_pos": torch.randn(2, 2),
        "backbone.layers.0.w_pos.weight": torch.randn(8, 2),
        "backbone.layers.0.w_pos.bias": torch.randn(8),
        "head.weight": torch.randn(1, 8),
    }


@pytest.mark.parametrize("tier", ["t0", "t1", "t2", "auto"])
def test_policy_hymeko_roundtrip_bit_exact(tmp_path: Path, tier: str) -> None:
    sd = _toy_state_dict()
    p = policy_to_hymeko(sd, tmp_path / "policy.hymeko", tier=tier)
    sd2 = hymeko_to_policy(p)
    assert set(sd2) == set(sd)
    for k in sd:
        assert sd2[k].shape == sd[k].shape
        assert torch.equal(sd2[k], sd[k]), f"{k} not bit-exact at tier {tier}"


def test_t2_writes_a_small_hymeko_plus_blob(tmp_path: Path) -> None:
    """T2: the .hymeko holds only structure + sha256 refs (small); the bulk floats go to a sibling .npz."""
    sd = {"big.weight": torch.randn(64, 64), "tiny": torch.randn(2)}  # a real-ish bulk tensor
    p = policy_to_hymeko(sd, tmp_path / "policy.hymeko", tier="t2")
    blob = p.with_suffix(".weights.npz")
    assert blob.is_file() and "blob" in p.read_text() and "sha256" in p.read_text()
    assert p.stat().st_size < blob.stat().st_size   # structure file << the weight blob


def test_t2_sha256_rejects_tampered_blob(tmp_path: Path) -> None:
    sd = _toy_state_dict()
    p = policy_to_hymeko(sd, tmp_path / "policy.hymeko", tier="t2")
    blob = p.with_suffix(".weights.npz")
    d = dict(np.load(blob))
    k0 = next(iter(d))
    d[k0] = d[k0] + np.float32(1.0)        # corrupt one tensor
    np.savez(blob, **d)
    with pytest.raises(ValueError, match="sha256"):
        hymeko_to_policy(p)


def test_invalid_tier_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        policy_to_hymeko(_toy_state_dict(), tmp_path / "p.hymeko", tier="binary")


def test_provenance_block_roundtrips(tmp_path: Path) -> None:
    """A `provenance` block (algo/backbone/strategy/…) is written and read back; absent ⇒ empty dict."""
    p = policy_to_hymeko(_toy_state_dict(), tmp_path / "p.hymeko",
                         meta={"algo": "sac", "backbone": "hsikan", "upright": 3.4, "seed": 1})
    prov = read_provenance(p)
    assert prov["algo"] == "sac" and prov["backbone"] == "hsikan"
    assert prov["upright"] == "3.4" and prov["seed"] == "1"
    assert read_provenance(policy_to_hymeko(_toy_state_dict(), tmp_path / "q.hymeko")) == {}


# ── unit: the T1 codec ────────────────────────────────────────────────────────
def test_b64_codec_roundtrips_float32_exact() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(257).astype(np.float32)
    assert np.array_equal(_decode_b64(_encode_b64(a)), a)


def test_t1_smaller_than_t0_and_auto_is_mixed(tmp_path: Path) -> None:
    # a tensor large enough that T1 (binary) beats T0 (decimal text) and auto routes it to binary.
    sd = {"big.weight": torch.randn(64, 64), "tiny": torch.randn(2)}
    t0 = policy_to_hymeko(sd, tmp_path / "t0.hymeko", tier="t0").stat().st_size
    t1 = policy_to_hymeko(sd, tmp_path / "t1.hymeko", tier="t1").stat().st_size
    assert t1 < 0.5 * t0, f"T1 {t1} not < 0.5×T0 {t0}"
    auto = (tmp_path / "auto.hymeko")
    policy_to_hymeko(sd, auto, tier="auto")
    txt = auto.read_text()
    assert "data_b64" in txt and "data [" in txt  # big tensor binary, tiny tensor decimal


@pytest.mark.parametrize("tier", ["t0", "t1", "t2", "auto"])
def test_stored_policy_is_valid_hymeko(tmp_path: Path, tier: str) -> None:
    """The written file is real HyMeKo — `hymeko validate` accepts every tier (skipped if CLI not built)."""
    cli = next((_REPO / "target" / p / "hymeko.exe" for p in ("debug", "release")
                if (_REPO / "target" / p / "hymeko.exe").is_file()), None)
    if cli is None:
        pytest.skip("hymeko CLI not built")
    p = policy_to_hymeko(_toy_state_dict(), tmp_path / "policy.hymeko", tier=tier)
    r = subprocess.run([str(cli), "validate", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, f"validate failed at tier {tier}: {r.stdout}\n{r.stderr}"
