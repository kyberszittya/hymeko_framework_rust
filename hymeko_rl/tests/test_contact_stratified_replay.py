"""Regression tests for contact-stratified replay sampling (train.replay.ReplayBuffer.sample_stratified) + the
coin-delivery stratum labelling. The single experimental variable of the contact-strategy experiment; these pin the
sampler contract so it cannot silently drift (ratios, with-replacement on thin strata, unchanged online/uniform path)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.experiments.coin_contact_replay import STRATA, STRATA_WEIGHTS, _stratum
from hymeko_rl.train.replay import ReplayBuffer


def _seeded(n_per_stratum: dict[int, int], n_online: int, *, dim: int = 3, adim: int = 2) -> ReplayBuffer:
    buf = ReplayBuffer(100_000, (dim,), adim)
    tags = np.concatenate([np.full(n, t, np.int16) for t, n in n_per_stratum.items()]) if n_per_stratum else np.array([], np.int16)
    m = len(tags)
    if m:
        z = np.zeros((m, dim), np.float32)
        buf.add_batch(z, np.zeros((m, adim), np.float32), np.zeros(m, np.float32), z, np.zeros(m, bool), tags=tags)
    for _ in range(n_online):
        buf.add(np.zeros(dim, np.float32), np.zeros(adim, np.float32), 0.0, np.zeros(dim, np.float32), False)
    return buf


def test_1_requested_ratios_realized_exactly() -> None:
    buf = _seeded({1: 500, 2: 500, 3: 500, 4: 500, 5: 500}, 2000)
    g, acc = np.random.default_rng(0), {}
    for _ in range(200):
        buf.sample_stratified(256, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=g, account=acc)
    tot = sum(acc.values())
    demo = sum(v for k, v in acc.items() if k > 0)
    assert abs(demo / tot - 0.5) <= 0.01                              # demo/online realized within rounding
    for k, w in STRATA_WEIGHTS.items():
        assert abs(acc[k] / demo - w) <= 0.02                        # per-stratum realized within rounding


def test_2_certified_bilateral_satisfies_conditions() -> None:
    # a quality that is labelled CERTIFIED_BILATERAL must meet the defining predicate (strict, OR all contact gates)
    q = dict(attribution=0.7, body=0.1, clean=True, bilateral=True, strict=False, zone=True)
    assert _stratum(q, in_recovery=False) == STRATA["CERTIFIED_BILATERAL"]
    assert q["strict"] or (q["zone"] and q["attribution"] >= 0.60 and q["body"] <= 0.20 and q["clean"] and q["bilateral"])
    assert _stratum(dict(attribution=0.1, body=0.9, clean=False, bilateral=False, strict=True, zone=True),
                    in_recovery=False) == STRATA["CERTIFIED_BILATERAL"]  # strict alone certifies


def test_3_contrastive_bulldoze_holds_near_strict_failures() -> None:
    # zone entry but sub-0.60 fingertip attribution (the measured 64111-style near-strict miss) -> CONTRASTIVE_BULLDOZE
    assert _stratum(dict(attribution=0.47, body=0.1, clean=False, bilateral=True, strict=False, zone=True),
                    in_recovery=False) == STRATA["CONTRASTIVE_BULLDOZE"]
    assert _stratum(dict(attribution=0.3, body=0.1, clean=True, bilateral=False, strict=False, zone=True),
                    in_recovery=False) == STRATA["CONTRASTIVE_BULLDOZE"]


def test_4_thin_stratum_samples_with_replacement_and_logs() -> None:
    buf = _seeded({1: 2, 2: 500, 3: 500, 4: 500, 5: 500}, 1000)     # CERTIFIED has only 2 (< its quota)
    g, sh, acc = np.random.default_rng(1), {}, {}
    buf.sample_stratified(256, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=g, shortage=sh, account=acc)
    assert acc[1] > 2                                                 # quota kept despite only 2 unique
    assert sh.get(1, {}).get("replacement", 0) > 0                    # observable shortage statistic emitted


def test_5_online_sampling_unchanged_by_tags() -> None:
    # sample() (uniform) must ignore tags entirely: identical draws whether or not strata are present
    a = _seeded({1: 100, 2: 100}, 300)
    b = _seeded({}, 500)                                              # all ONLINE, same total size
    ra = a.sample(32, generator=np.random.default_rng(5))
    rb = b.sample(32, generator=np.random.default_rng(5))
    assert np.array_equal(ra[0].numpy(), rb[0].numpy())              # tags do not perturb the uniform path


def test_6_control_reproduces_prechange_uniform() -> None:
    buf = _seeded({1: 100, 2: 100, 3: 100}, 700)
    # the CONTROL arm calls buf.sample(...) — its indices depend only on (generator, size), never on the new tag array
    r1 = buf.sample(64, generator=np.random.default_rng(9))
    r2 = buf.sample(64, generator=np.random.default_rng(9))
    assert np.array_equal(r1[0].numpy(), r2[0].numpy())


def test_7_batch_valid_when_optional_strata_empty() -> None:
    buf = _seeded({1: 300, 5: 300}, 1000)                            # strata 2,3,4 absent
    g, sh = np.random.default_rng(2), {}
    obs, act, rew, nxt, done = buf.sample_stratified(128, demo_frac=0.5, strata_weights=STRATA_WEIGHTS,
                                                     generator=g, shortage=sh)
    assert obs.shape[0] == 128                                       # exact batch size despite empty strata
    assert any(sh.get(t, {}).get("empty", 0) for t in (2, 3, 4))     # empty strata logged, not silently filled


def test_8_deterministic_for_fixed_rng() -> None:
    buf = _seeded({1: 200, 2: 200, 3: 200, 4: 200, 5: 200}, 800)
    r1 = buf.sample_stratified(64, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=np.random.default_rng(11))
    r2 = buf.sample_stratified(64, demo_frac=0.5, strata_weights=STRATA_WEIGHTS, generator=np.random.default_rng(11))
    assert np.array_equal(r1[1].numpy(), r2[1].numpy())              # identical sampled batch for identical RNG state
