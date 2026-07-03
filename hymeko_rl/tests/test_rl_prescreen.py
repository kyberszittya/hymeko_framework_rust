"""The fast RL architecture pre-screen: returns a per-backbone rank, a recommendation, and an honest caveat."""
from hymeko_rl.eval.rl_prescreen import prescreen


def test_prescreen_ranks_recommends_and_caveats() -> None:
    r = prescreen(("mlp", "sa_hsikan"), steps=60)
    assert set(r.arms) == {"mlp", "sa_hsikan"}
    for m in r.arms.values():
        assert "reward" in m and "deploy_ms" in m and m["params"] > 0
    assert r.recommend in ("mlp", "sa_hsikan")
    assert "stability" in r.caveat                                   # the latency screen states its limit
