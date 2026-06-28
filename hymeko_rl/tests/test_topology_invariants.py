"""Topology invariants (the walk-structure measures): frame coherence, balance frustration, spectral gap."""
from hymeko_rl.topology_invariants import balance_frustration, invariants, run_invariants
from hymeko_rl.topology_zoo import TOPOLOGIES


def test_petersen_is_tight_frame() -> None:
    inv = invariants(TOPOLOGIES["petersen"](seed=0))
    # SRG(10,3,0,1): every non-adjacent pair shares exactly 1 neighbour => |row_i.row_j|/3 = 1/3, sign-independent.
    assert abs(inv["frame_coherence"] - 1 / 3) < 5e-3
    assert inv["deg_irregularity"] == 0.0                 # 3-regular


def test_star_is_degenerate() -> None:
    inv = invariants(TOPOLOGIES["star"](9, seed=0))
    assert inv["frame_coherence"] == 1.0                  # hub: every spoke-pair overlaps fully


def test_frustration_caps_on_large_graphs() -> None:
    # Kneser K(9,2) has 36 vertices -> 2^36 brute-force is skipped (returns -1).
    assert balance_frustration(TOPOLOGIES["kneser"](9, seed=0)) == -1
    assert balance_frustration(TOPOLOGIES["ring"](9, seed=0)) == 0   # a single even cycle is balanceable


def test_run_invariants_covers_zoo_and_designs() -> None:
    r = run_invariants(n_nodes=9, seed=0)
    for name in ("petersen", "steiner9", "fano7", "star", "grotzsch"):
        assert name in r and "frame_coherence" in r[name]
    # the structural tight-frame trio
    for name in ("petersen", "steiner9", "fano7"):
        assert abs(r[name]["frame_coherence"] - 1 / 3) < 5e-3
