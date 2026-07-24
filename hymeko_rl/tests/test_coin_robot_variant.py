"""§1+§2 gates — the BALLTIP_INTERARM_FILTERED_V1 robot variant is HyMeKo-declared: spherical fingertips + simulator-level
inter-arm collision filtering via collision groups (not event-ignoring), with the collision manifest/query and the exact
ignored body pairs / masks. The frozen canonical baseline is unchanged."""
import mujoco

from hymeko_rl.coin_delivery.coin_robot_variant import (
    VARIANTS,
    build_variant_model,
    collision_manifest,
    min_interarm_clearance,
    read_collision_policy,
)


def test_collision_policy_from_hymeko():
    p = read_collision_policy()                                          # data/robotics/galambos_planar_balltip_v1.hymeko
    assert p["inter_arm"] == "filtered"
    assert p["left_contype"] == 4 and p["right_contype"] == 8 and p["conaffinity"] == 3
    assert p["coin_contype"] == 2 and p["coin_conaffinity"] == 2


def test_spherical_fingertips_radius_from_spec():
    # canonical tip 0.014; both ball-tip variants enlarge it to 0.020 (collision+visual one geom)
    assert collision_manifest(build_variant_model("canonical")[0])["fingertip_radii"] == [0.014]
    assert collision_manifest(build_variant_model("balltip_nofilter")[0])["fingertip_radii"] == [0.02]
    assert collision_manifest(build_variant_model("balltip_filtered")[0])["fingertip_radii"] == [0.02]


def test_inter_arm_filtering_via_collision_groups():
    m_canon = build_variant_model("canonical")[0]
    m_nofilt = build_variant_model("balltip_nofilter")[0]
    m_filt = build_variant_model("balltip_filtered")[0]
    for m, name in ((m_canon, "canonical"), (m_nofilt, "nofilter")):
        man = collision_manifest(m)
        assert man["inter_arm_collision"] == "enabled" and man["left_right_can_collide"] is True, name
    man = collision_manifest(m_filt)
    assert man["inter_arm_collision"] == "filtered" and man["left_right_can_collide"] is False   # left↔right DISABLED
    # the required policy holds for the filtered variant: arm↔coin and arm↔floor stay enabled
    assert man["arm_coin_enabled"] and man["arm_floor_enabled"]
    assert man["group_masks"]["left_contype"] == [4] and man["group_masks"]["right_contype"] == [8]


def test_filtered_body_pairs_and_masks_recorded():
    man = collision_manifest(build_variant_model("balltip_filtered")[0])
    # the exact ignored pairs are 3 left links × 3 right links = 9 (base/link1/link2 each side)
    assert len(man["filtered_body_pairs"]) == 9
    assert "base_left|base_right" in man["filtered_body_pairs"]
    assert all("_left|" in p and p.endswith("_right") for p in man["filtered_body_pairs"])


def test_min_interarm_clearance_diagnostic():
    m, _a = build_variant_model("balltip_filtered")
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    c = min_interarm_clearance(m, d)
    assert isinstance(c, float) and c > 0.0                              # arms rest well apart; diagnostic is a real distance


def test_variants_share_dof_layout():
    # all three variants have the SAME qpos layout (only geom radius + collision masks differ) -> a canonical start state
    # can be set into any variant env for a matched-panel comparison
    nqs = {v: build_variant_model(v)[0].nq for v in VARIANTS}
    assert len(set(nqs.values())) == 1, nqs
