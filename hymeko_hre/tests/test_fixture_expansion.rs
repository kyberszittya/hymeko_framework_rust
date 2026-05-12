//! End-to-end expansion tests that load real `.hymeko` fixture files,
//! run them through the `ModuleStore → Ir → HyperGraphView` pipeline, and
//! assert that `hymeko_hre::expansion::{star,clique}_expansion_coo` produce
//! the shapes we expect for the robotics models checked in under
//! `data/robotics/`.
//!
//! Complements the hand-built struct-literal tests in `test_expansion.rs`
//! by exercising the whole compiler stack, not just the tensor math.

mod common;

use hymeko::tensor::tensor_val::EdgeWScalar;
use hymeko_hre::expansion::{clique_expansion_coo, star_expansion_coo};

use crate::common::{load_and_lower, view_f32};

const MINI_ARM: &str = "../data/robotics/mini_arm.hymeko";
const MOVEO_ARM: &str = "../data/robotics/anthropomorphic_arm.hymeko";
const DIFF_ROBOT: &str = "../data/robotics/robot_4wh.hymeko";

// ---- mini_arm: 2 links + 1 continuous joint ------------------------------

#[test]
fn mini_arm_star_expansion_has_one_slice_per_hyperedge() {
    let (_store, compiled) = load_and_lower(MINI_ARM).expect("mini_arm should compile");
    let view = view_f32(&compiled);

    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, view.num_edges(), "one slice per hyperedge");
    assert_eq!(
        t.dim_i,
        view.num_nodes() + view.num_edges(),
        "dim = |V| + |E|"
    );
    assert_eq!(t.dim_j, view.num_nodes() + view.num_edges());
    assert!(
        !t.is_empty(),
        "mini_arm should produce non-empty star expansion"
    );
}

#[test]
fn mini_arm_clique_expansion_has_node_sized_slices() {
    let (_store, compiled) = load_and_lower(MINI_ARM).expect("mini_arm should compile");
    let view = view_f32(&compiled);

    let t = clique_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, view.num_edges());
    assert_eq!(t.dim_i, view.num_nodes(), "clique dim = |V|");
    assert_eq!(t.dim_j, view.num_nodes());
}

#[test]
fn mini_arm_star_expansion_entries_reference_valid_indices() {
    let (_store, compiled) = load_and_lower(MINI_ARM).expect("mini_arm should compile");
    let view = view_f32(&compiled);
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);

    let dim = view.num_nodes() + view.num_edges();
    for idx in 0..t.len() {
        let e = t.entry(idx);
        assert!(e.k < t.num_slices, "slice {} out of bounds", e.k);
        assert!(e.i < dim, "i={} out of bounds {}", e.i, dim);
        assert!(e.j < dim, "j={} out of bounds {}", e.j, dim);
    }
}

// ---- anthropomorphic_arm: 6 revolute + 1 fixed joint ---------------------

#[test]
fn moveo_star_expansion_slice_count_matches_edge_count() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo should compile");
    let view = view_f32(&compiled);
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, view.num_edges());
}

#[test]
fn moveo_star_expansion_non_empty_for_robot_joints() {
    // The `anthropomorphic_arm` fixture imports `meta_kinematics.hymeko`,
    // which contributes a handful of meta-hyperedges (joint templates,
    // plugin placeholders) that carry no concrete arcs — their star slices
    // are legitimately empty. The *robot's own* joints, however, must each
    // produce at least one incidence.
    //
    // moveo declares 7 robot joints: `j_fix`, `j0..j4`, `jtool`. The plus
    // control/simulation hyperedges (`gazebo_sim_system`,
    // `sim_control_plugin`, `joint_state_broadcaster`,
    // `joint_trajectory_controller`, `arm_joint_control`) also contribute,
    // so the lower bound is 7 — strictly looser than "every slice".
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo should compile");
    let view = view_f32(&compiled);
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);

    let num_edges = view.num_edges();
    let mut per_slice = vec![0usize; num_edges];
    for idx in 0..t.len() {
        per_slice[t.entry(idx).k] += 1;
    }
    let populated = per_slice.iter().filter(|&&cnt| cnt > 0).count();
    assert!(
        populated >= 7,
        "expected at least 7 populated star slices (one per robot joint), got {}",
        populated
    );
}

#[test]
fn moveo_clique_expansion_more_sparse_than_dense() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo should compile");
    let view = view_f32(&compiled);
    let t = clique_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);

    // Sanity: the nnz count stays well below dim_i^2 * num_slices.
    let dense = t.dim_i * t.dim_j * t.num_slices;
    assert!(
        t.len() < dense,
        "expected sparse fill, got nnz={} vs dense={}",
        t.len(),
        dense
    );
}

// ---- diff_robot: 4-wheel differential drive -----------------------------

#[test]
fn diff_robot_star_shape_consistent_with_view_counts() {
    let (_store, compiled) = load_and_lower(DIFF_ROBOT).expect("diff_robot should compile");
    let view = view_f32(&compiled);
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, view.num_edges());
    assert_eq!(t.dim_i, view.num_nodes() + view.num_edges());
}

// ---- alias parity on the expansion side ---------------------------------

#[test]
fn alias_and_baseline_fixtures_produce_equal_star_nnz() {
    // If `using ... as ...` is a pure desugaring, the aliased and baseline
    // fixtures must yield identical star-expansion shapes.
    let (_s1, c1) = load_and_lower(MOVEO_ARM).unwrap();
    let (_s2, c2) = load_and_lower("../data/robotics/anthropomorphic_arm_using.hymeko").unwrap();

    let t1 = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view_f32(&c1));
    let t2 = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view_f32(&c2));

    assert_eq!(t1.num_slices, t2.num_slices);
    assert_eq!(t1.dim_i, t2.dim_i);
    assert_eq!(t1.dim_j, t2.dim_j);
    assert_eq!(t1.len(), t2.len(), "aliased fixture must produce equal nnz");
}
