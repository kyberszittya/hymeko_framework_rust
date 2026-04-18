//! Simple, hand-built expansion tests.
//!
//! Build a tiny [`HyperGraphView`] by struct-literal construction (all fields
//! are `pub`) so we don't need to run the full `ModuleStore` → `Ir` →
//! `HyperGraphView::from_ir` pipeline just to verify expansion math.
//!
//! Fixture A — "fano-lite": 3 nodes `{n0, n1, n2}`, 1 hyperedge `e0` with
//! signs `(+n0, -n1, -n2)`. Hand-computed expectations:
//!   - Star expansion: dim = |V| + |E| = 4, slice 0 has 3 entries:
//!       (k=0, i=0, j=3, 1.0)  `+n0 -> e0`
//!       (k=0, i=3, j=1, 1.0)  `e0 -> -n1`
//!       (k=0, i=3, j=2, 1.0)  `e0 -> -n2`
//!   - Clique expansion: dim = |V| = 3, slice 0 has 4 entries:
//!       (+,-) `n0 -> n1` and `n0 -> n2`
//!       (-,-) neutral fan-out `n1 <-> n2`

use std::marker::PhantomData;

use hymeko::common::ids::{DeclId, EdgeId, NodeId};
use hymeko::tensor::tensor_val::EdgeWScalar;
use hymeko::traversal::hypergraphview::HyperGraphView;
use hymeko_hre::expansion::{clique_expansion_coo, star_expansion_coo, star_expansion_coo_normalized};

type View = HyperGraphView<f32, EdgeWScalar<f32>, f32>;

/// Build the fano-lite fixture described above.
fn fano_lite() -> View {
    HyperGraphView {
        node_decl: vec![DeclId::new(0), DeclId::new(1), DeclId::new(2)],
        edge_decl: vec![DeclId::new(3)],
        flat_node_edges: vec![EdgeId::new(0), EdgeId::new(0), EdgeId::new(0)],
        flat_node_sign: vec![1, -1, -1],
        node_offsets: vec![0, 1, 2, 3],
        flat_edge_nodes: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
        flat_edge_sign: vec![1, -1, -1],
        edge_offsets: vec![0, 3],
        flat_node_w: vec![1.0, 1.0, 1.0],
        flat_edge_w: vec![1.0, 1.0, 1.0],
        edge_weight: vec![EdgeWScalar(1.0)],
        _phantom: PhantomData,
    }
}

/// Two-edge fixture so we can check that slice-k dispatches correctly.
/// Edges: e0=(+n0,-n1), e1=(+n1,-n2). All signs are `+/-` so there's no
/// neutral doubling.
fn two_edge_chain() -> View {
    HyperGraphView {
        node_decl: vec![DeclId::new(0), DeclId::new(1), DeclId::new(2)],
        edge_decl: vec![DeclId::new(3), DeclId::new(4)],
        // node -> edges: n0 -> e0; n1 -> e0, e1; n2 -> e1
        flat_node_edges: vec![
            EdgeId::new(0),
            EdgeId::new(0),
            EdgeId::new(1),
            EdgeId::new(1),
        ],
        flat_node_sign: vec![1, -1, 1, -1],
        node_offsets: vec![0, 1, 3, 4],
        // edge -> nodes: e0 -> n0, n1; e1 -> n1, n2
        flat_edge_nodes: vec![
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
        ],
        flat_edge_sign: vec![1, -1, 1, -1],
        edge_offsets: vec![0, 2, 4],
        flat_node_w: vec![1.0, 1.0, 1.0, 1.0],
        flat_edge_w: vec![1.0, 1.0, 1.0, 1.0],
        edge_weight: vec![EdgeWScalar(1.0), EdgeWScalar(1.0)],
        _phantom: PhantomData,
    }
}

// ---- star_expansion_coo ---------------------------------------------------

#[test]
fn star_expansion_fano_lite_shape() {
    let view = fano_lite();
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, 1, "one hyperedge => one slice");
    assert_eq!(t.dim_i, 4, "dim = |V| + |E| = 3 + 1");
    assert_eq!(t.dim_j, 4);
    assert_eq!(t.len(), 3, "3 signed incidences, no neutral");
}

#[test]
fn star_expansion_fano_lite_entries() {
    let view = fano_lite();
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    // edge index in V* is num_nodes + edge_id = 3
    let triples: Vec<_> = (0..t.len())
        .map(|i| {
            let e = t.entry(i);
            (e.k, e.i, e.j, e.v)
        })
        .collect();
    assert!(triples.contains(&(0, 0, 3, 1.0)), "+n0 -> e0 missing");
    assert!(triples.contains(&(0, 3, 1, 1.0)), "e0 -> -n1 missing");
    assert!(triples.contains(&(0, 3, 2, 1.0)), "e0 -> -n2 missing");
}

#[test]
fn star_expansion_two_edge_chain_per_slice() {
    let view = two_edge_chain();
    let t = star_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, 2, "two edges => two slices");
    assert_eq!(t.dim_i, 5, "|V|=3 + |E|=2");
    // 2 incidences per edge, no neutrals
    assert_eq!(t.len(), 4);
    // Count per-slice
    let slice0 = (0..t.len()).filter(|&i| t.entry(i).k == 0).count();
    let slice1 = (0..t.len()).filter(|&i| t.entry(i).k == 1).count();
    assert_eq!(slice0, 2);
    assert_eq!(slice1, 2);
}

// ---- clique_expansion_coo -------------------------------------------------

#[test]
fn clique_expansion_fano_lite_shape() {
    let view = fano_lite();
    let t = clique_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    assert_eq!(t.num_slices, 1);
    assert_eq!(t.dim_i, 3, "clique dim = |V|");
    assert_eq!(t.dim_j, 3);
}

#[test]
fn clique_expansion_fano_lite_entries() {
    let view = fano_lite();
    let t = clique_expansion_coo::<f32, EdgeWScalar<f32>, f32>(&view);
    let triples: Vec<_> = (0..t.len())
        .map(|i| {
            let e = t.entry(i);
            (e.k, e.i, e.j, e.v)
        })
        .collect();
    // (+n0, -n1): push u=0 -> v=1
    assert!(triples.contains(&(0, 0, 1, 1.0)));
    // (+n0, -n2): push u=0 -> v=2
    assert!(triples.contains(&(0, 0, 2, 1.0)));
    // (-n1, -n2): neutral pair, both directions
    assert!(triples.contains(&(0, 1, 2, 1.0)));
    assert!(triples.contains(&(0, 2, 1, 1.0)));
    assert_eq!(t.len(), 4, "2 directed + 1 neutral pair (fired twice)");
}

// ---- star_expansion_coo_normalized ---------------------------------------

#[test]
fn star_expansion_normalized_weights_bounded() {
    let view = fano_lite();
    let t = star_expansion_coo_normalized::<f32, EdgeWScalar<f32>, f32>(&view, false, 1e-6);
    // With deg_v = 1 (each node has one edge), deg_e = 3 (edge has 3 nodes),
    // normalized weight = 1 / sqrt(1 * 3) = 0.5773...
    let expected = 1.0 / (1.0_f32 * 3.0).sqrt();
    for i in 0..t.len() {
        let e = t.entry(i);
        let diff = (e.v.abs() - expected).abs();
        assert!(
            diff < 1e-5,
            "normalized weight {} differs from expected {} by {}",
            e.v,
            expected,
            diff
        );
    }
}

#[test]
fn star_expansion_normalized_use_abs_flips_sign() {
    let view = fano_lite();
    let t_signed =
        star_expansion_coo_normalized::<f32, EdgeWScalar<f32>, f32>(&view, false, 1e-6);
    let t_abs = star_expansion_coo_normalized::<f32, EdgeWScalar<f32>, f32>(&view, true, 1e-6);
    // Every weight in the abs view is non-negative.
    for i in 0..t_abs.len() {
        assert!(t_abs.entry(i).v >= 0.0);
    }
    // Signed view contains at least one negative weight (the - incidences).
    let any_negative = (0..t_signed.len()).any(|i| t_signed.entry(i).v < 0.0);
    assert!(any_negative);
}
