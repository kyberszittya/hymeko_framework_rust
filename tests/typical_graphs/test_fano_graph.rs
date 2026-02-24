use hymeko_framework::{body, find_node};
use parser::ast::*;


#[test]
fn parse_fano_graph() {

    let path = "./data/typical_graphs/fano_graph.hymeko";
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let desc = parser::parse_description(&source_code).unwrap();

    // Top-level: "Fano_graph" név + üres header
    assert_eq!(desc.name, "Fano_graph");
    assert!(desc.header.is_empty(), "Expected empty header");

    // Top-levelben legyen a fano block (NodeDecl body-val)
    let fano = find_node(&desc.items, "fano");
    let fano_body = body(fano);

    // 7 node: n0..n6
    for i in 0..7 {
        let n = format!("n{}", i);
        let _ = find_node(fano_body, &n);
    }

    // 7 edge: e0..e6
    // Itt csak azt ellenőrizzük, hogy mindegyik EdgeDecl megvan,
    // és hogy a body-jában 1 arc van, és az 3 referenciát tartalmaz.
    for i in 0..7 {
        let ename = format!("e{}", i);

        let edge = fano_body
            .iter()
            .find_map(|it| match it {
                HyperItem::Edge(e) if e.inner.name == ename => Some(e),
                _ => None,
            })
            .unwrap_or_else(|| panic!("Expected Edge({}) in fano body", ename));

        // edge.inner.body : Vec<HyperItem>
        let arc_items: Vec<&parser::ast::HyperArc<&str>> = edge
            .inner
            .body
            .iter()
            .filter_map(|x| match x {
                HyperItem::Arc(a) => Some(a),
                _ => None,
            })
            .collect();

        assert_eq!(
            arc_items.len(),
            1,
            "Each edge should contain exactly 1 HyperArc statement; edge={}",
            ename
        );

        let arc = arc_items[0];

        // A te jelenlegi grammarodban:
        // HyperArc { inner: ArcInner { refs } }
        // ahol refs tipikusan Vec<SignedRef> vagy Vec<DirectedRef>.
        // A Fano input "~ n0" -> SignedRef::Neutral várható.
        assert_eq!(
            arc.inner.refs.len(),
            3,
            "Expected 3 endpoints in arc inside edge {}",
            edge.inner.name
        );
    }

    // extra sanity: a fano body-ban ne legyen top-level arc
    // (minden arc edge body-ban van)
    assert!(
        !fano_body.iter().any(|it| matches!(it, HyperItem::Arc(_))),
        "Did not expect HyperArc directly under `fano graph`"
    );
}