use parser::parse_description;
use parser::ast::*;
use super::constants::*;

fn must_parse<'a>(input: &'a str) -> AstStr<'a> {
    parse_description(input).unwrap()
}

// EdgeArcs now filters based on the reference lifetime
fn edge_arcs<'ast, 'slice>(e: &'slice EdgeDecl<'ast, &'ast str>) -> Vec<&'slice HyperArc<'ast, &'ast str>> {
    e.inner
        .body
        .iter()
        .filter_map(|it| match it {
            HyperItem::Arc(a) => Some(a),
            _ => None,
        })
        .collect()
}

#[test]
fn parses_minimal_description() {
    let d = must_parse(PARSE_DESC_SRC);

    assert_eq!(d.name, DESC_MY_DESC);
    assert_eq!(d.items.len(), BASE_ITEM_COUNT);

    match &d.items[0] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, NODE_A_NAME),
        _ => panic!("Expected Node(A)"),
    }
}

#[test]
fn parses_multiple_arcs_in_one_edge() {
    let d = must_parse(MULTI_ARC_DESC_SRC);

    // items: A, B, C, E1
    let e1 = match &d.items[MULTI_ARC_EDGE_INDEX] {
        HyperItem::Edge(e) => e,
        _ => panic!("Expected Edge at items[3]"),
    };

    let arcs = edge_arcs(e1);
    assert_eq!(arcs.len(), MULTI_ARC_COUNT);
}

#[test]
fn fails_if_arc_missing_semicolon() {
    let err = parse_description(MISSING_SEMI_DESC_SRC)
        .unwrap_err();

    let _ = err;
}