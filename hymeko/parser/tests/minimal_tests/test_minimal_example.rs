use parser::parse_description;
use parser::ast::*;

fn must_parse(input: &str) -> Description {
    parse_description(input).unwrap()
}

fn edge_arcs(e: &EdgeDecl) -> Vec<&HyperArc> {
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
    let d = must_parse(
        r#"
        MyDesc
        { }

        A ;
        B ;

        @E1 {
          (+A, -B);
        }
        "#
    );

    assert_eq!(d.name, "MyDesc");



    assert_eq!(d.items.len(), 3);

    match &d.items[0] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, "A"),
        _ => panic!("Expected Node(A)"),
    }
}

#[test]
fn parses_multiple_arcs_in_one_edge() {
    let d = must_parse(
        r#"
        D
        { }

        A ;
        B ;
        C ;

        @E1 {
          (+A, -B );
          (+A, -C );
        }
        "#
    );

    // items: A, B, C, E1
    let e1 = match &d.items[3] {
        HyperItem::Edge(e) => e,
        _ => panic!("Expected Edge at items[3]"),
    };

    let arcs = edge_arcs(e1);
    assert_eq!(arcs.len(), 2);
}

#[test]
fn fails_if_arc_missing_semicolon() {
    let err = parse_description(
        r#"
        D
        { }

        A ;
        B ;

        @E1 {
          +A -B
        }
        "#
    )
        .unwrap_err();

    let _ = err;
}