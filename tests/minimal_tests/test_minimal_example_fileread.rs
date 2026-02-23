use std::fs::File;
use memmap2::Mmap;
use parser::ast::*;
use parser::{parse_from_mmap};


#[test]
fn test_minimal_example() {
    let path = "./data/minimal_examples/minimal_example.hymeko";
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    // The AST is valid as long as 'mmap' is in scope.
    let d = parse_from_mmap(&mmap).unwrap();

    assert_eq!(d.name, "Minimal_Example");
    /*
    assert_eq!(d.meta.inner.name, "Minimal_Example");
    assert!(d.meta.inner.body.is_some());

     */

    assert_eq!(d.items.len(), 1);
    match &d.items[0] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, "context"),
        _ => panic!("Expected Node(context)"),
    }
}

#[test]
fn test_minimal_example_base_elements() {
    let path = "./data/minimal_examples/minimal_example_base_elements.hymeko";
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    // The AST is valid as long as 'mmap' is in scope.
    let d = parse_from_mmap(&mmap).unwrap();

    assert_eq!(d.name, "MyDesc");

    assert_eq!(d.items.len(), 3);

    // Hypernode A
    match &d.items[0] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, "A"),
        _ => panic!("Expected Node(A)"),
    }

    // Hypernode B
    match &d.items[1] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, "B"),
        _ => panic!("Expected Node(B)"),
    }

    // Edge E1: arcs már a body-ban vannak
    match &d.items[2] {
        HyperItem::Edge(e) => {
            assert_eq!(e.inner.name, "E1");

            // szedjük ki az arcokat a body-ból
            let arcs: Vec<&HyperArc<&str>> = e.inner.body.iter().filter_map(|it| {
                if let HyperItem::Arc(a) = it { Some(a) } else { None }
            }).collect();

            assert_eq!(arcs.len(), 1);
            assert_eq!(arcs[0].inner.refs.len(), 2);
        }
        _ => panic!("Expected Edge(E1)"),
    }
}