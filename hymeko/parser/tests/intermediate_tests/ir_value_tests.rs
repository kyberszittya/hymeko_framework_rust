#[cfg(test)]
mod ir_value_tests {
    use std::fs::File;
    use memmap2::Mmap;
    use parser::common::pathkey::PathKey;
    use parser::intern_pass::Interned;
    use parser::ir::ir::{ValueR};
    use parser::ir::lower::lower_to_ir;
    use parser::{parse_from_mmap};

    #[test]
    fn ir_value_resolution() {
        // This is a placeholder test to demonstrate how you might construct IR values.
        // In a real test, you would parse an actual graph and lower it to IR.
        let path = "./data/minimal_examples/minimal_example_with_fields.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        // The AST is valid as long as 'mmap' is in scope.
        let desc = parse_from_mmap(&mmap).unwrap();
        // Intern and resolve as needed, then lower to IR.
        let Interned { ast, mut interner } = parser::intern_pass::intern_ast(&desc);
        let idx = parser::resolve::build_index_sym(&ast, &interner).unwrap();
        let ir = lower_to_ir(&ast, &idx, &mut interner).unwrap();

        let sid_context = interner.intern("context");
        let sid_val0 = interner.intern("val0");
        let sid_val1 = interner.intern("val1");
        let sid_vector = interner.intern("vector");

        let did_context = *idx.by_path.get(&PathKey(vec![sid_context])).expect("context missing");
        let did_val0 = *idx.by_path.get(&PathKey(vec![sid_context, sid_val0])).expect("val0 missing");
        let did_val1 = *idx.by_path.get(&PathKey(vec![sid_context, sid_val1])).expect("val1 missing");
        let did_vector = *idx.by_path.get(&PathKey(vec![sid_context, sid_vector])).expect("vector missing");

        // tags + value
        let sid_int = interner.intern("int");
        assert!(ir.decl_anno[did_val0.0 as usize].tags.contains(&sid_int));
        assert_eq!(ir.decl_anno[did_val0.0 as usize].value, Some(ValueR::Num(56.0)));

        let sid_string = interner.intern("string");
        let sid_vakond = interner.intern("vakond");
        assert!(ir.decl_anno[did_val1.0 as usize].tags.contains(&sid_string));
        assert_eq!(ir.decl_anno[did_val1.0 as usize].value, Some(ValueR::Str(sid_vakond)));

        // vector list
        match ir.decl_anno[did_vector.0 as usize].value.as_ref().expect("vector has no value") {
            ValueR::List(xs) => assert_eq!(xs.len(), 7),
            other => panic!("vector should be list, got {other:?}"),
        }

        // val_undef: tags ok, value None
        let sid_val_undef = interner.intern("val_undef");
        let sid_real = interner.intern("real");
        let did_val_undef = *idx.by_path.get(&PathKey(vec![sid_context, sid_val_undef])).expect("val_undef missing");
        assert!(ir.decl_anno[did_val_undef.0 as usize].tags.contains(&sid_real));
        assert!(ir.decl_anno[did_val_undef.0 as usize].value.is_none());

        // sanity: context itself exists
        let _ = did_context;

    }
}