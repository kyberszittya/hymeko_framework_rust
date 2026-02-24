#[cfg(test)]
mod ir_value_tests {
    use std::f64;
    use hymeko_framework::common::pathkey::PathKey;
    use hymeko_framework::resolution::intern_pass::Interned;
    use hymeko_framework::ir::ir::{ValueR};
    use hymeko_framework::ir::lower::lower_to_ir;
    use hymeko_framework::resolution::intern_pass;
    use hymeko_framework::resolution::resolve::build_index_sym;

    #[test]
    fn ir_value_resolution() {
        // This is a placeholder test to demonstrate how you might construct IR values.
        // In a real test, you would parse an actual graph and lower it to IR.
        let path = "./data/minimal_examples/minimal_example_with_fields.hymeko";
        let source_code = parser::read_source_file(&path).expect("failed to read source file");

        // 2. Parse it, tying the AST lifetimes to the String
        let desc = parser::parse_description(&source_code).unwrap();
        // Intern and resolve as needed, then lower to IR.
        let Interned { ast, mut interner } = intern_pass::intern_ast(&desc);
        let idx = build_index_sym(&ast, &interner).unwrap();
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

        // negative scalar
        let sid_val_neg = interner.intern("val_neg");
        let did_val_neg = *idx.by_path.get(&PathKey(vec![sid_context, sid_val_neg])).expect("val_neg missing");
        assert_eq!(ir.decl_anno[did_val_neg.0 as usize].value, Some(ValueR::Num(-42.0)));

        // vector list
        match ir.decl_anno[did_vector.0 as usize].value.as_ref().expect("vector has no value") {
            ValueR::List(xs) => {
                assert_eq!(xs.len(), 7);
                if let Some(ValueR::Num(v)) = xs.get(1) {
                    assert!((*v + 17.8).abs() < f64::EPSILON, "second element should be -17.8, got {v}");
                } else {
                    panic!("second vector element should be numeric");
                }
            }
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