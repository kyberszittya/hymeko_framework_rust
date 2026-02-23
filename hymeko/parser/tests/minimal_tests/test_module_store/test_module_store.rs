#!cfg[(test)]
mod mod_test_module_store {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use parser::module_store::{ModuleStore};
    use parser::resolve::{build_index_sym_with_prefix, validate_all_refs_sym_with_prefix, Index};
    use parser::source_provider::MemProvider;
    use crate::minimal_tests::TestParser;

    #[test]
    fn module_store_loads_root_and_one_import() {
        // root.hmk importálja a dep.hmk-t A alias alatt
        let root_src = r#"
        Root {
            @"dep.hmk" -> A;
        }
        // items üres
    "#;

        let dep_src = r#"
        Dep { }
    "#;

        let fs = MemProvider::default()
            .with_file("root.hmk", Arc::<str>::from(root_src))
            .with_file("dep.hmk", Arc::<str>::from(dep_src));
        let fs_probe = fs.clone();
        // Ha nálad: ModuleStore::new(fs, parser)
        let mut ms = ModuleStore::new(fs, TestParser);

        let root_key = ms.load_recursive(Path::new("root.hmk")).unwrap();
        let root_entry = ms.get(&root_key).unwrap();

        assert_eq!(root_entry.deps.len(), 1);

        // ellenőrzés: dep is bent van a store-ban
        let dep_key = &root_entry.deps[0];
        assert!(ms.get(dep_key).is_some());

        assert_eq!(fs_probe.read_count("root.hmk"), 1);
        assert_eq!(fs_probe.read_count("dep.hmk"), 1);

        // ✅ cache ellenőrzés: újra betöltve nem olvasunk többet
        let _ = ms.load_recursive(Path::new("root.hmk")).unwrap();
        assert_eq!(fs_probe.read_count("root.hmk"), 1);
        assert_eq!(fs_probe.read_count("dep.hmk"), 1);
    }

    #[test]
    fn module_store_allows_resolving_imported_refs() {
        let root_src = r#"
Root { @"dep.hmk" -> A; }

elem {
  B{}
  @E {(+A.Dep.Foo);}
}
"#;

        let dep_src = r#"
dep_src{}
Dep { Foo; }
"#;

        let fs = MemProvider::default()
            .with_file("root.hmk", root_src)
            .with_file("dep.hmk", dep_src);

        let mut ms = ModuleStore::new(fs, TestParser);

        // 1) load modules
        let root_key = ms.load_recursive(Path::new("root.hmk")).unwrap();

        // ✅ klónozzuk ki az AST-t, ne referencia legyen a store-ból
        let root_ast = ms.get(&root_key).unwrap().ast.clone();

        let dep_key = ms.get(&root_key).unwrap().deps[0].clone();
        let dep_ast = ms.get(&dep_key).unwrap().ast.clone();

        // most már oké mut interner
        let a = ms.it.intern("A");

        let mut idx = Index { by_path: HashMap::new() };
        let mut next: u32 = 0;

        build_index_sym_with_prefix(&root_ast, &[], &ms.it, &mut idx, &mut next).unwrap();
        build_index_sym_with_prefix(&dep_ast, &[a], &ms.it, &mut idx, &mut next).unwrap();

        validate_all_refs_sym_with_prefix(&root_ast, &[], &idx, &mut ms.it).unwrap();
    }


}