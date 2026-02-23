

use std::path::{PathBuf};
use hymeko_framework::module_store::module_store::ModuleStore;
use hymeko_framework::module_store::source_provider::StdFsProvider;
use hymeko_framework::util::pretty_print::pretty_print_compiled;
use hymeko_framework::util::real_parser::RealParser;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| {
            eprintln!("Usage: hymeko <path-to-file.hymeko>");
            std::process::exit(2);
        });

    let root_path = PathBuf::from(path);

    let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);

    let compiled = ms.compile(&root_path).unwrap_or_else(|e| {
        eprintln!("compile failed: {e:?}");
        std::process::exit(1);
    });

    pretty_print_compiled(&ms.it, &compiled);
}