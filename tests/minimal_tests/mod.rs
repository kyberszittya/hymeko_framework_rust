use hymeko_framework::resolution::module_store::HymekoParser;
use parser::ast::AstStr;
use parser::hymeko::DescriptionParser;
use parser::lexer::simd::Lexer;

pub mod test_minimal_example;
pub mod test_minimal_example_with_fields;

pub mod test_minimal_example_fileread;

pub mod test_read_minimal_example;
mod test_minimal_example_basic_hierarchy;
mod test_smoke_test;
mod test_minimal_example_field_refs;
mod test_minimal_example_comments;
mod edges;

mod test_import;
mod test_module_store;


struct TestParser;

impl HymekoParser for TestParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        let p = DescriptionParser::new();
        p.parse(Lexer::new(src))
            .map_err(|e| format!("{e:?}"))
    }
}