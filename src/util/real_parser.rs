use parser::ast::AstStr;
use parser::hymeko::DescriptionParser;
use parser::lexer::simd::Lexer;
use crate::module_store::module_store::HymekoParser;

// ----------------------
// Parser adapter (LALRPOP + Lexer)
// ----------------------
pub struct RealParser;

impl HymekoParser for RealParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        let p = DescriptionParser::new();
        p.parse(Lexer::new(src))
            .map_err(|e| format!("{e:?}"))
    }
}