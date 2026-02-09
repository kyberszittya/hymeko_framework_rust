pub mod ast;
pub mod lexer;

// Must be pub
use crate::hymeko::DescriptionParser;
use lalrpop_util::lalrpop_mod;
use crate::ast::Description;
use crate::lexer::{LexError, Token};

lalrpop_mod!(pub hymeko);


pub fn parse_description(input: &str) -> Result<Description, lalrpop_util::ParseError<usize, Token, LexError>> {
    let lexer = crate::lexer::simd::Lexer::new(input);
    DescriptionParser::new().parse(lexer)
}


