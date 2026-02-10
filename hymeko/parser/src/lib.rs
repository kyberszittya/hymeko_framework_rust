pub mod ast;
pub mod lexer;

use crate::hymeko::DescriptionParser;
use lalrpop_util::lalrpop_mod;
use crate::ast::Description;
use crate::lexer::{LexError, Token};

lalrpop_mod!(pub hymeko);


pub fn parse_description(input: &str) -> Result<Description, lalrpop_util::ParseError<usize, Token, LexError>> {
    let lexer = crate::lexer::simd::Lexer::new(input);
    DescriptionParser::new().parse(lexer)
}


// Read file and parse
pub fn read_parse_file(path: &str) -> Result<Description, lalrpop_util::ParseError<usize, Token, LexError>>
{
    let input = std::fs::read_to_string(path).unwrap();
    parse_description(&input)
}