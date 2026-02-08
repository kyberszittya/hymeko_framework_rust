pub mod ast;

// Must be pub
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub hymeko);

pub fn parse_description(input: &str) -> Result<ast::Description, lalrpop_util::ParseError<usize, lalrpop_util::lexer::Token<'_>, &'static str>> {
    hymeko::DescriptionParser::new().parse(input)
}


