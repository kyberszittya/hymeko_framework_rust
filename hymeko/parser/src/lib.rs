
pub mod ast;
pub mod lexer;

use std::fs;
use std::path::Path;
use crate::hymeko::DescriptionParser;
use lalrpop_util::lalrpop_mod;
use crate::ast::{AstStr};
use crate::lexer::{LexError, Token};
use crate::lexer::simd::{Avx2Lexer, CoreLexer, ScalarLexer, Sse2Lexer};

lalrpop_mod!(pub hymeko);


// The generic boundary. The compiler will generate 3 highly optimized
// copies of this parser, one for each lexer type.
#[inline(always)]
fn parse_inner<'a, I>(
    iter: I,
) -> Result<AstStr<'a>, lalrpop_util::ParseError<usize, Token<'a>, LexError>>
where
    I: Iterator<Item = Result<(usize, Token<'a>, usize), LexError>>,
{
    DescriptionParser::new().parse(iter)
}

pub fn parse_description<'a>(
    input: &'a str,
) -> Result<AstStr<'a>, lalrpop_util::ParseError<usize, Token<'a>, LexError>> {
    let core = CoreLexer::new(input);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return parse_inner(Avx2Lexer(core));
        }
        if std::is_x86_feature_detected!("sse2") {
            return parse_inner(Sse2Lexer(core));
        }
    }

    // Fallback for non-x86 or older processors
    parse_inner(ScalarLexer(core))
}

pub fn read_source_file<P: AsRef<Path>>(path: P) -> std::io::Result<String> {
    fs::read_to_string(path)
}