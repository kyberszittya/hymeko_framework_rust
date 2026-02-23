
pub mod ast;
pub mod lexer;

use std::fs::File;
use crate::hymeko::DescriptionParser;
use lalrpop_util::lalrpop_mod;
use memmap2::Mmap;
use crate::ast::{AstStr};
use crate::lexer::{LexError, Token};

lalrpop_mod!(pub hymeko);



pub fn parse_description<'a>(input: &'a str) -> Result<AstStr<'a>, lalrpop_util::ParseError<usize, Token<'a>, LexError>> {
    let lexer = crate::lexer::simd::Lexer::new(input);

    // DescriptionParser now correctly returns AstStr<'a> (Description<'a, &'a str>)
    DescriptionParser::new().parse(lexer)
}

pub struct ParsedFile<'a> {
    pub mmap: Mmap,
    pub ast: AstStr<'a>,
}

pub fn read_parse_file(path: &str) -> Result<Mmap, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    // SAFETY: We assume the file is not being modified concurrently.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

pub fn parse_from_mmap<'a>(mmap: &'a Mmap) -> Result<AstStr<'a>, lalrpop_util::ParseError<usize, Token<'a>, LexError>> {
    let input = std::str::from_utf8(mmap).map_err(|_| {
        lalrpop_util::ParseError::User {
            error: LexError { at: 0, msg: "File is not valid UTF-8".into() }
        }
    })?;
    parse_description(input)
}



