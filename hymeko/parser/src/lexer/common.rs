use super::token::{Token, LexError};

pub type Location = usize;
pub type Spanned<T> = (Location, T, Location);
pub type LexItem = Result<Spanned<Token>, LexError>;

#[derive(Clone, Copy)]
pub struct Input<'i> {
    pub bytes: &'i [u8],
}

impl<'i> Input<'i> {
    #[inline(always)]
    pub fn new(s: &'i str) -> Self {
        Self { bytes: s.as_bytes() }
    }
}

pub trait LexerCore {
    fn next_item(&mut self) -> Option<super::common::LexItem>;
}




pub struct Lexer<'i> {
    input: super::common::Input<'i>,
    i: usize,
    n: usize,
}

impl<'i> Lexer<'i> {
    pub fn new(s: &'i str) -> Self {
        let input = super::common::Input::new(s);
        let n = input.bytes.len();
        Self { input, i: 0, n }
    }
}