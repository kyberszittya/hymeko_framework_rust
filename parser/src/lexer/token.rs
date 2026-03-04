use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    // punctuation / operators
    LBrace, RBrace, LParen, RParen, LBrack, RBrack, LAngle, RAngle,
    Comma, Semi, Dot, At,
    Colon, // for type annotations
    Plus, Minus, Tilde,
    Arrow,

    // literals bound to the input lifetime
    Ident(&'a str),
    Number(f64),
    Str(Cow<'a, str>),
    EOF,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub msg: String, // Heap allocation is acceptable for rare error paths
    pub at: usize,
}