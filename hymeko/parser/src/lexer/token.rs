
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // punctuation / operators
    LBrace, RBrace, LParen, RParen, LBrack, RBrack, LAngle, RAngle,
    Comma, Semi, Dot, At,
    Plus, Minus, Tilde,
    Arrow, // ->

    // literals
    Ident(String),
    Number(f64),
    Str(String),

    // end / error helper
    // (EOF-t LALRPOP nem mindig igényel; mi nem adjuk ki tokenként.)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub msg: String,
    pub at: usize, // byte offset
}