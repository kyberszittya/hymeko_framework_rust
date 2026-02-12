use crate::interner::Interner;
use super::token::{Token, LexError};


pub type Spanned<T> = (usize, T, usize);

pub struct Lexer<'input> {
    input: &'input str,
    bytes: &'input [u8],
    i: usize
}

impl<'input, 'a> Lexer<'input> {
    pub fn new(input: &'input str) -> Self {
        Self { input, bytes: input.as_bytes(), i: 0}
    }
}



impl<'input, 'a> Iterator for Lexer<'input> {
    type Item = Result<Spanned<Token>, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.bytes;

        while self.i < bytes.len() {
            // -------- skip whitespace --------
            if bytes[self.i].is_ascii_whitespace() {
                self.i += 1;
                continue;
            }

            // -------- line comment //... \n --------
            if bytes[self.i] == b'/' && self.i + 1 < bytes.len() && bytes[self.i + 1] == b'/' {
                self.i += 2;
                while self.i < bytes.len() && bytes[self.i] != b'\n' { self.i += 1; }
                continue;
            }

            let start = self.i;

            // -------- 2-char operator: -> --------
            if bytes[self.i] == b'-' && self.i + 1 < bytes.len() && bytes[self.i + 1] == b'>' {
                self.i += 2;
                return Some(Ok((start, Token::Arrow, self.i)));
            }

            // -------- 1-char tokens --------
            let one = match bytes[self.i] {
                b'{' => Some(Token::LBrace),
                b'}' => Some(Token::RBrace),
                b'(' => Some(Token::LParen),
                b')' => Some(Token::RParen),
                b'[' => Some(Token::LBrack),
                b']' => Some(Token::RBrack),
                b'<' => Some(Token::LAngle),
                b'>' => Some(Token::RAngle),
                b',' => Some(Token::Comma),
                b';' => Some(Token::Semi),
                b'.' => Some(Token::Dot),
                b'@' => Some(Token::At),
                b'+' => Some(Token::Plus),
                b'-' => Some(Token::Minus),
                b'~' => Some(Token::Tilde),
                _ => None,
            };
            if let Some(t) = one {
                self.i += 1;
                return Some(Ok((start, t, self.i)));
            }

            // -------- string literal: " ... " with escapes --------
            if bytes[self.i] == b'"' {
                self.i += 1;
                let mut s = String::new();
                while self.i < bytes.len() {
                    match bytes[self.i] {
                        b'\\' => {
                            if self.i + 1 >= bytes.len() {
                                return Some(Err(LexError{ msg:"Unterminated escape".into(), at:self.i }));
                            }
                            let esc = bytes[self.i + 1];
                            match esc {
                                b'"' => s.push('"'),
                                b'\\' => s.push('\\'),
                                b'n' => s.push('\n'),
                                b't' => s.push('\t'),
                                _ => return Some(Err(LexError{ msg: format!("Bad escape: \\{}", esc as char), at:self.i })),
                            }
                            self.i += 2;
                        }
                        b'"' => { self.i += 1; break; }
                        _ => {
                            s.push(bytes[self.i] as char);
                            self.i += 1;
                        }
                    }
                }
                if self.i > bytes.len() || bytes.get(self.i.wrapping_sub(1)) != Some(&b'"') {
                    return Some(Err(LexError{ msg:"Unterminated string".into(), at:start }));
                }
                return Some(Ok((start, Token::Str(s), self.i)));
            }

            // -------- number: 123 or 12.34 --------
            if bytes[self.i].is_ascii_digit() {
                self.i += 1;
                while self.i < bytes.len() && bytes[self.i].is_ascii_digit() { self.i += 1; }
                if self.i < bytes.len() && bytes[self.i] == b'.' {
                    self.i += 1;
                    while self.i < bytes.len() && bytes[self.i].is_ascii_digit() { self.i += 1; }
                }
                let text = &self.input[start..self.i];
                let num = match text.parse::<f64>() {
                    Ok(n) => n,
                    Err(_) => {
                        return Some(Err(LexError{ msg: format!("Bad number: {}", text), at:start }));
                    }
                };
                return Some(Ok((start, Token::Number(num), self.i)));
            }

            // -------- ident: [A-Za-z_][A-Za-z0-9_]* --------
            if bytes[self.i].is_ascii_alphabetic() || bytes[self.i] == b'_' {
                self.i += 1;
                while self.i < bytes.len() && (bytes[self.i].is_ascii_alphanumeric() || bytes[self.i] == b'_') {
                    self.i += 1;
                }
                let text = &self.input[start..self.i];

                return Some(Ok((start, Token::Ident(text.to_string()), self.i)));
            }

            // -------- unknown --------
            return Some(Err(LexError {
                msg: format!("Unexpected character: {:?}", bytes[self.i] as char),
                at: self.i
            }));
        }

        None
    }
}

