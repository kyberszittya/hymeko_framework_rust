use super::token::{Token, LexError};

pub type Location = usize;
pub type Spanned<T> = (Location, T, Location);
pub type LexItem = Result<Spanned<Token>, LexError>;

#[inline(always)]
fn is_ident_start(c: u8) -> bool {
    (c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z') || c == b'_'
}

#[inline(always)]
fn is_ident_cont(c: u8) -> bool {
    is_ident_start(c) || (c >= b'0' && c <= b'9')
}

/// A közös lexer “backend” trait.
/// A SIMD/simple lexer csak ezt implementálja + a 2 speciális hookot.
pub trait CommonLexer {
    fn bytes(&self) -> &[u8];
    fn pos(&self) -> usize;
    fn set_pos(&mut self, i: usize);

    #[inline(always)]
    fn byte_at(&self, idx: usize) -> Option<u8> {
        let b = self.bytes();
        if idx < b.len() { Some(b[idx]) } else { None }
    }

    #[inline(always)]
    fn peek2(&self) -> Option<(u8, u8)> {
        let i = self.pos();
        Some((self.byte_at(i)?, self.byte_at(i + 1)?))
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.bytes().len()
    }

    #[inline(always)]
    fn peek(&self) -> Option<u8> {
        let i = self.pos();
        let b = self.bytes();
        if i < b.len() { Some(b[i]) } else { None }
    }

    #[inline(always)]
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.set_pos(self.pos() + 1);
        Some(c)
    }

    /// Hook #1: whitespace skip (SIMD vagy scalar)
    fn skip_ws(&mut self);

    /// Hook #2: ident tail scan (SIMD vagy scalar)
    fn scan_ident_tail(&mut self);

    /// Opcionális: ha külön akarod kezelni a `/* */`-t és `//`-t
    #[inline(always)]
    fn skip_ws_and_comments(&mut self) -> Result<(), LexError> {
        loop {
            self.skip_ws();

            let i = self.pos();
            let Some((c0, c1)) = self.peek2() else { return Ok(()); };

            // line comment: //
            if c0 == b'/' && c1 == b'/' {
                self.set_pos(i + 2);
                while let Some(c) = self.peek() {
                    if c == b'\n' || c == b'\r' { break; }
                    self.set_pos(self.pos() + 1);
                }
                continue;
            }

            // block comment: /* ... */
            if c0 == b'/' && c1 == b'*' {
                let start = i;
                self.set_pos(i + 2);

                loop {
                    let p = self.pos();
                    let Some((a, b)) = self.peek2() else {
                        return Err(LexError {
                            msg: "Unterminated block comment /* ... */".to_string(),
                            at: start,
                        });
                    };

                    if a == b'*' && b == b'/' {
                        self.set_pos(p + 2);
                        break;
                    }
                    self.set_pos(p + 1);
                }

                continue;
            }

            return Ok(());
        }
    }

    #[inline(always)]
    fn lex_ident(&mut self, start: usize) -> Token {
        self.scan_ident_tail();
        let text = std::str::from_utf8(&self.bytes()[start..self.pos()]).unwrap();
        Token::Ident(text.to_string())
    }

    #[inline(always)]
    fn lex_number(&mut self, start: usize) -> Result<Token, LexError> {
        let mut i = self.pos();
        let mut seen_dot = false;

        while let Some(c) = self.byte_at(i) {
            if c.is_ascii_digit() {
                i += 1;
            } else if c == b'.' && !seen_dot {
                seen_dot = true;
                i += 1;
            } else {
                break;
            }
        }

        self.set_pos(i);

        let text = std::str::from_utf8(&self.bytes()[start..i]).unwrap();
        text.parse::<f64>()
            .map(Token::Number)
            .map_err(|_| LexError { at: start, msg: format!("Bad number literal: {}", text) })
    }

    #[inline(always)]
    fn lex_string(&mut self, start: usize) -> Result<Token, LexError> {
        let mut out = String::new();
        while let Some(c) = self.bump() {
            match c {
                b'"' => return Ok(Token::Str(out)),
                b'\\' => {
                    let esc = self.bump().ok_or(LexError{ at: self.pos(), msg: "Unterminated escape".into() })?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        other => out.push(other as char),
                    }
                }
                other => out.push(other as char),
            }
        }
        Err(LexError { at: start, msg: "Unterminated string literal".into() })
    }
}

/// Közös `next()` implementáció.
/// A lexer típusa adja a hookokat.
pub fn next_token<L: CommonLexer>(lex: &mut L) -> Option<LexItem> {
    if let Err(e) = lex.skip_ws_and_comments() {
        return Some(Err(e));
    }

    let start = lex.pos();
    let c = lex.bump()?;

    let tok = match c {
        b'{' => Token::LBrace,
        b'}' => Token::RBrace,
        b'(' => Token::LParen,
        b')' => Token::RParen,
        b'[' => Token::LBrack,
        b']' => Token::RBrack,
        b'<' => Token::LAngle,
        b'>' => Token::RAngle,
        b',' => Token::Comma,
        b';' => Token::Semi,
        b'.' => Token::Dot,
        b'@' => Token::At,
        b'+' => Token::Plus,
        b'~' => Token::Tilde,

        b'-' => {
            let i = lex.pos();
            if lex.byte_at(i) == Some(b'>') {
                lex.set_pos(i + 1);
                Token::Arrow
            } else {
                Token::Minus
            }
        }

        b'"' => match lex.lex_string(start) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        },

        d if d.is_ascii_digit() => match lex.lex_number(start) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        },

        a if is_ident_start(a) => lex.lex_ident(start),

        other => {
            return Some(Err(LexError {
                at: start,
                msg: format!("Unexpected char: {:?}", other as char),
            }))
        }
    };

    let end = lex.pos();
    Some(Ok((start, tok, end)))
}