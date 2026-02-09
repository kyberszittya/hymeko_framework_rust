use super::token::{Token, LexError};

pub type Spanned<T> = (usize, T, usize);

pub fn lex(input: &str) -> Result<Vec<Spanned<Token>>, LexError> {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut out: Vec<Spanned<Token>> = Vec::new();

    while i < bytes.len() {
        // -------- skip whitespace --------
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // -------- line comment //... \n --------
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
            continue;
        }

        let start = i;

        // -------- 2-char operator: -> --------
        if bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            i += 2;
            out.push((start, Token::Arrow, i));
            continue;
        }

        // -------- 1-char tokens --------
        let one = match bytes[i] {
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
            i += 1;
            out.push((start, t, i));
            continue;
        }

        // -------- string literal: " ... " with escapes --------
        if bytes[i] == b'"' {
            i += 1;
            let mut s = String::new();
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => {
                        if i + 1 >= bytes.len() {
                            return Err(LexError{ msg:"Unterminated escape".into(), at:i });
                        }
                        let esc = bytes[i + 1];
                        match esc {
                            b'"' => s.push('"'),
                            b'\\' => s.push('\\'),
                            b'n' => s.push('\n'),
                            b't' => s.push('\t'),
                            _ => return Err(LexError{ msg: format!("Bad escape: \\{}", esc as char), at:i }),
                        }
                        i += 2;
                    }
                    b'"' => { i += 1; break; }
                    _ => {
                        // ASCII-only safe push; ha kell UTF-8 string, akkor slice-olni kell.
                        s.push(bytes[i] as char);
                        i += 1;
                    }
                }
            }
            if i > bytes.len() || bytes.get(i-1) != Some(&b'"') {
                return Err(LexError{ msg:"Unterminated string".into(), at:start });
            }
            out.push((start, Token::Str(s), i));
            continue;
        }

        // -------- number: 123 or 12.34 --------
        if bytes[i].is_ascii_digit() {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            if i < bytes.len() && bytes[i] == b'.' {
                // decimal part
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            }
            let text = &input[start..i];
            let num = text.parse::<f64>().map_err(|_| LexError {
                msg: format!("Bad number: {}", text),
                at: start
            })?;
            out.push((start, Token::Number(num), i));
            continue;
        }

        // -------- ident: [A-Za-z_][A-Za-z0-9_]* --------
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let text = &input[start..i];
            out.push((start, Token::Ident(text.to_string()), i));
            continue;
        }

        // -------- unknown --------
        return Err(LexError {
            msg: format!("Unexpected character: {:?}", bytes[i] as char),
            at: i
        });
    }

    Ok(out)
}