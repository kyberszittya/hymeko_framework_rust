// src/lexer/simd.rs
use crate::lexer::{LexError, Token};

pub struct Lexer<'a> {
    s: &'a [u8],
    i: usize,
    n: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let b = input.as_bytes();
        Self { s: b, i: 0, n: b.len() }
    }

    #[inline(always)]
    fn peek(&self) -> Option<u8> {
        if self.i < self.n { Some(self.s[self.i]) } else { None }
    }

    #[inline(always)]
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }

    #[inline(always)]
    fn is_ident_start(c: u8) -> bool {
        (c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z') || c == b'_'
    }
    #[inline(always)]
    fn is_ident_cont(c: u8) -> bool {
        Self::is_ident_start(c) || (c >= b'0' && c <= b'9')
    }

    // ---------- SIMD: whitespace ----------
    #[inline(always)]
    fn skip_ws_and_line_comments(&mut self) -> Result<(), LexError> {
        loop {
            // Whitespace (SIMD / fallback)
            self.skip_ws();
            // Not enough char for comment prefixes
            if self.i + 1 >= self.n {
                return Ok(());
            }
            // Comment exclusion
            // Line comment
            if self.i + 1 < self.n && self.s[self.i] == b'/' &&
                self.s[self.i + 1] == b'/' {
                self.i += 2;
                while self.i < self.n {
                    let c = self.s[self.i];
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    self.i += 1;
                }
                // Collect \r\n
                continue;
            }
            // Block comment (/* */)
            if self.s[self.i] == b'/' && self.s[self.i + 1] == b'*' {
                let start = self.i;
                self.i += 2;

                let mut closed = false;

                while self.i + 1 < self.n {
                    if self.s[self.i] == b'*' && self.s[self.i + 1] == b'/' {
                        self.i += 2;
                        closed = true;
                        break;
                    }
                    self.i += 1;
                }
                // No closing before EOF
                if self.i + 1 >= self.n {
                    return Err(LexError{
                        msg: "Unterminated block comment /* ... */".to_string(),
                        at: start
                    });
                }
                if !closed {
                    return Err(LexError{
                        msg: "Unterminated block comment /* ... */".to_string(),
                        at: start
                    });

                }
                continue;
            }
            return Ok(());
        }
    }


    #[inline(always)]
    fn skip_ws(&mut self) {
        #[cfg(all(target_arch = "x86_64"))]
        {
            // runtime dispatch: ha van AVX2, használd; különben SSE2
            if std::is_x86_feature_detected!("avx2") {
                unsafe { self.skip_ws_avx2(); }
                return;
            } else {
                unsafe { self.skip_ws_sse2(); }
                return;
            }
        }

        // fallback (nem x86_64)
        while let Some(c) = self.peek() {
            match c {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn skip_ws_sse2(&mut self) {
        use std::arch::x86_64::*;
        while self.i < self.n {
            // scalar eleje
            let c = self.s[self.i];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.i += 1;
                continue;
            }

            let rem = self.n - self.i;
            if rem < 16 { break; }

            let p = self.s.as_ptr().add(self.i) as *const __m128i;
            let chunk = _mm_loadu_si128(p);

            let sp = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b' ' as i8));
            let tb = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\t' as i8));
            let nl = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\n' as i8));
            let cr = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\r' as i8));

            let ws = _mm_or_si128(_mm_or_si128(sp, tb), _mm_or_si128(nl, cr));
            let mask = _mm_movemask_epi8(ws) as u32;

            if mask == 0xFFFF {
                self.i += 16;
                continue;
            } else {
                // első nem-ws byte
                let not_ws = (!mask) & 0xFFFF;
                let adv = not_ws.trailing_zeros() as usize;
                self.i += adv;
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn skip_ws_avx2(&mut self) {
        use std::arch::x86_64::*;
        while self.i < self.n {
            let c = self.s[self.i];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.i += 1;
                continue;
            }

            let rem = self.n - self.i;
            if rem < 32 { break; }

            let p = self.s.as_ptr().add(self.i) as *const __m256i;
            let chunk = _mm256_loadu_si256(p);

            let sp = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
            let tb = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8));
            let nl = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\n' as i8));
            let cr = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\r' as i8));

            let ws = _mm256_or_si256(_mm256_or_si256(sp, tb), _mm256_or_si256(nl, cr));
            let mask = _mm256_movemask_epi8(ws) as u32;

            if mask == 0xFFFF_FFFF {
                self.i += 32;
                continue;
            } else {
                let not_ws = !mask;
                let adv = not_ws.trailing_zeros() as usize;
                self.i += adv;
                return;
            }
        }
    }

    // ---------- SIMD: ident tail ----------
    #[inline(always)]
    fn scan_ident_tail(&mut self) {
        // itt az a feltétel: [A-Za-z0-9_]
        #[cfg(all(target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe { self.scan_ident_tail_avx2(); }
                return;
            } else {
                unsafe { self.scan_ident_tail_sse2(); }
                return;
            }
        }

        // fallback
        while self.i < self.n && Self::is_ident_cont(self.s[self.i]) {
            self.i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn scan_ident_tail_sse2(&mut self) {
        use std::arch::x86_64::*;

        while self.i < self.n {
            // 0) ha a következő nem ident-cont, vége
            if !Self::is_ident_cont(self.s[self.i]) {
                return;
            }

            // 1) kis maradék: scalar
            let rem = self.n - self.i;
            if rem < 16 {
                while self.i < self.n && Self::is_ident_cont(self.s[self.i]) {
                    self.i += 1;
                }
                return;
            }

            let p = self.s.as_ptr().add(self.i) as *const __m128i;
            let x = _mm_loadu_si128(p);

            // in-range: (x>=lo && x<=hi)
            let ge = |lo: u8| _mm_cmpeq_epi8(_mm_max_epu8(x, _mm_set1_epi8(lo as i8)), x);
            let le = |hi: u8| _mm_cmpeq_epi8(_mm_min_epu8(x, _mm_set1_epi8(hi as i8)), x);

            let is_az = _mm_and_si128(ge(b'a'), le(b'z'));
            let is_AZ = _mm_and_si128(ge(b'A'), le(b'Z'));
            let is_09 = _mm_and_si128(ge(b'0'), le(b'9'));
            let is_us = _mm_cmpeq_epi8(x, _mm_set1_epi8(b'_' as i8));

            let ok = _mm_or_si128(_mm_or_si128(is_az, is_AZ), _mm_or_si128(is_09, is_us));
            let mask_ok = _mm_movemask_epi8(ok) as u32; // alsó 16 bit

            if (mask_ok & 0xFFFF) == 0xFFFF {
                // mind a 16 ok → ugorj tovább, és folytasd a while-t (jöhet újabb SIMD chunk)
                self.i += 16;
            } else {
                // első nem-ok pozíció → addig ugorj, és vége
                let not_ok = (!mask_ok) & 0xFFFF;
                let adv = not_ok.trailing_zeros() as usize;
                self.i += adv;
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_ident_tail_avx2(&mut self) {
        use std::arch::x86_64::*;

        while self.i < self.n {
            // Ha a következő nem ident-cont, vége.
            if !Self::is_ident_cont(self.s[self.i]) {
                return;
            }

            let rem = self.n - self.i;
            if rem < 32 {
                // kis maradék: scalar
                while self.i < self.n && Self::is_ident_cont(self.s[self.i]) {
                    self.i += 1;
                }
                return;
            }

            // SIMD: 32 byte ellenőrzés egyben
            let p = self.s.as_ptr().add(self.i) as *const __m256i;
            let x = _mm256_loadu_si256(p);

            // in-range: (x>=lo && x<=hi)
            let ge = |lo: u8| _mm256_cmpeq_epi8(_mm256_max_epu8(x, _mm256_set1_epi8(lo as i8)), x);
            let le = |hi: u8| _mm256_cmpeq_epi8(_mm256_min_epu8(x, _mm256_set1_epi8(hi as i8)), x);

            let is_az = _mm256_and_si256(ge(b'a'), le(b'z'));
            let is_AZ = _mm256_and_si256(ge(b'A'), le(b'Z'));
            let is_09 = _mm256_and_si256(ge(b'0'), le(b'9'));
            let is_us = _mm256_cmpeq_epi8(x, _mm256_set1_epi8(b'_' as i8));

            let ok = _mm256_or_si256(
                _mm256_or_si256(is_az, is_AZ),
                _mm256_or_si256(is_09, is_us),
            );

            let mask_ok = _mm256_movemask_epi8(ok) as u32;

            if mask_ok == 0xFFFF_FFFF {
                self.i += 32;
            } else {
                let not_ok = !mask_ok;
                let adv = not_ok.trailing_zeros() as usize;
                self.i += adv;
                return;
            }
        }
    }

    // ---------- lexers ----------
    fn lex_ident(&mut self, start: usize) -> Token {
        // self.i már az első char UTÁN van, tehát csak a tailt scanneljük
        self.scan_ident_tail();
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap().to_string();
        Token::Ident(text)
    }

    fn lex_number(&mut self, start: usize) -> Result<Token, LexError> {
        while self.i < self.n {
            let c = self.s[self.i];
            if (c >= b'0' && c <= b'9') || c == b'.' {
                self.i += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        match text.parse::<f64>() {
            Ok(v) => Ok(Token::Number(v)),
            Err(_) => Err(LexError { at: start, msg: format!("Bad number literal: {}", text) }),
        }
    }

    fn lex_string(&mut self, start: usize) -> Result<Token, LexError> {
        let mut out = String::new();
        while let Some(c) = self.bump() {
            match c {
                b'"' => return Ok(Token::Str(out)),
                b'\\' => {
                    let esc = self.bump().ok_or(LexError{at:self.i, msg:"Unterminated escape".into()})?;
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

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<(usize, Token, usize), LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Err(e) = self.skip_ws_and_line_comments() {
            return Some(Err(e));
        }
        let start = self.i;
        let c = self.bump()?;

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
                if self.i < self.n && self.s[self.i] == b'>' {
                    self.i += 1;
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            b'"' => match self.lex_string(start) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            },
            d if (d >= b'0' && d <= b'9') => match self.lex_number(start) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            },
            a if Lexer::is_ident_start(a) => self.lex_ident(start),
            other => {
                return Some(Err(LexError {
                    at: start,
                    msg: format!("Unexpected char: {:?}", other as char),
                }))
            }
        };

        let end = self.i;
        Some(Ok((start, tok, end)))
    }
}
