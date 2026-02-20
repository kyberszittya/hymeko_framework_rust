use super::common::{self, CommonLexer, LexItem};

// 1. Base state
pub struct CoreLexer<'a> {
    s: &'a [u8],
    i: usize,
    n: usize,
}

impl<'a> CoreLexer<'a> {
    #[inline(always)]
    fn is_ident_cont(c: u8) -> bool {
        (c >= b'a' && c <= b'z')
            || (c >= b'A' && c <= b'Z')
            || (c >= b'0' && c <= b'9')
            || c == b'_'
    }
}

// 2. Specific hardware variants
pub struct Avx2Lexer<'a>(CoreLexer<'a>);
pub struct Sse2Lexer<'a>(CoreLexer<'a>);
pub struct ScalarLexer<'a>(CoreLexer<'a>);

// 3. Public unified interface
pub enum Lexer<'a> {
    Avx2(Avx2Lexer<'a>),
    Sse2(Sse2Lexer<'a>),
    Scalar(ScalarLexer<'a>),
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let core = CoreLexer {
            s: input.as_bytes(),
            i: 0,
            n: input.len(),
        };

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return Lexer::Avx2(Avx2Lexer(core));
            }
            if std::is_x86_feature_detected!("sse2") {
                return Lexer::Sse2(Sse2Lexer(core));
            }
        }
        Lexer::Scalar(ScalarLexer(core))
    }
}

// 4. Iterator implementation delegates to monomorphized paths
impl<'a> Iterator for Lexer<'a> {
    type Item = LexItem<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Lexer::Avx2(l) => common::next_token(l),
            Lexer::Sse2(l) => common::next_token(l),
            Lexer::Scalar(l) => common::next_token(l),
        }
    }
}

// 5. Trait Implementations

// --- AVX2 ---
impl<'a> Avx2Lexer<'a> {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn skip_ws_avx2(&mut self) {
        use std::arch::x86_64::*;
        while self.0.i < self.0.n {
            let c = self.0.s[self.0.i];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.0.i += 1;
                continue;
            }

            let rem = self.0.n - self.0.i;
            if rem < 32 { break; }

            let p = self.0.s.as_ptr().add(self.0.i) as *const __m256i;
            let chunk = _mm256_loadu_si256(p);

            let sp = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
            let tb = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8));
            let nl = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\n' as i8));
            let cr = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\r' as i8));

            let ws = _mm256_or_si256(_mm256_or_si256(sp, tb), _mm256_or_si256(nl, cr));
            let mask = _mm256_movemask_epi8(ws) as u32;

            if mask == 0xFFFF_FFFF {
                self.0.i += 32;
                continue;
            } else {
                let not_ws = !mask;
                let adv = not_ws.trailing_zeros() as usize;
                self.0.i += adv;
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_ident_tail_avx2(&mut self) {
        use std::arch::x86_64::*;
        while self.0.i < self.0.n {
            if !CoreLexer::is_ident_cont(self.0.s[self.0.i]) {
                return;
            }

            let rem = self.0.n - self.0.i;
            if rem < 32 {
                while self.0.i < self.0.n && CoreLexer::is_ident_cont(self.0.s[self.0.i]) {
                    self.0.i += 1;
                }
                return;
            }

            let p = self.0.s.as_ptr().add(self.0.i) as *const __m256i;
            let x = _mm256_loadu_si256(p);

            let ge = |lo: u8| _mm256_cmpeq_epi8(_mm256_max_epu8(x, _mm256_set1_epi8(lo as i8)), x);
            let le = |hi: u8| _mm256_cmpeq_epi8(_mm256_min_epu8(x, _mm256_set1_epi8(hi as i8)), x);

            let is_az = _mm256_and_si256(ge(b'a'), le(b'z'));
            let is_az_upper = _mm256_and_si256(ge(b'A'), le(b'Z'));
            let is_09 = _mm256_and_si256(ge(b'0'), le(b'9'));
            let is_us = _mm256_cmpeq_epi8(x, _mm256_set1_epi8(b'_' as i8));

            let ok = _mm256_or_si256(
                _mm256_or_si256(is_az, is_az_upper),
                _mm256_or_si256(is_09, is_us),
            );

            let mask_ok = _mm256_movemask_epi8(ok) as u32;

            if mask_ok == 0xFFFF_FFFF {
                self.0.i += 32;
            } else {
                let not_ok = !mask_ok;
                let adv = not_ok.trailing_zeros() as usize;
                self.0.i += adv;
                return;
            }
        }
    }
}

impl<'a> CommonLexer<'a> for Avx2Lexer<'a> {
    #[inline(always)] fn bytes(&self) -> &'a[u8] { self.0.s }
    #[inline(always)] fn pos(&self) -> usize { self.0.i }
    #[inline(always)] fn set_pos(&mut self, v: usize) { self.0.i = v; }

    #[inline(always)]
    fn skip_ws(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe { self.skip_ws_avx2(); }
    }

    #[inline(always)]
    fn scan_ident_tail(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe { self.scan_ident_tail_avx2(); }
    }
}

// --- SSE2 ---
impl<'a> Sse2Lexer<'a> {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn skip_ws_sse2(&mut self) {
        use std::arch::x86_64::*;
        while self.0.i < self.0.n {
            let c = self.0.s[self.0.i];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.0.i += 1;
                continue;
            }

            let rem = self.0.n - self.0.i;
            if rem < 16 { break; }

            let p = self.0.s.as_ptr().add(self.0.i) as *const __m128i;
            let chunk = _mm_loadu_si128(p);

            let sp = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b' ' as i8));
            let tb = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\t' as i8));
            let nl = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\n' as i8));
            let cr = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'\r' as i8));

            let ws = _mm_or_si128(_mm_or_si128(sp, tb), _mm_or_si128(nl, cr));
            let mask = _mm_movemask_epi8(ws) as u32;

            if mask == 0xFFFF {
                self.0.i += 16;
                continue;
            } else {
                let not_ws = (!mask) & 0xFFFF;
                let adv = not_ws.trailing_zeros() as usize;
                self.0.i += adv;
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn scan_ident_tail_sse2(&mut self) {
        use std::arch::x86_64::*;
        while self.0.i < self.0.n {
            if !CoreLexer::is_ident_cont(self.0.s[self.0.i]) {
                return;
            }

            let rem = self.0.n - self.0.i;
            if rem < 16 {
                while self.0.i < self.0.n && CoreLexer::is_ident_cont(self.0.s[self.0.i]) {
                    self.0.i += 1;
                }
                return;
            }

            let p = self.0.s.as_ptr().add(self.0.i) as *const __m128i;
            let x = _mm_loadu_si128(p);

            let ge = |lo: u8| _mm_cmpeq_epi8(_mm_max_epu8(x, _mm_set1_epi8(lo as i8)), x);
            let le = |hi: u8| _mm_cmpeq_epi8(_mm_min_epu8(x, _mm_set1_epi8(hi as i8)), x);

            let is_az = _mm_and_si128(ge(b'a'), le(b'z'));
            let is_az_upper = _mm_and_si128(ge(b'A'), le(b'Z'));
            let is_09 = _mm_and_si128(ge(b'0'), le(b'9'));
            let is_us = _mm_cmpeq_epi8(x, _mm_set1_epi8(b'_' as i8));

            let ok = _mm_or_si128(_mm_or_si128(is_az, is_az_upper), _mm_or_si128(is_09, is_us));
            let mask_ok = _mm_movemask_epi8(ok) as u32;

            if (mask_ok & 0xFFFF) == 0xFFFF {
                self.0.i += 16;
            } else {
                let not_ok = (!mask_ok) & 0xFFFF;
                let adv = not_ok.trailing_zeros() as usize;
                self.0.i += adv;
                return;
            }
        }
    }
}

impl<'a> CommonLexer<'a> for Sse2Lexer<'a> {
    #[inline(always)] fn bytes(&self) -> &'a[u8] { self.0.s }
    #[inline(always)] fn pos(&self) -> usize { self.0.i }
    #[inline(always)] fn set_pos(&mut self, v: usize) { self.0.i = v; }

    #[inline(always)]
    fn skip_ws(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe { self.skip_ws_sse2(); }
    }

    #[inline(always)]
    fn scan_ident_tail(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe { self.scan_ident_tail_sse2(); }
    }
}

// --- SCALAR FALLBACK ---
impl<'a> CommonLexer<'a> for ScalarLexer<'a> {
    #[inline(always)] fn bytes(&self) -> &'a[u8] { self.0.s }
    #[inline(always)] fn pos(&self) -> usize { self.0.i }
    #[inline(always)] fn set_pos(&mut self, v: usize) { self.0.i = v; }

    #[inline(always)]
    fn skip_ws(&mut self) {
        while self.0.i < self.0.n {
            match self.0.s[self.0.i] {
                b' ' | b'\t' | b'\n' | b'\r' => self.0.i += 1,
                _ => break,
            }
        }
    }

    #[inline(always)]
    fn scan_ident_tail(&mut self) {
        while self.0.i < self.0.n && CoreLexer::is_ident_cont(self.0.s[self.0.i]) {
            self.0.i += 1;
        }
    }
}