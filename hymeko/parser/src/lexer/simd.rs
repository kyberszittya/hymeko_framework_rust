// src/lexer/simd.rs
use super::common::{self, CommonLexer, LexItem};

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
    fn is_ident_cont(c: u8) -> bool {
        (c >= b'a' && c <= b'z')
            || (c >= b'A' && c <= b'Z')
            || (c >= b'0' && c <= b'9')
            || c == b'_'
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




    #[inline(always)]
    fn scan_ident_tail_impl(&mut self) {
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
        while self.i < self.n && Self::is_ident_cont(self.s[self.i]) {
            self.i += 1;
        }
    }

    #[inline(always)]
    fn skip_ws_impl(&mut self) {
        // a te skip_ws() tartalmad (AVX2/SSE2 + fallback) mehet ide
        // (csak a név változik)
        #[cfg(all(target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe { self.skip_ws_avx2(); }
                return;
            } else {
                unsafe { self.skip_ws_sse2(); }
                return;
            }
        }
        while self.i < self.n {
            match self.s[self.i] {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }

}

impl<'a> CommonLexer for Lexer<'a> {
    #[inline(always)] fn bytes(&self) -> &[u8] { self.s }
    #[inline(always)] fn pos(&self) -> usize { self.i }
    #[inline(always)] fn set_pos(&mut self, v: usize) { self.i = v; }

    #[inline(always)] fn skip_ws(&mut self) { self.skip_ws_impl(); }
    #[inline(always)] fn scan_ident_tail(&mut self) { self.scan_ident_tail_impl(); }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = LexItem;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        common::next_token(self)
    }
}