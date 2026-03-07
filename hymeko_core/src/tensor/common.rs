use std::fmt::{Debug, Display};

pub trait Real:
Copy + PartialOrd +
core::ops::Add<Output=Self> +
core::ops::AddAssign +
core::ops::Sub<Output=Self> +
core::ops::SubAssign +
core::ops::Mul<Output=Self> +
core::ops::MulAssign +
core::ops::Div<Output=Self> +
Send + Sync +
Debug + Display + AsF64 + AsF32
{
    fn zero() -> Self;
    fn one() -> Self;
    fn neg_one() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn round(self) -> Self { self }
    /// Convert from any other real type, if needed (e.g. for mixed-precision).
    fn from_other<T: Real>(rhs: T) -> Self;

}

impl Real for f32 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn neg_one() -> Self { -1.0 }
    #[inline] fn abs(self) -> Self { self.abs() }
    #[inline] fn sqrt(self) -> Self { self.sqrt() }
    #[inline] fn max(self, rhs: Self) -> Self { self.max(rhs) }
    #[inline] fn min(self, rhs: Self) -> Self { self.min(rhs) }
    #[inline] fn round(self) -> Self { self.round() }

    #[inline] fn from_other<T: Real>(rhs: T) -> Self { rhs.as_f32() as f32 }

}

impl Real for f64 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn neg_one() -> Self { -1.0 }
    #[inline] fn abs(self) -> Self { self.abs() }
    #[inline] fn sqrt(self) -> Self { self.sqrt() }
    #[inline] fn max(self, rhs: Self) -> Self { self.max(rhs) }
    #[inline] fn min(self, rhs: Self) -> Self { self.min(rhs) }
    #[inline] fn round(self) -> Self { self.round() }
    #[inline] fn from_other<T: Real>(rhs: T) -> Self { rhs.as_f64() }
}

pub trait AsF64 { fn as_f64(self) -> f64; }
impl AsF64 for f32 { fn as_f64(self) -> f64 { self as f64 } }
impl AsF64 for f64 { fn as_f64(self) -> f64 { self } }

pub trait AsF32 { fn as_f32(self) -> f32; }
impl AsF32 for f32 { fn as_f32(self) -> f32 { self } }
impl AsF32 for f64 { fn as_f32(self) -> f32 { self as f32} }

#[inline(always)]
pub fn signed_incidence<F: Real>(sign: i8) -> F {
    match sign {
        1 => F::one(),
        -1 => F::neg_one(),
        _ => F::one(), // neutral: a "szimmetrikus/abs" nézetekben ez oké
    }
}



