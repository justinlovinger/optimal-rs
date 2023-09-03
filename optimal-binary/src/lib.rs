use num_traits::{pow, One, Zero};
use std::ops::{Add, Div, Mul, RangeInclusive, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Reduce innermost axis
/// to numbers within range.
/// Leftmost is least significant.
///
/// # Examples
///
/// ```
/// use optimal_binary::ToRealLE;
///
/// // It returns lower bound for empty arrays:
/// assert_eq!(ToRealLE::new(1.0..=2.0, 0).decode([]), 1.);
///
/// // It returns lower bound when all bits are false:
/// assert_eq!(ToRealLE::new(0.0..=1.0, 1).decode([false]), 0.);
/// assert_eq!(ToRealLE::new(1.0..=2.0, 2).decode([false, false]), 1.);
///
/// // It returns upper bound when all bits are true:
/// assert_eq!(ToRealLE::new(0.0..=1.0, 1).decode([true]), 1.);
/// assert_eq!(ToRealLE::new(1.0..=2.0, 2).decode([true, true]), 2.);
///
/// // It returns a number between lower and upper bound when some bits are true:
/// assert_eq!(ToRealLE::new(1.0..=4.0, 2).decode([true, false]), 2.);
/// assert_eq!(ToRealLE::new(1.0..=4.0, 2).decode([false, true]), 3.);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: One + Add<Output = T> + Deserialize<'de>"))
)]
pub struct ToRealLE<T> {
    #[cfg_attr(feature = "serde", serde(skip))]
    to_int: ToIntLE<T>,
    start: T,
    a: Option<T>,
}

impl<T> ToRealLE<T> {
    pub fn new(range: RangeInclusive<T>, len: usize) -> Self
    where
        T: Copy + One + Add<Output = T> + Sub<Output = T> + Div<Output = T>,
    {
        let to_int = ToIntLE::new();
        let (start, end) = range.into_inner();
        Self {
            a: if len > 0 {
                Some((end - start) / (pow(to_int.two, len) - T::one()))
            } else {
                None
            },
            start,
            to_int,
        }
    }

    pub fn decode(&self, bits: impl IntoIterator<Item = bool>) -> T
    where
        T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
    {
        match self.a {
            Some(a) => a * self.to_int.decode(bits) + self.start,
            None => self.start,
        }
    }
}

/// Reduce to base 10 integer representations of bits.
/// Leftmost is least significant.
///
/// # Examples
///
/// ```
/// use optimal_binary::ToIntLE;
///
/// let to_int_le: ToIntLE<u8> = ToIntLE::new();
///
/// // It returns 0 when empty:
/// assert_eq!(to_int_le.decode([]), 0_u8);
///
/// // It returns the base 10 integer represented by binary bits:
/// assert_eq!(to_int_le.decode([false]), 0_u8);
/// assert_eq!(to_int_le.decode([false, false]), 0_u8);
/// assert_eq!(to_int_le.decode([false, false, false]), 0_u8);
/// assert_eq!(to_int_le.decode([true]), 1_u8);
/// assert_eq!(to_int_le.decode([true, true]), 3_u8);
/// assert_eq!(to_int_le.decode([true, true, true]), 7_u8);
///
/// // It treats leftmost as least significant:
/// assert_eq!(to_int_le.decode([false, true]), 2_u8);
/// assert_eq!(to_int_le.decode([false, false, true]), 4_u8);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct ToIntLE<T> {
    #[cfg_attr(feature = "serde", serde(skip))]
    two: T,
}

impl<T> Default for ToIntLE<T>
where
    T: One + Add<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "serde")]
impl<'de, T> Deserialize<'de> for ToIntLE<T>
where
    T: One + Add<Output = T>,
{
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::new())
    }
}

impl<T> ToIntLE<T> {
    pub fn new() -> Self
    where
        T: One + Add<Output = T>,
    {
        Self {
            two: T::one() + T::one(),
        }
    }

    pub fn decode(&self, bits: impl IntoIterator<Item = bool>) -> T
    where
        T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
    {
        bits.into_iter()
            .fold((T::zero(), T::one()), |(acc, a), b| {
                (if b { acc + a } else { acc }, self.two * a)
            })
            .0
    }
}
