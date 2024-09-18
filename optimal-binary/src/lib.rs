//! Utilities for working with binary optimizers.

use num_traits::{pow, AsPrimitive};
use optimal_compute_core::{
    arg, arg1,
    cmp::Eq,
    control_flow::If,
    enumerate::Enumerate,
    math::{Add, Div, Mul, Pow, Sub},
    peano::{self, One, Suc, Zero},
    sum::Sum,
    val,
    zip::Zip3,
    Arg, Computation, Val,
};
use std::ops;

pub use self::from_bit::*;

type ToRealLe<Start, End, Bits, T> = If<
    Zip3<Start, End, Bits>,
    (&'static str, &'static str, &'static str),
    Eq<Val<Zero, usize>, Val<Zero, usize>>,
    Arg<Zero, T>,
    Add<
        Mul<Div<Sub<Arg<Zero, T>, Arg<Zero, T>>, Val<Zero, T>>, ToIntLe<Arg<One, bool>, T>>,
        Arg<Zero, T>,
    >,
>;

/// Reduce innermost axis
/// to numbers within range.
/// Leftmost is least significant.
/// `end` must be >= `start`.
/// `start` and `end` are inclusive.
///
/// # Examples
///
/// ```
/// use optimal_compute_core::{argvals, val, val1, Run};
/// use optimal_binary::to_real_le;
///
/// // It returns lower bound for empty arrays:
/// assert_eq!(to_real_le(0, val!(1.0), val!(2.0), val1!([])).run(argvals![]), 1.);
///
/// // It returns lower bound when all bits are false:
/// assert_eq!(to_real_le(1, val!(0.0), val!(1.0), val1!([false])).run(argvals![]), 0.);
/// assert_eq!(to_real_le(2, val!(1.0), val!(2.0), val1!([false, false])).run(argvals![]), 1.);
///
/// // It returns upper bound when all bits are true:
/// assert_eq!(to_real_le(1, val!(0.0), val!(1.0), val1!([true])).run(argvals![]), 1.);
/// assert_eq!(to_real_le(2, val!(1.0), val!(2.0), val1!([true, true])).run(argvals![]), 2.);
///
/// // It returns a number between lower and upper bound when some bits are true:
/// assert_eq!(to_real_le(2, val!(1.0), val!(4.0), val1!([true, false])).run(argvals![]), 2.);
/// assert_eq!(to_real_le(2, val!(1.0), val!(4.0), val1!([false, true])).run(argvals![]), 3.);
/// ```
pub fn to_real_le<Start, End, Bits, T>(
    len: usize,
    start: Start,
    end: End,
    bits: Bits,
) -> ToRealLe<Start, End, Bits, T>
where
    Start: Computation<Dim = Zero, Item = T>,
    End: Computation<Dim = Zero, Item = T>,
    Bits: Computation<Dim = peano::One, Item = bool>,
    T: 'static
        + Copy
        + ops::Add<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + ops::Div<Output = T>
        + num_traits::One
        + num_traits::Pow<T, Output = T>
        + std::iter::Sum,
    u8: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let two = 2_usize.as_();
    let a = pow(two, len) - T::one();
    Zip3(start, end, bits).if_(
        ("start", "end", "bits"),
        val!(len).eq(val!(0_usize)),
        arg!("start", T),
        ((arg!("end", T) - arg!("start", T)) / val!(a)) * to_int_le::<_, T>(arg1!("bits", bool))
            + arg!("start", T),
    )
    // The following would be possible
    // if we could raise to the power of a `usize`:
    // ```
    // let two = T::one() + T::one();
    // Zip3(start, end, bits).if_(
    //     ("start", "end", "bits"),
    //     arg1!("bits", bool).len().eq(val!(0_usize)),
    //     arg!("start", T),
    //     ((arg!("end", T) - arg!("start", T))
    //         / (val!(two).pow(arg1!("bits", bool).len()) - val!(T::one())))
    //         * to_int_le::<_, T>(arg1!("bits", bool))
    //         + arg!("start", T),
    // )
    // ```
}

type ToIntLe<Bits, T> =
    Sum<Enumerate<FromBit<Bits, T>, Mul<Arg<Suc<Zero>, T>, Pow<Val<Zero, T>, Arg<Suc<Zero>, T>>>>>;

/// Reduce to base 10 integer representations of bits.
/// Leftmost is least significant.
///
/// # Examples
///
/// ```
/// use optimal_compute_core::{argvals, val1, Run};
/// use optimal_binary::to_int_le;
///
/// // It returns 0 when empty:
/// assert_eq!(to_int_le::<_, u8>(val1!([])).run(argvals![]), 0_u8);
///
/// // It returns the base 10 integer represented by binary bits:
/// assert_eq!(to_int_le::<_, u8>(val1!([false])).run(argvals![]), 0_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([false, false])).run(argvals![]), 0_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([false, false, false])).run(argvals![]), 0_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([true])).run(argvals![]), 1_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([true, true])).run(argvals![]), 3_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([true, true, true])).run(argvals![]), 7_u8);
///
/// // It treats leftmost as least significant:
/// assert_eq!(to_int_le::<_, u8>(val1!([false, true])).run(argvals![]), 2_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([false, false, true])).run(argvals![]), 4_u8);
/// ```
pub fn to_int_le<Bits, T>(bits: Bits) -> ToIntLe<Bits, T>
where
    Bits: Computation<Dim = Suc<Zero>, Item = bool>,
    T: 'static
        + Copy
        + ops::Add<Output = T>
        + ops::Mul<Output = T>
        + num_traits::Pow<T, Output = T>
        + std::iter::Sum,
    u8: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let two = 2_usize.as_();
    FromBit::<_, T>::new(bits)
        .enumerate(arg1!("x", T) * val!(two).pow(arg1!("i", T)))
        .sum()
}

mod from_bit {
    use core::fmt;
    use std::marker::PhantomData;

    use num_traits::AsPrimitive;
    use optimal_compute_core::{impl_core_ops, Args, Computation, ComputationFn};

    #[derive(Clone, Copy, Debug)]
    pub struct FromBit<A, T>
    where
        Self: Computation,
    {
        pub child: A,
        ty: PhantomData<T>,
    }

    impl<A, T> FromBit<A, T>
    where
        Self: Computation,
    {
        pub fn new(child: A) -> Self {
            Self {
                child,
                ty: PhantomData,
            }
        }
    }

    impl<A, T> Computation for FromBit<A, T>
    where
        A: Computation<Item = bool>,
        T: 'static + Copy,
        u8: AsPrimitive<T>,
    {
        type Dim = A::Dim;
        type Item = T;
    }

    impl<A, T> ComputationFn for FromBit<A, T>
    where
        Self: Computation,
        A: ComputationFn,
    {
        fn args(&self) -> Args {
            self.child.args()
        }
    }

    impl_core_ops!(FromBit<A, T>);

    impl<A, T> fmt::Display for FromBit<A, T>
    where
        Self: Computation,
        A: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "from_bit({})", self.child)
        }
    }

    mod run {
        use optimal_compute_core::{
            peano::{One, Two, Zero},
            run::{ArgVals, Matrix, RunCore, Unwrap, Value},
        };

        use super::*;

        impl<A, T, Dim, AOut> RunCore for FromBit<A, T>
        where
            Self: Computation<Dim = Dim>,
            A: RunCore<Output = Value<AOut>>,
            AOut: BroadcastFromBit<Dim, T>,
        {
            type Output = Value<AOut::Output>;

            fn run_core(self, args: ArgVals) -> Self::Output {
                Value(self.child.run_core(args).unwrap().broadcast_from_bit())
            }
        }

        pub trait BroadcastFromBit<Dim, T> {
            type Output;

            fn broadcast_from_bit(self) -> Self::Output;
        }

        impl<T> BroadcastFromBit<Zero, T> for bool
        where
            T: 'static + Copy,
            u8: AsPrimitive<T>,
        {
            type Output = T;

            fn broadcast_from_bit(self) -> Self::Output {
                (self as u8).as_()
            }
        }

        impl<Lhs, T> BroadcastFromBit<One, T> for Lhs
        where
            Lhs: IntoIterator<Item = bool>,
            T: 'static + Copy,
            u8: AsPrimitive<T>,
        {
            type Output = std::iter::Map<Lhs::IntoIter, fn(bool) -> T>;

            fn broadcast_from_bit(self) -> Self::Output {
                self.into_iter().map(|b| (b as u8).as_())
            }
        }

        impl<Lhs, T> BroadcastFromBit<Two, T> for Matrix<Lhs>
        where
            Lhs: IntoIterator<Item = bool>,
            T: 'static + Copy,
            u8: AsPrimitive<T>,
        {
            type Output = Matrix<std::iter::Map<Lhs::IntoIter, fn(bool) -> T>>;

            fn broadcast_from_bit(self) -> Self::Output {
                // Neither shape nor the length of `inner` will change,
                // so they should still be fine.
                unsafe {
                    Matrix::new_unchecked(
                        self.shape(),
                        self.into_inner().into_iter().map(|b| (b as u8).as_()),
                    )
                }
            }
        }
    }
}
