//! Utilities for working with binary optimizers.

use computation_types::{
    arg, arg1,
    cmp::Eq,
    control_flow::{If, Then},
    enumerate::Enumerate,
    math::{Add, Div, Mul, Pow, SameOrZero, Sub},
    peano::{One, Zero},
    sum::Sum,
    val,
    zip::{Zip3, Zip4},
    AnyArg, Arg, Computation, ComputationFn, Function, Val,
};
use num_traits::{pow, AsPrimitive};
use std::ops;

pub use self::{chunks_to_int_le::*, from_bit::*};

pub type ChunksToRealLe<ToMin, ToMax, Bits, T> =
    Scale<Val<Zero, T>, ToMin, ToMax, ChunksToIntLe<Bits, T>>;

/// Return base 10 representations of each chunk of bits,
/// scaled to range `to_min..=to_max`.
/// Leftmost is least significant.
///
/// `to_max` must be >= `to_min`.
/// `chunk_size` must be > `0`.
/// `bits.len() % chunk_size` must be == `0`.
///
/// # Examples
///
/// ```
/// use computation_types::{arg1, val, val1, Function, Run};
/// use optimal_binary::chunks_to_real_le;
///
/// // It returns lower bound when all bits are false:
/// assert_eq!(chunks_to_real_le(1, val!(0.0), val!(1.0), val1!([false, false, false])).run(), [0., 0., 0.]);
/// assert_eq!(chunks_to_real_le(2, val!(1.0), val!(2.0), val1!([false, false, false, false])).run(), [1., 1.]);
///
/// // It returns upper bound when all bits are true:
/// assert_eq!(chunks_to_real_le(1, val!(0.0), val!(1.0), val1!([true, true, true])).run(), [1., 1., 1.]);
/// assert_eq!(chunks_to_real_le(2, val!(1.0), val!(2.0), val1!([true, true, true, true])).run(), [2., 2.]);
///
/// // It returns a number between lower and upper bound when some bits are true:
/// assert_eq!(chunks_to_real_le(2, val!(1.0), val!(4.0), val1!([true, false, false, true])).run(), [2., 3.]);
///
/// // It can construct a computation-function:
/// assert_eq!(Function::anonymous("point", chunks_to_real_le(2, val!(0.0), val!(3.0), arg1!("point", bool))).call(vec![true, false]), [1.]);
/// ```
pub fn chunks_to_real_le<ToMin, ToMax, Bits, T>(
    chunk_size: usize,
    to_min: ToMin,
    to_max: ToMax,
    bits: Bits,
) -> ChunksToRealLe<ToMin, ToMax, Bits, T>
where
    ToMin: Computation<Item = T>,
    ToMax: Computation<Item = T>,
    Bits: Computation<Dim = One, Item = bool>,
    T: 'static
        + Copy
        + AnyArg
        + ops::Add<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + ops::Div<Output = T>
        + num_traits::One
        + num_traits::Pow<T, Output = T>
        + std::iter::Sum,
    u8: AsPrimitive<T>,
    usize: AsPrimitive<T>,
    ToMax::Dim: SameOrZero<ToMin::Dim>,
    <ToMax::Dim as SameOrZero<ToMin::Dim>>::Max: SameOrZero<Zero>,
    <<ToMax::Dim as SameOrZero<ToMin::Dim>>::Max as SameOrZero<Zero>>::Max:
        SameOrZero<One, Max = One>,
    One: SameOrZero<ToMin::Dim, Max = One>,
    Arg<ToMin::Dim, T>: ComputationFn<Dim = ToMin::Dim, Item = T>,
    <Arg<ToMin::Dim, T> as ComputationFn>::Filled: Computation<Dim = ToMin::Dim, Item = T>,
    Arg<ToMax::Dim, T>: ComputationFn<Dim = ToMax::Dim, Item = T>,
    <Arg<ToMax::Dim, T> as ComputationFn>::Filled: Computation<Dim = ToMax::Dim, Item = T>,
{
    scale(
        val!(to_int_max(chunk_size)),
        to_min,
        to_max,
        ChunksToIntLe::new(chunk_size, bits),
    )
}

pub type ToRealLe<ToMin, ToMax, Bits, T> = If<
    Zip3<ToMin, ToMax, Bits>,
    (&'static str, &'static str, &'static str),
    Eq<Val<Zero, usize>, Val<Zero, usize>>,
    Arg<Zero, T>,
    Scale<Val<Zero, T>, Arg<Zero, T>, Arg<Zero, T>, ToIntLe<Arg<One, bool>, T>>,
>;

/// Return base 10 representations of bits scaled to range `to_min..=to_max`.
/// Leftmost is least significant.
///
/// `to_max` must be >= `to_min`.
///
/// # Examples
///
/// ```
/// use computation_types::{arg1, val, val1, Function, Run};
/// use optimal_binary::to_real_le;
///
/// // It returns lower bound for empty arrays:
/// assert_eq!(to_real_le(0, val!(1.0), val!(2.0), val1!([])).run(), 1.);
///
/// // It returns lower bound when all bits are false:
/// assert_eq!(to_real_le(1, val!(0.0), val!(1.0), val1!([false])).run(), 0.);
/// assert_eq!(to_real_le(2, val!(1.0), val!(2.0), val1!([false, false])).run(), 1.);
///
/// // It returns upper bound when all bits are true:
/// assert_eq!(to_real_le(1, val!(0.0), val!(1.0), val1!([true])).run(), 1.);
/// assert_eq!(to_real_le(2, val!(1.0), val!(2.0), val1!([true, true])).run(), 2.);
///
/// // It returns a number between lower and upper bound when some bits are true:
/// assert_eq!(to_real_le(2, val!(1.0), val!(4.0), val1!([true, false])).run(), 2.);
/// assert_eq!(to_real_le(2, val!(1.0), val!(4.0), val1!([false, true])).run(), 3.);
///
/// // It can construct a computation-function:
/// assert_eq!(Function::anonymous("point", to_real_le(2, val!(0.0), val!(3.0), arg1!("point", bool))).call(vec![true, false]), 1.);
/// ```
pub fn to_real_le<ToMin, ToMax, Bits, T>(
    len: usize,
    to_min: ToMin,
    to_max: ToMax,
    bits: Bits,
) -> ToRealLe<ToMin, ToMax, Bits, T>
where
    ToMin: Computation<Dim = Zero, Item = T>,
    ToMax: Computation<Dim = Zero, Item = T>,
    Bits: Computation<Dim = One, Item = bool>,
    T: 'static
        + Copy
        + AnyArg
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
    Zip3(to_min, to_max, bits).if_(
        ("to_min", "to_max", "bits"),
        val!(len).eq(val!(0_usize)),
        arg!("to_min", T),
        scale(
            val!(to_int_max(len)),
            arg!("to_min", T),
            arg!("to_max", T),
            to_int_le::<_, T>(arg1!("bits", bool)),
        ),
    )
}

pub type Scale<FromMax, ToMin, ToMax, Num> = Then<
    Zip4<FromMax, ToMin, ToMax, Num>,
    (&'static str, &'static str, &'static str, &'static str),
    Add<
        Mul<
            Div<
                Sub<
                    Arg<<ToMax as Computation>::Dim, <ToMax as Computation>::Item>,
                    Arg<<ToMin as Computation>::Dim, <ToMin as Computation>::Item>,
                >,
                Arg<<FromMax as Computation>::Dim, <FromMax as Computation>::Item>,
            >,
            Arg<<Num as Computation>::Dim, <Num as Computation>::Item>,
        >,
        Arg<<ToMin as Computation>::Dim, <ToMin as Computation>::Item>,
    >,
>;

/// Scale numbers from `0..=from_max` to `to_min..=to_max`.
///
/// This is meant to be used alongside a function to convert bits to integers.
/// In which case,
/// `from_max` can be obtained using [`to_int_max`].
///
/// `to_max` must be >= `to_min`.
/// Passing `0` for `from_max` is an error.
/// Passing a number > `from_max` will result in an output > `to_max`.
pub fn scale<FromMax, ToMin, ToMax, Num>(
    from_max: FromMax,
    to_min: ToMin,
    to_max: ToMax,
    num: Num,
) -> Scale<FromMax, ToMin, ToMax, Num>
where
    FromMax: Computation<Item = Num::Item>,
    ToMin: Computation<Item = Num::Item>,
    ToMax: Computation<Item = Num::Item>,
    Num: Computation,
    Num::Item: AnyArg
        + ops::Add<Output = Num::Item>
        + ops::Sub<Output = Num::Item>
        + ops::Mul<Output = Num::Item>
        + ops::Div<Output = Num::Item>,
    ToMax::Dim: SameOrZero<ToMin::Dim>,
    <ToMax::Dim as SameOrZero<ToMin::Dim>>::Max: SameOrZero<FromMax::Dim>,
    <<ToMax::Dim as SameOrZero<ToMin::Dim>>::Max as SameOrZero<FromMax::Dim>>::Max:
        SameOrZero<Num::Dim, Max = Num::Dim>,
    Num::Dim: SameOrZero<ToMin::Dim, Max = Num::Dim>,
    Arg<Num::Dim, Num::Item>: ComputationFn<Dim = Num::Dim, Item = Num::Item>,
    <Arg<Num::Dim, Num::Item> as ComputationFn>::Filled:
        Computation<Dim = Num::Dim, Item = Num::Item>,
    Arg<ToMin::Dim, Num::Item>: ComputationFn<Dim = ToMin::Dim, Item = Num::Item>,
    <Arg<ToMin::Dim, Num::Item> as ComputationFn>::Filled:
        Computation<Dim = ToMin::Dim, Item = Num::Item>,
    Arg<FromMax::Dim, Num::Item>: ComputationFn<Dim = FromMax::Dim, Item = Num::Item>,
    <Arg<FromMax::Dim, Num::Item> as ComputationFn>::Filled:
        Computation<Dim = FromMax::Dim, Item = Num::Item>,
    Arg<ToMax::Dim, Num::Item>: ComputationFn<Dim = ToMax::Dim, Item = Num::Item>,
    <Arg<ToMax::Dim, Num::Item> as ComputationFn>::Filled:
        Computation<Dim = ToMax::Dim, Item = Num::Item>,
{
    Zip4(from_max, to_min, to_max, num).then(Function::anonymous(
        ("from_max", "to_min", "to_max", "num"),
        (((Arg::<ToMax::Dim, ToMax::Item>::new("to_max")
            .sub(Arg::<ToMin::Dim, ToMin::Item>::new("to_min")))
        .div(Arg::<FromMax::Dim, FromMax::Item>::new("from_max")))
        .mul(Arg::<Num::Dim, Num::Item>::new("num")))
        .add(Arg::<ToMin::Dim, ToMin::Item>::new("to_min")),
    ))
}

/// Return the largest integer `to_int_...` can return.
pub fn to_int_max<T>(len: usize) -> T
where
    T: 'static + Copy + num_traits::One + ops::Sub<Output = T>,
    usize: AsPrimitive<T>,
{
    pow(2.as_(), len) - T::one()
}

pub type ToIntLe<Bits, T> =
    Sum<Enumerate<FromBit<Bits, T>, Mul<Arg<One, T>, Pow<Val<Zero, T>, Arg<One, T>>>>>;

/// Return base 10 integer representations of bits.
/// Leftmost is least significant.
///
/// # Examples
///
/// ```
/// use computation_types::{arg1, val1, Function, Run};
/// use optimal_binary::to_int_le;
///
/// // It returns 0 when empty:
/// assert_eq!(to_int_le::<_, u8>(val1!([])).run(), 0_u8);
///
/// // It returns the base 10 integer represented by binary bits:
/// assert_eq!(to_int_le::<_, u8>(val1!([false])).run(), 0_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([false, false])).run(), 0_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([false, false, false])).run(), 0_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([true])).run(), 1_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([true, true])).run(), 3_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([true, true, true])).run(), 7_u8);
///
/// // It treats leftmost as least significant:
/// assert_eq!(to_int_le::<_, u8>(val1!([false, true])).run(), 2_u8);
/// assert_eq!(to_int_le::<_, u8>(val1!([false, false, true])).run(), 4_u8);
///
/// // It can construct a computation-function:
/// assert_eq!(Function::anonymous("point", to_int_le::<_, u8>(arg1!("point", bool))).call(vec![false]), 0_u8);
/// ```
pub fn to_int_le<Bits, T>(bits: Bits) -> ToIntLe<Bits, T>
where
    Bits: Computation<Dim = One, Item = bool>,
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
        .enumerate(Function::anonymous(
            ("x", "i"),
            arg1!("x", T) * val!(two).pow(arg1!("i", T)),
        ))
        .sum()
}

mod chunks_to_int_le {
    use core::fmt;
    use std::marker::PhantomData;

    use computation_types::{
        impl_core_ops, peano::One, Computation, ComputationFn, NamedArgs, Names,
    };

    #[derive(Clone, Copy, Debug)]
    pub struct ChunksToIntLe<A, T>
    where
        Self: Computation,
    {
        pub chunk_size: usize,
        pub child: A,
        ty: PhantomData<T>,
    }

    impl<A, T> ChunksToIntLe<A, T>
    where
        Self: Computation,
    {
        pub fn new(chunk_size: usize, child: A) -> Self {
            Self {
                chunk_size,
                child,
                ty: PhantomData,
            }
        }
    }

    impl<A, T> Computation for ChunksToIntLe<A, T>
    where
        A: Computation<Dim = One, Item = bool>,
    {
        type Dim = One;
        type Item = T;
    }

    impl<A, T> ComputationFn for ChunksToIntLe<A, T>
    where
        Self: Computation,
        A: ComputationFn,
        ChunksToIntLe<A::Filled, T>: Computation,
    {
        type Filled = ChunksToIntLe<A::Filled, T>;

        fn fill(self, named_args: NamedArgs) -> Self::Filled {
            ChunksToIntLe {
                chunk_size: self.chunk_size,
                child: self.child.fill(named_args),
                ty: self.ty,
            }
        }

        fn arg_names(&self) -> Names {
            self.child.arg_names()
        }
    }

    impl_core_ops!(ChunksToIntLe<A, T>);

    impl<A, T> fmt::Display for ChunksToIntLe<A, T>
    where
        Self: Computation,
        A: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "chunks_to_int_le({}, {})", self.chunk_size, self.child)
        }
    }

    mod run {
        use std::ops;

        use computation_types::{run::RunCore, val1, Run};
        use itertools::Itertools;
        use num_traits::AsPrimitive;

        use crate::to_int_le;

        use super::*;

        impl<A, T> RunCore for ChunksToIntLe<A, T>
        where
            Self: Computation,
            A: RunCore,
            A::Output: IntoIterator<Item = bool>,
            T: 'static
                + Copy
                + fmt::Debug
                + ops::Add<Output = T>
                + ops::Mul<Output = T>
                + num_traits::Pow<T, Output = T>
                + std::iter::Sum,
            u8: AsPrimitive<T>,
            usize: AsPrimitive<T>,
        {
            type Output = Vec<T>;

            fn run_core(self) -> Self::Output {
                self.child
                    .run_core()
                    .into_iter()
                    .chunks(self.chunk_size)
                    .into_iter()
                    .map(|bits| to_int_le(val1!(bits)).run())
                    .collect()
            }
        }
    }
}

mod from_bit {
    use core::fmt;
    use std::marker::PhantomData;

    use computation_types::{impl_core_ops, Computation, ComputationFn, NamedArgs, Names};
    use num_traits::AsPrimitive;

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
        FromBit<A::Filled, T>: Computation,
    {
        type Filled = FromBit<A::Filled, T>;

        fn fill(self, named_args: NamedArgs) -> Self::Filled {
            FromBit {
                child: self.child.fill(named_args),
                ty: self.ty,
            }
        }

        fn arg_names(&self) -> Names {
            self.child.arg_names()
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
        use computation_types::{
            peano::{One, Two, Zero},
            run::{Matrix, RunCore},
        };

        use super::*;

        impl<A, T, Dim, Out> RunCore for FromBit<A, T>
        where
            Self: Computation<Dim = Dim>,
            A: RunCore<Output = Out>,
            Out: BroadcastFromBit<Dim, T>,
        {
            type Output = Out::Output;

            fn run_core(self) -> Self::Output {
                self.child.run_core().broadcast_from_bit()
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
