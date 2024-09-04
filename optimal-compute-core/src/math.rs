use core::{fmt, ops};

use num_traits::Signed;
use paste::paste;

use crate::{impl_core_ops, impl_display_for_inline_binary, Computation, ComputationFn};

pub use self::same_or_zero::*;

mod same_or_zero {
    use crate::peano::{Suc, Zero};

    pub trait SameOrZero<B> {
        type Max;
    }

    impl SameOrZero<Zero> for Zero {
        type Max = Zero;
    }

    impl<A> SameOrZero<Suc<A>> for Zero {
        type Max = Suc<A>;
    }

    impl<A> SameOrZero<Zero> for Suc<A> {
        type Max = Suc<A>;
    }

    impl<A> SameOrZero<Suc<A>> for Suc<A> {
        type Max = Suc<A>;
    }
}

macro_rules! impl_binary_op {
    ( $op:ident ) => {
        impl_binary_op!($op, ops);
    };
    ( $op:ident, $package:ident ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct $op<A, B>(pub(crate) A, pub(crate) B);

            impl<A, B> Computation for $op<A, B>
            where
                A: Computation,
                B: Computation,
                A::Dim: SameOrZero<B::Dim>,
                A::Item: $package::$op<B::Item>,
            {
                type Dim = <A::Dim as SameOrZero<B::Dim>>::Max;
                type Item = <A::Item as $package::$op<B::Item>>::Output;
            }

            impl<A, B> ComputationFn for $op<A, B>
            where
                Self: Computation,
                A: ComputationFn,
                B: ComputationFn,
            {
                fn args(&self) -> crate::Args {
                    self.0.args().union(self.1.args())
                }
            }

            impl_core_ops!($op<A, B>);
        }
    };
}

macro_rules! impl_unary_op {
    ( $op:ident ) => {
        impl_unary_op!($op, ops);
    };
    ( $op:ident, $package:ident ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct $op<A>(pub(crate) A);

            impl<A> Computation for $op<A>
            where
                A: Computation,
                A::Item: $package::$op,
            {
                type Dim = A::Dim;
                type Item = <A::Item as $package::$op>::Output;
            }

            impl<A> ComputationFn for $op<A>
            where
                Self: Computation,
                A: ComputationFn,
            {
                fn args(&self) -> crate::Args {
                    self.0.args()
                }
            }

            impl_core_ops!($op<A>);
        }
    };
}

impl_binary_op!(Add);
impl_binary_op!(Sub);
impl_binary_op!(Mul);
impl_binary_op!(Div);
impl_binary_op!(Pow, num_traits);
impl_unary_op!(Neg);

impl_display_for_inline_binary!(Add, "+");
impl_display_for_inline_binary!(Sub, "-");
impl_display_for_inline_binary!(Mul, "*");
impl_display_for_inline_binary!(Div, "/");
impl_display_for_inline_binary!(Pow, "^");

impl<A> fmt::Display for Neg<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "-{}", self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Abs<A>(pub(crate) A);

impl<A> Computation for Abs<A>
where
    A: Computation,
    A::Item: Signed,
{
    type Dim = A::Dim;
    type Item = A::Item;
}

impl<A> ComputationFn for Abs<A>
where
    Self: Computation,
    A: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.0.args()
    }
}

impl_core_ops!(Abs<A>);

impl<A> fmt::Display for Abs<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.abs()", self.0)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{val, Computation};

    macro_rules! assert_op_display {
        ( $x:ident $op:tt $y:ident ) => {
            prop_assert_eq!((val!($x) $op val!($y)).to_string(), format!("({} {} {})", val!($x), stringify!($op), val!($y)));
        };
        ( $x:ident . $op:ident ( $y:ident ) ) => {
            prop_assert_eq!(val!($x).$op(val!($y)).to_string(), format!("{}.{}({})", val!($x), stringify!($op), val!($y)));
        };
    }

    #[proptest]
    fn add_should_display(x: i32, y: i32) {
        assert_op_display!(x + y);
    }

    #[proptest]
    fn sub_should_display(x: i32, y: i32) {
        assert_op_display!(x - y);
    }

    #[proptest]
    fn mul_should_display(x: i32, y: i32) {
        assert_op_display!(x * y);
    }

    #[proptest]
    fn div_should_display(x: i32, y: i32) {
        assert_op_display!(x / y);
    }

    #[proptest]
    fn pow_should_display(x: i32, y: u32) {
        prop_assert_eq!(
            val!(x).pow(val!(y)).to_string(),
            format!("({} ^ {})", val!(x), val!(y))
        );
    }

    #[proptest]
    fn neg_should_display(x: i32) {
        prop_assert_eq!((-val!(x)).to_string(), format!("-{}", val!(x)));
    }

    #[proptest]
    fn abs_should_display(x: i32) {
        prop_assert_eq!(val!(x).abs().to_string(), format!("{}.abs()", val!(x)));
    }
}
