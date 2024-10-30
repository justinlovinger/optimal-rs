use core::{fmt, ops};

use paste::paste;

use crate::{impl_core_ops, impl_display_for_inline_binary, Computation, ComputationFn};

pub use self::{same_or_zero::*, trig::*};

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
        impl_binary_op!($op, $package, $op);
    };
    ( $op:ident, $package:ident, $bound:ident ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct $op<A, B>(pub A, pub B)
            where
                Self: Computation;

            impl<A, B, ADim, AItem> Computation for $op<A, B>
            where
                A: Computation<Dim = ADim, Item = AItem>,
                B: Computation,
                ADim: SameOrZero<B::Dim>,
                AItem: $package::$bound<B::Item>,
            {
                type Dim = ADim::Max;
                type Item = AItem::Output;
            }

            impl<A, B> ComputationFn for $op<A, B>
            where
                Self: Computation,
                A: ComputationFn,
                B: ComputationFn,
            {
                fn arg_names(&self) -> crate::Names {
                    self.0.arg_names().union(self.1.arg_names())
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
        impl_unary_op!($op, $package, $op);
    };
    ( $op:ident, $package:ident, $bound:ident ) => {
        impl_unary_op!($op, $package, $bound, Item::Output);
    };
    ( $op:ident, $package:ident, $bound:ident, Item $( :: $Output:ident )? ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct $op<A>(pub A)
            where
                Self: Computation;


            impl<A, Item> Computation for $op<A>
            where
                A: Computation<Item = Item>,
                Item: $package::$bound,
            {
                type Dim = A::Dim;
                type Item = Item $( ::$Output )?;
            }

            impl<A> ComputationFn for $op<A>
            where
                Self: Computation,
                A: ComputationFn,
            {
                fn arg_names(&self) -> crate::Names {
                    self.0.arg_names()
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
impl_unary_op!(Abs, num_traits, Signed, Item);

impl_display_for_inline_binary!(Add, "+");
impl_display_for_inline_binary!(Sub, "-");
impl_display_for_inline_binary!(Mul, "*");
impl_display_for_inline_binary!(Div, "/");
impl_display_for_inline_binary!(Pow, "^");

impl<A> fmt::Display for Neg<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "-{}", self.0)
    }
}

impl<A> fmt::Display for Abs<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.abs()", self.0)
    }
}

mod trig {
    use num_traits::real;

    use super::*;

    impl_unary_op!(Sin, real, Real, Item);
    impl_unary_op!(Cos, real, Real, Item);
    impl_unary_op!(Tan, real, Real, Item);
    impl_unary_op!(Asin, real, Real, Item);
    impl_unary_op!(Acos, real, Real, Item);
    impl_unary_op!(Atan, real, Real, Item);

    macro_rules! impl_display {
        ( $op:ident ) => {
            paste::paste! {
                impl<A> fmt::Display for $op<A>
                where
                    Self: Computation,
                    A: fmt::Display,
                {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, "{}.{}()", self.0, stringify!([<$op:lower>]))
                    }
                }
            }
        };
    }

    impl_display!(Sin);
    impl_display!(Cos);
    impl_display!(Tan);
    impl_display!(Asin);
    impl_display!(Acos);
    impl_display!(Atan);
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

    mod trig {
        use super::*;

        #[proptest]
        fn sin_should_display(x: f32) {
            prop_assert_eq!(val!(x).sin().to_string(), format!("{}.sin()", val!(x)));
        }

        #[proptest]
        fn cos_should_display(x: f32) {
            prop_assert_eq!(val!(x).cos().to_string(), format!("{}.cos()", val!(x)));
        }

        #[proptest]
        fn tan_should_display(x: f32) {
            prop_assert_eq!(val!(x).tan().to_string(), format!("{}.tan()", val!(x)));
        }

        #[proptest]
        fn asin_should_display(x: f32) {
            prop_assert_eq!(val!(x).asin().to_string(), format!("{}.asin()", val!(x)));
        }

        #[proptest]
        fn acos_should_display(x: f32) {
            prop_assert_eq!(val!(x).acos().to_string(), format!("{}.acos()", val!(x)));
        }

        #[proptest]
        fn atan_should_display(x: f32) {
            prop_assert_eq!(val!(x).atan().to_string(), format!("{}.atan()", val!(x)));
        }
    }
}
