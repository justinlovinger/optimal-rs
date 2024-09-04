use core::fmt;

use paste::paste;

use crate::{
    impl_core_ops, impl_display_for_inline_binary,
    peano::{Suc, Zero},
    Args, Computation, ComputationFn,
};

macro_rules! impl_cmp_op {
    ( $op:ident ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct $op<A, B>(pub(crate) A, pub(crate) B);

            impl<A, B> Computation for $op<A, B>
            where
                A: Computation<Dim = Zero>,
                B: Computation<Dim = Zero>,
            {
                type Dim = Zero;
                type Item = bool;
            }

            impl<A, B> ComputationFn for $op<A, B>
            where
                Self: Computation,
                A: ComputationFn,
                B: ComputationFn,
            {
                fn args(&self) -> Args {
                    self.0.args().union(self.1.args())
                }
            }

            impl_core_ops!($op<A, B>);
        }
    };
}

impl_cmp_op!(Eq);
impl_cmp_op!(Ne);
impl_cmp_op!(Lt);
impl_cmp_op!(Le);
impl_cmp_op!(Gt);
impl_cmp_op!(Ge);

impl_display_for_inline_binary!(Eq, "==");
impl_display_for_inline_binary!(Ne, "!=");
impl_display_for_inline_binary!(Lt, "<");
impl_display_for_inline_binary!(Le, "<=");
impl_display_for_inline_binary!(Gt, ">");
impl_display_for_inline_binary!(Ge, ">=");

#[derive(Clone, Copy, Debug)]
pub struct Max<A>(pub(crate) A);

impl<A, D> Computation for Max<A>
where
    A: Computation<Dim = Suc<D>>,
    A::Item: PartialOrd,
{
    type Dim = Zero;
    type Item = A::Item;
}

impl<A> ComputationFn for Max<A>
where
    Self: Computation,
    A: ComputationFn,
{
    fn args(&self) -> Args {
        self.0.args()
    }
}

impl_core_ops!(Max<A>);

impl<A> fmt::Display for Max<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.max()", self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Not<A>(pub(crate) A);

impl<A> Computation for Not<A>
where
    A: Computation<Dim = Zero, Item = bool>,
{
    type Dim = Zero;
    type Item = bool;
}

impl<A> ComputationFn for Not<A>
where
    Self: Computation,
    A: ComputationFn,
{
    fn args(&self) -> Args {
        self.0.args()
    }
}

impl_core_ops!(Not<A>);

impl<A> fmt::Display for Not<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "!{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{val, val1};

    use super::*;

    macro_rules! test_display {
        ( $op:ident, $inline:tt ) => {
            paste! {
                #[proptest]
                fn [<$op _should_ display>](x: i32, y: i32) {
                    prop_assert_eq!(val!(x).$op(val!(y)).to_string(), format!("({} {} {})", x, stringify!($inline), y));
                }
            }
        };
    }

    test_display!(eq, ==);
    test_display!(ne, !=);
    test_display!(lt, <);
    test_display!(le, <=);
    test_display!(gt, >);
    test_display!(ge, >=);

    #[proptest]
    fn max_should_display(x: i32, y: i32, z: i32) {
        prop_assert_eq!(
            val1!([x, y, z]).max().to_string(),
            format!("{}.max()", val1!([x, y, z]))
        );
    }

    #[proptest]
    fn not_should_display(x: i32, y: i32) {
        let inp = val!(x).lt(val!(y));
        prop_assert_eq!(inp.not().to_string(), format!("!{}", inp));
    }
}
