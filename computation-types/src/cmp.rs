use core::fmt;

use paste::paste;

use crate::{
    impl_computation_fn_for_binary, impl_computation_fn_for_unary, impl_core_ops,
    impl_display_for_inline_binary,
    peano::{Suc, Zero},
    Computation, ComputationFn, NamedArgs,
};

macro_rules! impl_cmp_op {
    ( $op:ident, $where:ident ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct $op<A, B>(pub A, pub B)
            where
                Self: Computation;

            impl<A, B> Computation for $op<A, B>
            where
                A: Computation<Dim = Zero>,
                B: Computation<Dim = Zero>,
                A::Item: $where<B::Item>
            {
                type Dim = Zero;
                type Item = bool;
            }

            impl_computation_fn_for_binary!($op);

            impl_core_ops!($op<A, B>);
        }
    };
}

impl_cmp_op!(Eq, PartialEq);
impl_cmp_op!(Ne, PartialEq);
impl_cmp_op!(Lt, PartialOrd);
impl_cmp_op!(Le, PartialOrd);
impl_cmp_op!(Gt, PartialOrd);
impl_cmp_op!(Ge, PartialOrd);

impl_display_for_inline_binary!(Eq, "==");
impl_display_for_inline_binary!(Ne, "!=");
impl_display_for_inline_binary!(Lt, "<");
impl_display_for_inline_binary!(Le, "<=");
impl_display_for_inline_binary!(Gt, ">");
impl_display_for_inline_binary!(Ge, ">=");

#[derive(Clone, Copy, Debug)]
pub struct Max<A>(pub A)
where
    Self: Computation;

impl<A, D> Computation for Max<A>
where
    A: Computation<Dim = Suc<D>>,
    A::Item: PartialOrd,
{
    type Dim = Zero;
    type Item = A::Item;
}

impl_computation_fn_for_unary!(Max);

impl_core_ops!(Max<A>);

impl<A> fmt::Display for Max<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.max()", self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Not<A>(pub A)
where
    Self: Computation;

impl<A> Computation for Not<A>
where
    A: Computation<Dim = Zero, Item = bool>,
{
    type Dim = Zero;
    type Item = bool;
}

impl_computation_fn_for_unary!(Not);

impl_core_ops!(Not<A>);

impl<A> fmt::Display for Not<A>
where
    Self: Computation,
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
