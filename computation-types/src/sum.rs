use core::{fmt, ops};

use crate::{
    impl_computation_fn_for_unary, impl_core_ops,
    peano::{Suc, Zero},
    Computation, ComputationFn, NamedArgs,
};

#[derive(Clone, Copy, Debug)]
pub struct Sum<A>(pub A)
where
    Self: Computation;

impl<A, D> Computation for Sum<A>
where
    A: Computation<Dim = Suc<D>>,
    A::Item: ops::Add,
{
    type Dim = Zero;
    type Item = <A::Item as ops::Add>::Output;
}

impl_computation_fn_for_unary!(Sum);

impl_core_ops!(Sum<A>);

impl<A> fmt::Display for Sum<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.sum()", self.0)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{val1, Computation};

    #[proptest]
    fn sum_should_display(xs: Vec<i32>) {
        prop_assert_eq!(
            val1!(xs.clone()).sum().to_string(),
            format!("{}.sum()", val1!(xs))
        );
    }
}
