use core::ops;

use crate::{run::RunCore, sum::Sum, Computation, Unwrap, Value};

impl<A, Out> RunCore for Sum<A>
where
    Self: Computation,
    A: RunCore<Output = Value<Out>>,
    Out: IntoIterator,
    Out::Item: ops::Add,
    <Out::Item as ops::Add>::Output: std::iter::Sum<Out::Item>,
{
    type Output = Value<<Out::Item as ops::Add>::Output>;

    fn run_core(self) -> Self::Output {
        Value(self.0.run_core().unwrap().into_iter().sum())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{run::Matrix, val, val1, val2, Computation, Run};

    #[proptest]
    fn sum_should_sum_vectors(
        #[strategy(-100000..100000)] x: i32,
        #[strategy(-100000..100000)] y: i32,
        #[strategy(-100000..100000)] z: i32,
    ) {
        prop_assert_eq!(val1!([x, y, z]).sum().run(), x + y + z);
    }

    #[proptest]
    fn sum_should_sum_matrices(
        #[strategy(-100000..100000)] x: i32,
        #[strategy(-100000..100000)] y: i32,
        #[strategy(-100000..100000)] z: i32,
        #[strategy(-100000..100000)] q: i32,
    ) {
        prop_assert_eq!(
            val2!(Matrix::from_vec((2, 2), vec![x, y, z, q]).unwrap())
                .sum()
                .run(),
            x + y + z + q
        );
    }

    #[proptest]
    fn sum_should_return_a_scalar(
        #[strategy(-100000..100000)] x: i32,
        #[strategy(-100000..100000)] y: i32,
        #[strategy(-100000..100000)] z: i32,
    ) {
        prop_assert_eq!((val1!([x, y, z]).sum() + val!(1)).run(), x + y + z + 1);
    }
}
