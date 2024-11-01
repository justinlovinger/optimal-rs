use core::ops;

use crate::{run::RunCore, sum::Sum, Computation};

impl<A, Item> RunCore for Sum<A>
where
    Self: Computation,
    A: RunCore,
    A::Output: IntoIterator<Item = Item>,
    Item: ops::Add,
    Item::Output: std::iter::Sum<Item>,
{
    type Output = Item::Output;

    fn run_core(self) -> Self::Output {
        self.0.run_core().into_iter().sum()
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
