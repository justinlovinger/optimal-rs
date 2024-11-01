use num_traits::AsPrimitive;

use crate::{enumerate::Enumerate, run::RunCore, Computation, ComputationFn, Run};

impl<A, F, Item> RunCore for Enumerate<A, F>
where
    Self: Computation,
    A: Run<Output = Vec<Item>>,
    Item: 'static + Copy + std::fmt::Debug,
    usize: AsPrimitive<Item>,
    F: ComputationFn,
    F::Filled: RunCore,
{
    type Output = <F::Filled as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        let xs = self.child.run();
        let enumerated = 0..xs.len();
        self.f.call_core((
            xs,
            enumerated
                .map(|i| i.as_())
                .collect::<Vec<<A::Output as IntoIterator>::Item>>(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg1, val1, Computation, Function, Run};

    #[proptest]
    fn enumerate_should_provide_indices(xs: Vec<usize>) {
        let (xs_, is) = val1!(xs.clone())
            .enumerate(Function::anonymous(("x", "i"), arg1!("x").zip(arg1!("i"))))
            .run();
        prop_assert_eq!(
            (xs_, is),
            xs.into_iter().enumerate().map(|(i, x)| (x, i)).unzip()
        );
    }

    #[proptest]
    fn enumerate_should_call_the_given_function_with_indices(xs: Vec<usize>) {
        prop_assert_eq!(
            val1!(xs.clone())
                .enumerate(Function::anonymous(
                    ("x", "i"),
                    arg1!("x", usize) + arg1!("i", usize)
                ))
                .run(),
            xs.into_iter()
                .enumerate()
                .map(|(i, x)| i + x)
                .collect::<Vec<_>>()
        );
    }
}
