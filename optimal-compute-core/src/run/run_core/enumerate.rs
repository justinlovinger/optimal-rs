use num_traits::AsPrimitive;

use crate::{
    named_args,
    enumerate::Enumerate,
    run::{NamedArgs, RunCore},
    Computation, Run,
};

impl<A, F, Item> RunCore for Enumerate<A, F>
where
    Self: Computation,
    A: Run<Output = Vec<Item>>,
    Item: 'static + Copy + std::fmt::Debug,
    usize: AsPrimitive<Item>,
    F: RunCore,
{
    type Output = F::Output;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        let xs = self.child.run(args);
        let enumerated = 0..xs.len();
        self.f.run_core(named_args![
            ("x", xs),
            (
                "i",
                enumerated
                    .map(|i| i.as_())
                    .collect::<Vec<<A::Output as IntoIterator>::Item>>()
            )
        ])
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg1, named_args, val1, Computation, Run};

    #[proptest]
    fn enumerate_should_provide_indices(xs: Vec<usize>) {
        let (xs_, is) = val1!(xs.clone())
            .enumerate(arg1!("x").zip(arg1!("i")))
            .run(named_args![]);
        prop_assert_eq!(
            (xs_, is),
            xs.into_iter().enumerate().map(|(i, x)| (x, i)).unzip()
        );
    }

    #[proptest]
    fn enumerate_should_call_the_given_function_with_indices(xs: Vec<usize>) {
        prop_assert_eq!(
            val1!(xs.clone())
                .enumerate(arg1!("x", usize) + arg1!("i", usize))
                .run(named_args![]),
            xs.into_iter()
                .enumerate()
                .map(|(i, x)| i + x)
                .collect::<Vec<_>>()
        );
    }
}
