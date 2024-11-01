use crate::{black_box::BlackBox, run::RunCore, Computation, Run};

impl<A, F, FOut, FDim, FItem> RunCore for BlackBox<A, F, FDim, FItem>
where
    A: Computation + Run,
    F: FnOnce(A::Output) -> FOut,
{
    type Output = FOut;

    fn run_core(self) -> Self::Output {
        (self.f)(self.child.run())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{peano::Zero, val, Computation, Run, Value};

    #[proptest]
    fn black_box_should_run_its_function_and_pass_the_result_to_next(
        #[strategy(-100000..10000)] x: i32,
    ) {
        prop_assert_eq!(
            (val!(x).black_box::<_, Zero, i32>(|x: i32| Value(x + 1)) + val!(1)).run(),
            x + 2
        );
    }

    #[proptest]
    fn black_box_should_support_returning_tuples(#[strategy(-100000..10000)] x: i32) {
        prop_assert_eq!(
            val!(x)
                .black_box::<_, (Zero, Zero), (i32, i32)>(|x: i32| (Value(x + 1), Value(x + 2)))
                .run(),
            (x + 1, x + 2)
        );
    }
}
