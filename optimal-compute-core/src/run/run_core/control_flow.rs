use crate::{
    control_flow::{If, LoopWhile, Then},
    run::{ArgVals, Collect, FromArgsVals, RunCore},
    Computation, Run,
};

impl<A, Args, P, FTrue, FFalse, Collected, Out> RunCore for If<A, Args, P, FTrue, FFalse>
where
    A: Computation + RunCore,
    A::Output: Collect<A::Dim, Collected = Collected>,
    Collected: Clone,
    Args: Clone,
    ArgVals: FromArgsVals<Args, Collected>,
    P: Run<Output = bool>,
    FTrue: Computation + RunCore,
    FTrue::Output: Collect<FTrue::Dim, Collected = Out>,
    FFalse: Computation + RunCore,
    FFalse::Output: Collect<FFalse::Dim, Collected = Out>,
{
    type Output = Out;

    fn run_core(self, args: ArgVals) -> Self::Output {
        let vals = self.child.run_core(args).collect();
        if self
            .predicate
            .run(ArgVals::from_args_vals(self.args.clone(), vals.clone()))
        {
            self.f_true
                .run_core(ArgVals::from_args_vals(self.args, vals))
                .collect()
        } else {
            self.f_false
                .run_core(ArgVals::from_args_vals(self.args, vals))
                .collect()
        }
    }
}

impl<A, Args, F, P, Collected> RunCore for LoopWhile<A, Args, F, P>
where
    A: Computation + RunCore,
    A::Output: Collect<A::Dim, Collected = Collected>,
    Collected: Clone,
    Args: Clone,
    ArgVals: FromArgsVals<Args, Collected>,
    F: Clone + Computation + RunCore,
    F::Output: Collect<F::Dim, Collected = Collected>,
    P: Clone + Run<Output = bool>,
{
    type Output = Collected;

    fn run_core(self, args: ArgVals) -> Self::Output {
        let mut out = self.child.run_core(args).collect();
        loop {
            if !self
                .predicate
                .clone()
                .run(ArgVals::from_args_vals(self.args.clone(), out.clone()))
            {
                return out;
            }
            out = self
                .f
                .clone()
                .run_core(ArgVals::from_args_vals(self.args.clone(), out))
                .collect();
        }
    }
}

impl<A, Args, F, Collected> RunCore for Then<A, Args, F>
where
    A: Computation + RunCore,
    A::Output: Collect<A::Dim, Collected = Collected>,
    ArgVals: FromArgsVals<Args, Collected>,
    F: RunCore,
{
    type Output = F::Output;

    fn run_core(self, args: ArgVals) -> Self::Output {
        self.f.run_core(ArgVals::from_args_vals(
            self.args,
            self.child.run_core(args).collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg, arg1, arg2, argvals, run::Matrix, val, val1, val2, Computation, Run};

    #[test]
    fn if_should_run_f_true_if_predicate_is_true() {
        assert_eq!(run_if_zero(0), 1);
    }

    #[test]
    fn if_should_run_f_false_if_predicate_is_false() {
        assert_eq!(run_if_zero(-1), -2);
    }

    fn run_if_zero(x: i32) -> i32 {
        val!(x)
            .if_(
                "x",
                arg!("x", i32).ge(val!(0)),
                arg!("x", i32) + val!(1),
                arg!("x", i32) - val!(1),
            )
            .run(argvals![])
    }

    #[test]
    fn if_should_run_f_true_if_predicate_is_true_given_a_vector() {
        assert_eq!(run_if_one([-1, 1]), [0, 2]);
    }

    #[test]
    fn if_should_run_f_false_if_predicate_is_false_given_a_vector() {
        assert_eq!(run_if_one([-1, 0]), [-2, -1]);
    }

    fn run_if_one(xs: impl IntoIterator<Item = i32>) -> Vec<i32> {
        val1!(xs.into_iter().collect::<Vec<_>>())
            .if_(
                "x",
                arg1!("x", i32).sum().ge(val!(0)),
                arg1!("x", i32) + val!(1),
                arg1!("x", i32) - val!(1),
            )
            .run(argvals![])
    }

    #[proptest]
    fn loop_while_should_iterate_a_scalar_until_predicate_is_false(#[strategy(-1000..9)] x: i32) {
        prop_assert_eq!(
            val!(x)
                .loop_while("x", arg!("x", i32) + val!(1), arg!("x", i32).lt(val!(10)))
                .run(argvals![]),
            10
        );
    }

    #[proptest]
    fn loop_while_should_iterate_a_vector_until_predicate_is_false(#[strategy(-1000..9)] x: i32) {
        prop_assert_eq!(
            val1!(vec![x, x])
                .loop_while(
                    "x",
                    arg1!("x", i32) + val!(1),
                    arg1!("x", i32).sum().lt(val!(19))
                )
                .run(argvals![]),
            [10, 10]
        );
    }

    #[proptest]
    fn loop_while_should_iterate_a_matrix_until_predicate_is_false(#[strategy(-1000..9)] x: i32) {
        prop_assert_eq!(
            val2!(Matrix::from_vec((2, 2), vec![x, x, x, x]).unwrap())
                .loop_while(
                    "x",
                    arg2!("x", i32) + val!(1),
                    arg2!("x", i32).sum().lt(val!(37))
                )
                .run(argvals![]),
            Matrix::from_vec((2, 2), vec![10, 10, 10, 10]).unwrap()
        );
    }

    #[proptest]
    fn loop_while_should_iterate_a_tuple_until_predicate_is_false(#[strategy(-1000..9)] x: i32) {
        prop_assert_eq!(
            val!(x)
                .zip(val!(0))
                .loop_while(
                    ("x", "y"),
                    (arg!("x", i32) + val!(1)).zip(arg!("y", i32) + val!(1)),
                    arg!("x", i32).lt(val!(10))
                )
                .run(argvals![]),
            (10, 10 - x)
        );
    }

    #[proptest]
    fn then_should_chain_the_first_into_the_second(#[strategy(-1000..1000)] x: i32) {
        prop_assert_eq!(
            (val!(x) + val!(1))
                .then("x", arg!("x", i32) + val!(1))
                .run(argvals![]),
            x + 2
        );
    }

    #[proptest]
    fn then_should_enable_extracting_from_tuples(
        #[strategy(-1000..1000)] x: i32,
        #[strategy(-1000..1000)] y: i32,
    ) {
        prop_assert_eq!(
            val!(x)
                .zip(val!(y))
                .then(("x", "y"), arg!("x", i32))
                .run(argvals![]),
            x
        );
        prop_assert_eq!(
            val!(x)
                .zip(val!(y))
                .then(("x", "y"), arg!("y", i32))
                .run(argvals![]),
            y
        );
    }

    #[proptest]
    fn then_should_enable_reordering_tuples(
        #[strategy(-1000..1000)] x: i32,
        #[strategy(-1000..1000)] y: i32,
    ) {
        prop_assert_eq!(
            val!(x)
                .zip(val!(y))
                .then(("x", "y"), arg!("y", i32).zip(arg!("x", i32)))
                .run(argvals![]),
            (y, x)
        );
    }

    #[proptest]
    fn then_should_enable_extracting_from_nested_tuples(
        #[strategy(-1000..1000)] x: i32,
        #[strategy(-1000..1000)] y: i32,
        #[strategy(-1000..1000)] z: i32,
    ) {
        prop_assert_eq!(
            val!(x)
                .zip(val!(y).zip(val!(z)))
                .then(("x", ("y", "z")), arg!("x", i32))
                .run(argvals![]),
            x
        );
        prop_assert_eq!(
            val!(x)
                .zip(val!(y).zip(val!(z)))
                .then(("x", ("y", "z")), arg!("y", i32))
                .run(argvals![]),
            y
        );
        prop_assert_eq!(
            val!(x)
                .zip(val!(y).zip(val!(z)))
                .then(("x", ("y", "z")), arg!("z", i32))
                .run(argvals![]),
            z
        );
    }
}
