use crate::{
    control_flow::{If, LoopWhile, Then},
    run::{Collect, RunCore},
    Computation, ComputationFn, Function, NamedArgs, Run,
};

impl<A, ArgNames, P, FTrue, FFalse, Collected, Out> RunCore for If<A, ArgNames, P, FTrue, FFalse>
where
    Self: Computation,
    A: Computation + RunCore,
    A::Output: Collect<A::Dim, Collected = Collected>,
    Collected: Clone,
    ArgNames: Clone,
    (ArgNames, Collected): Into<NamedArgs>,
    P: ComputationFn,
    P::Filled: Run<Output = bool>,
    FTrue: ComputationFn,
    FTrue::Filled: RunCore,
    <FTrue::Filled as RunCore>::Output: Collect<FTrue::Dim, Collected = Out>,
    FFalse: ComputationFn,
    FFalse::Filled: RunCore,
    <FFalse::Filled as RunCore>::Output: Collect<FFalse::Dim, Collected = Out>,
{
    type Output = Out;

    fn run_core(self) -> Self::Output {
        let vals = self.child.run_core().collect();
        if Function::anonymous(self.arg_names.clone(), self.predicate).call(vals.clone()) {
            Function::anonymous(self.arg_names.clone(), self.f_true)
                .call_core(vals)
                .collect()
        } else {
            Function::anonymous(self.arg_names.clone(), self.f_false)
                .call_core(vals)
                .collect()
        }
    }
}

impl<A, ArgNames, F, P, Collected> RunCore for LoopWhile<A, ArgNames, F, P>
where
    Self: Computation,
    A: Computation + RunCore,
    A::Output: Collect<A::Dim, Collected = Collected>,
    Collected: Clone,
    ArgNames: Clone,
    (ArgNames, Collected): Into<NamedArgs>,
    F: Clone + ComputationFn,
    F::Filled: RunCore,
    <F::Filled as RunCore>::Output: Collect<F::Dim, Collected = Collected>,
    P: Clone + ComputationFn,
    P::Filled: Run<Output = bool>,
{
    type Output = Collected;

    fn run_core(self) -> Self::Output {
        let mut out = self.child.run_core().collect();
        loop {
            if !Function::anonymous(self.arg_names.clone(), self.predicate.clone())
                .call(out.clone())
            {
                return out;
            }
            out = Function::anonymous(self.arg_names.clone(), self.f.clone())
                .call_core(out)
                .collect();
        }
    }
}

impl<A, ArgNames, F, Collected> RunCore for Then<A, ArgNames, F>
where
    Self: Computation,
    A: Computation + RunCore,
    A::Output: Collect<A::Dim, Collected = Collected>,
    (ArgNames, Collected): Into<NamedArgs>,
    F: ComputationFn,
    F::Filled: RunCore,
{
    type Output = <F::Filled as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        self.f.call_core(self.child.run_core().collect())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{
        arg, arg1, arg2, function::Function, run::Matrix, val, val1, val2, Computation, Run,
    };

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
            .run()
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
            .run()
    }

    #[proptest]
    fn loop_while_should_iterate_a_scalar_until_predicate_is_false(#[strategy(-1000..9)] x: i32) {
        prop_assert_eq!(
            val!(x)
                .loop_while("x", arg!("x", i32) + val!(1), arg!("x", i32).lt(val!(10)))
                .run(),
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
                .run(),
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
                .run(),
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
                .run(),
            (10, 10 - x)
        );
    }

    #[proptest]
    fn then_should_chain_the_first_into_the_second(#[strategy(-1000..1000)] x: i32) {
        prop_assert_eq!(
            (val!(x) + val!(1))
                .then(Function::anonymous("x", arg!("x", i32) + val!(1)))
                .run(),
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
                .then(Function::anonymous(("x", "y"), arg!("x", i32)))
                .run(),
            x
        );
        prop_assert_eq!(
            val!(x)
                .zip(val!(y))
                .then(Function::anonymous(("x", "y"), arg!("y", i32)))
                .run(),
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
                .then(Function::anonymous(
                    ("x", "y"),
                    arg!("y", i32).zip(arg!("x", i32))
                ))
                .run(),
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
                .then(Function::anonymous(("x", ("y", "z")), arg!("x", i32)))
                .run(),
            x
        );
        prop_assert_eq!(
            val!(x)
                .zip(val!(y).zip(val!(z)))
                .then(Function::anonymous(("x", ("y", "z")), arg!("y", i32)))
                .run(),
            y
        );
        prop_assert_eq!(
            val!(x)
                .zip(val!(y).zip(val!(z)))
                .then(Function::anonymous(("x", ("y", "z")), arg!("z", i32)))
                .run(),
            z
        );
    }
}
