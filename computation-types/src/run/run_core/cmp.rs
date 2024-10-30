use paste::paste;

use crate::{
    cmp::*,
    run::{DistributeArgs, NamedArgs, RunCore, Unwrap, Value},
    Computation,
};

macro_rules! impl_run_core_for_comparison {
    ( $op:ident, $where:ident ) => {
        paste! {
            impl<A, B, AOut, BOut> RunCore for $op<A, B>
            where
                Self: Computation,
                (A, B): DistributeArgs<Output = (Value<AOut>, Value<BOut>)>,
                AOut: $where<BOut>
            {
                type Output = Value<bool>;

                fn run_core(self, args: NamedArgs) -> Self::Output {
                    let (x, y) = (self.0, self.1).distribute(args);
                    Value(x.unwrap().[<$op:lower>](&y.unwrap()))
                }
            }
        }
    };
}

impl_run_core_for_comparison!(Eq, PartialEq);
impl_run_core_for_comparison!(Ne, PartialEq);
impl_run_core_for_comparison!(Lt, PartialOrd);
impl_run_core_for_comparison!(Le, PartialOrd);
impl_run_core_for_comparison!(Gt, PartialOrd);
impl_run_core_for_comparison!(Ge, PartialOrd);

impl<A, Out> RunCore for Max<A>
where
    Self: Computation,
    A: RunCore<Output = Value<Out>>,
    Out: IntoIterator,
    Out::Item: PartialOrd,
{
    type Output = Value<Out::Item>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        Value(
            self.0
                .run_core(args)
                .unwrap()
                .into_iter()
                .reduce(|x, y| if x > y { x } else { y })
                .unwrap(),
        )
    }
}

impl<A> RunCore for Not<A>
where
    Self: Computation,
    A: RunCore<Output = Value<bool>>,
{
    type Output = Value<bool>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        Value(!self.0.run_core(args).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{named_args, val, val1, Run};

    use super::*;

    macro_rules! test_comparison {
        ( $op:ident, $inline:tt ) => {
            paste! {
                #[proptest]
                fn [<$op _should_ $op>](#[strategy(-10..10)] x: i32, #[strategy(-10..10)] y: i32) {
                    prop_assert_eq!(val!(x).$op(val!(y)).run(named_args![]), x $inline y);
                }
            }
        };
    }

    test_comparison!(eq, ==);
    test_comparison!(ne, !=);
    test_comparison!(lt, <);
    test_comparison!(le, <=);
    test_comparison!(gt, >);
    test_comparison!(ge, >=);

    #[proptest]
    fn max_should_max(x: i32, y: i32, z: i32) {
        prop_assert_eq!(val1!([x, y, z]).max().run(named_args![]), x.max(y).max(z));
    }

    #[proptest]
    fn max_should_return_a_scalar(x: i32, y: i32, z: i32) {
        prop_assert_eq!(
            (val1!([x, y, z]).max() + val!(1)).run(named_args![]),
            x.max(y).max(z) + 1
        );
    }

    #[proptest]
    fn not_should_not(x: i32, y: i32) {
        let inp = val!(x).lt(val!(y));
        prop_assert_eq!(inp.not().run(named_args![]), x >= y);
    }
}
