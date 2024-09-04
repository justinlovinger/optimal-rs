use paste::paste;

use crate::ComputationFn;

use super::{ArgVals, RunCore};

pub trait DistributeArgs {
    type Output;

    fn distribute(self, args: ArgVals) -> Self::Output;
}

impl<A, B> DistributeArgs for (A, B)
where
    A: ComputationFn + RunCore,
    B: ComputationFn + RunCore,
{
    type Output = (A::Output, B::Output);

    fn distribute(self, args: ArgVals) -> Self::Output {
        let (lhs_args, rhs_args) = args
            .partition(&self.0.args(), &self.1.args())
            .unwrap_or_else(|e| panic!("{}", e,));
        (self.0.run_core(lhs_args), self.1.run_core(rhs_args))
    }
}

macro_rules! impl_distribute_args_for_n_tuple {
    ( $n:expr, $( $i:expr ),* ) => {
        paste! {
            impl< $( [<T $i>] ),* > DistributeArgs for ( $( [<T $i>] ),* )
            where
                $( [<T $i>]: ComputationFn + RunCore ),*
            {
                type Output = ( $( [<T $i>]::Output ),* );

                fn distribute(self, args: ArgVals) -> Self::Output {
                    let ( $( [<args_ $i>] ),* ) = args
                        .[<partition $n>]( $( &self.$i.args() ),* )
                        .unwrap_or_else(|e| panic!("{}", e,));
                    ( $( self.$i.run_core([<args_ $i>]) ),* )
                }
            }
        }
    };
}

impl_distribute_args_for_n_tuple!(3, 0, 1, 2);
impl_distribute_args_for_n_tuple!(4, 0, 1, 2, 3);
impl_distribute_args_for_n_tuple!(5, 0, 1, 2, 3, 4);
impl_distribute_args_for_n_tuple!(6, 0, 1, 2, 3, 4, 5);
impl_distribute_args_for_n_tuple!(7, 0, 1, 2, 3, 4, 5, 6);
impl_distribute_args_for_n_tuple!(8, 0, 1, 2, 3, 4, 5, 6, 7);
impl_distribute_args_for_n_tuple!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_distribute_args_for_n_tuple!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_distribute_args_for_n_tuple!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_distribute_args_for_n_tuple!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_distribute_args_for_n_tuple!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_distribute_args_for_n_tuple!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_distribute_args_for_n_tuple!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_distribute_args_for_n_tuple!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg, argvals, run::Value, val};

    use super::*;

    #[proptest]
    fn distribute_args_should_properly_route_args(x: i32, y: i32) {
        prop_assert_eq!(
            (val!(x), val!(y)).distribute(argvals![]),
            (Value(x), Value(y))
        );
        prop_assert_eq!(
            (arg!("foo"), val!(y)).distribute(argvals![("foo", x)]),
            (Value(x), Value(y))
        );
        prop_assert_eq!(
            (val!(x), arg!("bar")).distribute(argvals![("bar", y)]),
            (Value(x), Value(y))
        );
        prop_assert_eq!(
            (arg!("foo"), arg!("bar")).distribute(argvals![("foo", x), ("bar", y)]),
            (Value(x), Value(y))
        );
    }

    #[proptest]
    fn distribute_args_should_properly_route_args_for_three_tuples(x: i32, y: i32, z: i32) {
        prop_assert_eq!(
            (val!(x), val!(y), val!(z)).distribute(argvals![]),
            (Value(x), Value(y), Value(z))
        );
        prop_assert_eq!(
            (arg!("foo"), val!(y), val!(z)).distribute(argvals![("foo", x)]),
            (Value(x), Value(y), Value(z))
        );
        prop_assert_eq!(
            (val!(x), arg!("bar"), val!(z)).distribute(argvals![("bar", y)]),
            (Value(x), Value(y), Value(z))
        );
        prop_assert_eq!(
            (arg!("foo"), arg!("bar"), val!(z)).distribute(argvals![("foo", x), ("bar", y)]),
            (Value(x), Value(y), Value(z))
        );
        prop_assert_eq!(
            (arg!("foo"), val!(y), arg!("baz")).distribute(argvals![("foo", x), ("baz", z)]),
            (Value(x), Value(y), Value(z))
        );
        prop_assert_eq!(
            (val!(x), arg!("bar"), arg!("baz")).distribute(argvals![("bar", y), ("baz", z)]),
            (Value(x), Value(y), Value(z))
        );
        prop_assert_eq!(
            (arg!("foo"), arg!("bar"), arg!("baz")).distribute(argvals![
                ("foo", x),
                ("bar", y),
                ("baz", z)
            ]),
            (Value(x), Value(y), Value(z))
        );
    }
}
