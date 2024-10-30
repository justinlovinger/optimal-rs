mod distribute_args;
mod into_vec;
mod matrix;
mod named_args;
mod run_core;

use crate::ComputationFn;

pub use self::{
    collect::*, distribute_args::*, into_vec::*, matrix::*, named_args::*, run_core::*, unwrap::*,
};

/// A computation that can be run
/// without additional compilation.
///
/// This trait is automatically implemented
/// for types implementing [`RunCore`].
pub trait Run {
    type Output;

    fn run(self, args: NamedArgs) -> Self::Output;
}

impl<T, Collected> Run for T
where
    T: ComputationFn + RunCore,
    T::Output: Collect<T::Dim, Collected = Collected>,
    Collected: Unwrap,
{
    type Output = Collected::Unwrapped;

    fn run(self, args: NamedArgs) -> Self::Output {
        self.run_core(args).collect().unwrap()
    }
}

mod function {
    use crate::function::Function;

    use super::{FromNamesArgs, NamedArgs, Run, RunCore};

    impl<ArgNames, Body> Function<ArgNames, Body> {
        pub fn call<Args>(self, args: Args) -> Body::Output
        where
            NamedArgs: FromNamesArgs<ArgNames, Args>,
            Body: Run,
        {
            self.body
                .run(NamedArgs::from_names_args(self.arg_names, args))
        }

        pub fn call_core<Args>(self, args: Args) -> Body::Output
        where
            NamedArgs: FromNamesArgs<ArgNames, Args>,
            Body: RunCore,
        {
            self.body
                .run_core(NamedArgs::from_names_args(self.arg_names, args))
        }
    }
}

mod collect {
    use paste::paste;

    use crate::peano::{One, Two, Zero};

    use super::{IntoVec, Matrix, Value};

    pub trait Collect<OutDims> {
        type Collected;

        fn collect(self) -> Self::Collected;
    }

    impl<T> Collect<Zero> for Value<T> {
        type Collected = Value<T>;

        fn collect(self) -> Self::Collected {
            self
        }
    }

    impl<T> Collect<One> for Value<T>
    where
        T: IntoVec,
    {
        type Collected = Value<Vec<T::Item>>;

        fn collect(self) -> Self::Collected {
            Value(self.0.into_vec())
        }
    }

    impl<V> Collect<Two> for Value<Matrix<V>>
    where
        V: IntoVec,
    {
        type Collected = Value<Matrix<Vec<V::Item>>>;

        fn collect(self) -> Self::Collected {
            // Neither shape nor the length of `inner` will change,
            // so they should still be fine.
            Value(unsafe { Matrix::new_unchecked(self.0.shape(), self.0.into_inner().into_vec()) })
        }
    }

    macro_rules! impl_collect_for_n_tuple {
        ( $n:expr, $( $i:expr ),* ) => {
            paste! {
                impl< $( [<T $i>] ),* , $( [<DimT $i>] ),* > Collect<( $( [<DimT $i>] ),* )> for ( $( [<T $i>] ),* )
                where
                    $( [<T $i>]: Collect< [<DimT $i>] > ),*
                {
                    type Collected = ( $( [<T $i>]::Collected ),* );

                    fn collect(self) -> Self::Collected {
                        ( $( self.$i.collect() ),* )
                    }
                }
            }
        };
    }

    impl_collect_for_n_tuple!(2, 0, 1);
    impl_collect_for_n_tuple!(3, 0, 1, 2);
    impl_collect_for_n_tuple!(4, 0, 1, 2, 3);
    impl_collect_for_n_tuple!(5, 0, 1, 2, 3, 4);
    impl_collect_for_n_tuple!(6, 0, 1, 2, 3, 4, 5);
    impl_collect_for_n_tuple!(7, 0, 1, 2, 3, 4, 5, 6);
    impl_collect_for_n_tuple!(8, 0, 1, 2, 3, 4, 5, 6, 7);
    impl_collect_for_n_tuple!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_collect_for_n_tuple!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    impl_collect_for_n_tuple!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    impl_collect_for_n_tuple!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    impl_collect_for_n_tuple!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    impl_collect_for_n_tuple!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    impl_collect_for_n_tuple!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    impl_collect_for_n_tuple!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

mod unwrap {
    use paste::paste;

    use super::Value;

    pub trait Unwrap {
        type Unwrapped;

        fn unwrap(self) -> Self::Unwrapped;
    }

    impl<T> Unwrap for Value<T> {
        type Unwrapped = T;

        fn unwrap(self) -> Self::Unwrapped {
            self.0
        }
    }

    macro_rules! impl_unwrap_for_n_tuple {
        ( $n:expr, $( $i:expr ),* ) => {
            paste! {
                impl< $( [<T $i>] ),* > Unwrap for ( $( [<T $i>] ),* )
                where
                    $( [<T $i>]: Unwrap ),*
                {
                    type Unwrapped = ( $( [<T $i>]::Unwrapped ),* );

                    fn unwrap(self) -> Self::Unwrapped {
                        ( $( self.$i.unwrap() ),* )
                    }
                }
            }
        };
    }

    impl_unwrap_for_n_tuple!(2, 0, 1);
    impl_unwrap_for_n_tuple!(3, 0, 1, 2);
    impl_unwrap_for_n_tuple!(4, 0, 1, 2, 3);
    impl_unwrap_for_n_tuple!(5, 0, 1, 2, 3, 4);
    impl_unwrap_for_n_tuple!(6, 0, 1, 2, 3, 4, 5);
    impl_unwrap_for_n_tuple!(7, 0, 1, 2, 3, 4, 5, 6);
    impl_unwrap_for_n_tuple!(8, 0, 1, 2, 3, 4, 5, 6, 7);
    impl_unwrap_for_n_tuple!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_unwrap_for_n_tuple!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    impl_unwrap_for_n_tuple!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    impl_unwrap_for_n_tuple!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    impl_unwrap_for_n_tuple!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    impl_unwrap_for_n_tuple!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    impl_unwrap_for_n_tuple!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    impl_unwrap_for_n_tuple!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg, named_args, val, val1};

    use super::*;

    #[proptest]
    fn operations_should_combine(
        #[strategy(-1000..1000)] x: i32,
        #[strategy(-1000..1000)] y: i32,
        #[strategy(-1000..1000)] z: i32,
    ) {
        prop_assume!((y - z) != 0);
        prop_assume!(z != 0);
        prop_assert_eq!(
            (val!(x) / (val!(y) - val!(z)) + -(val!(z) * val!(y))).run(named_args![]),
            x / (y - z) + -(z * y)
        );
        prop_assert_eq!(
            (-(((val!(x) + val!(y) - val!(z)) / val!(z)) * val!(y))).run(named_args![]),
            -(((x + y - z) / z) * y)
        );
        prop_assert_eq!(-(-val!(x)).run(named_args![]), -(-x));
        prop_assert_eq!(
            (val1!([x, y]) / (val!(y) - val!(z)) + -(val!(z) * val!(y))).run(named_args![]),
            [x / (y - z) + -(z * y), y / (y - z) + -(z * y)]
        );
    }

    #[proptest]
    fn args_should_propagate_correctly(
        #[strategy(-1000..1000)] x: i32,
        #[strategy(-1000..1000)] y: i32,
        #[strategy(-1000..1000)] z: i32,
        #[strategy(-1000..1000)] in_x: i32,
        #[strategy(-1000..1000)] in_y: i32,
        #[strategy(-1000..1000)] in_z: i32,
    ) {
        prop_assume!((x - in_y) != 0);
        prop_assume!(z != 0);
        prop_assert_eq!(
            (arg!("foo", i32) / (val!(x) - arg!("bar", i32))
                + -(val!(z) * val!(y) + arg!("baz", i32)))
            .run(named_args![("foo", in_x), ("bar", in_y), ("baz", in_z)]),
            in_x / (x - in_y) + -(z * y + in_z)
        );
        prop_assert_eq!(
            (arg!("foo", i32)
                + (((val!(x) + val!(y) - arg!("bar", i32)) / -val!(z)) * arg!("baz", i32)))
            .run(named_args![("foo", in_x), ("bar", in_y), ("baz", in_z)]),
            in_x + (((x + y - in_y) / -z) * in_z)
        );
        prop_assert_eq!(-(-arg!("foo", i32)).run(named_args![("foo", x)]), -(-x));
    }
}
