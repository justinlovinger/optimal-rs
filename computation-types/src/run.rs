mod into_vec;
mod matrix;
mod run_core;

use crate::{ComputationFn, NamedArgs};

pub use self::{collect::*, into_vec::*, matrix::*, run_core::*};

/// A computation that can be run
/// without additional compilation.
///
/// This trait is automatically implemented
/// for types implementing [`RunCore`].
pub trait Run {
    type Output;

    fn run(self) -> Self::Output;
}

impl<T, Collected> Run for T
where
    T: ComputationFn + RunCore,
    T::Output: Collect<T::Dim, Collected = Collected>,
{
    type Output = Collected;

    fn run(self) -> Self::Output {
        self.run_core().collect()
    }
}

mod function {
    use crate::{function::Function, ComputationFn, FromNamesArgs};

    use super::{NamedArgs, Run, RunCore};

    impl<ArgNames, Body> Function<ArgNames, Body> {
        pub fn call<Args>(self, args: Args) -> <Body::Filled as Run>::Output
        where
            NamedArgs: FromNamesArgs<ArgNames, Args>,
            Body: ComputationFn,
            Body::Filled: Run,
        {
            self.fill(args).run()
        }

        pub fn call_core<Args>(self, args: Args) -> <Body::Filled as RunCore>::Output
        where
            NamedArgs: FromNamesArgs<ArgNames, Args>,
            Body: ComputationFn,
            Body::Filled: RunCore,
        {
            self.fill(args).run_core()
        }
    }
}

mod collect {
    use paste::paste;

    use crate::peano::{One, Two, Zero};

    use super::{IntoVec, Matrix};

    pub trait Collect<OutDims> {
        type Collected;

        fn collect(self) -> Self::Collected;
    }

    impl<T> Collect<Zero> for T {
        type Collected = T;

        fn collect(self) -> Self::Collected {
            self
        }
    }

    impl<T> Collect<One> for T
    where
        T: IntoVec,
    {
        type Collected = Vec<T::Item>;

        fn collect(self) -> Self::Collected {
            self.into_vec()
        }
    }

    impl<V> Collect<Two> for Matrix<V>
    where
        V: IntoVec,
    {
        type Collected = Matrix<Vec<V::Item>>;

        fn collect(self) -> Self::Collected {
            // Neither shape nor the length of `inner` will change,
            // so they should still be fine.
            unsafe { Matrix::new_unchecked(self.shape(), self.into_inner().into_vec()) }
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{val, val1};

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
            (val!(x) / (val!(y) - val!(z)) + -(val!(z) * val!(y))).run(),
            x / (y - z) + -(z * y)
        );
        prop_assert_eq!(
            (-(((val!(x) + val!(y) - val!(z)) / val!(z)) * val!(y))).run(),
            -(((x + y - z) / z) * y)
        );
        prop_assert_eq!(-(-val!(x)).run(), -(-x));
        prop_assert_eq!(
            (val1!([x, y]) / (val!(y) - val!(z)) + -(val!(z) * val!(y))).run(),
            [x / (y - z) + -(z * y), y / (y - z) + -(z * y)]
        );
    }
}
