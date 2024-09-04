use paste::paste;

use crate::{
    run::{ArgVals, DistributeArgs, RunCore},
    zip::*,
    ComputationFn,
};

impl<A, B, Out> RunCore for Zip<A, B>
where
    A: ComputationFn,
    B: ComputationFn,
    (A, B): DistributeArgs<Output = Out>,
{
    type Output = Out;

    fn run_core(self, args: ArgVals) -> Self::Output {
        (self.0, self.1).distribute(args)
    }
}

impl<A, OutA, OutB> RunCore for Fst<A>
where
    A: RunCore<Output = (OutA, OutB)>,
{
    type Output = OutA;

    fn run_core(self, args: ArgVals) -> Self::Output {
        self.0.run_core(args).0
    }
}

impl<A, OutA, OutB> RunCore for Snd<A>
where
    A: RunCore<Output = (OutA, OutB)>,
{
    type Output = OutB;

    fn run_core(self, args: ArgVals) -> Self::Output {
        self.0.run_core(args).1
    }
}

macro_rules! impl_intocpu_for_zip_n {
    ( $n:expr, $( $i:expr ),* ) => {
        paste! {
            impl< $( [<T $i>] ),* , Out > RunCore for [<Zip $n>]< $( [<T $i>] ),* >
            where
                ( $( [<T $i>] ),* ): DistributeArgs<Output = Out>,
            {
                type Output = Out;

                fn run_core(self, args: ArgVals) -> Self::Output {
                    ( $( self.$i ),* ).distribute(args)
                }
            }
        }
    };
}

impl_intocpu_for_zip_n!(3, 0, 1, 2);
impl_intocpu_for_zip_n!(4, 0, 1, 2, 3);
impl_intocpu_for_zip_n!(5, 0, 1, 2, 3, 4);
impl_intocpu_for_zip_n!(6, 0, 1, 2, 3, 4, 5);
impl_intocpu_for_zip_n!(7, 0, 1, 2, 3, 4, 5, 6);
impl_intocpu_for_zip_n!(8, 0, 1, 2, 3, 4, 5, 6, 7);
impl_intocpu_for_zip_n!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_intocpu_for_zip_n!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_intocpu_for_zip_n!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_intocpu_for_zip_n!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_intocpu_for_zip_n!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_intocpu_for_zip_n!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_intocpu_for_zip_n!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_intocpu_for_zip_n!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{argvals, val, Computation, Run};

    use super::*;

    #[proptest]
    fn zip_should_return_a_tuple_when_run(x: usize, y: usize) {
        prop_assert_eq!(val!(x).zip(val!(y)).run(argvals![]), (x, y));
    }

    #[proptest]
    fn fst_should_return_the_first_item_in_a_tuple_when_run(x: usize, y: usize) {
        prop_assert_eq!(val!(x).zip(val!(y)).fst().run(argvals![]), x);
    }

    #[proptest]
    fn snd_should_return_the_second_item_in_a_tuple_when_run(x: usize, y: usize) {
        prop_assert_eq!(val!(x).zip(val!(y)).snd().run(argvals![]), y);
    }

    #[proptest]
    fn zip3_should_return_a_three_tuple_when_run(x: usize, y: usize, z: usize) {
        prop_assert_eq!(
            Zip3::new(val!(x), val!(y), val!(z)).run(argvals![]),
            (x, y, z)
        );
    }
}
