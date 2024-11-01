use paste::paste;

use crate::{run::RunCore, zip::*, Computation};

impl<A, B> RunCore for Zip<A, B>
where
    Self: Computation,
    A: RunCore,
    B: RunCore,
{
    type Output = (A::Output, B::Output);

    fn run_core(self) -> Self::Output {
        (self.0.run_core(), self.1.run_core())
    }
}

impl<A, OutA, OutB> RunCore for Fst<A>
where
    Self: Computation,
    A: RunCore<Output = (OutA, OutB)>,
{
    type Output = OutA;

    fn run_core(self) -> Self::Output {
        self.0.run_core().0
    }
}

impl<A, OutA, OutB> RunCore for Snd<A>
where
    Self: Computation,
    A: RunCore<Output = (OutA, OutB)>,
{
    type Output = OutB;

    fn run_core(self) -> Self::Output {
        self.0.run_core().1
    }
}

macro_rules! impl_intocpu_for_zip_n {
    ( $n:expr, $( $i:expr ),* ) => {
        paste! {
            impl< $( [<T $i>] ),* > RunCore for [<Zip $n>]< $( [<T $i>] ),* >
            where
                Self: Computation,
                $( [<T $i>]: RunCore ),*,
            {
                type Output = ( $( [<T $i>]::Output ),* );

                fn run_core(self) -> Self::Output {
                    ( $( self.$i.run_core() ),* )
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

    use crate::{val, Computation, Run};

    use super::*;

    #[proptest]
    fn zip_should_return_a_tuple_when_run(x: usize, y: usize) {
        prop_assert_eq!(val!(x).zip(val!(y)).run(), (x, y));
    }

    #[proptest]
    fn fst_should_return_the_first_item_in_a_tuple_when_run(x: usize, y: usize) {
        prop_assert_eq!(val!(x).zip(val!(y)).fst().run(), x);
    }

    #[proptest]
    fn snd_should_return_the_second_item_in_a_tuple_when_run(x: usize, y: usize) {
        prop_assert_eq!(val!(x).zip(val!(y)).snd().run(), y);
    }

    #[proptest]
    fn zip3_should_return_a_three_tuple_when_run(x: usize, y: usize, z: usize) {
        prop_assert_eq!(Zip3(val!(x), val!(y), val!(z)).run(), (x, y, z));
    }
}
