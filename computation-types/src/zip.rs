use core::fmt;

use paste::paste;

use crate::{
    impl_computation_fn_for_binary, impl_computation_fn_for_unary, impl_core_ops, Computation,
    ComputationFn, NamedArgs, Names,
};

#[derive(Clone, Copy, Debug)]
pub struct Zip<A, B>(pub A, pub B)
where
    Self: Computation;

impl<A, B> Computation for Zip<A, B>
where
    A: Computation,
    B: Computation,
{
    type Dim = (A::Dim, B::Dim);
    type Item = (A::Item, B::Item);
}

impl_core_ops!(Zip<A, B>);

impl_computation_fn_for_binary!(Zip);

impl<A, B> fmt::Display for Zip<A, B>
where
    Self: Computation,
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fst<A>(pub A)
where
    Self: Computation;

impl<A, DimA, DimB, ItemA, ItemB> Computation for Fst<A>
where
    A: Computation<Dim = (DimA, DimB), Item = (ItemA, ItemB)>,
{
    type Dim = DimA;
    type Item = ItemA;
}

impl_core_ops!(Fst<A>);

impl_computation_fn_for_unary!(Fst);

impl<A> fmt::Display for Fst<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.0", self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Snd<A>(pub A)
where
    Self: Computation;

impl<A, DimA, DimB, ItemA, ItemB> Computation for Snd<A>
where
    A: Computation<Dim = (DimA, DimB), Item = (ItemA, ItemB)>,
{
    type Dim = DimB;
    type Item = ItemB;
}

impl_computation_fn_for_unary!(Snd);

impl_core_ops!(Snd<A>);

impl<A> fmt::Display for Snd<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.1", self.0)
    }
}

macro_rules! zip_n {
    ( $n:expr, $i_first:expr, $( $i_rest:expr ),* ) => {
        zip_n!(@combined $n, { $i_first, $( $i_rest ),* } { $i_first, $( $i_rest ),* });
    };
    ( @combined $n:expr, { $i_first:expr, $( $i_rest:expr ),* } { $( $i:expr ),* } ) => {
        paste! {
            #[derive(Clone, Copy, Debug)]
            pub struct [<Zip $n>]< $( [<T $i>] ),* >( $( pub [<T $i>] ),* )
            where
                Self: Computation;

            impl< $( [<T $i>] ),* > Computation for [<Zip $n>]< $( [<T $i>] ),* >
            where
                $( [<T $i>]: Computation ),*
            {
                type Dim = ( $( [<T $i>]::Dim ),* );
                type Item = ( $( [<T $i>]::Item ),* );
            }

            impl< $( [<T $i>] ),* > ComputationFn for [<Zip $n>]< $( [<T $i>] ),* >
            where
                Self: Computation,
                $( [<T $i>]: ComputationFn ),*,
                [<Zip $n>]< $( [<T $i>]::Filled ),* >: Computation
            {
                type Filled = [<Zip $n>]< $( [<T $i>]::Filled ),* >;

                fn fill(self, named_args: NamedArgs) -> Self::Filled {
                    let ( $( [<args_ $i>] ),* ) = named_args
                        .[<partition $n>]( $( &self.$i.arg_names() ),* )
                        .unwrap_or_else(|e| panic!("{}", e,));
                    [<Zip $n>]( $( self.$i.fill([<args_ $i>]) ),* )
                }

                fn arg_names(&self) -> Names {
                    Names::union_many([ $( &self.$i.arg_names() ),* ])
                }
            }

            impl_core_ops!([<Zip $n>]< $( [<T $i>] ),* >);

            impl< $( [<T $i>] ),* > fmt::Display for [<Zip $n>]< $( [<T $i>] ),* >
            where
                Self: Computation,
                $( [<T $i>]: fmt::Display ),*
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    "(".fmt(f)?;
                    self.$i_first.fmt(f)?;
                    $(
                      ", ".fmt(f)?;
                      self.$i_rest.fmt(f)?;
                    )*
                    ")".fmt(f)
                }
            }
        }
    };
}

zip_n!(3, 0, 1, 2);
zip_n!(4, 0, 1, 2, 3);
zip_n!(5, 0, 1, 2, 3, 4);
zip_n!(6, 0, 1, 2, 3, 4, 5);
zip_n!(7, 0, 1, 2, 3, 4, 5, 6);
zip_n!(8, 0, 1, 2, 3, 4, 5, 6, 7);
zip_n!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
zip_n!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
zip_n!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
zip_n!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
zip_n!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
zip_n!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
zip_n!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
zip_n!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{val, Computation};

    use super::*;

    #[proptest]
    fn zip_should_display(x: usize, y: usize) {
        prop_assert_eq!(
            val!(x).zip(val!(y)).to_string(),
            format!("({}, {})", val!(x), val!(y))
        );
    }

    #[proptest]
    fn fst_should_display(x: usize, y: usize) {
        let inp = val!(x).zip(val!(y));
        prop_assert_eq!(inp.fst().to_string(), format!("{}.0", inp));
    }

    #[proptest]
    fn snd_should_display(x: usize, y: usize) {
        let inp = val!(x).zip(val!(y));
        prop_assert_eq!(inp.snd().to_string(), format!("{}.1", inp));
    }

    #[proptest]
    fn zip3_should_display(x: usize, y: usize, z: usize) {
        prop_assert_eq!(
            Zip3(val!(x), val!(y), val!(z)).to_string(),
            format!("({}, {}, {})", val!(x), val!(y), val!(z))
        );
    }
}
