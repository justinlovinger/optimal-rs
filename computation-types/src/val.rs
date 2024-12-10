use core::fmt;
use std::marker::PhantomData;

use crate::{
    impl_core_ops,
    peano::{One, Suc, Two, Zero},
    Computation, ComputationFn, NamedArgs, Names,
};

#[derive(Clone, Copy, Debug)]
pub struct Val<Dim, T>
where
    Self: Computation,
{
    dim: PhantomData<Dim>,
    pub inner: T,
}

pub type Val0<T> = Val<Zero, T>;
pub type Val1<T> = Val<One, T>;
pub type Val2<T> = Val<Two, T>;

impl<Dim, T> Val<Dim, T>
where
    Self: Computation,
{
    pub fn new(value: T) -> Self {
        Val {
            dim: PhantomData,
            inner: value,
        }
    }
}

#[macro_export]
macro_rules! val {
    ( $value:expr ) => {
        $crate::Val0::new($value)
    };
}

#[macro_export]
macro_rules! val1 {
    ( $value:expr ) => {
        $crate::Val1::new($value)
    };
}

#[macro_export]
macro_rules! val2 {
    ( $value:expr ) => {
        $crate::Val2::new($value)
    };
}

impl<T> Computation for Val<Zero, T> {
    type Dim = Zero;
    type Item = T;
}

impl<D, T> Computation for Val<Suc<D>, T>
where
    T: IntoIterator,
{
    type Dim = Suc<D>;
    type Item = T::Item;
}

impl<D, T> ComputationFn for Val<D, T>
where
    Val<D, T>: Computation,
{
    type Filled = Self;

    fn fill(self, _named_args: NamedArgs) -> Self::Filled {
        self
    }

    fn arg_names(&self) -> Names {
        Names::new()
    }
}

impl_core_ops!(Val<Dim, T>);

impl<T> fmt::Display for Val<Zero, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl<D, T> fmt::Display for Val<Suc<D>, T>
where
    Self: Computation,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    #[proptest]
    fn val_should_display_inner(x: i32) {
        prop_assert_eq!(val!(x).to_string(), x.to_string())
    }

    #[proptest]
    fn val1_should_display_items(xs: Vec<i32>) {
        prop_assert_eq!(val1!(xs.clone()).to_string(), format!("{:?}", xs.clone()));
    }
}
