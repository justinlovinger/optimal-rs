use core::fmt;
use std::marker::PhantomData;

use crate::{
    impl_core_ops,
    peano::{One, Two, Zero},
    AnyArg, Computation, ComputationFn, NamedArgs, Names, Val,
};

#[derive(Clone, Copy, Debug)]
pub struct Arg<Dim, T>
where
    Self: Computation,
{
    pub name: &'static str,
    dim: PhantomData<Dim>,
    elem: PhantomData<T>,
}

pub type Arg0<T> = Arg<Zero, T>;
pub type Arg1<T> = Arg<One, T>;
pub type Arg2<T> = Arg<Two, T>;

impl<Dim, T> Arg<Dim, T> {
    pub fn new(name: &'static str) -> Self {
        Arg {
            name,
            dim: PhantomData,
            elem: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! arg {
    ( $name:literal ) => {
        $crate::Arg0::new($name)
    };
    ( $name:literal, $elem:ty ) => {
        $crate::Arg0::<$elem>::new($name)
    };
}

#[macro_export]
macro_rules! arg1 {
    ( $name:literal ) => {
        $crate::Arg1::new($name)
    };
    ( $name:literal, $elem:ty ) => {
        $crate::Arg1::<$elem>::new($name)
    };
}

#[macro_export]
macro_rules! arg2 {
    ( $name:literal ) => {
        $crate::Arg2::new($name)
    };
    ( $name:literal, $elem:ty ) => {
        $crate::Arg2::<$elem>::new($name)
    };
}

impl<D, T> Computation for Arg<D, T> {
    type Dim = D;
    type Item = T;
}

impl<T> ComputationFn for Arg<Zero, T>
where
    Self: Computation,
    T: 'static + AnyArg,
{
    type Filled = Val<Zero, T>;

    fn fill(self, mut named_args: NamedArgs) -> Self::Filled {
        Val::new(
            named_args
                .pop(self.name)
                .unwrap_or_else(|e| panic!("{}", e)),
        )
    }

    fn arg_names(&self) -> Names {
        Names::singleton(self.name)
    }
}

impl<T> ComputationFn for Arg<One, T>
where
    Self: Computation,
    T: 'static + Clone + AnyArg,
{
    type Filled = Val<One, Vec<T>>;

    fn fill(self, mut named_args: NamedArgs) -> Self::Filled {
        Val::new(
            named_args
                .pop(self.name)
                .unwrap_or_else(|e| panic!("{}", e)),
        )
    }

    fn arg_names(&self) -> Names {
        Names::singleton(self.name)
    }
}

impl<T> ComputationFn for Arg<Two, T>
where
    Self: Computation,
    T: 'static + Clone + AnyArg,
{
    type Filled = Val<Two, crate::run::Matrix<Vec<T>>>;

    fn fill(self, mut named_args: NamedArgs) -> Self::Filled {
        Val::new(
            named_args
                .pop(self.name)
                .unwrap_or_else(|e| panic!("{}", e)),
        )
    }

    fn arg_names(&self) -> Names {
        Names::singleton(self.name)
    }
}

impl_core_ops!(Arg<Dim, T>);

impl<Dim, T> fmt::Display for Arg<Dim, T>
where
    Self: Computation,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn arg_should_display_placeholder() {
        assert_eq!(arg!("foo", i32).to_string(), "foo");
        assert_eq!(arg1!("bar", i32).to_string(), "bar");
    }
}
