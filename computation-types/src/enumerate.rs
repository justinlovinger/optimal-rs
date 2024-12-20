use core::fmt;

use crate::{
    impl_core_ops, peano::One, Computation, ComputationFn, Function, Name, NamedArgs, Names,
};

#[derive(Clone, Copy, Debug)]
pub struct Enumerate<A, F>
where
    Self: Computation,
{
    pub child: A,
    pub f: Function<(Name, Name), F>,
}

impl<A, F> Computation for Enumerate<A, F>
where
    A: Computation<Dim = One>,
    F: Computation,
{
    type Dim = F::Dim;
    type Item = F::Item;
}

impl<A, F> ComputationFn for Enumerate<A, F>
where
    Self: Computation,
    A: ComputationFn,
    Enumerate<A::Filled, F>: Computation,
{
    type Filled = Enumerate<A::Filled, F>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        Enumerate {
            child: self.child.fill(named_args),
            f: self.f,
        }
    }

    fn arg_names(&self) -> Names {
        self.child.arg_names()
    }
}

impl_core_ops!(Enumerate<A, F>);

impl<A, F> fmt::Display for Enumerate<A, F>
where
    Self: Computation,
    A: fmt::Display,
    F: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.enumerate({})", self.child, self.f.body)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg1, val1, Computation, Function};

    #[proptest]
    fn enumerate_should_display(xs: Vec<usize>) {
        let inp = val1!(xs.iter().cloned());
        let f = Function::anonymous(("x", "i"), arg1!("x", usize) + arg1!("i", usize));
        prop_assert_eq!(
            inp.clone().enumerate(f).to_string(),
            format!("{}.enumerate({})", inp, f.body)
        );
    }
}
