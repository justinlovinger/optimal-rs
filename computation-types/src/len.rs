use core::fmt;

use crate::{
    impl_core_ops,
    peano::{One, Zero},
    Computation, ComputationFn, NamedArgs, Names,
};

#[derive(Clone, Copy, Debug)]
pub struct Len<A>(pub A)
where
    Self: Computation;

impl<A> Computation for Len<A>
where
    A: Computation<Dim = One>,
{
    type Dim = Zero;
    type Item = usize;
}

impl<A> ComputationFn for Len<A>
where
    Self: Computation,
    A: ComputationFn,
    Len<A::Filled>: Computation,
{
    type Filled = Len<A::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        Len(self.0.fill(named_args))
    }

    fn arg_names(&self) -> Names {
        self.0.arg_names()
    }
}

impl_core_ops!(Len<A>);

impl<A> fmt::Display for Len<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.len()", self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::val1;

    use super::*;

    #[test]
    fn len_should_display() {
        let inp = val1!(vec![0, 1]);
        assert_eq!(inp.clone().len().to_string(), format!("{}.len()", inp));
    }
}
