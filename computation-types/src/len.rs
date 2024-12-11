use core::fmt;

use crate::{
    impl_computation_fn_for_unary, impl_core_ops,
    peano::{One, Zero},
    Computation, ComputationFn, NamedArgs,
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

impl_computation_fn_for_unary!(Len);

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
