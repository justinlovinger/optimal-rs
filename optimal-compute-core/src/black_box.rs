use core::fmt;
use std::marker::PhantomData;

use crate::{impl_core_ops, Args, Computation, ComputationFn};

/// See [`Computation::black_box`].
#[derive(Clone, Copy, Debug)]
pub struct BlackBox<A, F, FDim, FItem>
where
    Self: Computation,
{
    pub child: A,
    pub f: F,
    pub(super) f_dim: PhantomData<FDim>,
    pub(super) f_item: PhantomData<FItem>,
}

impl<A, F, FDim, FItem> Computation for BlackBox<A, F, FDim, FItem>
where
    A: Computation,
{
    type Dim = FDim;
    type Item = FItem;
}

impl<A, F, FDim, FItem> ComputationFn for BlackBox<A, F, FDim, FItem>
where
    Self: Computation,
    A: ComputationFn,
{
    fn args(&self) -> Args {
        self.child.args()
    }
}

impl_core_ops!(BlackBox<A, F, FDim, FItem>);

impl<A, F, FDim, FItem> fmt::Display for BlackBox<A, F, FDim, FItem>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f({})", self.child)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{peano::Zero, val, Computation};

    #[proptest]
    fn black_box_should_display(x: i32) {
        let inp = val!(x);
        prop_assert_eq!(
            inp.black_box::<_, Zero, i32>(|x: i32| x + 1).to_string(),
            format!("f({})", inp)
        );
    }
}
