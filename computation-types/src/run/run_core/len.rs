use crate::{len::Len, Computation};

use super::RunCore;

impl<A> RunCore for Len<A>
where
    Self: Computation,
    A: RunCore,
    A::Output: IntoIterator,
    <A::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Output = usize;

    fn run_core(self) -> Self::Output {
        self.0.run_core().into_iter().len()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{val1, Computation, Run};

    #[proptest]
    fn len_should_return_length(xs: Vec<i32>) {
        prop_assert_eq!(val1!(xs.clone()).len().run(), xs.len());
    }
}
