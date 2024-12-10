use crate::{Computation, Val};

use super::RunCore;

impl<Dim, T> RunCore for Val<Dim, T>
where
    Self: Computation,
{
    type Output = T;

    fn run_core(self) -> Self::Output {
        self.inner
    }
}
