mod black_box;
mod cmp;
mod control_flow;
mod enumerate;
mod linalg;
mod math;
mod rand;
mod sum;
mod zip;

use crate::{Computation, Len, Val};

/// See [`super::Run`].
///
/// Unlike `run`,
/// `run_core` may return iterators
/// or difficult-to-use types.
/// Outside implementing `RunCore`,
/// `run` should be used.
pub trait RunCore {
    type Output;

    fn run_core(self) -> Self::Output;
}

impl<T> RunCore for &T
where
    T: ToOwned + ?Sized,
    T::Owned: RunCore,
{
    type Output = <T::Owned as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        self.to_owned().run_core()
    }
}

impl<T> RunCore for &mut T
where
    T: ToOwned + ?Sized,
    T::Owned: RunCore,
{
    type Output = <T::Owned as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        self.to_owned().run_core()
    }
}

impl<T> RunCore for Box<T>
where
    T: RunCore + ?Sized,
{
    type Output = T::Output;

    fn run_core(self) -> Self::Output {
        (*self).run_core()
    }
}

impl<T> RunCore for std::rc::Rc<T>
where
    T: ToOwned + ?Sized,
    T::Owned: RunCore,
{
    type Output = <T::Owned as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        self.as_ref().to_owned().run_core()
    }
}

impl<T> RunCore for std::sync::Arc<T>
where
    T: ToOwned + ?Sized,
    T::Owned: RunCore,
{
    type Output = <T::Owned as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        self.as_ref().to_owned().run_core()
    }
}

impl<T> RunCore for std::borrow::Cow<'_, T>
where
    T: ToOwned + ?Sized,
    T::Owned: RunCore,
{
    type Output = <T::Owned as RunCore>::Output;

    fn run_core(self) -> Self::Output {
        self.into_owned().run_core()
    }
}

impl<Dim, A> RunCore for Val<Dim, A>
where
    Self: Computation,
{
    type Output = A;

    fn run_core(self) -> Self::Output {
        self.inner
    }
}

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
