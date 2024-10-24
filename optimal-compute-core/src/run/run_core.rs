mod black_box;
mod cmp;
mod control_flow;
mod enumerate;
mod linalg;
mod math;
mod rand;
mod sum;
mod zip;

use crate::{
    peano::{One, Two, Zero},
    run::{ArgVal, ArgVals},
    Arg, Computation, Len, Val,
};

use super::{Matrix, Unwrap, Value};

/// See [`super::Run`].
///
/// Unlike `run`,
/// `run_core` may return iterators
/// or difficult-to-use types.
/// Outside implementing `RunCore`,
/// `run` should be used.
pub trait RunCore {
    type Output;

    fn run_core(self, args: ArgVals) -> Self::Output;
}

impl<T> RunCore for &T
where
    T: RunCore + ?Sized,
{
    type Output = T::Output;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        unimplemented!("Unsized types cannot run")
    }
}

impl<T> RunCore for &mut T
where
    T: RunCore + ?Sized,
{
    type Output = T::Output;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        unimplemented!("Unsized types cannot run")
    }
}

impl<T> RunCore for Box<T>
where
    T: RunCore + ?Sized,
{
    type Output = T::Output;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        unimplemented!("Unsized types cannot run")
    }
}

impl<T> RunCore for std::rc::Rc<T>
where
    T: RunCore + ?Sized,
{
    type Output = T::Output;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        unimplemented!("Unsized types cannot run")
    }
}

impl<T> RunCore for std::sync::Arc<T>
where
    T: RunCore + ?Sized,
{
    type Output = T::Output;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        unimplemented!("Unsized types cannot run")
    }
}

impl<'a, T> RunCore for std::borrow::Cow<'a, T>
where
    T: RunCore + ToOwned + ?Sized,
{
    type Output = T::Output;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        unimplemented!("Unsized types cannot run, only associated types are available.")
    }
}

impl<Dim, A> RunCore for Val<Dim, A>
where
    Self: Computation,
{
    type Output = Value<A>;

    fn run_core(self, _args: ArgVals) -> Self::Output {
        Value(self.inner)
    }
}

impl<A> RunCore for Arg<Zero, A>
where
    Self: Computation,
    A: 'static + ArgVal,
{
    type Output = Value<A>;

    fn run_core(self, mut args: ArgVals) -> Self::Output {
        Value(args.pop(self.name).unwrap_or_else(|e| panic!("{}", e)))
    }
}

impl<A> RunCore for Arg<One, A>
where
    Vec<A>: ArgVal,
    A: 'static,
{
    type Output = Value<Vec<A>>;

    fn run_core(self, mut args: ArgVals) -> Self::Output {
        Value(args.pop(self.name).unwrap_or_else(|e| panic!("{}", e)))
    }
}

impl<A> RunCore for Arg<Two, A>
where
    Matrix<Vec<A>>: ArgVal,
    A: 'static,
{
    type Output = Value<Matrix<Vec<A>>>;

    fn run_core(self, mut args: ArgVals) -> Self::Output {
        Value(args.pop(self.name).unwrap_or_else(|e| panic!("{}", e)))
    }
}

impl<A, Out, It> RunCore for Len<A>
where
    Self: Computation,
    A: RunCore<Output = Value<Out>>,
    Out: IntoIterator<IntoIter = It>,
    It: ExactSizeIterator,
{
    type Output = Value<usize>;

    fn run_core(self, args: ArgVals) -> Self::Output {
        Value(self.0.run_core(args).unwrap().into_iter().len())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{argvals, val1, Computation, Run};

    #[proptest]
    fn len_should_return_length(xs: Vec<i32>) {
        prop_assert_eq!(val1!(xs.clone()).len().run(argvals![]), xs.len());
    }
}
