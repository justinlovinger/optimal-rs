//! Mathematical optimization framework.
//!
//! An optimizer is split between a static configuration
//! and a dynamic state.
//! Traits are defined on the configuration
//! and may take a state.

pub mod derivative;
pub mod derivative_free;
mod iterator;

use std::borrow::Cow;

use blanket::blanket;
use ndarray::prelude::*;
use streaming_iterator::StreamingIterator;

use crate::prelude::Problem;

pub use self::iterator::*;

/// An optimizer
/// capable of solving an optimization problem,
/// automatically implemented on compatible [`OptimizerConfig`]s.
pub trait Optimizer<P>
where
    P: Problem,
{
    /// Return point that attempts to minimize the given problem.
    ///
    /// How well the point minimizes the problem
    /// depends on the optimizer.
    fn argmin(&self) -> Array1<P::PointElem>;
}

impl<P, T> Optimizer<P> for T
where
    P: Problem,
    P::PointElem: Clone,
    for<'a> &'a T: OptimizerConfig<Problem = P>,
    for<'a> <&'a T as OptimizerConfig>::Optimizer: Convergent,
{
    fn argmin(&self) -> Array1<P::PointElem> {
        self.start()
            .into_streaming_iter()
            .find(|o| o.is_done())
            .expect("should converge")
            .best_point()
            .to_owned()
    }
}

/// An optimizer configuration.
pub trait OptimizerConfig {
    /// Problem to optimize.
    type Problem: Problem;

    /// Optimizer this config can initialize.
    type Optimizer: RunningOptimizer<Problem = Self::Problem>;

    /// Return a running optimizer.
    ///
    /// This may be nondeterministic.
    fn start(self) -> Self::Optimizer;

    /// Return problem to optimize.
    fn problem(&self) -> &Self::Problem;
}

/// An optimizer configuration
/// for an optimizer
/// requiring a source of randomness.
pub trait StochasticOptimizerConfig<R>: OptimizerConfig {
    /// Return a running optimizer
    /// initialized using `rng`.
    fn start_using(self, rng: &mut R) -> Self::Optimizer;
}

/// A fully defined running optimizer.
///
/// A type implementing this
/// should also implement one of
/// `PointBased`
/// or `PopulationBased`.
pub trait RunningOptimizer: RunningOptimizerBase + RunningOptimizerStep {}

impl<T> RunningOptimizer for T where T: RunningOptimizerBase + RunningOptimizerStep {}

/// Running optimizer methods requiring only a reference.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait RunningOptimizerBase {
    /// Problem to optimize.
    type Problem: Problem;

    /// Return problem being optimized.
    fn problem(&self) -> &Self::Problem;

    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<<Self::Problem as Problem>::PointElem, Ix1>;

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    ///
    /// Most optimizers cannot return the best point value
    /// until at least one step has been performed.
    ///
    /// If an optimizer never stores the best point value,
    /// this will always return `None`.
    fn stored_best_point_value(&self) -> Option<&<Self::Problem as Problem>::PointValue>;
}

/// Step methods of a running optimizer.
#[blanket(derive(Mut, Box))]
pub trait RunningOptimizerStep: RunningOptimizerBase {
    /// Perform an optimization step.
    fn step(&mut self);
}

/// An automatically implemented extension to [`RunningOptimizer`].
pub trait RunningOptimizerExt<'a>: RunningOptimizerBase {
    /// Return the value of the best point discovered,
    /// evaluating the best point
    /// if necessary.
    fn best_point_value(&self) -> Cow<<Self::Problem as Problem>::PointValue>
    where
        <Self::Problem as Problem>::PointValue: Clone;
}

impl<'a, T> RunningOptimizerExt<'a> for T
where
    T: RunningOptimizerBase,
{
    fn best_point_value(&self) -> Cow<<Self::Problem as Problem>::PointValue>
    where
        <Self::Problem as Problem>::PointValue: Clone,
    {
        self.stored_best_point_value().map_or_else(
            || Cow::Owned(self.problem().evaluate(self.best_point())),
            Cow::Borrowed,
        )
    }
}

/// A running optimizer able to efficiently provide a view
/// of a point to be evaluated.
/// For optimizers evaluating at most one point per step.
pub trait PointBased: RunningOptimizerBase {
    /// Return point to be evaluated.
    fn point(&self) -> Option<ArrayView1<<Self::Problem as Problem>::PointElem>>;
}

/// A running optimizer able to efficiently provide a view
/// of points to be evaluated.
/// For optimizers evaluating more than one point per step.
pub trait PopulationBased: RunningOptimizerBase {
    /// Return points to be evaluated.
    fn points(&self) -> ArrayView2<<Self::Problem as Problem>::PointElem>;
}

/// A running optimizer that may be done.
/// This does *not* guarantee the optimizer *will* converge,
/// only that it *may*.
pub trait Convergent: RunningOptimizerBase {
    /// Return if optimizer is done.
    fn is_done(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! is_object_safe {
        ( $trait:ident $( < $( $bound:tt $( = $type:ty )?  ),* > )? ) => {
            paste::paste! {
                fn [< _ $trait:snake _is_object_safe >](_: &dyn $trait $( < $( $bound $( = $type )? ),* > )?) {}
            }
        }
    }

    is_object_safe!(OptimizerConfig<Problem = (), Optimizer = ()>);
    is_object_safe!(StochasticOptimizerConfig<(), Problem = (), Optimizer = ()>);
    is_object_safe!(RunningOptimizer<Problem = ()>);
    is_object_safe!(PointBased<Problem = ()>);
    is_object_safe!(Convergent<Problem = ()>);
}
