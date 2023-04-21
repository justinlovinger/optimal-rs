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

use crate::prelude::Problem;

pub use self::iterator::*;

/// An optimizer configuration.
// #[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerConfig {
    /// Problem to optimize.
    type Problem;

    /// Optimizer this config can initialize.
    type Optimizer;

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

/// A fully defined optimizer.
///
/// A type implementing this
/// should also implement one of
/// `PointBased`
/// or `PopulationBased`.
pub trait Optimizer: OptimizerBase + OptimizerStep + OptimizerDeinitialization {}

impl<T> Optimizer for T where T: OptimizerBase + OptimizerStep + OptimizerDeinitialization {}

/// Optimizer methods requiring only a reference.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerBase {
    /// Problem to optimize.
    type Problem: Problem;

    /// Configuration for this optimizer.
    type Config;

    /// State of this optimizer.
    type State;

    /// Return optimizer configuration.
    fn config(&self) -> &Self::Config;

    /// Return state of optimizer.
    fn state(&self) -> &Self::State;

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

/// Step methods of an optimizer.
#[blanket(derive(Mut, Box))]
pub trait OptimizerStep: OptimizerBase {
    /// Perform an optimization step.
    fn step(&mut self);
}

/// Standard deinitialization of an optimizer.
#[blanket(derive(Box))]
pub trait OptimizerDeinitialization: OptimizerBase {
    /// Stop optimization run,
    /// returning configuration and state.
    fn stop(self) -> (Self::Config, Self::State);
}

/// An automatically implemented extension to [`RunningOptimizer`].
pub trait RunningOptimizerExt<'a>: OptimizerBase {
    /// Return the value of the best point discovered,
    /// evaluating the best point
    /// if necessary.
    fn best_point_value(&self) -> Cow<<Self::Problem as Problem>::PointValue>
    where
        <Self::Problem as Problem>::PointValue: Clone;

    /// Return problem to optimize.
    fn problem(&'a self) -> &'a Self::Problem;
}

impl<'a, T> RunningOptimizerExt<'a> for T
where
    T: OptimizerBase,
    Self::Config: OptimizerConfig<Problem = Self::Problem> + 'a,
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

    fn problem(&'a self) -> &'a Self::Problem {
        self.config().problem()
    }
}

/// A running optimizer able to efficiently provide a view
/// of a point to be evaluated.
/// For optimizers evaluating at most one point per step.
pub trait PointBased: OptimizerBase {
    /// Return point to be evaluated.
    fn point(&self) -> Option<ArrayView1<<Self::Problem as Problem>::PointElem>>;
}

/// A running optimizer able to efficiently provide a view
/// of points to be evaluated.
/// For optimizers evaluating more than one point per step.
pub trait PopulationBased: OptimizerBase {
    /// Return points to be evaluated.
    fn points(&self) -> ArrayView2<<Self::Problem as Problem>::PointElem>;
}

/// A running optimizer that may be done.
/// This does *not* guarantee the optimizer *will* converge,
/// only that it *may*.
pub trait Convergent: OptimizerBase {
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
    is_object_safe!(Optimizer<Problem = (), Config = (), State = ()>);
    is_object_safe!(PointBased<Problem = (), Config = (), State = ()>);
    is_object_safe!(Convergent<Problem = (), Config = (), State = ()>);
}
