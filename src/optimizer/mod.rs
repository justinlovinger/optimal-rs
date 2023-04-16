//! Mathematical optimization framework.
//!
//! An optimizer is split between a static configuration
//! and a dynamic state.
//! Traits are defined on the configuration
//! and may take a state.

pub mod derivative;
pub mod derivative_free;
mod iterator;

use ndarray::prelude::*;
use rand::Rng;

use crate::prelude::Problem;

pub use self::iterator::*;

/// An optimizer configuration.
pub trait OptimizerConfig {
    /// Problem to optimize.
    type Problem;

    /// Return problem to optimize.
    fn problem(&self) -> &Self::Problem;
}

/// An optimizer in the process of optimization.
pub trait RunningOptimizer {
    /// Elements in points.
    type PointElem;

    /// Value returned by problem
    /// when point is evaluated.
    type PointValue;

    /// Configuration for this optimizer.
    type Config;

    /// State of this optimizer.
    type State;

    /// Initialize this optimizer using the given configuration.
    ///
    /// This may be nondeterministic.
    fn new(config: Self::Config) -> Self;

    /// Perform an optimization step.
    fn step(&mut self);

    /// Return optimizer configuration.
    fn config(&self) -> &Self::Config;

    /// Return state of optimizer.
    fn state(&self) -> &Self::State;

    /// Stop optimization run,
    /// returning configuration and state.
    fn stop(self) -> (Self::Config, Self::State);

    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<Self::PointElem, Ix1>;

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    ///
    /// Most optimizers cannot return the best point value
    /// until at least one step has been performed.
    ///
    /// If an optimizer never stores the best point value,
    /// this will always return `None`.
    fn stored_best_point_value(&self) -> Option<&Self::PointValue>;
}

/// An optimizer
/// requiring a source of randomness
/// to initialize.
pub trait StochasticRunningOptimizer: RunningOptimizer {
    /// Initialize this optimizer
    /// using `rng`.
    fn new_using<R>(config: Self::Config, rng: &mut R) -> Self
    where
        R: Rng;
}

/// An automatically implemented extension to [`RunningOptimizer`].
pub trait RunningOptimizerExt<'a, A, B, P, C, S> {
    /// Return the value of the best point discovered,
    /// evaluating the best point
    /// if necessary.
    fn best_point_value(&'a self) -> B;

    /// Return problem to optimize.
    fn problem(&'a self) -> &'a P;
}

impl<'a, A, B, P, C, S, T> RunningOptimizerExt<'a, A, B, P, C, S> for T
where
    B: Clone,
    P: Problem<PointElem = A, PointValue = B> + 'a,
    C: OptimizerConfig<Problem = P> + 'a,
    T: RunningOptimizer<PointElem = A, PointValue = B, Config = C, State = S>,
{
    fn best_point_value(&'a self) -> B {
        self.stored_best_point_value()
            .map_or_else(|| self.problem().evaluate(self.best_point()), |x| x.clone())
    }

    fn problem(&'a self) -> &'a P {
        self.config().problem()
    }
}

/// A running optimizer able to efficiently provide a view
/// of a point to be evaluated.
/// For optimizers evaluating at most one point per step.
pub trait PointBased: RunningOptimizer {
    /// Return point to be evaluated.
    fn point(&self) -> Option<ArrayView1<Self::PointElem>>;
}

/// A running optimizer able to efficiently provide a view
/// of points to be evaluated.
/// For optimizers evaluating more than one point per step.
pub trait PopulationBased: RunningOptimizer {
    /// Return points to be evaluated.
    fn points(&self) -> ArrayView2<Self::PointElem>;
}

/// A running optimizer that may be done.
/// This does *not* guarantee the optimizer *will* converge,
/// only that it *may*.
pub trait Convergent: RunningOptimizer {
    /// Return if optimizer is done.
    fn is_done(&self) -> bool;
}
