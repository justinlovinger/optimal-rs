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

/// An optimizer configuration
/// for an optimizer
/// requiring a source of randomness.
pub trait StochasticOptimizerConfig<O> {
    /// Return a running optimizer
    /// initialized using `rng`.
    fn start_using<R>(self, rng: &mut R) -> O
    where
        R: Rng;
}

/// An optimizer configuration.
pub trait OptimizerConfig<'a, O, P> {
    /// Return a running optimizer.
    ///
    /// This may be nondeterministic.
    fn start(self) -> O;

    /// Return problem to optimize.
    fn problem(&'a self) -> &'a P;
}

/// An optimizer in the process of optimization.
pub trait RunningOptimizer<A, B, C, S> {
    /// Perform an optimization step.
    fn step(&mut self);

    /// Return state of optimizer.
    fn state(&self) -> &S;

    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<A, Ix1>;

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    ///
    /// Most optimizers cannot return the best point value
    /// until at least one step has been performed.
    ///
    /// If an optimizer never stores the best point value,
    /// this will always return `None`.
    fn stored_best_point_value(&self) -> Option<B>;

    /// Return optimizer configuration.
    fn config(&self) -> &C;
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
    P: Problem<A, B> + 'a,
    C: OptimizerConfig<'a, T, P> + 'a,
    T: RunningOptimizer<A, B, C, S>,
{
    fn best_point_value(&'a self) -> B {
        self.stored_best_point_value()
            .unwrap_or_else(|| self.problem().evaluate(self.best_point()))
    }

    fn problem(&'a self) -> &'a P {
        self.config().problem()
    }
}

/// A running optimizer able to efficiently provide a view
/// of a point to be evaluated.
/// For optimizers evaluating at most one point per step.
pub trait PointBased<A> {
    /// Return point to be evaluated.
    fn point(&self) -> Option<ArrayView1<A>>;
}

/// A running optimizer able to efficiently provide a view
/// of points to be evaluated.
/// For optimizers evaluating more than one point per step.
pub trait PopulationBased<A> {
    /// Return points to be evaluated.
    fn points(&self) -> ArrayView2<A>;
}

/// A running optimizer that may be done.
/// This does *not* guarantee the optimizer *will* converge,
/// only that it *may*.
pub trait Convergent {
    /// Return if optimizer is done.
    fn is_done(&self) -> bool;
}
