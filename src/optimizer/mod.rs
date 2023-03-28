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
pub trait OptimizerConfig<O> {
    /// Return a running optimizer.
    ///
    /// This may be nondeterministic.
    fn start(self) -> O;
}

/// An optimizer in the process of optimization.
pub trait RunningOptimizer<A> {
    /// Perform an optimization step.
    fn step(&mut self);

    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<A, Ix1>;
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

/// A running optimizer
/// able to efficiently return the value
/// of the best point discovered.
///
/// Most optimizers cannot return the best point value
/// until at least one step has been performed.
pub trait BestPointValue<A> {
    /// Return the value of the best point discovered.
    fn best_point_value(&self) -> Option<A>;
}
