use ndarray::prelude::*;

use crate::prelude::*;

/// An optimizer configuration.
// TODO: use `blanket` when <https://github.com/althonos/blanket/issues/8> is fixed:
// #[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerConfig<P> {
    /// Error returned when this configuration fails to validate.
    type Err;

    /// State this config can initialize.
    type State;

    /// Error returned when a state fails to validate.
    type StateErr;

    /// Result of evaluation.
    type Evaluation;

    // TODO: this method may be unnecessary.
    // Configuration parameters can be validated when bulding a config.
    // This method is only necessary
    // if config parameters depend on problem.
    /// Return whether `self` is valid
    /// for the given `problem`.
    fn validate(&self, problem: &P) -> Result<(), Self::Err>;

    // TODO: we may be able to remove this method
    // and replace it with something like
    // `state.point().len() == problem.len()`
    // in `Optimizer`.
    // However,
    // it will also need to check whether all points making up a state
    // are within bounds.
    /// Return whether `state` is valid
    /// for `self`
    /// and the given `problem`.
    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr>;

    /// Return a valid initial state.
    ///
    /// This may be nondeterministic.
    ///
    /// # Safety
    ///
    /// `self` must be valid
    /// for the given `problem`.
    unsafe fn initial_state(&self, problem: &P) -> Self::State;

    /// Evaluate the given state
    /// with the given problem.
    ///
    /// # Safety
    ///
    /// `self` must be valid
    /// for the given `problem`,
    /// and `state` must be valid
    /// for `self`
    /// and the given `problem`.
    unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation;

    /// Finish an optimization step
    /// given an evaluation of the current state.
    ///
    /// # Safety
    ///
    /// `self` must be valid
    /// for the given `problem`,
    /// `state` must be valid
    /// for `self`,
    /// and `evaluation` must be from `self`
    /// and the given `state`.
    unsafe fn step_from_evaluated(
        &self,
        evaluation: Self::Evaluation,
        state: Self::State,
    ) -> Self::State;
}

/// An optimizer configuration
/// for an optimizer
/// requiring a source of randomness.
// TODO: use `blanket` when <https://github.com/althonos/blanket/issues/8> is fixed:
// #[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait StochasticOptimizerConfig<P, R>: OptimizerConfig<P> {
    /// Return a valid initial state
    /// initialized using `rng`.
    ///
    /// # Safety
    ///
    /// `self` must be valid
    /// for the given `problem`.
    unsafe fn initial_state_using(&self, problem: &P, rng: &mut R) -> Self::State;
}

/// A config for an optimizer that may be done.
/// This does *not* guarantee the optimizer *will* converge,
/// only that it *may*.
// TODO: use `blanket` when <https://github.com/althonos/blanket/issues/8> is fixed:
// #[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Convergent<P>: OptimizerConfig<P> {
    /// Return if optimizer is done.
    ///
    /// # Safety
    ///
    /// `state` must be valid
    /// for `self`.
    unsafe fn is_done(&self, state: &Self::State) -> bool;
}

/// An optimizer state.
// TODO: use `blanket` when <https://github.com/althonos/blanket/issues/8> is fixed
// and can support associated type generics:
// #[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerState<P>
where
    P: Problem,
{
    /// Data to be evaluated.
    type Evaluatee<'a>
    where
        Self: 'a;

    /// Return data to be evaluated.
    fn evaluatee(&self) -> Self::Evaluatee<'_>;

    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<P::PointElem, Ix1>;

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    ///
    /// Most optimizers cannot return the best point value
    /// until at least one step has been performed.
    ///
    /// If an optimizer never stores the best point value,
    /// this will always return `None`.
    fn stored_best_point_value(&self) -> Option<&P::PointValue> {
        None
    }
}
