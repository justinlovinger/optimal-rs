use blanket::blanket;

use crate::prelude::*;

pub use self::extensions::*;

/// An optimizer configuration.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerConfig<P> {
    /// State this config can initialize.
    type State;

    /// Error returned when a state fails to validate.
    type StateErr;

    /// Result of evaluation.
    type Evaluation;

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
    fn initial_state(&self, problem: &P) -> Self::State;

    /// Evaluate the given state
    /// with the given problem.
    ///
    /// # Safety
    ///
    /// `state` must be valid
    /// for `self`
    /// and the given `problem`.
    unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation;

    /// Finish an optimization step
    /// given an evaluation of the current state.
    ///
    /// # Safety
    ///
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
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait StochasticOptimizerConfig<P, R>: OptimizerConfig<P> {
    /// Return a valid initial state
    /// initialized using `rng`.
    ///
    /// # Safety
    fn initial_state_using(&self, problem: &P, rng: &mut R) -> Self::State;
}

/// An optimizer state.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
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
    fn best_point(&self) -> P::Point<'_>;

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    ///
    /// Most optimizers cannot return the best point value
    /// until at least one step has been performed.
    ///
    /// If an optimizer never stores the best point value,
    /// this will always return `None`.
    fn stored_best_point_value(&self) -> Option<&P::Value> {
        None
    }
}

mod extensions {
    use super::*;

    /// Automatically implemented extensions to `OptimizerConfig`.
    pub trait OptimizerConfigExt<P>: OptimizerConfig<P> {
        /// Return this optimizer
        /// running on the given problem.
        ///
        /// This may be nondeterministic.
        fn start(self, problem: P) -> RunningOptimizer<P, Self>
        where
            Self: Sized,
        {
            RunningOptimizer::start(self, problem)
        }

        /// Return this optimizer
        /// running on the given problem.
        /// if the given `state` is valid.
        #[allow(clippy::type_complexity)]
        fn start_from(
            self,
            problem: P,
            state: Self::State,
        ) -> Result<RunningOptimizer<P, Self>, (P, Self, Self::State, Self::StateErr)>
        where
            Self: Sized,
        {
            RunningOptimizer::start_from(self, problem, state)
        }

        /// Return this optimizer default
        /// running on the given problem.
        fn start_default_for(problem: P) -> RunningOptimizer<P, Self>
        where
            for<'a> Self: Sized + DefaultFor<&'a P>,
        {
            RunningOptimizer::default_for(problem)
        }
    }

    impl<P, T> OptimizerConfigExt<P> for T where T: OptimizerConfig<P> {}

    /// Automatically implemented extensions to `StochasticOptimizerConfig`.
    pub trait StochasticOptimizerConfigExt<P, R>: StochasticOptimizerConfig<P, R> {
        /// Return this optimizer
        /// running on the given problem
        /// initialized using `rng`.
        fn start_using(self, problem: P, rng: &mut R) -> RunningOptimizer<P, Self>
        where
            Self: Sized,
        {
            RunningOptimizer::start_using(self, problem, rng)
        }
    }

    impl<P, R, T> StochasticOptimizerConfigExt<P, R> for T where T: StochasticOptimizerConfig<P, R> {}
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use super::*;

    assert_obj_safe!(OptimizerConfig<(), State = (), StateErr = (), Evaluation = ()>);
    assert_obj_safe!(StochasticOptimizerConfig<(), (), State = (), StateErr = (), Evaluation = ()>);
}
