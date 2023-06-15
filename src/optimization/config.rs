use ndarray::prelude::*;

use crate::prelude::*;

pub use self::extensions::*;

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

mod extensions {
    use super::*;

    /// Error returned when `start_from` is given an invalid problem or state.
    #[derive(Clone, Debug, thiserror::Error, PartialEq)]
    pub enum StartFromError<P, S> {
        /// Error returned when `start_from` is given an invalid problem.
        ProblemError(P),
        /// Error returned when `start_from` is given an invalid state.
        StateError(S),
    }

    /// Automatically implemented extensions to `OptimizerConfig`.
    pub trait OptimizerConfigExt<P>: OptimizerConfig<P> {
        /// Return this optimizer
        /// running on the given problem.
        ///
        /// This may be nondeterministic.
        #[allow(clippy::type_complexity)]
        fn start(
            self,
            problem: P,
        ) -> Result<RunningOptimizer<P, Self, Optimizer<P, Self>>, (P, Self, Self::Err)>
        where
            P: Problem,
            Self: Sized,
        {
            Optimizer::new(problem, self).map(|x| x.start())
        }

        /// Return this optimizer
        /// running on the given problem.
        /// if the given `state` is valid.
        #[allow(clippy::type_complexity)]
        fn start_from(
            self,
            problem: P,
            state: Self::State,
        ) -> Result<
            RunningOptimizer<P, Self, Optimizer<P, Self>>,
            (
                P,
                Self,
                Self::State,
                StartFromError<Self::Err, Self::StateErr>,
            ),
        >
        where
            P: Problem,
            Self: Sized,
        {
            match Optimizer::new(problem, self) {
                Ok(x) => x.start_from(state).map_err(|(o, s, e)| {
                    let (p, c) = o.into_inner();
                    (p, c, s, StartFromError::StateError(e))
                }),
                Err((p, c, e)) => Err((p, c, state, StartFromError::ProblemError(e))),
            }
        }

        /// Return this optimizer default
        /// running on the given problem.
        fn start_default_for(problem: P) -> RunningOptimizer<P, Self, Optimizer<P, Self>>
        where
            P: Problem,
            for<'a> Self: Sized + DefaultFor<&'a P>,
        {
            Optimizer::default_for(problem).start()
        }
    }

    impl<P, T> OptimizerConfigExt<P> for T where T: OptimizerConfig<P> {}

    /// Automatically implemented extensions to `StochasticOptimizerConfig`.
    pub trait StochasticOptimizerConfigExt<P, R>: StochasticOptimizerConfig<P, R> {
        /// Return this optimizer
        /// running on the given problem
        /// initialized using `rng`.
        #[allow(clippy::type_complexity)]
        fn start_using(
            self,
            problem: P,
            rng: &mut R,
        ) -> Result<RunningOptimizer<P, Self, Optimizer<P, Self>>, (P, Self, Self::Err)>
        where
            P: Problem,
            Self: Sized,
        {
            Optimizer::new(problem, self).map(|x| x.start_using(rng))
        }
    }

    impl<P, R, T> StochasticOptimizerConfigExt<P, R> for T where T: StochasticOptimizerConfig<P, R> {}
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use super::*;

    assert_obj_safe!(OptimizerConfig<(), Err = (), State = (), StateErr = (), Evaluation = ()>);
    assert_obj_safe!(StochasticOptimizerConfig<(), (), Err = (), State = (), StateErr = (), Evaluation = ()>);
}
