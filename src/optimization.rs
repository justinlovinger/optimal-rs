//! Mathematical optimization framework.
//!
//! An optimizer configuration should remain static during operation
//! and may optionally depend on a given problem.
//! A problem and configuration
//! can be used to start a running optimizer.
//! A running optimizer has state
//! that depends on its problem
//! and configuration.

use std::{
    borrow::{Borrow, Cow},
    marker::PhantomData,
};

use blanket::blanket;
use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use streaming_iterator::StreamingIterator;

use crate::prelude::*;

#[cfg(any(test, feature = "serde"))]
use serde::{Deserialize, Serialize};

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

// TODO: maybe add a `start`-like method
// to return a `Box<dyn RunningOptimizerMethod>`.
/// Optimizer methods
/// independent of configuration.
///
/// Consider this trait sealed.
/// It should not be implemented
/// outside the package defining it.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerConfigless<P>: OptimizerProblem<P> + OptimizerArgmin<P>
where
    P: Problem,
{
}

/// Optimizer `problem`,
/// independent of configuration.
///
/// Consider this trait sealed.
/// It should not be implemented
/// outside the package defining it.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerProblem<P> {
    /// Return problem to optimize.
    fn problem(&self) -> &P;
}

/// Optimizer `argmin`,
/// independent of configuration.
///
/// Consider this trait sealed.
/// It should not be implemented
/// outside the package defining it.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait OptimizerArgmin<P>
where
    P: Problem,
{
    /// Return point that attempts to minimize the given problem.
    ///
    /// How well the point minimizes the problem
    /// depends on the optimizer.
    fn argmin(&self) -> Array1<P::PointElem>;
}

/// Running optimizer methods
/// independent of configuration
/// and state.
///
/// Consider this trait sealed.
/// It should not be implemented
/// outside the package defining it.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait RunningOptimizerConfigless<P>
where
    P: Problem,
{
    /// Return problem being optimized.
    fn problem(&self) -> &P;

    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<P::PointElem, Ix1>;

    /// Return the value of the best point discovered,
    /// evaluating the best point
    /// if necessary.
    fn best_point_value(&self) -> Cow<P::PointValue>
    where
        P::PointValue: Clone;
}

/// An optimizer
/// capable of solving a given optimization problem.
#[derive(Clone, Debug)]
#[cfg_attr(any(test, feature = "serde"), derive(Serialize, Deserialize))]
pub struct Optimizer<P, C> {
    problem: P,
    config: C,
}

/// A running optimizer.
///
/// Initial optimizer state is emitted before stepping,
/// meaning the first call to `advance` or `next` will not change state.
/// For example,
/// `nth(100)` will step `99` times,
/// returning the 100th state.
#[derive(Clone, Debug)]
#[cfg_attr(any(test, feature = "serde"), derive(Serialize, Deserialize))]
pub struct RunningOptimizer<P, C, O>
where
    C: OptimizerConfig<P>,
{
    problem: PhantomData<P>,
    config: PhantomData<C>,
    optimizer: O,
    state: C::State,
    skipped_first_step: bool,
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    evaluation_cache: OnceCell<C::Evaluation>,
}

impl<P, C> Optimizer<P, C> {
    /// Return a new optimizer
    /// if given `config` is valid
    /// for the given `problem`.
    pub fn new(problem: P, config: C) -> Result<Self, (P, C, C::Err)>
    where
        C: OptimizerConfig<P>,
    {
        match config.validate(&problem) {
            Ok(_) => Ok(Self { problem, config }),
            Err(e) => Err((problem, config, e)),
        }
    }

    /// Return optimizer configuration.
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Return problem and configuration.
    pub fn into_inner(self) -> (P, C) {
        (self.problem, self.config)
    }
}

impl<P, C> DefaultFor<P> for Optimizer<P, C>
where
    for<'a> C: DefaultFor<&'a P>,
{
    fn default_for(x: P) -> Self {
        Self {
            config: C::default_for(&x),
            problem: x,
        }
    }
}

impl<P, C> Optimizer<P, C>
where
    P: Problem,
    C: OptimizerConfig<P>,
{
    /// Return a running optimizer.
    ///
    /// This may be nondeterministic.
    pub fn start(self) -> RunningOptimizer<P, C, Optimizer<P, C>> {
        RunningOptimizer::new(self.initial_state(), self)
    }

    fn initial_state(&self) -> C::State {
        // This operation is safe
        // because `self.config` was validated
        // when `self` was constructed.
        unsafe { self.config.initial_state(&self.problem) }
    }

    /// Return a running optimizer
    /// initialized using `rng`.
    pub fn start_using<R>(self, rng: &mut R) -> RunningOptimizer<P, C, Optimizer<P, C>>
    where
        C: StochasticOptimizerConfig<P, R>,
    {
        RunningOptimizer::new(self.initial_state_using(rng), self)
    }

    fn initial_state_using<R>(&self, rng: &mut R) -> C::State
    where
        C: StochasticOptimizerConfig<P, R>,
    {
        // This operation is safe
        // because `self.config` was validated
        // when `self` was constructed.
        unsafe { self.config.initial_state_using(&self.problem, rng) }
    }

    /// Return a running optimizer
    /// if the given `state` is valid
    /// for this optimizer.
    #[allow(clippy::type_complexity)]
    pub fn start_from(
        self,
        state: C::State,
    ) -> Result<RunningOptimizer<P, C, Optimizer<P, C>>, (Self, C::StateErr)>
    where
        C::State: OptimizerState<P>,
    {
        match self.config.validate_state(&self.problem, &state) {
            Ok(_) => Ok(RunningOptimizer::new(state, self)),
            Err(e) => Err((self, e)),
        }
    }
}

impl<P, C> OptimizerConfigless<P> for Optimizer<P, C>
where
    P: Problem,
    P::PointElem: Clone,
    C: OptimizerConfig<P> + Convergent<P>,
    C::State: OptimizerState<P>,
{
}

impl<P, C> OptimizerProblem<P> for Optimizer<P, C> {
    fn problem(&self) -> &P {
        &self.problem
    }
}

impl<P, C> OptimizerArgmin<P> for Optimizer<P, C>
where
    P: Problem,
    P::PointElem: Clone,
    C: OptimizerConfig<P> + Convergent<P>,
    C::State: OptimizerState<P>,
{
    fn argmin(&self) -> Array1<P::PointElem> {
        RunningOptimizer::new(self.initial_state(), self)
            .find(|o| o.is_done())
            .expect("should converge")
            .best_point()
            .to_owned()
    }
}

impl<P, C, O> RunningOptimizer<P, C, O>
where
    C: OptimizerConfig<P>,
{
    fn new(state: C::State, optimizer: O) -> Self {
        Self {
            problem: PhantomData,
            config: PhantomData,
            optimizer,
            state,
            skipped_first_step: false,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Stop optimization run,
    /// returning problem,
    /// configuration,
    /// and state.
    pub fn into_inner(self) -> (O, C::State) {
        (self.optimizer, self.state)
    }

    /// Return state of optimizer.
    pub fn state(&self) -> &C::State {
        &self.state
    }
}

impl<P, C, O> RunningOptimizer<P, C, O>
where
    C: OptimizerConfig<P>,
    O: Borrow<Optimizer<P, C>>,
{
    /// Return optimizer configuration.
    pub fn config(&self) -> &C {
        &self.optimizer.borrow().config
    }

    /// Return if optimizer is done.
    pub fn is_done(&self) -> bool
    where
        C: Convergent<P>,
    {
        // This operation is safe
        // because `self.state` was validated
        // when constructing `self`.
        unsafe { self.optimizer.borrow().config.is_done(&self.state) }
    }
}

impl<P, C, O> RunningOptimizer<P, C, O>
where
    P: Problem,
    C: OptimizerConfig<P>,
    O: Borrow<Optimizer<P, C>>,
{
    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &C::Evaluation {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    /// Perform an optimization step.
    fn step(&mut self) {
        let evaluation = self
            .evaluation_cache
            .take()
            .unwrap_or_else(|| self.evaluate());
        replace_with::replace_with_or_abort(&mut self.state, |state| {
            // This operation is safe
            // because `self.evaluate` returns a valid evaluation of `self.state`
            // and `self.config` was validated
            // when constructing the optimizer `self` was from.
            unsafe {
                self.optimizer
                    .borrow()
                    .config
                    .step_from_evaluated(evaluation, state)
            }
        });
    }

    fn evaluate(&self) -> C::Evaluation {
        // This operation is safe
        // because `self.state` was validated
        // when constructing `self`
        // and `self.config` was validated
        // when constructing the optimizer `self` was from.
        unsafe {
            self.optimizer
                .borrow()
                .config
                .evaluate(&self.optimizer.borrow().problem, &self.state)
        }
    }
}

impl<P, C, O> RunningOptimizerConfigless<P> for RunningOptimizer<P, C, O>
where
    P: Problem,
    C: OptimizerConfig<P>,
    C::State: OptimizerState<P>,
    O: Borrow<Optimizer<P, C>>,
{
    fn problem(&self) -> &P {
        &self.optimizer.borrow().problem
    }

    fn best_point(&self) -> CowArray<P::PointElem, Ix1> {
        self.state.best_point()
    }

    fn best_point_value(&self) -> Cow<P::PointValue>
    where
        P::PointValue: Clone,
    {
        self.state.stored_best_point_value().map_or_else(
            || Cow::Owned(self.problem().evaluate(self.best_point())),
            Cow::Borrowed,
        )
    }
}

impl<P, C, O> StreamingIterator for RunningOptimizer<P, C, O>
where
    P: Problem,
    C: OptimizerConfig<P>,
    O: Borrow<Optimizer<P, C>>,
{
    type Item = Self;

    fn advance(&mut self) {
        // `advance` is called before the first `get`,
        // but we want to emit the initial state.
        if self.skipped_first_step {
            self.step()
        } else {
            self.skipped_first_step = true;
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use ndarray::prelude::*;
    use rand::prelude::*;
    use rand_xoshiro::SplitMix64;

    use crate::{optimizer::derivative_free::pbil, prelude::*};

    assert_obj_safe!(OptimizerConfigless<()>);
    assert_obj_safe!(RunningOptimizerConfigless<()>);
    assert_obj_safe!(OptimizerConfig<(), Err = (), State = (), StateErr = (), Evaluation = ()>);
    assert_obj_safe!(StochasticOptimizerConfig<(), (), Err = (), State = (), StateErr = (), Evaluation = ()>);
    assert_obj_safe!(Convergent<(), Err = (), State = (), StateErr = (), Evaluation = ()>);

    #[test]
    fn running_optimizer_streaming_iterator_emits_initial_state() {
        let seed = 0;
        let config = pbil::Pbil::default_for(Count);
        assert_eq!(
            config
                .clone()
                .start_using(&mut SplitMix64::seed_from_u64(seed))
                .next()
                .unwrap()
                .state(),
            config
                .start_using(&mut SplitMix64::seed_from_u64(seed))
                .state()
        );
    }

    #[test]
    fn running_optimizer_streaming_iterator_runs_for_same_number_of_steps() {
        let seed = 0;
        let steps = 100;
        let config = pbil::Pbil::default_for(Count);
        let mut o = config
            .clone()
            .start_using(&mut SplitMix64::seed_from_u64(seed));
        for _ in 0..steps {
            o.step();
        }
        assert_eq!(
            config
                .start_using(&mut SplitMix64::seed_from_u64(seed))
                .nth(steps)
                .unwrap()
                .state(),
            o.state()
        );
    }

    #[derive(Clone, Debug)]
    struct Count;

    impl Problem for Count {
        type PointElem = bool;
        type PointValue = u64;

        fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
            point.fold(0, |acc, b| acc + *b as u64)
        }
    }

    impl FixedLength for Count {
        fn len(&self) -> usize {
            16
        }
    }
}
