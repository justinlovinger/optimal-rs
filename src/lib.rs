#![allow(clippy::needless_doctest_main)]

//! Mathematical optimization and machine learning framework
//! and algorithms.
//!
//! Optimal provides a composable framework
//! for mathematical optimization
//! and machine learning
//! from the optimization perspective,
//! in addition to algorithm implementations.
//!
//! The framework consists of runners,
//! optimizers,
//! and problems,
//! with a chain of dependency as follows:
//! `runner -> optimizer -> problem`.
//! Most optimizers can support many problems
//! and most runners can support many optimizers.
//!
//! A problem defines a mathematical optimization problem.
//! An optimizer defines the steps for solving a problem,
//! usually as an infinite series of state transitions
//! incrementally improving a solution.
//! A runner defines the stopping criteria for an optimizer
//! and may affect the optimization sequence
//! in other ways.
//!
//! # Examples
//!
//! Minimize the `Count` problem
//! using a PBIL optimizer:
//!
//! ```
//! use ndarray::prelude::*;
//! use optimal::{optimizer::derivative_free::pbil, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut o = pbil::Config::start_default_for(Count);
//!     let point = pbil::UntilConvergedConfig::default().argmin(&mut o);
//!     let point_value = Count.evaluate(point.view().into());
//!     println!("f({point}) = {point_value}");
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type Point<'a> = CowArray<'a, bool, Ix1>;
//!     type Value = u64;
//!
//!     fn evaluate(&self, point: Self::Point<'_>) -> Self::Value {
//!         point.fold(0, |acc, b| acc + *b as u64)
//!     }
//! }
//!
//! impl FixedLength for Count {
//!     fn len(&self) -> usize {
//!         16
//!     }
//! }
//! ```
//!
//! Minimize a problem
//! one step at a time:
//!
//! ```
//! # use ndarray::prelude::*;
//! # use optimal::{optimizer::derivative_free::pbil, prelude::*};
//! # use streaming_iterator::StreamingIterator;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Count;
//! #
//! # impl Problem for Count {
//! #     type Point<'a> = CowArray<'a, bool, Ix1>;
//! #     type Value = u64;
//! #
//! #     fn evaluate(&self, point: Self::Point<'_>) -> Self::Value {
//! #         point.fold(0, |acc, b| acc + *b as u64)
//! #     }
//! # }
//! #
//! # impl FixedLength for Count {
//! #     fn len(&self) -> usize {
//! #         16
//! #     }
//! # }
//! #
//! let mut it = pbil::UntilConvergedConfig::default().start(pbil::Config::start_default_for(Count));
//! while let Some(o) = it.next() {
//!     println!("{:?}", o.state());
//! }
//! let o = it.into_inner().0;
//! println!("f({}) = {}", o.best_point(), o.best_point_value());
//! ```

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

mod derive;
mod optimization;
pub mod optimizer;
pub mod prelude;
mod problem;
mod traits;

#[cfg(test)]
mod tests {
    // These tests checks API flexibility
    // and usability,
    // not implementation details.
    // As such,
    // whether or not the desired use-case can be expressed,
    // and the code compiles,
    // is more important
    // than particular values.

    use std::fmt::Debug;

    use replace_with::replace_with_or_abort;
    use serde::{Deserialize, Serialize};

    use super::prelude::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockProblem;

    impl Problem for MockProblem {
        type Point<'a> = usize;
        type Value = usize;

        fn evaluate<'a>(&'a self, point: Self::Point<'a>) -> Self::Value {
            point + 1
        }
    }

    macro_rules! mock_optimizer {
        ( $id:ident ) => {
            paste::paste! {
                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockConfig $id >];

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockState $id >](usize);

                impl<P> OptimizerConfig<P> for [< MockConfig $id >]
                where
                    for<'a> P: Problem<Point<'a> = usize, Value = usize> + 'a,
                {
                    type State = [< MockState $id >];
                    type StateErr = ();
                    type Evaluation = P::Value;

                    fn validate_state(&self, _problem: &P, _state: &Self::State) -> Result<(), Self::StateErr> {
                        Ok(())
                    }

                    fn initial_state(&self, _problem: &P) -> Self::State {
                        [< MockState $id >](1)
                    }

                    unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation {
                        problem.evaluate(state.0)
                    }

                    unsafe fn step_from_evaluated(&self, evaluation: Self::Evaluation, mut state: Self::State) -> Self::State {
                        state.0 += evaluation;
                        state
                    }
                }

                impl<P> DefaultFor<P> for [< MockConfig $id >] {
                    fn default_for(_problem: P) -> Self {
                        Self
                    }
                }

                impl<P> OptimizerState<P> for [< MockState $id >]
                where
                    for<'a> P: Problem<Point<'a> = usize> + 'a,
                {
                    type Evaluatee<'a> = ();

                    fn evaluatee(&self) -> Self::Evaluatee<'_> {}

                    fn best_point(&self) -> P::Point<'_> {
                        self.0
                    }

                    fn stored_best_point_value(&self) -> Option<&P::Value> {
                        None
                    }
                }
            }
        };
    }

    mock_optimizer!(A);
    mock_optimizer!(B);

    struct MaxStepsConfig(usize);

    struct MaxStepsState(usize);

    impl<I> RunnerConfig<I> for MaxStepsConfig {
        type State = MaxStepsState;

        fn initial_state(&self) -> Self::State {
            MaxStepsState(0)
        }

        fn is_done(&self, _it: &I, state: &Self::State) -> bool {
            state.0 >= self.0
        }

        fn update(&self, _it: &I, state: &mut Self::State) {
            state.0 += 1;
        }
    }

    #[test]
    fn optimizers_should_be_easily_comparable() {
        type BoxedOptimizer<P> =
            Box<dyn StreamingIterator<Item = dyn RunningOptimizerConfigless<P>>>;

        fn best_optimizer<P, I>(optimizers: I) -> usize
        where
            P: Problem,
            P::Value: Ord,
            I: IntoIterator<Item = BoxedOptimizer<P>>,
        {
            optimizers
                .into_iter()
                .enumerate()
                .map(|(i, mut o)| {
                    let o = o.nth(10).unwrap();
                    (o.problem().evaluate(o.best_point()), i)
                })
                .min()
                .expect("`optimizers` should be non-empty")
                .1
        }

        best_optimizer([
            Box::new(
                MockConfigA::start_default_for(MockProblem)
                    .map_ref(|x| x as &dyn RunningOptimizerConfigless<MockProblem>),
            ) as BoxedOptimizer<MockProblem>,
            Box::new(
                MockConfigB::start_default_for(MockProblem)
                    .map_ref(|x| x as &dyn RunningOptimizerConfigless<MockProblem>),
            ) as BoxedOptimizer<MockProblem>,
        ]);
    }

    #[test]
    fn parallel_optimization_runs_should_be_easy() {
        use std::{sync::Arc, thread::spawn};

        fn parallel<P, C, A>(problem: P, config: C)
        where
            P: Problem + Clone + Debug + Send + Sync + 'static,
            for<'a> P::Point<'a>: ToOwned<Owned = A>,
            A: Clone + Send + 'static,
            C: OptimizerConfig<P> + Clone + Debug + Send + Sync + 'static,
            C::State: OptimizerState<P>,
        {
            let problem2 = problem.clone();
            let config2 = config.clone();
            let handler1 = spawn(move || {
                #[allow(clippy::redundant_clone)] // False positive.
                MaxStepsConfig(10)
                    .argmin(&mut config2.start(problem2))
                    .to_owned()
            });
            let handler2 = spawn(move || {
                #[allow(clippy::redundant_clone)] // False positive.
                MaxStepsConfig(10)
                    .argmin(&mut config.start(problem))
                    .to_owned()
            });
            handler1.join().unwrap();
            handler2.join().unwrap();
        }

        parallel(
            Arc::new(MockProblem),
            Arc::new(MockConfigA::default_for(MockProblem)),
        );
    }

    #[test]
    fn examining_state_and_corresponding_evaluations_should_be_easy() {
        MockConfigA::start_default_for(MockProblem)
            .inspect(|o| println!("state: {:?}, evaluation: {:?}", o.state(), o.evaluation()))
            .nth(10);
    }

    #[test]
    fn optimizers_should_be_able_to_restart_automatically() {
        // This is a partial implementation
        // of a restart mixin,
        // missing best point tracking.

        trait Restart {
            fn restart(&mut self);
        }

        impl<P, C> Restart for RunningOptimizer<P, C>
        where
            P: Problem,
            C: OptimizerConfig<P>,
        {
            fn restart(&mut self) {
                replace_with_or_abort(self, |o| {
                    let (c, p, _) = o.into_inner();
                    c.start(p)
                })
            }
        }

        impl<I, C> Restart for Runner<I, C>
        where
            I: Restart,
            C: RunnerConfig<I>,
        {
            fn restart(&mut self) {
                replace_with_or_abort(self, |x| {
                    let (mut it, c, _) = x.into_inner();
                    it.restart();
                    c.start(it)
                })
            }
        }

        struct RestarterConfig {
            max_restarts: usize,
        }

        struct RestarterState {
            restarts: usize,
        }

        impl<I, C> RunnerConfig<Runner<I, C>> for RestarterConfig
        where
            I: Restart,
            C: RunnerConfig<I>,
        {
            type State = RestarterState;

            fn initial_state(&self) -> Self::State {
                RestarterState { restarts: 0 }
            }

            fn is_done(&self, _it: &Runner<I, C>, state: &Self::State) -> bool {
                state.restarts >= self.max_restarts
            }

            fn advance(&self, it: &mut Runner<I, C>, state: &mut Self::State)
            where
                Runner<I, C>: StreamingIterator,
            {
                if it.is_done() {
                    state.restarts += 1;
                    it.restart();
                } else {
                    it.advance()
                }
            }
        }

        let _ = RestarterConfig { max_restarts: 10 }
            .start(MaxStepsConfig(10).start(MockConfigA::start_default_for(MockProblem)))
            .nth(100);
    }

    // Applications may need to select an optimizer at runtime,
    // run it for less than a full optimization,
    // save the partial run,
    // and resume it later.
    #[test]
    fn dynamic_optimizers_should_be_partially_runable() {
        use std::hint::unreachable_unchecked;

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockConfig {
            A(MockConfigA),
            B(MockConfigB),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockState {
            A(MockStateA),
            B(MockStateB),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockStateError<P>
        where
            for<'a> P: Problem<Point<'a> = usize, Value = usize> + 'a,
        {
            WrongState,
            A(<MockConfigA as OptimizerConfig<P>>::StateErr),
            B(<MockConfigB as OptimizerConfig<P>>::StateErr),
        }

        enum MockEvaluation<P>
        where
            for<'a> P: Problem<Point<'a> = usize, Value = usize> + 'a,
        {
            A(<MockConfigA as OptimizerConfig<P>>::Evaluation),
            B(<MockConfigB as OptimizerConfig<P>>::Evaluation),
        }

        impl<P> OptimizerConfig<P> for MockConfig
        where
            for<'a> P: Problem<Point<'a> = usize, Value = usize> + 'a,
        {
            type State = MockState;
            type StateErr = MockStateError<P>;
            type Evaluation = MockEvaluation<P>;

            fn validate_state(
                &self,
                problem: &P,
                state: &Self::State,
            ) -> Result<(), Self::StateErr> {
                match self {
                    Self::A(c) => match state {
                        MockState::A(s) => c.validate_state(problem, s).map_err(MockStateError::A),
                        _ => Err(MockStateError::WrongState),
                    },
                    Self::B(c) => match state {
                        MockState::B(s) => c.validate_state(problem, s).map_err(MockStateError::B),
                        _ => Err(MockStateError::WrongState),
                    },
                }
            }

            fn initial_state(&self, problem: &P) -> Self::State {
                // `initial_state` is safe if this method is safe,
                // because the inner config was validated
                // when `self` was validated.
                match self {
                    Self::A(c) => MockState::A(c.initial_state(problem)),
                    Self::B(c) => MockState::B(c.initial_state(problem)),
                }
            }

            unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation {
                // `evaluate` is safe if this method is safe,
                // because the inner config was validated
                // when `self` was validated
                // and inner state was validated
                // when `state` was validated.
                // `unreachable_unchecked` is safe if this method is safe
                // because state was verified to match `self`
                // in `validate_state`.
                match self {
                    Self::A(c) => match state {
                        #[allow(clippy::unit_arg)]
                        MockState::A(s) => MockEvaluation::A(unsafe { c.evaluate(problem, s) }),
                        _ => unsafe { unreachable_unchecked() },
                    },
                    Self::B(c) => match state {
                        #[allow(clippy::unit_arg)]
                        MockState::B(s) => MockEvaluation::B(unsafe { c.evaluate(problem, s) }),
                        _ => unsafe { unreachable_unchecked() },
                    },
                }
            }

            unsafe fn step_from_evaluated(
                &self,
                evaluation: Self::Evaluation,
                state: Self::State,
            ) -> Self::State {
                // `step_from_evaluated` is safe if this method is safe,
                // because the inner config was validated
                // when `self` was validated,
                // `evaluation` came from `state.inner`,
                // and inner state was validated
                // when `state` was validated.
                // `unreachable_unchecked` is safe if this method is safe
                // because `state` was verified to match `self`
                // in `validate_state`
                match self {
                    Self::A(c) => match (evaluation, state) {
                        (MockEvaluation::A(e), MockState::A(s)) => MockState::A(unsafe {
                            <MockConfigA as OptimizerConfig<P>>::step_from_evaluated(c, e, s)
                        }),
                        _ => unsafe { unreachable_unchecked() },
                    },
                    Self::B(c) => match (evaluation, state) {
                        (MockEvaluation::B(e), MockState::B(s)) => MockState::B(unsafe {
                            <MockConfigB as OptimizerConfig<P>>::step_from_evaluated(c, e, s)
                        }),
                        _ => unsafe { unreachable_unchecked() },
                    },
                }
            }
        }

        impl<P> OptimizerState<P> for MockState
        where
            P: Problem,
            for<'a> MockStateA: OptimizerState<P, Evaluatee<'a> = ()>,
            for<'a> MockStateB: OptimizerState<P, Evaluatee<'a> = ()>,
        {
            type Evaluatee<'a> = ();

            fn evaluatee(&self) -> Self::Evaluatee<'_> {
                match self {
                    Self::A(x) => x.evaluatee(),
                    Self::B(x) => x.evaluatee(),
                }
            }

            fn best_point(&self) -> P::Point<'_> {
                match self {
                    Self::A(x) => x.best_point(),
                    Self::B(x) => x.best_point(),
                }
            }

            fn stored_best_point_value(&self) -> Option<&P::Value> {
                match self {
                    Self::A(x) => x.stored_best_point_value(),
                    Self::B(x) => x.stored_best_point_value(),
                }
            }
        }

        let mut o = MockConfig::A(MockConfigA).start(MockProblem);
        o.next();
        let store = serde_json::to_string(&o).unwrap();
        o = serde_json::from_str(&store).unwrap();
        o.next();
        o.best_point_value();
    }
}
