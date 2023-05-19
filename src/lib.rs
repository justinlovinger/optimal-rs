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
//!     let point = pbil::PbilDoneWhenConverged::default_for(Count).argmin();
//!     let point_value = Count.evaluate(point.view().into());
//!     println!("f({}) = {}", point, point_value);
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type PointElem = bool;
//!     type PointValue = u64;
//!
//!     fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
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
//! #     type PointElem = bool;
//! #     type PointValue = u64;
//! #
//! #     fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
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
//! let mut o = pbil::PbilDoneWhenConverged::default_for(Count)
//!     .start()
//!     .inspect(|o| println!("{:?}", o.state()));
//! let last = o
//!     .find(|o| o.is_done())
//!     .expect("should converge");
//! println!("f({}) = {}", last.best_point(), last.best_point_value());
//! ```

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

mod derive;
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

    use ndarray::prelude::*;
    use num_traits::Zero;
    use serde::{Deserialize, Serialize};

    use super::prelude::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockProblem;

    impl Problem for MockProblem {
        type PointElem = usize;
        type PointValue = usize;
        fn evaluate(&self, _point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
            0
        }
    }

    macro_rules! mock_optimizer {
        ( $id:ident ) => {
            paste::paste! {
                type [< MockOptimizer $id >]<P> = Optimizer<P, [< MockConfig $id >]>;

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockConfig $id >];

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockState $id >];

                impl<P> OptimizerConfig<P> for [< MockConfig $id >] {
                    type Err = ();
                    type State = [< MockState $id >];
                    type StateErr = ();
                    type Evaluation = ();

                    fn validate(&self, _problem: &P) -> Result<(), Self::Err> {
                        Ok(())
                    }

                    fn validate_state(&self, _problem: &P, _state: &Self::State) -> Result<(), Self::StateErr> {
                        Ok(())
                    }

                    unsafe fn initial_state(&self, _problem: &P) -> Self::State {
                        [< MockState $id >]
                    }

                    unsafe fn evaluate(&self, _problem: &P, _state: &Self::State) -> Self::Evaluation {}

                    unsafe fn step_from_evaluated(&self, _evaluation: Self::Evaluation, _state: Self::State) -> Self::State {
                        [< MockState $id >]
                    }
                }

                impl<P> DefaultFor<&P> for [< MockConfig $id >] {
                    fn default_for(_problem: &P) -> Self {
                        Self
                    }
                }

                impl<P> Convergent<P> for [< MockConfig $id >] {
                    fn is_done(&self, _state: &Self::State) -> bool {
                        true
                    }
                }

                impl<P> OptimizerState<P> for [< MockState $id >]
                where
                    P: Problem,
                    P::PointElem: Clone + Zero,
                {
                    type Evaluatee<'a> = ();

                    fn evaluatee(&self) -> Self::Evaluatee<'_> {}

                    fn best_point(&self) -> CowArray<P::PointElem, Ix1> {
                        Array::from_elem(1, P::PointElem::zero()).into()
                    }

                    fn stored_best_point_value(&self) -> Option<&P::PointValue> {
                        None
                    }
                }
            }
        };
    }

    mock_optimizer!(A);
    mock_optimizer!(B);

    #[test]
    fn optimizers_should_be_easily_comparable() {
        fn best_optimizer<P, I>(optimizers: I) -> usize
        where
            P: Problem,
            P::PointValue: Ord,
            I: IntoIterator<Item = Box<dyn OptimizerConfigless<P>>>,
        {
            optimizers
                .into_iter()
                .enumerate()
                .map(|(i, o)| (o.problem().evaluate(o.argmin().into()), i))
                .min()
                .expect("`optimizers` should be non-empty")
                .1
        }

        best_optimizer([
            Box::new(MockOptimizerA::default_for(MockProblem))
                as Box<dyn OptimizerConfigless<MockProblem>>,
            Box::new(MockOptimizerB::default_for(MockProblem)),
        ]);
    }

    #[test]
    fn parallel_optimization_runs_should_be_easy() {
        use std::{sync::Arc, thread::spawn};

        fn parallel<P, C>(optimizer: Arc<Optimizer<P, C>>)
        where
            P: Problem + Send + Sync + 'static,
            P::PointElem: Clone + Send,
            C: OptimizerConfig<P> + Convergent<P> + Send + Sync + 'static,
            C::State: OptimizerState<P>,
        {
            let optimizer2 = Arc::clone(&optimizer);
            let handler1 = spawn(move || optimizer2.argmin());
            let handler2 = spawn(move || optimizer.argmin());
            handler1.join().unwrap();
            handler2.join().unwrap();
        }

        parallel(Arc::new(MockOptimizerA::default_for(MockProblem)));
    }

    #[test]
    fn examining_state_and_corresponding_evaluations_should_be_easy() {
        MockOptimizerA::default_for(MockProblem)
            .start()
            .inspect(|o| println!("state: {:?}, evaluation: {:?}", o.state(), o.evaluation()))
            .find(|o| o.is_done());
    }

    #[test]
    fn optimizers_should_be_able_to_restart_automatically() {
        // This is a partial implementation
        // of a restart mixin,
        // missing best point tracking
        // and a `Convergent` implementation.

        use std::marker::PhantomData;

        #[derive(Clone, Debug)]
        struct RestarterConfig<C> {
            inner: C,
            _max_restarts: usize,
        }

        #[derive(Clone, Debug)]
        struct RestarterState<P, S>
        where
            P: Problem,
        {
            inner: S,
            restarts: usize,
            problem: PhantomData<P>,
            _best_point: Option<Array1<P::PointElem>>,
            _best_point_value: Option<P::PointValue>,
        }

        enum RestarterEvaluation<S, E> {
            InitialState(S),
            InnerEvaluation(E),
        }

        impl<C> RestarterConfig<C> {
            fn new(inner: C, max_restarts: usize) -> Self {
                Self {
                    inner,
                    _max_restarts: max_restarts,
                }
            }
        }

        impl<P, C> OptimizerConfig<P> for RestarterConfig<C>
        where
            P: Problem,
            C: OptimizerConfig<P> + Convergent<P>,
        {
            type Err = C::Err;
            type State = RestarterState<P, C::State>;
            type StateErr = C::StateErr;
            type Evaluation = RestarterEvaluation<C::State, C::Evaluation>;

            fn validate(&self, problem: &P) -> Result<(), Self::Err> {
                self.inner.validate(problem)
            }

            fn validate_state(
                &self,
                problem: &P,
                state: &Self::State,
            ) -> Result<(), Self::StateErr> {
                // Note: should also check and report if restarts is greater than max restarts.
                self.inner.validate_state(problem, &state.inner)
            }

            unsafe fn initial_state(&self, problem: &P) -> Self::State {
                RestarterState {
                    inner: self.inner.initial_state(problem),
                    restarts: 0,
                    problem: PhantomData,
                    _best_point: None,
                    _best_point_value: None,
                }
            }

            unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation {
                // `initial_state` and `evaluate` are safe if this method is safe,
                // because `self.inner` was validated
                // when `self` was validated
                // and `state.inner` was validated
                // when `state` was validated.
                if self.inner.is_done(&state.inner) {
                    RestarterEvaluation::InitialState(unsafe { self.inner.initial_state(problem) })
                } else {
                    RestarterEvaluation::InnerEvaluation(unsafe {
                        self.inner.evaluate(problem, &state.inner)
                    })
                }
            }

            unsafe fn step_from_evaluated(
                &self,
                evaluation: Self::Evaluation,
                mut state: Self::State,
            ) -> Self::State {
                match evaluation {
                    RestarterEvaluation::InitialState(s) => {
                        state.restarts += 1;
                        state.inner = s;
                    }
                    RestarterEvaluation::InnerEvaluation(e) => {
                        // This operation is safe if this method is safe,
                        // because `state.inner` was validated
                        // when `state` was validated
                        // and `evaluation` came from `state.inner`.
                        state.inner = unsafe { self.inner.step_from_evaluated(e, state.inner) };
                    }
                }
                state
            }
        }

        impl<P, S> OptimizerState<P> for RestarterState<P, S>
        where
            P: Problem,
            S: OptimizerState<P>,
        {
            type Evaluatee<'a> = S::Evaluatee<'a>
            where
                P: 'a,
                S: 'a;

            fn evaluatee(&self) -> Self::Evaluatee<'_> {
                self.inner.evaluatee()
            }

            fn best_point(&self) -> CowArray<P::PointElem, Ix1> {
                unimplemented!()
            }

            fn stored_best_point_value(&self) -> Option<&P::PointValue> {
                unimplemented!()
            }
        }

        let mut o = Optimizer::new(MockProblem, RestarterConfig::new(MockConfigA, 10))
            .unwrap()
            .start();
        o.next();
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
        enum MockError<P> {
            A(<MockConfigA as OptimizerConfig<P>>::Err),
            B(<MockConfigB as OptimizerConfig<P>>::Err),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockState {
            A(MockStateA),
            B(MockStateB),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockStateError<P> {
            WrongState,
            A(<MockConfigA as OptimizerConfig<P>>::StateErr),
            B(<MockConfigB as OptimizerConfig<P>>::StateErr),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockEvaluation<P> {
            A(<MockConfigA as OptimizerConfig<P>>::Evaluation),
            B(<MockConfigB as OptimizerConfig<P>>::Evaluation),
        }

        impl<P> OptimizerConfig<P> for MockConfig {
            type Err = MockError<P>;
            type State = MockState;
            type StateErr = MockStateError<P>;
            type Evaluation = MockEvaluation<P>;

            fn validate(&self, problem: &P) -> Result<(), Self::Err> {
                match self {
                    Self::A(c) => c.validate(problem).map_err(MockError::A),
                    Self::B(c) => c.validate(problem).map_err(MockError::B),
                }
            }

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

            unsafe fn initial_state(&self, problem: &P) -> Self::State {
                // `initial_state` is safe if this method is safe,
                // because the inner config was validated
                // when `self` was validated.
                match self {
                    Self::A(c) => MockState::A(unsafe { c.initial_state(problem) }),
                    Self::B(c) => MockState::B(unsafe { c.initial_state(problem) }),
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
            P::PointElem: Clone + Zero,
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

            fn best_point(&self) -> CowArray<P::PointElem, Ix1> {
                match self {
                    Self::A(x) => x.best_point(),
                    Self::B(x) => x.best_point(),
                }
            }

            fn stored_best_point_value(&self) -> Option<&P::PointValue> {
                match self {
                    Self::A(x) => x.stored_best_point_value(),
                    Self::B(x) => x.stored_best_point_value(),
                }
            }
        }

        let mut o = Optimizer::new(MockProblem, MockConfig::A(MockConfigA))
            .unwrap()
            .start();
        o.next();
        let store = serde_json::to_string(&o).unwrap();
        o = serde_json::from_str(&store).unwrap();
        o.next();
        o.best_point_value();
    }
}
