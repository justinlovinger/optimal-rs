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
//! use optimal::{optimizer::derivative_free::pbil::*, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut o = Config::start_default_for(Count);
//!     let point = UntilConvergedConfig::default().argmin(&mut o);
//!     let point_value = Count.evaluate(point.view().into_shape((1, Count.len())).unwrap())[0];
//!     println!("f({point}) = {point_value}");
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type Value = u64;
//!
//!     fn evaluate(&self, points: ArrayView2<bool>) -> Array1<Self::Value> {
//!         points.fold_axis(Axis(1), 0, |acc, b| acc + *b as u64)
//!     }
//!
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
//! # use optimal::{optimizer::derivative_free::pbil::*, prelude::*};
//! # use streaming_iterator::StreamingIterator;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Count;
//! #
//! # impl Problem for Count {
//! #     type Value = u64;
//! #
//! #     fn evaluate(&self, points: ArrayView2<bool>) -> Array1<Self::Value> {
//! #         points.fold_axis(Axis(1), 0, |acc, b| acc + *b as u64)
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         16
//! #     }
//! # }
//! #
//! let mut it = UntilConvergedConfig::default().start(Config::start_default_for(Count));
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

    fn mock_obj_func(x: usize) -> usize {
        x + 1
    }

    macro_rules! mock_optimizer {
        ( $id:ident ) => {
            paste::paste! {
                trait [< Problem $id >] {
                    fn evaluate(&self, x: usize) -> usize;
                }

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockProblem $id >];

                impl [< Problem $id >] for [< MockProblem $id >] {
                    fn evaluate(&self, x: usize) -> usize {
                        mock_obj_func(x)
                    }
                }

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockOptimizer $id >]<P> {
                    problem: P,
                    state: usize,
                }

                impl<P> [< MockOptimizer $id >]<P> {
                    fn new(problem: P) -> Self {
                        Self { problem, state: 0 }
                    }

                    #[allow(dead_code)]
                    fn evaluation(&self) -> usize
                    where
                        P: [< Problem $id >],
                    {
                        self.problem.evaluate(self.state)
                    }
                }

                impl<P> StreamingIterator for [< MockOptimizer $id >]<P>
                where
                    P: [< Problem $id >],
                {
                    type Item = Self;

                    fn advance(&mut self) {
                        self.state += self.problem.evaluate(self.state)
                    }

                    fn get(&self) -> Option<&Self::Item> {
                        Some(self)
                    }
                }

                impl<P> Optimizer for [< MockOptimizer $id >]<P> {
                    type Point = usize;

                    fn best_point(&self) -> Self::Point {
                        self.state
                    }
                }
            }
        };
    }

    mock_optimizer!(A);
    mock_optimizer!(B);

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MaxStepsConfig(usize);

    #[derive(Clone, Debug, Serialize, Deserialize)]
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
        type BoxedOptimizer<A> = Box<dyn StreamingIterator<Item = dyn Optimizer<Point = A>>>;

        fn best_optimizer<A, B, F, I>(obj_func: F, optimizers: I) -> usize
        where
            B: Ord,
            F: Fn(A) -> B,
            I: IntoIterator<Item = BoxedOptimizer<A>>,
        {
            optimizers
                .into_iter()
                .enumerate()
                .map(|(i, mut o)| {
                    let o = o.nth(10).unwrap();
                    (obj_func(o.best_point()), i)
                })
                .min()
                .expect("`optimizers` should be non-empty")
                .1
        }

        best_optimizer(
            mock_obj_func,
            [
                Box::new(
                    MockOptimizerA::new(MockProblemA)
                        .map_ref(|x| x as &dyn Optimizer<Point = usize>),
                ) as BoxedOptimizer<usize>,
                Box::new(
                    MockOptimizerB::new(MockProblemB)
                        .map_ref(|x| x as &dyn Optimizer<Point = usize>),
                ) as BoxedOptimizer<usize>,
            ],
        );
    }

    #[test]
    fn parallel_optimization_runs_should_be_easy() {
        use std::thread::spawn;

        fn parallel<A, O, F>(start: F)
        where
            A: Send + 'static,
            O: StreamingIterator + Optimizer<Point = A> + Send + 'static,
            F: Fn() -> O,
        {
            let mut o1 = start();
            let mut o2 = start();
            #[allow(clippy::redundant_clone)] // False positive.
            let handler1 = spawn(move || MaxStepsConfig(10).argmin(&mut o1));
            #[allow(clippy::redundant_clone)] // False positive.
            let handler2 = spawn(move || MaxStepsConfig(10).argmin(&mut o2));
            handler1.join().unwrap();
            handler2.join().unwrap();
        }

        parallel(|| MockOptimizerA::new(MockProblemA));
    }

    #[test]
    fn examining_state_and_corresponding_evaluations_should_be_easy() {
        // Note,
        // this is a bit of a hack
        // because methods providing evaluations are current a convention,
        // not a part of the API.
        MockOptimizerA::new(MockProblemA)
            .inspect(|o| println!("state: {:?}, evaluation: {:?}", o.state, o.evaluation()))
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

        impl<P> Restart for MockOptimizerA<P> {
            fn restart(&mut self) {
                replace_with_or_abort(self, |this| MockOptimizerA::new(this.problem))
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
            .start(MaxStepsConfig(10).start(MockOptimizerA::new(MockProblemA)))
            .nth(100);
    }

    // Applications may need to select an optimizer at runtime,
    // run it for less than a full optimization,
    // save the partial run,
    // and resume it later.
    #[test]
    fn dynamic_optimizers_should_be_partially_runable() {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum DynOptimizer {
            A(MockOptimizerA<MockProblemA>),
            B(MockOptimizerB<MockProblemB>),
        }

        impl StreamingIterator for DynOptimizer {
            type Item = Self;

            fn advance(&mut self) {
                match self {
                    DynOptimizer::A(x) => x.advance(),
                    DynOptimizer::B(x) => x.advance(),
                }
            }

            fn get(&self) -> Option<&Self::Item> {
                Some(self)
            }
        }

        impl Optimizer for DynOptimizer {
            type Point = usize;

            fn best_point(&self) -> Self::Point {
                match self {
                    DynOptimizer::A(x) => x.best_point(),
                    DynOptimizer::B(x) => x.best_point(),
                }
            }
        }

        let mut o = MaxStepsConfig(10).start(DynOptimizer::A(MockOptimizerA::new(MockProblemA)));
        o.next();
        let store = serde_json::to_string(&o).unwrap();
        o = serde_json::from_str(&store).unwrap();
        o.next();
        o.into_inner().0.best_point();
    }
}
