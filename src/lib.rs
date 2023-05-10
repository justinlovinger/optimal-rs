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
//!     let point = pbil::DoneWhenConvergedConfig::default(Count).argmin();
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
//! let mut iter = pbil::DoneWhenConvergedConfig::default(Count)
//!     .start()
//!     .into_streaming_iter()
//!     .inspect(|o| println!("{:?}", o.state()));
//! let o = iter
//!     .find(|o| o.is_done())
//!     .expect("should converge");
//! println!("f({}) = {}", o.best_point(), o.best_point_value());
//! ```

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

mod derive;
pub mod optimizer;
pub mod prelude;
mod problem;

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

    use std::{borrow::Borrow, marker::PhantomData};

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
                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockConfig $id >]<P>(P);

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockState $id >];

                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockRunning $id >]<P, C>{
                    problem: PhantomData<P>,
                    config: C,
                }

                impl<P, C> [< MockRunning $id >]<P, C> {
                    fn new(config: C) -> Self {
                        Self {
                            problem: PhantomData,
                            config,
                        }
                    }
                }

                impl<P> OptimizerConfig for [< MockConfig $id >]<P>
                where
                    P: Problem,
                    P::PointElem: Clone + Zero,
                {
                    type Problem = P;
                    type Optimizer = [< MockRunning $id >]<P, Self>;

                    fn start(self) -> Self::Optimizer {
                        [< MockRunning $id >]::new(self)
                    }

                    fn problem(&self) -> &Self::Problem {
                        &self.0
                    }
                }

                impl<P> OptimizerConfig for &[< MockConfig $id >]<P>
                where
                    P: Problem,
                    P::PointElem: Clone + Zero,
                {
                    type Problem = P;
                    type Optimizer = [< MockRunning $id >]<P, Self>;

                    fn start(self) -> Self::Optimizer {
                        [< MockRunning $id >]::new(self)
                    }

                    fn problem(&self) -> &Self::Problem {
                        &self.0
                    }
                }

                impl<P, C> RunningOptimizerBase for [< MockRunning $id >]<P, C>
                where
                    P: Problem,
                    P::PointElem: Clone + Zero,
                    C: Borrow<[< MockConfig $id >]<P>>,
                {
                    type Problem = P;

                    fn problem(&self) -> &Self::Problem {
                        &self.config.borrow().0
                    }

                    fn best_point(&self) -> CowArray<<Self::Problem as Problem>::PointElem, Ix1> {
                        Array::from_elem(1, <Self::Problem as Problem>::PointElem::zero()).into()
                    }

                    fn stored_best_point_value(&self) -> Option<&<Self::Problem as Problem>::PointValue> {
                        None
                    }
                }

                impl<P, C> RunningOptimizerStep for [< MockRunning $id >]<P, C>
                where
                    P: Problem,
                    P::PointElem: Clone + Zero,
                    C: Borrow<[< MockConfig $id >]<P>>,
                {
                    fn step(&mut self) {}
                }

                impl<P, C> Convergent for [< MockRunning $id >]<P, C>
                where
                    P: Problem,
                    P::PointElem: Clone + Zero,
                    C: Borrow<[< MockConfig $id >]<P>>,
                {
                    fn is_done(&self) -> bool {
                        true
                    }
                }
            }
        };
    }

    mock_optimizer!(A);
    mock_optimizer!(B);

    #[test]
    fn optimizers_should_be_easily_comparable() {
        fn best_optimizer<P, I>(problem: P, optimizers: I) -> usize
        where
            P: Problem,
            P::PointValue: Ord,
            I: IntoIterator<Item = Box<dyn Optimizer<P>>>,
        {
            optimizers
                .into_iter()
                .enumerate()
                .map(|(i, o)| (problem.evaluate(o.argmin().into()), i))
                .min()
                .expect("`optimizers` should be non-empty")
                .1
        }

        best_optimizer(
            MockProblem,
            [
                Box::new(MockConfigA(MockProblem)) as Box<dyn Optimizer<MockProblem>>,
                Box::new(MockConfigB(MockProblem)) as Box<dyn Optimizer<MockProblem>>,
            ],
        );
    }

    #[test]
    fn optimizers_should_be_able_to_restart_automatically() {
        struct Restarter<C>
        where
            C: OptimizerConfig,
        {
            config: C,
            inner: C::Optimizer,
            restarts: usize,
        }

        // We only implement `step`
        // to minimize code.
        impl<C> Restarter<C>
        where
            C: Clone + OptimizerConfig,
        {
            fn new(config: C) -> Self {
                Self {
                    restarts: 0,
                    inner: config.clone().start(),
                    config,
                }
            }

            fn step(&mut self)
            where
                C::Optimizer: RunningOptimizer + Convergent,
            {
                if self.inner.is_done() {
                    self.restarts += 1;
                    // Ideally,
                    // we would not need to clone.
                    self.inner = self.config.clone().start();
                } else {
                    self.inner.step()
                }
            }
        }

        let mut o = Restarter::new(MockConfigA(MockProblem));
        o.step();
    }

    #[test]
    fn parallel_optimization_runs_should_be_easy() {
        use std::{sync::Arc, thread::spawn};

        fn parallel(config: Arc<dyn Optimizer<MockProblem> + Send + Sync>) {
            let config1 = Arc::clone(&config);
            let handler1 = spawn(move || config1.argmin());
            let handler2 = spawn(move || config.argmin());
            handler1.join().unwrap();
            handler2.join().unwrap();
        }

        parallel(Arc::new(MockConfigA(MockProblem)));
    }

    // Applications may need to select an optimizer at runtime,
    // run it for less than a full optimization,
    // save the partial run,
    // and resume it later.
    #[test]
    fn dynamic_optimizers_should_be_partially_runable() {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockConfig {
            A(MockConfigA<MockProblem>),
            B(MockConfigB<MockProblem>),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockState {
            A(MockStateA),
            B(MockStateB),
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum MockRunning {
            A(MockRunningA<MockProblem, MockConfigA<MockProblem>>),
            B(MockRunningB<MockProblem, MockConfigB<MockProblem>>),
        }

        impl OptimizerConfig for MockConfig {
            type Problem = MockProblem;
            type Optimizer = MockRunning;

            fn start(self) -> Self::Optimizer {
                match self {
                    Self::A(x) => MockRunning::A(x.start()),
                    Self::B(x) => MockRunning::B(x.start()),
                }
            }

            fn problem(&self) -> &Self::Problem {
                match self {
                    Self::A(x) => x.problem(),
                    Self::B(x) => x.problem(),
                }
            }
        }

        impl RunningOptimizerBase for MockRunning {
            type Problem = MockProblem;

            fn problem(&self) -> &Self::Problem {
                match self {
                    Self::A(x) => x.problem(),
                    Self::B(x) => x.problem(),
                }
            }

            fn best_point(&self) -> CowArray<<Self::Problem as Problem>::PointElem, Ix1> {
                match self {
                    Self::A(x) => x.best_point(),
                    Self::B(x) => x.best_point(),
                }
            }

            fn stored_best_point_value(&self) -> Option<&<Self::Problem as Problem>::PointValue> {
                match self {
                    Self::A(x) => x.stored_best_point_value(),
                    Self::B(x) => x.stored_best_point_value(),
                }
            }
        }

        impl RunningOptimizerStep for MockRunning {
            fn step(&mut self) {
                match self {
                    Self::A(x) => x.step(),
                    Self::B(x) => x.step(),
                }
            }
        }

        let mut o = MockConfig::A(MockConfigA(MockProblem)).start();
        o.step();
        let store = serde_json::to_string(&o).unwrap();
        o = serde_json::from_str(&store).unwrap();
        o.step();
        o.best_point_value();
    }
}
