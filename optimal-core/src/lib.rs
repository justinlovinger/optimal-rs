#![allow(clippy::needless_doctest_main)]
#![cfg_attr(test, feature(unboxed_closures))]
#![cfg_attr(test, feature(fn_traits))]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Core traits and types for Optimal.
//!
//! Most optimizers are expected to adhere to particular conventions.
//! An optimizer configuration should remain static during operation.
//! A problem and configuration
//! can be used to start a running optimizer.
//! A running optimizer has state
//! that depends on its problem
//! and configuration.

use blanket::blanket;
use streaming_iterator::StreamingIterator;

/// Running optimizer methods
/// independent of configuration
/// and state.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Optimizer {
    /// A point in the problem space being optimized.
    type Point;

    /// Return the best point discovered.
    fn best_point(&self) -> Self::Point;
}

/// An extension trait adding methods to `StreamingIterator`.
pub trait StreamingIteratorExt: StreamingIterator {
    /// Return the last item
    /// in this iterator,
    /// assuming `get()` returns `Some`
    /// when `is_done()` returns `true`.
    fn last(&mut self) -> Option<&Self::Item> {
        while !self.is_done() {
            self.advance()
        }
        (*self).get()
    }
}

/// An extension trait adding methods to `StreamingIterator`
/// for optimization.
pub trait OptimizerExt: StreamingIteratorExt {
    /// Return point that attempts to minimize a problem
    /// by running to completion.
    ///
    /// How well the point minimizes the problem
    /// depends on the optimizer.
    fn argmin(&mut self) -> Option<<Self::Item as Optimizer>::Point>
    where
        Self::Item: Optimizer,
    {
        self.last().map(|x| x.best_point())
    }
}

impl<T> StreamingIteratorExt for T where T: StreamingIterator {}
impl<T> OptimizerExt for T where T: StreamingIterator {}

/// Useful traits,
/// types,
/// and functions
/// unlikely to conflict with existing definitions.
pub mod prelude {
    pub use streaming_iterator::StreamingIterator;

    pub use super::{Optimizer, OptimizerExt, StreamingIteratorExt};
}

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
    use static_assertions::assert_obj_safe;

    use crate::prelude::*;

    assert_obj_safe!(Optimizer<Point = ()>);

    fn mock_obj_func(x: usize) -> usize {
        x + 1
    }

    macro_rules! mock_optimizer {
        ( $id:ident ) => {
            paste::paste! {
                #[derive(Clone, Debug, Serialize, Deserialize)]
                struct [< MockOptimizer $id >]<F> {
                    obj_func: F,
                    state: usize,
                }

                impl<F> [< MockOptimizer $id >]<F> {
                    fn new(obj_func: F) -> Self {
                        Self { obj_func, state: 0 }
                    }

                    fn evaluation(&self) -> usize
                    where
                        F: Fn(usize) -> usize
                    {
                        (self.obj_func)(self.state)
                    }
                }

                impl<F> StreamingIterator for [< MockOptimizer $id >]<F>
                where
                    F: Fn(usize) -> usize
                {
                    type Item = Self;

                    fn advance(&mut self) {
                        self.state += self.evaluation()
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
    struct MaxSteps<I> {
        max_i: usize,
        i: usize,
        it: I,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MaxStepsConfig(usize);

    impl MaxStepsConfig {
        fn start<I>(self, it: I) -> MaxSteps<I> {
            MaxSteps {
                i: 0,
                max_i: self.0,
                it,
            }
        }
    }

    impl<I> StreamingIterator for MaxSteps<I>
    where
        I: StreamingIterator,
    {
        type Item = I::Item;

        fn advance(&mut self) {
            self.it.advance();
            self.i += 1;
        }

        fn get(&self) -> Option<&Self::Item> {
            self.it.get()
        }

        fn is_done(&self) -> bool {
            self.it.is_done() || self.i >= self.max_i
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
                    MockOptimizerA::new(mock_obj_func)
                        .map_ref(|x| x as &dyn Optimizer<Point = usize>),
                ) as BoxedOptimizer<usize>,
                Box::new(
                    MockOptimizerB::new(mock_obj_func)
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
            O: StreamingIterator + Send + 'static,
            O::Item: Optimizer<Point = A>,
            F: Fn() -> O,
        {
            let o1 = start();
            let o2 = start();
            let handler1 = spawn(move || MaxStepsConfig(10).start(o1).argmin());
            let handler2 = spawn(move || MaxStepsConfig(10).start(o2).argmin());
            handler1.join().unwrap();
            handler2.join().unwrap();
        }

        parallel(|| MockOptimizerA::new(mock_obj_func));
    }

    #[test]
    fn examining_state_and_corresponding_evaluations_should_be_easy() {
        // Note,
        // this is a bit of a hack
        // because methods providing evaluations are current a convention,
        // not a part of the API.
        MockOptimizerA::new(mock_obj_func)
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
                replace_with_or_abort(self, |this| MockOptimizerA::new(this.obj_func))
            }
        }

        impl<I> Restart for MaxSteps<I>
        where
            I: Restart,
        {
            fn restart(&mut self) {
                replace_with_or_abort(self, |this| {
                    let mut it = this.it;
                    it.restart();
                    MaxStepsConfig(this.max_i).start(it)
                })
            }
        }

        struct Restarter<I> {
            max_restarts: usize,
            restarts: usize,
            it: I,
        }

        struct RestarterConfig {
            max_restarts: usize,
        }

        impl RestarterConfig {
            fn start<I>(self, it: I) -> Restarter<I> {
                Restarter {
                    max_restarts: self.max_restarts,
                    restarts: 0,
                    it,
                }
            }
        }

        impl<I> StreamingIterator for Restarter<I>
        where
            I: StreamingIterator + Restart,
        {
            type Item = I::Item;

            fn advance(&mut self) {
                if self.restarts < self.max_restarts && self.it.is_done() {
                    self.restarts += 1;
                    self.it.restart();
                } else {
                    self.it.advance()
                }
            }

            fn get(&self) -> Option<&Self::Item> {
                self.it.get()
            }
        }

        let _ = RestarterConfig { max_restarts: 10 }
            .start(MaxStepsConfig(10).start(MockOptimizerA::new(mock_obj_func)))
            .nth(100);
    }

    // Applications may need to select an optimizer at runtime,
    // run it for less than a full optimization,
    // save the partial run,
    // and resume it later.
    #[test]
    fn dynamic_optimizers_should_be_partially_runable() {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        enum DynOptimizer<F> {
            A(MockOptimizerA<F>),
            B(MockOptimizerB<F>),
        }

        impl<F> StreamingIterator for DynOptimizer<F>
        where
            F: Fn(usize) -> usize,
        {
            type Item = Self;

            fn advance(&mut self) {
                match self {
                    Self::A(x) => x.advance(),
                    Self::B(x) => x.advance(),
                }
            }

            fn get(&self) -> Option<&Self::Item> {
                Some(self)
            }
        }

        impl<F> Optimizer for DynOptimizer<F> {
            type Point = usize;

            fn best_point(&self) -> Self::Point {
                match self {
                    Self::A(x) => x.best_point(),
                    Self::B(x) => x.best_point(),
                }
            }
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        struct MockObjFunc;

        impl FnOnce<(usize,)> for MockObjFunc {
            type Output = usize;
            extern "rust-call" fn call_once(self, args: (usize,)) -> Self::Output {
                mock_obj_func(args.0)
            }
        }

        impl FnMut<(usize,)> for MockObjFunc {
            extern "rust-call" fn call_mut(&mut self, args: (usize,)) -> Self::Output {
                mock_obj_func(args.0)
            }
        }

        impl Fn<(usize,)> for MockObjFunc {
            extern "rust-call" fn call(&self, args: (usize,)) -> Self::Output {
                mock_obj_func(args.0)
            }
        }

        let mut o = MaxStepsConfig(10).start(DynOptimizer::A(MockOptimizerA::new(MockObjFunc)));
        o.next();
        let store = serde_json::to_string(&o).unwrap();
        o = serde_json::from_str(&store).unwrap();
        o.next();
        o.get().unwrap().best_point();
    }
}
