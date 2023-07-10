use blanket::blanket;
use once_cell::sync::OnceCell;
use streaming_iterator::StreamingIterator;

use crate::prelude::*;

#[cfg(any(test, feature = "serde"))]
use serde::{Deserialize, Serialize};

/// Running optimizer methods
/// independent of configuration
/// and state.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Optimizer<P>
where
    P: Problem,
{
    /// Return the best point discovered.
    fn best_point(&self) -> P::Point<'_>;

    /// Return the value of the best point discovered,
    /// evaluating the best point
    /// if necessary.
    fn best_point_value(&self) -> P::Value;
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
pub struct RunningOptimizer<P, C>
where
    C: OptimizerConfig<P>,
{
    problem: P,
    config: C,
    state: C::State,
    skipped_first_step: bool,
    #[cfg_attr(any(test, feature = "serde"), serde(skip))]
    evaluation_cache: OnceCell<C::Evaluation>,
}

impl<P, C> RunningOptimizer<P, C>
where
    C: OptimizerConfig<P>,
{
    /// Return a running optimizer    // This takes `state` first
    // because `problem` and `config` are used
    // to get `state`.
    // Taking `state` first makes it easier to use.
    pub(crate) fn new(state: C::State, config: C, problem: P) -> Self {
        Self {
            problem,
            config,
            state,
            skipped_first_step: false,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Stop optimization run,
    /// returning problem,
    /// configuration,
    /// and state.
    pub fn into_inner(self) -> (C, P, C::State) {
        (self.config, self.problem, self.state)
    }

    /// Return optimizer configuration.
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Return problem being optimized.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// Return state of optimizer.
    pub fn state(&self) -> &C::State {
        &self.state
    }
}

impl<P, C> RunningOptimizer<P, C>
where
    P: Problem,
    C: OptimizerConfig<P>,
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
            unsafe { self.config.step_from_evaluated(evaluation, state) }
        });
    }

    fn evaluate(&self) -> C::Evaluation {
        // This operation is safe
        // because `self.state` was validated
        // when constructing `self`
        // and `self.config` was validated
        // when constructing the optimizer `self` was from.
        unsafe { self.config.evaluate(&self.problem, &self.state) }
    }
}

impl<P, C> StreamingIterator for RunningOptimizer<P, C>
where
    P: Problem,
    C: OptimizerConfig<P>,
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

impl<P, C> Optimizer<P> for RunningOptimizer<P, C>
where
    P: Problem,
    C: OptimizerConfig<P>,
    C::State: OptimizerState<P>,
{
    fn best_point(&self) -> P::Point<'_> {
        self.state.best_point()
    }

    fn best_point_value(&self) -> P::Value {
        self.state
            .stored_best_point_value()
            .unwrap_or_else(|| self.problem().evaluate(self.best_point()))
    }
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use ndarray::prelude::*;
    use rand::prelude::*;
    use rand_xoshiro::SplitMix64;

    use crate::{optimizer::derivative_free::pbil, prelude::*};

    assert_obj_safe!(Optimizer<()>);

    #[test]
    fn running_optimizer_streaming_iterator_emits_initial_state() {
        let seed = 0;
        let config = pbil::Config::default_for(&Count);
        assert_eq!(
            config
                .clone()
                .start_using(Count, &mut SplitMix64::seed_from_u64(seed))
                .next()
                .unwrap()
                .state(),
            config
                .start_using(Count, &mut SplitMix64::seed_from_u64(seed))
                .state()
        );
    }

    #[test]
    fn running_optimizer_streaming_iterator_runs_for_same_number_of_steps() {
        let seed = 0;
        let steps = 100;
        let config = pbil::Config::default_for(&Count);
        let mut o = config
            .clone()
            .start_using(Count, &mut SplitMix64::seed_from_u64(seed));
        for _ in 0..steps {
            o.step();
        }
        assert_eq!(
            config
                .start_using(Count, &mut SplitMix64::seed_from_u64(seed))
                .nth(steps)
                .unwrap()
                .state(),
            o.state()
        );
    }

    #[derive(Clone, Debug)]
    struct Count;

    impl Problem for Count {
        type Point<'a> = CowArray<'a, bool, Ix1>;
        type Value = u64;

        fn evaluate<'a>(&'a self, point: Self::Point<'a>) -> Self::Value {
            point.fold(0, |acc, b| acc + *b as u64)
        }
    }

    impl FixedLength for Count {
        fn len(&self) -> usize {
            16
        }
    }
}
