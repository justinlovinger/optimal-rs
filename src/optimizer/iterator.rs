use std::marker::PhantomData;

use streaming_iterator::StreamingIterator;

use crate::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl<A, B, C, S, O> IntoStreamingIterator<A, B, C, S> for O
where
    O: RunningOptimizer<A, B, C, S>,
{
    fn into_streaming_iter(self) -> StepIterator<A, B, C, S, O> {
        StepIterator::new(self)
    }
}

/// An automatically implemented extension to [`RunningOptimizer`]
/// providing an iterator-based API.
///
/// Initial optimizer state is emitted before stepping,
/// meaning the first call to `advance` or `next` will not change state.
/// For example,
/// `nth(100)` will step `99` times.
pub trait IntoStreamingIterator<A, B, C, S> {
    /// Return an iterator over optimizer states.
    fn into_streaming_iter(self) -> StepIterator<A, B, C, S, Self>
    where
        Self: Sized;
}

/// An iterator returned by [`into_streaming_iter`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StepIterator<A, B, C, S, O> {
    point_elem: PhantomData<A>,
    point_value: PhantomData<B>,
    config: PhantomData<C>,
    state: PhantomData<S>,
    inner: O,
    skipped_first_step: bool,
}

impl<A, B, C, S, O> StepIterator<A, B, C, S, O> {
    fn new(optimizer: O) -> Self {
        Self {
            point_elem: PhantomData,
            point_value: PhantomData,
            config: PhantomData,
            state: PhantomData,
            inner: optimizer,
            skipped_first_step: false,
        }
    }

    /// Return inner optimizer.
    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<A, B, C, S, O> StreamingIterator for StepIterator<A, B, C, S, O>
where
    O: RunningOptimizer<A, B, C, S>,
{
    type Item = O;

    fn advance(&mut self) {
        // `advance` is called before the first `get`,
        // but we want to emit the initial state.
        if self.skipped_first_step {
            self.inner.step()
        } else {
            self.skipped_first_step = true;
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterate_emits_initial_state() {
        assert_eq!(
            MockConfig::default()
                .start()
                .into_streaming_iter()
                .next()
                .unwrap()
                .state
                .steps,
            0
        )
    }

    #[test]
    fn iterate_emits_done_state() {
        let config = MockConfig::default();
        assert_eq!(
            MockConfig::default()
                .start()
                .into_streaming_iter()
                .find(|o| o.is_done())
                .unwrap()
                .state
                .steps,
            config.max_steps
        )
    }

    struct MockRunning {
        pub config: MockConfig,
        pub state: MockState,
    }

    struct MockConfig {
        max_steps: usize,
    }

    struct MockState {
        steps: usize,
    }

    impl MockRunning {
        fn new(config: MockConfig, state: MockState) -> Self {
            Self { config, state }
        }
    }

    impl OptimizerConfig<'_, MockRunning, ()> for MockConfig {
        fn start(self) -> MockRunning {
            MockRunning::new(self, MockState::new())
        }

        fn problem(&self) -> &() {
            unimplemented!()
        }
    }

    impl RunningOptimizer<f64, f64, MockConfig, MockState> for MockRunning {
        fn step(&mut self) {
            self.state.steps += 1;
        }

        fn config(&self) -> &MockConfig {
            &self.config
        }

        fn state(&self) -> &MockState {
            &self.state
        }

        fn stop(self) -> (MockConfig, MockState) {
            (self.config, self.state)
        }

        fn best_point(&self) -> ndarray::CowArray<f64, ndarray::Ix1> {
            unimplemented!()
        }

        fn stored_best_point_value(&self) -> Option<f64> {
            unimplemented!()
        }
    }

    impl Convergent for MockRunning {
        fn is_done(&self) -> bool {
            self.state.steps >= self.config.max_steps
        }
    }

    impl Default for MockConfig {
        fn default() -> Self {
            MockConfig { max_steps: 10 }
        }
    }

    impl MockState {
        fn new() -> Self {
            Self { steps: 0 }
        }
    }
}
