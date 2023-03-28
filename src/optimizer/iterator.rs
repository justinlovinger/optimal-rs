use std::marker::PhantomData;

use streaming_iterator::StreamingIterator;

use crate::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl<A, O> IntoStreamingIterator<A> for O
where
    O: RunningOptimizer<A>,
{
    fn into_streaming_iter(self) -> StepIterator<A, O> {
        StepIterator::new(self)
    }
}

/// An automatically implemented extension to [`Step`]
/// providing an iterator-based API.
///
/// Initial optimizer state is emitted before stepping,
/// meaning the first call to `advance` or `next` will not change state.
/// For example,
/// `nth(100)` will step `99` times.
pub trait IntoStreamingIterator<A> {
    /// Return an iterator over optimizer states.
    fn into_streaming_iter(self) -> StepIterator<A, Self>
    where
        Self: Sized;
}

/// An iterator returned by [`into_streaming_iter`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StepIterator<A, O> {
    point_elem: PhantomData<A>,
    inner: O,
    skipped_first_step: bool,
}

impl<A, O> StepIterator<A, O> {
    fn new(optimizer: O) -> Self {
        Self {
            point_elem: PhantomData,
            inner: optimizer,
            skipped_first_step: false,
        }
    }

    /// Return inner optimizer.
    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<A, O> StreamingIterator for StepIterator<A, O>
where
    O: RunningOptimizer<A>,
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
            (MockOptimizer {
                config: MockConfig::default(),
                state: MockState::new(),
            })
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
            (MockOptimizer {
                config: MockConfig::default(),
                state: MockState::new(),
            })
            .into_streaming_iter()
            .find(|o| o.is_done())
            .unwrap()
            .state
            .steps,
            config.max_steps
        )
    }

    struct MockOptimizer {
        pub config: MockConfig,
        pub state: MockState,
    }

    struct MockConfig {
        max_steps: usize,
    }

    struct MockState {
        steps: usize,
    }

    impl RunningOptimizer<f64> for MockOptimizer {
        fn step(&mut self) {
            self.state.steps += 1;
        }

        fn best_point(&self) -> ndarray::CowArray<f64, ndarray::Ix1> {
            unimplemented!()
        }
    }

    impl Convergent for MockOptimizer {
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
