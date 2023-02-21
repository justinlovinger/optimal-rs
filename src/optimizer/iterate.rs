use ndarray::prelude::*;
use replace_with::replace_with_or_abort;
use std::marker::PhantomData;
use streaming_iterator::StreamingIterator;

use crate::prelude::*;

impl<A, B, S, C> Iterate<A, B, S> for C
where
    C: Step<A, B, S, S>,
{
    fn iterate<F>(&self, f: F, state: S) -> StepIterator<A, B, C, F, S>
    where
        F: Fn(CowArray<A, Ix2>) -> Array1<B>,
    {
        StepIterator::new(self, f, state)
    }
}

/// An automatically implemented extension to [`Step`]
/// providing an iterator-based API.
pub trait Iterate<A, B, S> {
    /// Return an iterator over optimizer states.
    fn iterate<F>(&self, f: F, state: S) -> StepIterator<A, B, Self, F, S>
    where
        Self: Sized,
        F: Fn(CowArray<A, Ix2>) -> Array1<B>;
}

/// An iterator returned by [`Iterate`].
pub struct StepIterator<'a, A, B, C, F, S> {
    point_elem: PhantomData<A>,
    point_value: PhantomData<B>,
    config: &'a C,
    f: F,
    state: S,
    skipped_first_step: bool,
}

impl<'a, A, B, C, F, S> StepIterator<'a, A, B, C, F, S> {
    fn new(config: &'a C, f: F, state: S) -> Self {
        Self {
            point_elem: PhantomData,
            point_value: PhantomData,
            config,
            f,
            state,
            skipped_first_step: false,
        }
    }
}

impl<'a, A, B, C, F, S> StreamingIterator for StepIterator<'a, A, B, C, F, S>
where
    C: Step<A, B, S, S>,
    F: Fn(CowArray<A, Ix2>) -> Array1<B>,
{
    type Item = S;

    fn advance(&mut self) {
        // `advance` is called before the first `get`,
        // but we want to emit the initial state.
        if self.skipped_first_step {
            replace_with_or_abort(&mut self.state, |state| self.config.step(&self.f, state));
        } else {
            self.skipped_first_step = true;
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(&self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Data;

    #[test]
    fn iterate_emits_initial_state() {
        assert_eq!(
            MockConfig::default()
                .iterate(f, MockState::new())
                .next()
                .unwrap()
                .steps,
            0
        )
    }

    #[test]
    fn iterate_emits_done_state() {
        let config = MockConfig::default();
        assert_eq!(
            config
                .iterate(f, MockState::new())
                .find(|s| config.is_done(s))
                .unwrap()
                .steps,
            config.max_steps
        )
    }

    fn f(xs: CowArray<f64, Ix2>) -> Array1<f64> {
        xs.sum_axis(Axis(xs.ndim() - 1))
    }

    struct MockConfig {
        max_steps: usize,
    }

    impl Default for MockConfig {
        fn default() -> Self {
            MockConfig { max_steps: 10 }
        }
    }

    struct MockState {
        steps: usize,
    }

    impl MockState {
        fn new() -> Self {
            Self { steps: 0 }
        }
    }

    impl Points<f64, MockState> for MockConfig {
        fn points<'a>(&'a self, _state: &'a MockState) -> CowArray<f64, Ix2> {
            Array2::zeros((0, 0)).into()
        }
    }

    impl StepFromEvaluated<f64, MockState, MockState> for MockConfig {
        fn step_from_evaluated<S: Data<Elem = f64>>(
            &self,
            _point_values: ArrayBase<S, Ix1>,
            state: MockState,
        ) -> MockState {
            MockState {
                steps: state.steps + 1,
            }
        }
    }

    impl IsDone<MockState> for MockConfig {
        fn is_done(&self, state: &MockState) -> bool {
            state.steps >= self.max_steps
        }
    }
}
