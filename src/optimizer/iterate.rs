use streaming_iterator::StreamingIterator;

use crate::prelude::*;

impl<O> Iterate for O
where
    O: Step,
{
    fn iterate(self) -> StepIterator<O> {
        StepIterator::new(self)
    }
}

/// An automatically implemented extension to [`Step`]
/// providing an iterator-based API.
pub trait Iterate {
    /// Return an iterator over optimizer states.
    fn iterate(self) -> StepIterator<Self>
    where
        Self: Sized;
}

/// An iterator returned by [`Iterate`].
pub struct StepIterator<O> {
    optimizer: O,
    skipped_first_step: bool,
}

impl<O> StepIterator<O> {
    fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            skipped_first_step: false,
        }
    }
}

impl<O> StreamingIterator for StepIterator<O>
where
    O: Step,
{
    type Item = O;

    fn advance(&mut self) {
        // `advance` is called before the first `get`,
        // but we want to emit the initial state.
        if self.skipped_first_step {
            self.optimizer.step()
        } else {
            self.skipped_first_step = true;
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(&self.optimizer)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{prelude::*, Data};

    use super::*;

    #[test]
    fn iterate_emits_initial_state() {
        assert_eq!(
            (MockOptimizer {
                config: MockConfig::default(),
                state: MockState::new(),
                f,
            })
            .iterate()
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
                f,
            })
            .iterate()
            .find(|o| o.is_done())
            .unwrap()
            .state
            .steps,
            config.max_steps
        )
    }

    fn f(xs: CowArray<f64, Ix2>) -> Array1<f64> {
        xs.sum_axis(Axis(xs.ndim() - 1))
    }

    struct MockOptimizer<F> {
        pub config: MockConfig,
        pub state: MockState,
        pub f: F,
    }

    impl<F> Step for MockOptimizer<F>
    where
        F: Fn(CowArray<f64, Ix2>) -> Array1<f64>,
    {
        fn step(&mut self) {
            self.step_from_evaluated((self.f)(self.points()))
        }
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

    impl<F> Points<f64> for MockOptimizer<F> {
        fn points(&self) -> CowArray<f64, Ix2> {
            Array2::zeros((0, 0)).into()
        }
    }

    impl<F> StepFromEvaluated<f64> for MockOptimizer<F> {
        fn step_from_evaluated<S>(&mut self, _point_values: ArrayBase<S, Ix1>)
        where
            S: Data<Elem = f64>,
        {
            self.state.steps += 1;
        }
    }

    impl<F> IsDone for MockOptimizer<F> {
        fn is_done(&self) -> bool {
            self.state.steps >= self.config.max_steps
        }
    }
}
