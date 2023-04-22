use std::marker::PhantomData;

use streaming_iterator::StreamingIterator;

use crate::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl<P, C, S, O> IntoStreamingIterator<P, C, S> for O {
    fn into_streaming_iter(self) -> StepIterator<P, C, S, O> {
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
pub trait IntoStreamingIterator<P, C, S> {
    /// Return an iterator over optimizer states.
    fn into_streaming_iter(self) -> StepIterator<P, C, S, Self>
    where
        Self: Sized;
}

/// An iterator returned by [`into_streaming_iter`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StepIterator<P, C, S, O> {
    problem: PhantomData<P>,
    config: PhantomData<C>,
    state: PhantomData<S>,
    inner: O,
    skipped_first_step: bool,
}

impl<P, C, S, O> StepIterator<P, C, S, O> {
    fn new(optimizer: O) -> Self {
        Self {
            problem: PhantomData,
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

impl<P, C, S, O> StreamingIterator for StepIterator<P, C, S, O>
where
    O: RunningOptimizerStep<Problem = P, Config = C, State = S>,
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
    use ndarray::prelude::*;
    use rand::prelude::*;
    use rand_xoshiro::SplitMix64;

    use crate::optimizer::derivative_free::pbil;

    use super::*;

    #[test]
    fn iterator_emits_initial_state() {
        let seed = 0;
        let config = pbil::Config::default(Count);
        assert_eq!(
            config
                .clone()
                .start_using(&mut SplitMix64::seed_from_u64(seed))
                .into_streaming_iter()
                .next()
                .unwrap()
                .state(),
            config
                .start_using(&mut SplitMix64::seed_from_u64(seed))
                .state()
        );
    }

    #[test]
    fn iterator_runs_for_same_number_of_steps() {
        let seed = 0;
        let steps = 100;
        let config = pbil::Config::default(Count);
        let mut o = config
            .clone()
            .start_using(&mut SplitMix64::seed_from_u64(seed));
        for _ in 0..steps {
            o.step();
        }
        assert_eq!(
            config
                .start_using(&mut SplitMix64::seed_from_u64(seed))
                .into_streaming_iter()
                .nth(steps)
                .unwrap()
                .state(),
            o.state()
        );
    }

    #[derive(Clone, Debug)]
    struct Count;

    impl Problem for Count {
        type PointElem = bool;
        type PointValue = u64;

        fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
            point.fold(0, |acc, b| acc + *b as u64)
        }
    }

    impl FixedLength for Count {
        fn len(&self) -> usize {
            16
        }
    }
}
