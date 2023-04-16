use std::marker::PhantomData;

use streaming_iterator::StreamingIterator;

use crate::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl<A, B, C, S, O> IntoStreamingIterator<A, B, C, S> for O
where
    O: RunningOptimizer<PointElem = A, PointValue = B, Config = C, State = S>,
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
    O: RunningOptimizer<PointElem = A, PointValue = B, Config = C, State = S>,
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
    use ndarray::{prelude::*, Data};
    use rand::prelude::*;

    use crate::optimizer::derivative_free::pbil;

    use super::*;

    #[test]
    fn iterator_emits_initial_state() {
        let config = pbil::Config::default(Count);
        assert_eq!(
            pbil::Running::new_using(&config, &mut StdRng::seed_from_u64(0))
                .into_streaming_iter()
                .next()
                .unwrap()
                .state(),
            pbil::Running::new_using(&config, &mut StdRng::seed_from_u64(0)).state()
        );
    }

    #[test]
    fn iterator_runs_for_same_number_of_steps() {
        let steps = 100;
        let config = pbil::Config::default(Count);
        let mut o = pbil::Running::new_using(&config, &mut StdRng::seed_from_u64(0));
        for _ in 0..steps {
            o.step();
        }
        assert_eq!(
            pbil::Running::new_using(&config, &mut StdRng::seed_from_u64(0))
                .into_streaming_iter()
                .nth(steps)
                .unwrap()
                .state(),
            o.state()
        );
    }

    struct Count;

    impl Problem<bool, u64> for Count {
        fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> u64
        where
            S: ndarray::RawData<Elem = bool> + Data,
        {
            point.fold(0, |acc, b| acc + *b as u64)
        }
    }

    impl FixedLength for Count {
        fn len(&self) -> usize {
            16
        }
    }
}
