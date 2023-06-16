use blanket::blanket;
use streaming_iterator::StreamingIterator;

pub use self::extensions::*;

/// A runner configuration.
///
/// A runner can determine when an optimization sequence is done
/// and run it to completion.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait RunnerConfig<I> {
    /// Type of this runners state.
    type State;

    /// Return the initial state of this runner.
    fn initial_state(&self) -> Self::State;

    /// Return if optimization is done.
    fn is_done(&self, it: &I, state: &Self::State) -> bool;

    /// Update the state of this runner.
    fn update(&self, _it: &I, _state: &mut Self::State) {}

    /// Advance the optimization sequence,
    /// updating the state of this runner
    /// if necessary.
    fn advance(&self, it: &mut I, _state: &mut Self::State)
    where
        I: StreamingIterator,
    {
        it.advance()
    }
}

mod extensions {
    use ndarray::prelude::*;

    use crate::prelude::*;

    use super::*;

    /// Automatically implemented extensions to `RunnerConfig`.
    pub trait RunnerConfigExt<I>: RunnerConfig<I> {
        /// Start the optimization run
        /// defined by this config
        /// and the given optimization sequence.
        fn start(self, it: I) -> Runner<I, Self>
        where
            Self: Sized,
        {
            Runner {
                inner: it,
                state: self.initial_state(),
                config: self,
            }
        }

        /// Run the given sequence to completion.
        fn run(self, it: I) -> I
        where
            Self: Sized,
            I: StreamingIterator,
        {
            let mut it = self.start(it);
            while it.next().is_some() {}
            it.into_inner().0
        }

        /// Return point that attempts to minimize a problem
        /// by running the given optimizer to completion.
        ///
        /// How well the point minimizes the problem
        /// depends on the optimizer.
        fn argmin<P>(self, it: I) -> Array1<P::PointElem>
        where
            Self: Sized,
            P: Problem,
            P::PointElem: Clone,
            I: StreamingIterator + RunningOptimizerConfigless<P>,
        {
            self.run(it).best_point().to_owned()
        }
    }

    impl<I, T> RunnerConfigExt<I> for T where T: RunnerConfig<I> {}

    /// An optimization run.
    #[derive(Clone, Debug)]
    #[cfg_attr(
        any(test, feature = "serde"),
        derive(serde::Serialize, serde::Deserialize)
    )]
    pub struct Runner<I, C>
    where
        C: RunnerConfig<I>,
    {
        inner: I,
        config: C,
        state: C::State,
    }

    impl<I, C> Runner<I, C>
    where
        C: RunnerConfig<I>,
    {
        /// Stop optimization run,
        /// returning inner optimization sequence,
        /// runner configuration,
        /// and runner state.
        pub fn into_inner(self) -> (I, C, C::State) {
            (self.inner, self.config, self.state)
        }
    }

    impl<I, C> StreamingIterator for Runner<I, C>
    where
        I: StreamingIterator,
        C: RunnerConfig<I>,
    {
        type Item = I::Item;

        fn advance(&mut self) {
            self.config.update(&self.inner, &mut self.state);
            self.config.advance(&mut self.inner, &mut self.state);
        }

        fn get(&self) -> Option<&Self::Item> {
            if self.config.is_done(&self.inner, &self.state) {
                None
            } else {
                self.inner.get()
            }
        }
    }
}
