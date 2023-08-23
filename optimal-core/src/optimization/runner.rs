use streaming_iterator::StreamingIterator;

use crate::prelude::Optimizer;

/// A runner
/// able to determine when an optimization sequence is done
/// and run it to completion.
pub trait Runner: StreamingIterator {
    /// Type of inner iterator.
    type It;

    /// Stop the optimization run,
    /// returning inner optimization sequence.
    fn stop(self) -> Self::It
    where
        Self: Sized,
        Self::It: Sized;

    /// Run to completion.
    fn run(mut self) -> Self::It
    where
        Self: Sized,
        Self::It: Sized,
    {
        while self.next().is_some() {}
        self.stop()
    }

    /// Return point that attempts to minimize a problem
    /// by running to completion.
    ///
    /// How well the point minimizes the problem
    /// depends on the optimizer.
    fn argmin(self) -> <Self::It as Optimizer>::Point
    where
        Self: Sized,
        Self::It: Sized + Optimizer,
    {
        self.run().best_point()
    }
}
