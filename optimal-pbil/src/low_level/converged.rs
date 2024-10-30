use core::fmt;

use computation_types::{
    impl_core_ops,
    peano::{One, Zero},
    Computation, ComputationFn, Names,
};

use super::{Probability, ProbabilityThreshold};

/// Return whether all probabilities are above the given threshold
/// or below its inverse.
#[derive(Clone, Copy, Debug)]
pub struct Converged<T, P>
where
    Self: Computation,
{
    /// Computation representing [`ProbabilityThreshold`].
    pub threshold: T,
    /// Computation representing probabilities to check.
    pub probabilities: P,
}

impl<T, P> Converged<T, P>
where
    Self: Computation,
{
    #[allow(missing_docs)]
    pub fn new(threshold: T, probabilities: P) -> Self {
        Self {
            threshold,
            probabilities,
        }
    }
}

impl<T, P> Computation for Converged<T, P>
where
    T: Computation<Dim = Zero, Item = ProbabilityThreshold>,
    P: Computation<Dim = One, Item = Probability>,
{
    type Dim = Zero;
    type Item = bool;
}

impl<T, P> ComputationFn for Converged<T, P>
where
    Self: Computation,
    T: ComputationFn,
    P: ComputationFn,
{
    fn arg_names(&self) -> Names {
        self.threshold
            .arg_names()
            .union(self.probabilities.arg_names())
    }
}

impl_core_ops!(Converged<T, P>);

impl<T, P> fmt::Display for Converged<T, P>
where
    Self: Computation,
    T: fmt::Display,
    P: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "converged({}, {})", self.threshold, self.probabilities)
    }
}

mod run {
    use computation_types::{
        run::{DistributeArgs, NamedArgs, RunCore, Unwrap, Value},
        Computation,
    };

    use crate::low_level::{Probability, ProbabilityThreshold};

    use super::Converged;

    impl<T, P, POut> RunCore for Converged<T, P>
    where
        Self: Computation,
        (T, P): DistributeArgs<Output = (Value<ProbabilityThreshold>, Value<POut>)>,
        POut: IntoIterator<Item = Probability>,
    {
        type Output = Value<bool>;

        fn run_core(self, args: NamedArgs) -> Self::Output {
            let (threshold, probabilities) = (self.threshold, self.probabilities)
                .distribute(args)
                .unwrap();
            Value(converged(threshold, probabilities))
        }
    }

    fn converged(
        threshold: ProbabilityThreshold,
        probabilities: impl IntoIterator<Item = Probability>,
    ) -> bool {
        probabilities
            .into_iter()
            .all(|p| p > threshold.upper_bound() || p < threshold.lower_bound())
    }
}
