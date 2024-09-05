use optimal_compute_core::{impl_core_ops, peano::One, Args, Computation, ComputationFn};

use super::Probability;

/// Estimate the best sample discovered
/// from a set of probabilities.
#[derive(Clone, Copy, Debug)]
pub struct PointFrom<P>
where
    Self: Computation,
{
    /// Computation representing probabilities to make a point from.
    pub probabilities: P,
}

impl<P> PointFrom<P>
where
    Self: Computation,
{
    #[allow(missing_docs)]
    pub fn new(probabilities: P) -> Self {
        Self { probabilities }
    }
}

impl<P> Computation for PointFrom<P>
where
    P: Computation<Dim = One, Item = Probability>,
{
    type Dim = One;
    type Item = bool;
}

impl<P> ComputationFn for PointFrom<P>
where
    Self: Computation,
    P: ComputationFn,
{
    fn args(&self) -> Args {
        self.probabilities.args()
    }
}

impl_core_ops!(PointFrom<P>);

mod run {
    use optimal_compute_core::{
        run::{RunCore, Unwrap, Value},
        Computation,
    };

    use crate::low_level::Probability;

    use super::PointFrom;

    impl<P, POut> RunCore for PointFrom<P>
    where
        Self: Computation,
        P: RunCore<Output = Value<POut>>,
        POut: IntoIterator<Item = Probability>,
    {
        type Output = Value<std::iter::Map<POut::IntoIter, fn(Probability) -> bool>>;

        fn run_core(self, args: optimal_compute_core::run::ArgVals) -> Self::Output {
            Value(
                self.probabilities
                    .run_core(args)
                    .unwrap()
                    .into_iter()
                    .map(|p| f64::from(p) >= 0.5),
            )
        }
    }
}
