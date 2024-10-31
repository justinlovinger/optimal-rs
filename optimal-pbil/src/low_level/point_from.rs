use core::fmt;

use computation_types::{impl_core_ops, peano::One, Computation, ComputationFn, NamedArgs, Names};

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
    PointFrom<P::Filled>: Computation,
{
    type Filled = PointFrom<P::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        PointFrom {
            probabilities: self.probabilities.fill(named_args),
        }
    }

    fn arg_names(&self) -> Names {
        self.probabilities.arg_names()
    }
}

impl_core_ops!(PointFrom<P>);

impl<P> fmt::Display for PointFrom<P>
where
    Self: Computation,
    P: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "point_from({})", self.probabilities)
    }
}

mod run {
    use computation_types::{run::RunCore, Computation, NamedArgs, Unwrap, Value};

    use crate::low_level::Probability;

    use super::PointFrom;

    impl<P, POut> RunCore for PointFrom<P>
    where
        Self: Computation,
        P: RunCore<Output = Value<POut>>,
        POut: IntoIterator<Item = Probability>,
    {
        type Output = Value<std::iter::Map<POut::IntoIter, fn(Probability) -> bool>>;

        fn run_core(self, args: NamedArgs) -> Self::Output {
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
