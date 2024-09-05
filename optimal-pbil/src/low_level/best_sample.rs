use optimal_compute_core::{
    impl_core_ops,
    peano::{One, Zero},
    Args, Computation, ComputationFn,
};
use rand::Rng;

use super::{NumSamples, Probability};

/// See [`Sampleable::best_sample`].
#[derive(Clone, Copy, Debug)]
pub struct BestSample<N, F, P, R>
where
    Self: Computation,
{
    /// Computation representing [`NumSamples`].
    pub num_samples: N,
    /// Computation representing objective function.
    pub obj_func: F,
    /// Computation representing probabilities to sample from.
    pub probabilities: P,
    /// Computation representing a source of randomness.
    pub rng: R,
}

impl<N, F, P, R> BestSample<N, F, P, R>
where
    Self: Computation,
{
    #[allow(missing_docs)]
    pub fn new(num_samples: N, obj_func: F, probabilities: P, rng: R) -> Self {
        Self {
            num_samples,
            obj_func,
            probabilities,
            rng,
        }
    }
}

impl<N, F, P, R> Computation for BestSample<N, F, P, R>
where
    N: Computation<Dim = Zero, Item = NumSamples>,
    F: ComputationFn<Dim = Zero>,
    F::Item: PartialOrd,
    P: Computation<Dim = One, Item = Probability>,
    R: Computation<Dim = Zero>,
    R::Item: Rng,
{
    type Dim = (Zero, One);
    type Item = (R::Item, bool);
}

impl<N, F, P, R> ComputationFn for BestSample<N, F, P, R>
where
    Self: Computation,
    N: ComputationFn,
    P: ComputationFn,
    R: ComputationFn,
{
    fn args(&self) -> Args {
        self.num_samples
            .args()
            .union(self.probabilities.args())
            .union(self.rng.args())
    }
}

impl_core_ops!(BestSample<N, F, P, R>);

mod run {
    use optimal_compute_core::{
        argvals,
        peano::One,
        run::{Collect, DistributeArgs, RunCore, Unwrap, Value},
        Computation, Run,
    };
    use rand::Rng;

    use crate::low_level::{NumSamples, Probability, Sampleable};

    use super::BestSample;

    impl<N, F, P, R, POut, ROut> RunCore for BestSample<N, F, P, R>
    where
        Self: Computation,
        (N, P, R): DistributeArgs<Output = (Value<NumSamples>, POut, Value<ROut>)>,
        POut: Collect<One, Collected = Value<Vec<Probability>>>,
        ROut: Rng,
        F: Clone + Run,
        F::Output: PartialOrd,
    {
        type Output = (Value<ROut>, Value<Vec<bool>>);

        fn run_core(self, args: optimal_compute_core::run::ArgVals) -> Self::Output {
            let (num_samples, probabilities, mut rng) =
                (self.num_samples, self.probabilities, self.rng)
                    .distribute(args)
                    .collect()
                    .unwrap();
            let out = Sampleable::new(&probabilities).best_sample(
                num_samples,
                |sample| {
                    // Note,
                    // this is likely inefficient,
                    // but it lets us treat a computation-function
                    // like a closure.
                    self.obj_func
                        .clone()
                        .run(argvals![("sample", sample.to_owned())])
                },
                &mut rng,
            );
            (Value(rng), Value(out))
        }
    }
}
