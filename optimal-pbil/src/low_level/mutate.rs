use optimal_compute_core::{
    impl_core_ops,
    peano::{One, Zero},
    Args, Computation, ComputationFn,
};
use rand::Rng;

use super::{MutationAdjustRate, MutationChance, Probability};

/// With `chance`,
/// adjust each probability towards a random probability
/// at `adjust_rate`.
#[derive(Clone, Copy, Debug)]
pub struct Mutate<C, A, P, R> {
    pub(crate) chance: C,
    pub(crate) adjust_rate: A,
    pub(crate) probabilities: P,
    pub(crate) rng: R,
}

impl<C, A, P, R> Mutate<C, A, P, R> {
    #[allow(missing_docs)]
    pub fn new(chance: C, adjust_rate: A, probabilities: P, rng: R) -> Self
    where
        Self: Computation,
    {
        Self {
            chance,
            adjust_rate,
            probabilities,
            rng,
        }
    }
}

impl<C, A, P, R> Computation for Mutate<C, A, P, R>
where
    C: Computation<Dim = Zero, Item = MutationChance>,
    A: Computation<Dim = Zero, Item = MutationAdjustRate>,
    P: Computation<Dim = One, Item = Probability>,
    R: Computation<Dim = Zero>,
    R::Item: Rng,
{
    type Dim = (Zero, One);
    type Item = (R::Item, Probability);
}

impl<C, A, P, R> ComputationFn for Mutate<C, A, P, R>
where
    Self: Computation,
    C: ComputationFn,
    A: ComputationFn,
    P: ComputationFn,
    R: ComputationFn,
{
    fn args(&self) -> Args {
        self.chance
            .args()
            .union(self.adjust_rate.args())
            .union(self.probabilities.args())
            .union(self.rng.args())
    }
}

impl_core_ops!(Mutate<C, A, P, R>);

mod run {
    use optimal_compute_core::run::{ArgVals, DistributeArgs, RunCore, Unwrap, Value};
    use rand::{distributions::Standard, Rng};

    use crate::low_level::{adjust, MutationAdjustRate, MutationChance, Probability};

    use super::Mutate;

    impl<C, A, P, R, POut, ROut> RunCore for Mutate<C, A, P, R>
    where
        (C, A, P, R): DistributeArgs<
            Output = (
                Value<MutationChance>,
                Value<MutationAdjustRate>,
                Value<POut>,
                Value<ROut>,
            ),
        >,
        POut: IntoIterator<Item = Probability>,
        ROut: Rng,
    {
        type Output = (Value<ROut>, Value<Vec<Probability>>);

        fn run_core(self, args: ArgVals) -> Self::Output {
            let (chance, adjust_rate, probabilities, mut rng) =
                (self.chance, self.adjust_rate, self.probabilities, self.rng)
                    .distribute(args)
                    .unwrap();
            let out = mutate_probabilities(chance, adjust_rate, probabilities, &mut rng).collect();
            (Value(rng), Value(out))
        }
    }

    fn mutate_probabilities<'a, R>(
        chance: MutationChance,
        adjust_rate: MutationAdjustRate,
        probabilities: impl IntoIterator<Item = Probability> + 'a,
        rng: &'a mut R,
    ) -> impl Iterator<Item = Probability> + 'a
    where
        R: Rng,
    {
        let distr = chance.into_distr();
        probabilities.into_iter().map(move |p| {
            if rng.sample(distr) {
                // `Standard` distribution excludes `1`,
                // but it more efficient
                // than `Uniform::new_inclusive(0., 1.)`.
                // This operation is safe
                // because Probability is closed under `adjust`
                // with rate in [0,1].
                unsafe {
                    Probability::new_unchecked(adjust(
                        adjust_rate.into(),
                        f64::from(p),
                        rng.sample(Standard),
                    ))
                }
            } else {
                p
            }
        })
    }
}
