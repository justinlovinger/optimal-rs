use core::fmt;

use optimal_compute_core::{
    impl_core_ops,
    peano::{One, Zero},
    Names, Computation, ComputationFn,
};
use rand::Rng;

use super::{MutationAdjustRate, MutationChance, Probability};

/// With `chance`,
/// adjust each probability towards a random probability
/// at `adjust_rate`.
#[derive(Clone, Copy, Debug)]
pub struct Mutate<C, A, P, R>
where
    Self: Computation,
{
    /// Computation representing [`MutationChance`].
    pub chance: C,
    /// Computation representing [`MutationAdjustRate`].
    pub adjust_rate: A,
    /// Computation representing probabilities to mutate.
    pub probabilities: P,
    /// Computation representing a source of randomness.
    pub rng: R,
}

impl<C, A, P, R> Mutate<C, A, P, R>
where
    Self: Computation,
{
    #[allow(missing_docs)]
    pub fn new(chance: C, adjust_rate: A, probabilities: P, rng: R) -> Self {
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
    fn arg_names(&self) -> Names {
        self.chance
            .arg_names()
            .union(self.adjust_rate.arg_names())
            .union(self.probabilities.arg_names())
            .union(self.rng.arg_names())
    }
}

impl_core_ops!(Mutate<C, A, P, R>);

impl<C, A, P, R> fmt::Display for Mutate<C, A, P, R>
where
    Self: Computation,
    C: fmt::Display,
    A: fmt::Display,
    P: fmt::Display,
    R: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mutate({}, {}, {}, {})",
            self.chance, self.adjust_rate, self.probabilities, self.rng
        )
    }
}

mod run {
    use optimal_compute_core::{
        run::{ArgVals, DistributeArgs, RunCore, Unwrap, Value},
        Computation,
    };
    use rand::{distributions::Standard, Rng};

    use crate::low_level::{adjust, MutationAdjustRate, MutationChance, Probability};

    use super::Mutate;

    impl<C, A, P, R, POut, ROut> RunCore for Mutate<C, A, P, R>
    where
        Self: Computation,
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
