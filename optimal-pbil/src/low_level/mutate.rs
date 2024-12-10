use core::fmt;

use computation_types::{
    impl_core_ops,
    peano::{One, Zero},
    Computation, ComputationFn, NamedArgs, Names,
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
    type Dim = (One, Zero);
    type Item = (Probability, R::Item);
}

impl<C, A, P, R> ComputationFn for Mutate<C, A, P, R>
where
    Self: Computation,
    C: ComputationFn,
    A: ComputationFn,
    P: ComputationFn,
    R: ComputationFn,
    Mutate<C::Filled, A::Filled, P::Filled, R::Filled>: Computation,
{
    type Filled = Mutate<C::Filled, A::Filled, P::Filled, R::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        let (args_0, args_1, args_2, args_3) = named_args
            .partition4(
                &self.chance.arg_names(),
                &self.adjust_rate.arg_names(),
                &self.probabilities.arg_names(),
                &self.rng.arg_names(),
            )
            .unwrap_or_else(|e| panic!("{}", e,));
        Mutate {
            chance: self.chance.fill(args_0),
            adjust_rate: self.adjust_rate.fill(args_1),
            probabilities: self.probabilities.fill(args_2),
            rng: self.rng.fill(args_3),
        }
    }

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
    use computation_types::{run::RunCore, Computation};
    use rand::{distributions::Standard, Rng};

    use crate::low_level::{adjust, MutationAdjustRate, MutationChance, Probability};

    use super::Mutate;

    impl<C, A, P, R> RunCore for Mutate<C, A, P, R>
    where
        Self: Computation,
        C: RunCore<Output = MutationChance>,
        A: RunCore<Output = MutationAdjustRate>,
        P: RunCore,
        R: RunCore,
        P::Output: IntoIterator<Item = Probability>,
        R::Output: Rng,
    {
        type Output = (Vec<Probability>, R::Output);

        fn run_core(self) -> Self::Output {
            let mut rng = self.rng.run_core();
            let out = mutate_probabilities(
                self.chance.run_core(),
                self.adjust_rate.run_core(),
                self.probabilities.run_core(),
                &mut rng,
            )
            .collect();
            (out, rng)
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
