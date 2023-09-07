use partial_min_max::min;
use rand::{
    distributions::{Bernoulli, Standard},
    prelude::*,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

use self::{sampleable::Sampleable, valued::Valued};

use super::types::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynState<B> {
    Started(Started),
    SampledFirst(SampledFirst),
    EvaluatedFirst(EvaluatedFirst<B>),
    Sampled(Sampled<B>),
    Evaluated(Evaluated<B>),
    Compared(Compared<B>),
    Adjusted(Adjusted),
    Mutated(Mutated),
    Finished(Finished),
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Started {
    pub probabilities: Vec<Probability>,
    pub rng: Xoshiro256PlusPlus,
}

/// PBIL state ready to start sampling.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InitializedSampling {
    pub probabilities: Sampleable,
    pub rng: Xoshiro256PlusPlus,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SampledFirst {
    pub probabilities: Sampleable,
    pub rng: Xoshiro256PlusPlus,
    pub sample: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvaluatedFirst<B> {
    pub probabilities: Sampleable,
    pub rng: Xoshiro256PlusPlus,
    pub sample: Valued<Vec<bool>, B>,
}

/// PBIL state for sampling
/// and adjusting probabilities
/// based on samples.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sampled<B> {
    pub probabilities: Sampleable,
    pub rng: Xoshiro256PlusPlus,
    pub samples_generated: usize,
    pub best_sample: Valued<Vec<bool>, B>,
    pub sample: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Evaluated<B> {
    pub probabilities: Sampleable,
    pub rng: Xoshiro256PlusPlus,
    pub samples_generated: usize,
    pub best_sample: Valued<Vec<bool>, B>,
    pub sample: Valued<Vec<bool>, B>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Compared<B> {
    pub probabilities: Sampleable,
    pub rng: Xoshiro256PlusPlus,
    pub samples_generated: usize,
    pub best_sample: Valued<Vec<bool>, B>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Adjusted {
    pub probabilities: Vec<Probability>,
    pub rng: Xoshiro256PlusPlus,
}

/// PBIL state for mutating probabilities.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mutated {
    pub probabilities: Vec<Probability>,
    pub rng: Xoshiro256PlusPlus,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Finished {
    pub probabilities: Vec<Probability>,
    pub rng: Xoshiro256PlusPlus,
}

impl<B> DynState<B> {
    pub fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self::Started(Started::new(probabilities, rng))
    }
}

impl Started {
    fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self { probabilities, rng }
    }

    pub fn into_initialized_sampling(self) -> InitializedSampling {
        InitializedSampling::new(self.probabilities, self.rng)
    }
}

impl InitializedSampling {
    fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self {
            probabilities: Sampleable::new(probabilities),
            rng,
        }
    }

    pub fn into_sampled_first(self) -> SampledFirst {
        SampledFirst::new(self.probabilities, self.rng)
    }
}

impl SampledFirst {
    fn new(probabilities: Sampleable, mut rng: Xoshiro256PlusPlus) -> Self {
        Self {
            sample: probabilities.sample(&mut rng),
            probabilities,
            rng,
        }
    }

    pub fn into_evaluated_first<B>(self, f: impl FnOnce(&[bool]) -> B) -> EvaluatedFirst<B> {
        EvaluatedFirst::new(self.probabilities, self.rng, self.sample, f)
    }
}

impl<B> EvaluatedFirst<B> {
    fn new(
        probabilities: Sampleable,
        rng: Xoshiro256PlusPlus,
        sample: Vec<bool>,
        f: impl FnOnce(&[bool]) -> B,
    ) -> Self {
        Self {
            probabilities,
            rng,
            sample: Valued::new(sample, f),
        }
    }

    pub fn into_sampled(self) -> Sampled<B> {
        Sampled::new(self.probabilities, self.rng, 1, self.sample)
    }
}

impl<B> Sampled<B> {
    fn new(
        probabilities: Sampleable,
        mut rng: Xoshiro256PlusPlus,
        samples_generated: usize,
        best_sample: Valued<Vec<bool>, B>,
    ) -> Self {
        Self {
            sample: probabilities.sample(&mut rng),
            probabilities,
            rng,
            best_sample,
            samples_generated: samples_generated + 1,
        }
    }

    pub fn into_evaluated(self, f: impl FnOnce(&[bool]) -> B) -> Evaluated<B> {
        Evaluated::new(
            self.probabilities,
            self.rng,
            self.samples_generated,
            self.best_sample,
            self.sample,
            f,
        )
    }
}

impl<B> Evaluated<B> {
    fn new(
        probabilities: Sampleable,
        rng: Xoshiro256PlusPlus,
        samples_generated: usize,
        best_sample: Valued<Vec<bool>, B>,
        sample: Vec<bool>,
        f: impl FnOnce(&[bool]) -> B,
    ) -> Self {
        Self {
            probabilities,
            rng,
            samples_generated,
            best_sample,
            sample: Valued::new(sample, f),
        }
    }

    pub fn into_compared(self) -> Compared<B>
    where
        B: PartialOrd,
    {
        Compared::new(
            self.probabilities,
            self.rng,
            self.samples_generated,
            self.best_sample,
            self.sample,
        )
    }
}

impl<B> Compared<B> {
    fn new(
        probabilities: Sampleable,
        rng: Xoshiro256PlusPlus,
        samples_generated: usize,
        best_sample: Valued<Vec<bool>, B>,
        sample: Valued<Vec<bool>, B>,
    ) -> Self
    where
        B: PartialOrd,
    {
        Self {
            probabilities,
            rng,
            samples_generated,
            best_sample: min(sample, best_sample),
        }
    }

    pub fn into_sampled(self) -> Sampled<B> {
        Sampled::new(
            self.probabilities,
            self.rng,
            self.samples_generated,
            self.best_sample,
        )
    }

    pub fn into_adjusted(self, adjust_rate: AdjustRate) -> Adjusted {
        Adjusted::new(
            adjust_rate,
            self.probabilities.into_probabilities(),
            self.rng,
            self.best_sample.into_parts().0,
        )
    }
}

impl Adjusted {
    fn new(
        adjust_rate: AdjustRate,
        mut probabilities: Vec<Probability>,
        rng: Xoshiro256PlusPlus,
        best_sample: Vec<bool>,
    ) -> Self {
        adjust_probabilities(&mut probabilities, adjust_rate.into(), best_sample);
        Self { probabilities, rng }
    }

    pub fn into_mutated(self, chance: MutationChance, adjust_rate: MutationAdjustRate) -> Mutated {
        Mutated::new(chance, adjust_rate, self.probabilities, self.rng)
    }

    pub fn into_finished(self) -> Finished {
        Finished::new(self.probabilities, self.rng)
    }
}

impl Mutated {
    /// Step to 'Mutated' state
    /// by randomly mutating probabilities.
    ///
    /// With chance `chance`,
    /// adjust each probability towards a random probability
    /// at rate `adjust_rate`.
    pub fn new(
        chance: MutationChance,
        adjust_rate: MutationAdjustRate,
        mut probabilities: Vec<Probability>,
        mut rng: Xoshiro256PlusPlus,
    ) -> Self {
        let distr: Bernoulli = chance.into();
        probabilities.iter_mut().for_each(|p| {
            if rng.sample(distr) {
                // `Standard` distribution excludes `1`,
                // but it more efficient
                // than `Uniform::new_inclusive(0., 1.)`.
                // This operation is safe
                // because Probability is closed under `adjust`
                // with rate in [0,1].
                *p = unsafe {
                    Probability::new_unchecked(adjust(
                        adjust_rate.into(),
                        f64::from(*p),
                        rng.sample(Standard),
                    ))
                }
            }
        });
        Self { probabilities, rng }
    }

    pub fn into_finished(self) -> Finished {
        Finished::new(self.probabilities, self.rng)
    }
}

impl Finished {
    pub fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self { probabilities, rng }
    }

    pub fn into_started(self) -> Started {
        Started::new(self.probabilities, self.rng)
    }
}

fn adjust_probabilities(probabilities: &mut [Probability], adjust_rate: f64, sample: Vec<bool>) {
    probabilities.iter_mut().zip(&sample).for_each(|(p, b)| {
        // This operation is safe
        // because Probability is closed under `adjust`
        // with rate in [0,1].
        *p = unsafe {
            Probability::new_unchecked(adjust(adjust_rate, f64::from(*p), *b as u8 as f64))
        }
    });
}

/// Adjust a number from `x` to `y`
/// at given rate.
fn adjust<T>(rate: T, x: T, y: T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    x + rate * (y - x)
}

mod sampleable {
    use rand::{distributions::Bernoulli, prelude::*};

    use crate::Probability;

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, PartialEq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct Sampleable {
        probabilities: Vec<Probability>,
        distrs: Vec<Bernoulli>,
    }

    impl Sampleable {
        pub fn new(probabilities: Vec<Probability>) -> Self {
            Self {
                distrs: probabilities
                    .iter()
                    .map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"))
                    .collect::<Vec<_>>(),
                probabilities,
            }
        }

        pub fn probabilities(&self) -> &[Probability] {
            &self.probabilities
        }

        pub fn into_probabilities(self) -> Vec<Probability> {
            self.probabilities
        }
    }

    impl Distribution<Vec<bool>> for Sampleable {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<bool> {
            self.distrs.iter().map(|d| rng.sample(d)).collect()
        }
    }
}

mod valued {
    use std::borrow::Borrow;

    use derive_getters::{Dissolve, Getters};

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, PartialEq, Eq, Dissolve, Getters)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[dissolve(rename = "into_parts")]
    pub struct Valued<T, B> {
        x: T,
        value: B,
    }

    impl<T, B> Valued<T, B> {
        pub fn new<Borrowed, F>(x: T, f: F) -> Self
        where
            Borrowed: ?Sized,
            T: Borrow<Borrowed>,
            F: FnOnce(&Borrowed) -> B,
        {
            Self {
                value: f(x.borrow()),
                x,
            }
        }
    }

    impl<T, B> PartialOrd for Valued<T, B>
    where
        T: PartialEq,
        B: PartialOrd,
    {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::bounds::{LowerBounded, UpperBounded};
    use proptest::{prelude::*, test_runner::FileFailurePersistence};
    use test_strategy::proptest;

    #[proptest(failure_persistence = Some(Box::new(FileFailurePersistence::Off)))]
    fn probabilities_are_valid_after_adjusting(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        adjust_rate: AdjustRate,
    ) {
        let state = InitializedSampling::new(
            initial_probabilities,
            Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let state = state.into_sampled_first();
        let state = state.into_evaluated_first(f).into_sampled();
        let state = state
            .into_evaluated(f)
            .into_compared()
            .into_adjusted(adjust_rate);
        prop_assert!(are_valid(&state.probabilities));
    }

    #[proptest(failure_persistence = Some(Box::new(FileFailurePersistence::Off)))]
    fn probabilities_are_valid_after_mutating(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) {
        let state = Mutated::new(
            mutation_chance,
            mutation_adjust_rate,
            initial_probabilities,
            Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        prop_assert!(are_valid(&state.probabilities));
    }

    macro_rules! arbitrary_from_bounded {
        ( $from:ident, $to:ident ) => {
            impl Arbitrary for $to {
                type Parameters = ();
                type Strategy = BoxedStrategy<$to>;
                fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
                    ($from::from($to::min_value())..=$to::max_value().into())
                        .prop_map(|x| x.try_into().unwrap())
                        .boxed()
                }
            }
        };
    }

    arbitrary_from_bounded!(f64, Probability);
    arbitrary_from_bounded!(f64, AdjustRate);
    arbitrary_from_bounded!(f64, MutationChance);
    arbitrary_from_bounded!(f64, MutationAdjustRate);

    fn f(point: &[bool]) -> usize {
        point.iter().filter(|x| **x).count()
    }

    fn are_valid(probabilities: &[Probability]) -> bool {
        probabilities
            .iter()
            .all(|p| Probability::try_from(f64::from(*p)).is_ok())
    }
}
