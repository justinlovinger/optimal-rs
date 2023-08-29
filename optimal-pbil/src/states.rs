use super::types::*;
use ndarray::{prelude::*, Data};
use ndarray_rand::RandomExt;
use rand::{
    distributions::{Bernoulli, Standard},
    prelude::SeedableRng,
    Rng,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// PBIL state ready to start sampling.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ready {
    probabilities: Array1<Probability>,
    rng: Xoshiro256PlusPlus,
    distrs: Array1<Bernoulli>,
    sample: Array1<bool>,
}

/// PBIL state for sampling
/// and adjusting probabilities
/// based on samples.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sampling<B> {
    probabilities: Array1<Probability>,
    rng: Xoshiro256PlusPlus,
    distrs: Array1<Bernoulli>,
    best_sample: Array1<bool>,
    best_value: B,
    sample: Array1<bool>,
    samples_generated: usize,
}

/// PBIL state for mutating probabilities.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mutating {
    probabilities: Array1<Probability>,
    rng: Xoshiro256PlusPlus,
}

impl Ready {
    /// Return recommended initial state.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each sample
    pub(super) fn initial(num_bits: usize) -> Self {
        Self::new(
            Array::from_elem(num_bits, Probability::default()),
            Xoshiro256PlusPlus::from_entropy(),
        )
    }

    /// Return recommended initial state.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each sample
    /// - `rng`: source of randomness
    pub(super) fn initial_using<R>(num_bits: usize, rng: R) -> Self
    where
        R: Rng,
    {
        Self::new(
            Array::from_elem(num_bits, Probability::default()),
            Xoshiro256PlusPlus::from_rng(rng).expect("RNG should initialize"),
        )
    }

    /// Return custom initial state.
    pub(super) fn new(probabilities: Array1<Probability>, mut rng: Xoshiro256PlusPlus) -> Self {
        let distrs =
            probabilities.map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"));
        Self {
            sample: distrs.map(|d| rng.sample(d)),
            probabilities,
            rng,
            distrs,
        }
    }

    /// Step to a 'Sampling' state
    /// by evaluating the first sample.
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn to_sampling<B>(mut self, value: B) -> Sampling<B> {
        let distrs = self
            .probabilities
            .map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"));
        Sampling {
            sample: distrs.map(|d| self.rng.sample(d)),
            probabilities: self.probabilities,
            rng: self.rng,
            distrs,
            best_sample: self.sample,
            best_value: value,
            samples_generated: 2, // This includes `sample` and `best_sample`.
        }
    }

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }

    /// Return sample to evaluate.
    pub fn sample(&self) -> &Array1<bool> {
        &self.sample
    }
}

impl<B> Sampling<B> {
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn to_sampling(mut self, value: B) -> Sampling<B>
    where
        B: PartialOrd,
    {
        let (best_sample, best_value) = if value > self.best_value {
            (self.sample, value)
        } else {
            (self.best_sample, self.best_value)
        };
        Sampling {
            sample: self.distrs.map(|d| self.rng.sample(d)),
            probabilities: self.probabilities,
            rng: self.rng,
            distrs: self.distrs,
            best_sample,
            best_value,
            samples_generated: self.samples_generated + 1,
        }
    }

    /// Step to 'Mutating' state
    /// by adjusting probabilities
    /// towards the best sample.
    ///
    /// # Arguments
    ///
    /// - `point_values`: value of each sample,
    ///    each element corresponding to a row of `samples`.
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn to_mutating(mut self, adjust_rate: AdjustRate, value: B) -> Mutating
    where
        B: PartialOrd,
    {
        let best_sample = if value > self.best_value {
            self.sample
        } else {
            self.best_sample
        };
        adjust_probabilities(&mut self.probabilities, adjust_rate.into(), best_sample);
        Mutating {
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    /// Step to 'Ready' state,
    /// skipping 'Mutating'.
    ///
    /// # Arguments
    ///
    /// - `point_values`: value of each sample,
    ///    each element corresponding to a row of `samples`.
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn to_ready(self, adjust_rate: AdjustRate, value: B) -> Ready
    where
        B: PartialOrd,
    {
        let x = self.to_mutating(adjust_rate, value);
        Ready::new(x.probabilities, x.rng)
    }

    pub(super) fn samples_generated(&self) -> usize {
        self.samples_generated
    }

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }

    /// Return sample to evaluate.
    pub fn sample(&self) -> &Array1<bool> {
        &self.sample
    }
}

impl Mutating {
    /// Step to 'Ready' state
    /// by randomly mutating probabilities.
    ///
    /// With chance `chance`,
    /// adjust each probability towards a random probability
    /// at rate `adjust_rate`.
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn to_ready(
        mut self,
        chance: MutationChance,
        adjust_rate: MutationAdjustRate,
    ) -> Ready {
        let distr: Bernoulli = chance.into();
        self.probabilities.zip_mut_with(
            &Array::random_using(self.probabilities.len(), distr, &mut self.rng),
            |p, mutate| {
                if *mutate {
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
                            self.rng.sample(Standard),
                        ))
                    }
                }
            },
        );
        Ready::new(self.probabilities, self.rng)
    }

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }
}

fn adjust_probabilities<S>(
    probabilities: &mut Array1<Probability>,
    adjust_rate: f64,
    sample: ArrayBase<S, Ix1>,
) where
    S: Data<Elem = bool>,
{
    probabilities.zip_mut_with(&sample, |p, b| {
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
        let state = Ready::new(
            initial_probabilities.into(),
            Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let value = f(state.sample().view());
        let state = state.to_sampling(value);
        let value = f(state.sample().view());
        prop_assert!(are_valid(
            state.to_ready(adjust_rate, value).probabilities()
        ));
    }

    #[proptest(failure_persistence = Some(Box::new(FileFailurePersistence::Off)))]
    fn probabilities_are_valid_after_mutating(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) {
        let state = (Mutating {
            probabilities: initial_probabilities.into(),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        })
        .to_ready(mutation_chance, mutation_adjust_rate);
        prop_assert!(are_valid(state.probabilities()));
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

    fn f(point: ArrayView1<bool>) -> usize {
        point.iter().filter(|x| **x).count()
    }

    fn are_valid(probabilities: &Array1<Probability>) -> bool {
        probabilities
            .iter()
            .all(|p| Probability::try_from(f64::from(*p)).is_ok())
    }
}
