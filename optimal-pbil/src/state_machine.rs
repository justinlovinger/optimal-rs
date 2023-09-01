use rand::{
    distributions::{Bernoulli, Standard},
    prelude::*,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

use super::types::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynState<B> {
    Ready(State<Ready>),
    Sampling(State<Sampling<B>>),
    Mutating(State<Mutating>),
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct State<T> {
    probabilities: Vec<Probability>,
    rng: Xoshiro256PlusPlus,
    inner: T,
}

/// PBIL state ready to start sampling.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ready {
    distrs: Vec<Bernoulli>,
    sample: Vec<bool>,
}

/// PBIL state for sampling
/// and adjusting probabilities
/// based on samples.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sampling<B> {
    distrs: Vec<Bernoulli>,
    best_sample: Vec<bool>,
    best_value: B,
    sample: Vec<bool>,
    samples_generated: usize,
}

/// PBIL state for mutating probabilities.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mutating;

impl<B> DynState<B> {
    pub fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self::Ready(State::<Ready>::new(probabilities, rng))
    }
}

impl<T> State<T> {
    /// Return probabilities.
    pub fn probabilities(&self) -> &[Probability] {
        &self.probabilities
    }
}

impl State<Ready> {
    /// Return custom initial state.
    fn new(probabilities: Vec<Probability>, mut rng: Xoshiro256PlusPlus) -> Self {
        let distrs = probabilities
            .iter()
            .map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"))
            .collect::<Vec<_>>();
        Self {
            inner: Ready {
                sample: distrs.iter().map(|d| rng.sample(d)).collect(),
                distrs,
            },
            probabilities,
            rng,
        }
    }

    /// Step to a 'Sampling' state
    /// by evaluating the first sample.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_sampling<B>(mut self, value: B) -> State<Sampling<B>> {
        let distrs = self
            .probabilities
            .iter()
            .map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"))
            .collect::<Vec<_>>();
        State {
            inner: Sampling {
                sample: distrs.iter().map(|d| self.rng.sample(d)).collect(),
                distrs,
                best_sample: self.inner.sample,
                best_value: value,
                samples_generated: 2, // This includes `sample` and `best_sample`.
            },
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    /// Return sample to evaluate.
    pub fn sample(&self) -> &[bool] {
        &self.inner.sample
    }
}

impl<B> State<Sampling<B>> {
    #[allow(clippy::wrong_self_convention)]
    pub fn to_sampling(mut self, value: B) -> Self
    where
        B: PartialOrd,
    {
        let (best_sample, best_value) = if value > self.inner.best_value {
            (self.inner.sample, value)
        } else {
            (self.inner.best_sample, self.inner.best_value)
        };

        Self {
            inner: Sampling {
                sample: self
                    .inner
                    .distrs
                    .iter()
                    .map(|d| self.rng.sample(d))
                    .collect(),
                distrs: self.inner.distrs,
                best_sample,
                best_value,
                samples_generated: self.inner.samples_generated + 1,
            },
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    /// Step to 'Mutating' state
    /// by adjusting probabilities
    /// towards the best sample.
    ///
    /// # Arguments
    ///
    /// - `value`: value of sample.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_mutating(mut self, adjust_rate: AdjustRate, value: B) -> State<Mutating>
    where
        B: PartialOrd,
    {
        let best_sample = if value > self.inner.best_value {
            self.inner.sample
        } else {
            self.inner.best_sample
        };
        adjust_probabilities(&mut self.probabilities, adjust_rate.into(), best_sample);

        State {
            inner: Mutating,
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    /// Step to 'Ready' state,
    /// skipping 'Mutating'.
    ///
    /// # Arguments
    ///
    /// - `value`: value of sample.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_ready(self, adjust_rate: AdjustRate, value: B) -> State<Ready>
    where
        B: PartialOrd,
    {
        let state = self.to_mutating(adjust_rate, value);
        State::<Ready>::new(state.probabilities, state.rng)
    }

    pub fn samples_generated(&self) -> usize {
        self.inner.samples_generated
    }

    /// Return sample to evaluate.
    pub fn sample(&self) -> &[bool] {
        &self.inner.sample
    }
}

impl State<Mutating> {
    /// Step to 'Ready' state
    /// by randomly mutating probabilities.
    ///
    /// With chance `chance`,
    /// adjust each probability towards a random probability
    /// at rate `adjust_rate`.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_ready(
        mut self,
        chance: MutationChance,
        adjust_rate: MutationAdjustRate,
    ) -> State<Ready> {
        let distr: Bernoulli = chance.into();
        self.probabilities.iter_mut().for_each(|p| {
            if self.rng.sample(distr) {
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
        });
        State::<Ready>::new(self.probabilities, self.rng)
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
        let state = State::<Ready>::new(
            initial_probabilities,
            Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let value = f(state.sample());
        let state = state.to_sampling(value);
        let value = f(state.sample());
        let state = state.to_ready(adjust_rate, value);
        prop_assert!(are_valid(state.probabilities()));
    }

    #[proptest(failure_persistence = Some(Box::new(FileFailurePersistence::Off)))]
    fn probabilities_are_valid_after_mutating(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) {
        let state = State {
            inner: Mutating,
            probabilities: initial_probabilities,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
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

    fn f(point: &[bool]) -> usize {
        point.iter().filter(|x| **x).count()
    }

    fn are_valid(probabilities: &[Probability]) -> bool {
        probabilities
            .iter()
            .all(|p| Probability::try_from(f64::from(*p)).is_ok())
    }
}
