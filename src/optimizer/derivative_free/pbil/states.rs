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

/// PBIL state ready to start a new iteration.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ready {
    probabilities: Array1<Probability>,
    rng: Xoshiro256PlusPlus,
}

/// PBIL state for sampling
/// and adjusting probabilities
/// based on samples.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sampling {
    probabilities: Array1<Probability>,
    rng: Xoshiro256PlusPlus,
    samples: Array2<bool>,
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
    pub fn initial(num_bits: usize) -> Self {
        Self {
            probabilities: Array::from_elem(num_bits, Probability::default()),
            rng: Xoshiro256PlusPlus::from_entropy(),
        }
    }

    /// Return recommended initial state.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each sample
    /// - `rng`: source of randomness
    pub fn initial_using<R>(num_bits: usize, rng: R) -> Self
    where
        R: Rng,
    {
        Self {
            probabilities: Array::from_elem(num_bits, Probability::default()),
            rng: Xoshiro256PlusPlus::from_rng(rng).expect("RNG should initialize"),
        }
    }

    /// Return custom initial state.
    pub fn new(probabilities: Array1<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self { probabilities, rng }
    }

    /// Step to a 'Sampling' state
    /// by generating samples.
    pub fn to_sampling(mut self, num_samples: NumSamples) -> Sampling {
        Sampling {
            samples: self.samples(num_samples),
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    fn samples(&mut self, num_samples: NumSamples) -> Array2<bool> {
        self.probabilities
            .map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"))
            .broadcast((num_samples.into(), self.probabilities.len()))
            .unwrap()
            .map(|distr| self.rng.sample(distr))
    }

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }
}

impl Sampling {
    /// Step to 'Mutating' state
    /// by adjusting probabilities
    /// towards the best sample.
    ///
    /// # Arguments
    ///
    /// - `point_values`: value of each sample,
    ///    each element corresponding to a row of `samples`.
    pub fn to_mutating<A, S>(
        mut self,
        adjust_rate: AdjustRate,
        sample_values: ArrayBase<S, Ix1>,
    ) -> Mutating
    where
        A: Debug + PartialOrd,
        S: Data<Elem = A>,
    {
        adjust_probabilities(
            &mut self.probabilities,
            adjust_rate.into(),
            best_sample(&self.samples, sample_values),
        );
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
    pub fn to_ready<A, S>(self, adjust_rate: AdjustRate, sample_values: ArrayBase<S, Ix1>) -> Ready
    where
        A: Debug + PartialOrd,
        S: Data<Elem = A>,
    {
        let x = self.to_mutating(adjust_rate, sample_values);
        Ready {
            probabilities: x.probabilities,
            rng: x.rng,
        }
    }

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }

    /// Return samples to evaluate.
    pub fn samples(&self) -> &Array2<bool> {
        &self.samples
    }
}

impl Mutating {
    /// Step to 'Ready' state
    /// by randomly mutating probabilities.
    ///
    /// With chance `chance`,
    /// adjust each probability towards a random probability
    /// at rate `adjust_rate`.
    pub fn to_ready(mut self, chance: MutationChance, adjust_rate: MutationAdjustRate) -> Ready {
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
        Ready {
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }
}

fn best_sample<A, S1, S2>(
    points: &ArrayBase<S1, Ix2>,
    point_values: ArrayBase<S2, Ix1>,
) -> ArrayView1<bool>
where
    A: Debug + PartialOrd,
    S1: Data<Elem = bool>,
    S2: Data<Elem = A>,
{
    points.row(
        point_values
            .iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| {
                x.partial_cmp(y)
                    .unwrap_or_else(|| panic!("Cannot compare {x:?} with {y:?}"))
            })
            .map(|(i, _)| i)
            .expect("should have samples"),
    )
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

/// Finalize probabilities to bits.
pub fn finalize(probabilities: &Array1<Probability>) -> Array1<bool> {
    probabilities.map(|p| f64::from(*p) >= 0.5)
}

/// Have probabilities converged?
pub fn converged(threshold: &ConvergedThreshold, probabilities: &Array1<Probability>) -> bool {
    probabilities
        .iter()
        .all(|p| p > &threshold.upper_bound() || p < &threshold.lower_bound())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Data, RemoveAxis};
    use num_traits::bounds::{LowerBounded, UpperBounded};
    use proptest::{prelude::*, test_runner::FileFailurePersistence};
    use test_strategy::proptest;

    #[proptest(failure_persistence = Some(Box::new(FileFailurePersistence::Off)))]
    fn probabilities_are_valid_after_adjusting(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        num_samples: NumSamples,
        adjust_rate: AdjustRate,
    ) {
        let state = Ready::new(
            initial_probabilities.into(),
            Xoshiro256PlusPlus::seed_from_u64(seed),
        )
        .to_sampling(num_samples);
        let point_values = f(state.samples().view());
        prop_assert!(are_valid(
            state.to_ready(adjust_rate, point_values).probabilities()
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

    impl Arbitrary for NumSamples {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (NumSamples::min_value().into()..100)
                .prop_map(|x| x.try_into().unwrap())
                .boxed()
        }
    }

    fn f<S, D>(bss: ArrayBase<S, D>) -> Array<u64, D::Smaller>
    where
        S: Data<Elem = bool>,
        D: Dimension + RemoveAxis,
    {
        bss.fold_axis(Axis(bss.ndim() - 1), 0, |acc, b| acc + *b as u64)
    }

    fn are_valid(probabilities: &Array1<Probability>) -> bool {
        probabilities
            .iter()
            .all(|p| Probability::try_from(f64::from(*p)).is_ok())
    }
}
