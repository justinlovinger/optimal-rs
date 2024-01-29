//! Low-level functions offering greater flexibility.
//!
//! # Examples
//!
//! ```
//! use optimal_pbil::{low_level::*, types::*};
//! use rand::prelude::*;
//!
//! fn main() {
//!     let len = 2;
//!
//!     let num_samples = NumSamples::default();
//!     let adjust_rate = AdjustRate::default();
//!     let mutation_chance = MutationChance::default_for(len);
//!     let mutation_adjust_rate = MutationAdjustRate::default();
//!     let threshold = ProbabilityThreshold::default();
//!
//!     let mut rng = SmallRng::from_entropy();
//!     let mut probabilities = std::iter::repeat(Probability::default())
//!         .take(len)
//!         .collect::<Vec<_>>();
//!     while !converged(threshold, probabilities.iter().copied()) {
//!         probabilities = mutate_probabilities(
//!             &mutation_chance,
//!             mutation_adjust_rate,
//!             adjust_probabilities(
//!                 adjust_rate,
//!                 Sampleable::new(&probabilities).best_sample(num_samples, obj_func, &mut rng),
//!                 probabilities,
//!             ),
//!             &mut rng,
//!         )
//!         .collect();
//!     }
//!
//!     println!("{:?}", point_from(probabilities).collect::<Vec<_>>());
//! }
//!
//! fn obj_func(point: &[bool]) -> usize {
//!     point.iter().filter(|x| **x).count()
//! }
//! ```

use std::ops::{Add, Mul, Sub};

use rand::{distributions::Standard, prelude::*};

use crate::types::*;

pub use self::sampleable::Sampleable;

/// Adjust each probability towards corresponding `sample` bit
/// at `rate`.
pub fn adjust_probabilities(
    rate: AdjustRate,
    sample: impl IntoIterator<Item = bool>,
    probabilities: impl IntoIterator<Item = Probability>,
) -> impl Iterator<Item = Probability> {
    probabilities.into_iter().zip(sample).map(move |(p, b)| {
        // This operation is safe
        // because Probability is closed under `adjust`
        // with rate in [0,1].
        unsafe { Probability::new_unchecked(adjust(rate.into(), f64::from(p), b as u8 as f64)) }
    })
}

/// Adjust a number from `x` to `y`
/// at given rate.
fn adjust<T>(rate: T, x: T, y: T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    x + rate * (y - x)
}

/// With `chance`,
/// adjust each probability towards a random probability
/// at `adjust_rate`.
pub fn mutate_probabilities<'a, R>(
    chance: &'a MutationChance,
    adjust_rate: MutationAdjustRate,
    probabilities: impl IntoIterator<Item = Probability> + 'a,
    rng: &'a mut R,
) -> impl Iterator<Item = Probability> + 'a
where
    R: Rng,
{
    probabilities.into_iter().map(move |p| {
        if rng.sample(chance) {
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

/// Return whether all probabilities are above the given threshold
/// or below its inverse.
pub fn converged(
    threshold: ProbabilityThreshold,
    probabilities: impl IntoIterator<Item = Probability>,
) -> bool {
    probabilities
        .into_iter()
        .all(|p| p > threshold.upper_bound() || p < threshold.lower_bound())
}

/// Estimate the best sample discovered
/// from a set of probabilities.
pub fn point_from(
    probabilities: impl IntoIterator<Item = Probability>,
) -> impl Iterator<Item = bool> {
    probabilities.into_iter().map(|p| f64::from(p) >= 0.5)
}

mod sampleable {
    use rand::{distributions::Bernoulli, prelude::*};

    use crate::types::{NumSamples, Probability};

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    /// Probabilities prepared to sample.
    #[derive(Clone, Debug, PartialEq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct Sampleable<P> {
        probabilities: P,
        distrs: Vec<Bernoulli>,
    }

    impl<P> Sampleable<P> {
        #[allow(missing_docs)]
        pub fn new(probabilities: P) -> Self
        where
            P: AsRef<[Probability]>,
        {
            Self {
                distrs: probabilities
                    .as_ref()
                    .iter()
                    .map(|p| Bernoulli::new(f64::from(*p)).expect("Invalid probability"))
                    .collect(),
                probabilities,
            }
        }

        /// Return the sample that minimizes the objective function
        /// among `num_samples` samples
        /// sampled from `probabilities`.
        pub fn best_sample<F, V, R>(
            &self,
            num_samples: NumSamples,
            obj_func: F,
            rng: &mut R,
        ) -> Vec<bool>
        where
            F: Fn(&[bool]) -> V,
            V: PartialOrd,
            R: Rng,
        {
            std::iter::repeat_with(|| self.sample(rng))
                .take(num_samples.into())
                .map(|sample| (obj_func(&sample), sample))
                .min_by(|(value, _), (other, _)| value.partial_cmp(other).unwrap())
                .map(|(_, sample)| sample)
                .unwrap()
        }

        #[allow(missing_docs)]
        pub fn probabilities(&self) -> &P {
            &self.probabilities
        }

        #[allow(missing_docs)]
        pub fn into_probabilities(self) -> P {
            self.probabilities
        }
    }

    impl<P> Distribution<Vec<bool>> for Sampleable<P> {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<bool> {
            self.distrs.iter().map(|d| rng.sample(d)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::bounds::{LowerBounded, UpperBounded};
    use proptest::prelude::*;
    use test_strategy::proptest;

    #[proptest()]
    fn probabilities_are_valid_after_adjusting(
        adjust_rate: AdjustRate,
        #[strategy(Vec::<Probability>::arbitrary().prop_flat_map(|xs| (prop::collection::vec(bool::arbitrary(), xs.len()), Just(xs))))]
        args: (Vec<bool>, Vec<Probability>),
    ) {
        let (sample, probabilities) = args;
        prop_assert!(are_valid(adjust_probabilities(
            adjust_rate,
            sample,
            probabilities
        )));
    }

    #[proptest()]
    fn probabilities_are_valid_after_mutating(
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
        seed: u64,
        probabilities: Vec<Probability>,
    ) {
        prop_assert!(are_valid(mutate_probabilities(
            &mutation_chance,
            mutation_adjust_rate,
            probabilities,
            &mut SmallRng::seed_from_u64(seed),
        )));
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

    fn are_valid(probabilities: impl IntoIterator<Item = Probability>) -> bool {
        probabilities
            .into_iter()
            .all(|p| Probability::try_from(f64::from(p)).is_ok())
    }
}
