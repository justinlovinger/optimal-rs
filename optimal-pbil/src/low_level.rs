//! Low-level functions offering greater flexibility.
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{*, peano::*, run::Value};
//! use optimal_pbil::{low_level::*, types::*};
//! use ::rand::prelude::*;
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
//!     let rng = SmallRng::from_entropy();
//!     let probabilities = std::iter::repeat(Probability::default())
//!         .take(len)
//!         .collect::<Vec<_>>();
//!
//!     let pbil = PointFrom::new(
//!         val!(rng)
//!             .zip(val1!(probabilities))
//!             .loop_while(
//!                 ("rng", "probabilities"),
//!                 arg1!("probabilities", Probability)
//!                     .zip(BestSample::new(
//!                         val!(num_samples),
//!                         arg1!("sample").black_box::<_, Zero, usize>(|sample: Vec<bool>| {
//!                             Value(obj_func(&sample))
//!                         }),
//!                         arg1!("probabilities", Probability),
//!                         arg!("rng", SmallRng),
//!                     ))
//!                     .then(
//!                         ("probabilities", ("rng", "sample")),
//!                         Mutate::new(
//!                             val!(mutation_chance),
//!                             val!(mutation_adjust_rate),
//!                             Adjust::new(val!(adjust_rate), arg1!("probabilities"), arg1!("sample")),
//!                             arg!("rng", SmallRng),
//!                         ),
//!                     ),
//!                 Converged::new(val!(threshold), arg1!("probabilities", Probability)).not(),
//!             )
//!             .then(
//!                 ("rng", "probabilities"),
//!                 arg1!("probabilities", Probability),
//!             ),
//!     );
//!
//!     println!("{:?}", pbil.run(argvals![]));
//! }
//!
//! fn obj_func(point: &[bool]) -> usize {
//!     point.iter().filter(|x| **x).count()
//! }
//! ```

mod adjust;
mod best_sample;
mod converged;
mod mutate;
mod point_from;

use std::ops::{Add, Mul, Sub};

use crate::types::*;

pub use self::{
    adjust::Adjust, best_sample::BestSample, converged::Converged, mutate::Mutate,
    point_from::PointFrom, sampleable::Sampleable,
};

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
    use num_traits::bounds::{LowerBounded, UpperBounded};
    use optimal_compute_core::{argvals, val, val1, Computation, Run};
    use proptest::prelude::*;
    use rand::{rngs::SmallRng, SeedableRng};
    use test_strategy::proptest;

    use super::*;

    #[proptest()]
    fn probabilities_are_valid_after_adjusting(
        adjust_rate: AdjustRate,
        #[strategy(Vec::<Probability>::arbitrary().prop_flat_map(|xs| (prop::collection::vec(bool::arbitrary(), xs.len()), Just(xs))))]
        args: (Vec<bool>, Vec<Probability>),
    ) {
        let (sample, probabilities) = args;
        prop_assert!(are_valid(
            Adjust::new(val!(adjust_rate), val1!(probabilities), val1!(sample)).run(argvals![])
        ));
    }

    #[proptest()]
    fn probabilities_are_valid_after_mutating(
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
        seed: u64,
        probabilities: Vec<Probability>,
    ) {
        prop_assert!(are_valid(
            Mutate::new(
                val!(mutation_chance),
                val!(mutation_adjust_rate),
                val1!(probabilities),
                val!(SmallRng::seed_from_u64(seed)),
            )
            .snd()
            .run(argvals![])
        ));
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
