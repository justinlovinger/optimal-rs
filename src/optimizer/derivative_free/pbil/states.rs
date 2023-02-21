use super::types::*;
use ndarray::{prelude::*, Data};
use ndarray_rand::RandomExt;
use rand::{
    distributions::{Bernoulli, Standard},
    prelude::{SeedableRng, SmallRng},
    Rng,
};
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Initial and post-evaluation state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Init<R> {
    probabilities: Array1<Probability>,
    rng: R,
}

impl Init<SmallRng> {
    /// Return recommended initial state.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each sample
    pub fn default(num_bits: usize) -> Self {
        Self {
            probabilities: Array::from_elem(num_bits, Probability::default()),
            rng: SmallRng::from_entropy(),
        }
    }
}

impl<R: Rng> Init<R> {
    /// Return custom initial state.
    pub fn new(probabilities: Array1<Probability>, rng: R) -> Self {
        Self { probabilities, rng }
    }

    /// With chance 'chance',
    /// adjust each probability towards a random probability
    /// at rate 'adjust_rate'.
    pub fn mutate(&mut self, chance: MutationChance, adjust_rate: MutationAdjustRate) {
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
    }

    /// Step to a 'PreEval' state.
    pub fn to_pre_eval(mut self, num_samples: NumSamples) -> PreEval<R> {
        PreEval {
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
}

impl<R> Init<R> {
    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }
}

/// State with samples ready for evaluation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PreEval<R> {
    probabilities: Array1<Probability>,
    rng: R,
    samples: Array2<bool>,
}

impl<R> PreEval<R> {
    /// Step to 'Init' state
    /// by adjusting probabilities
    /// towards the best sample.
    ///
    /// # Arguments
    ///
    /// - `point_values`: value of each sample,
    ///    each element corresponding to a row of `samples`.
    pub fn to_init<A: Debug + PartialOrd, S: Data<Elem = A>>(
        mut self,
        adjust_rate: AdjustRate,
        sample_values: ArrayBase<S, Ix1>,
    ) -> Init<R> {
        adjust_probabilities(
            &mut self.probabilities,
            adjust_rate.into(),
            best_sample(&self.samples, sample_values),
        );
        Init {
            probabilities: self.probabilities,
            rng: self.rng,
        }
    }

    // TODO: add `to_mutate` and `Mutate` state;
    // move `mutate` out of `Init`.

    /// Return probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        &self.probabilities
    }

    /// Return samples to evaluate.
    pub fn samples(&self) -> &Array2<bool> {
        &self.samples
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
                    .unwrap_or_else(|| panic!("Cannot compare {:?} with {:?}", x, y))
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
    fn probabilities_are_valid_after_adjust(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        num_samples: NumSamples,
        adjust_rate: AdjustRate,
    ) {
        let init = Init::new(initial_probabilities.into(), SmallRng::seed_from_u64(seed));
        let pre_eval = init.to_pre_eval(num_samples);
        let point_values = f(pre_eval.samples().view());
        let init = pre_eval.to_init(adjust_rate, point_values);
        prop_assert!(are_valid(init.probabilities()));
    }

    #[proptest(failure_persistence = Some(Box::new(FileFailurePersistence::Off)))]
    fn probabilities_are_valid_after_mutate(
        initial_probabilities: Vec<Probability>,
        seed: u64,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) {
        let mut init = Init::new(initial_probabilities.into(), SmallRng::seed_from_u64(seed));
        init.mutate(mutation_chance, mutation_adjust_rate);
        prop_assert!(are_valid(init.probabilities()));
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
