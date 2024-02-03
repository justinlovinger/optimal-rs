//! Types for PBIL.

use core::convert::TryFrom;
use std::f64::EPSILON;

use derive_more::{Display, Into};
use derive_num_bounded::{
    derive_from_str_from_try_into, derive_into_inner, derive_new_from_bounded_float,
    derive_new_from_lower_bounded, derive_try_from_from_new,
};
use num_traits::bounds::{LowerBounded, UpperBounded};
use rand::distributions::Bernoulli;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Number of samples generated
/// during steps.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "usize"))]
#[cfg_attr(feature = "serde", serde(try_from = "usize"))]
pub struct NumSamples(usize);

impl Default for NumSamples {
    fn default() -> Self {
        Self(20)
    }
}

impl LowerBounded for NumSamples {
    fn min_value() -> Self {
        Self(2)
    }
}

derive_new_from_lower_bounded!(NumSamples(usize));
derive_into_inner!(NumSamples(usize));
derive_try_from_from_new!(NumSamples(usize));
derive_from_str_from_try_into!(NumSamples(usize));

/// Degree to adjust probabilities towards best point
/// during steps.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "f64"))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
pub struct AdjustRate(f64);

impl Default for AdjustRate {
    fn default() -> Self {
        Self(0.1)
    }
}

impl LowerBounded for AdjustRate {
    fn min_value() -> Self {
        Self(EPSILON)
    }
}

impl UpperBounded for AdjustRate {
    fn max_value() -> Self {
        Self(1.)
    }
}

derive_new_from_bounded_float!(AdjustRate(f64));
derive_into_inner!(AdjustRate(f64));
derive_try_from_from_new!(AdjustRate(f64));
derive_from_str_from_try_into!(AdjustRate(f64));

impl Eq for AdjustRate {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for AdjustRate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `f64` has total ordering for the the range of values allowed by this type.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

/// Probability for each probability to mutate,
/// independently.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "f64"))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
pub struct MutationChance(f64);

derive_new_from_bounded_float!(MutationChance(f64));
derive_into_inner!(MutationChance(f64));
derive_try_from_from_new!(MutationChance(f64));
derive_from_str_from_try_into!(MutationChance(f64));

impl Eq for MutationChance {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for MutationChance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `f64` has total ordering for the the range of values allowed by this type.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

impl LowerBounded for MutationChance {
    fn min_value() -> Self {
        Self(0.0)
    }
}

impl UpperBounded for MutationChance {
    fn max_value() -> Self {
        Self(1.0)
    }
}

impl From<MutationChance> for Bernoulli {
    fn from(value: MutationChance) -> Self {
        Bernoulli::new(value.0).unwrap()
    }
}

impl MutationChance {
    /// Return recommended default mutation chance,
    /// average of one mutation per step.
    pub fn default_for(len: usize) -> Self {
        if len == 0 {
            Self(1.0)
        } else {
            Self(1. / len as f64)
        }
    }

    /// Return whether no chance to mutate.
    pub fn is_zero(&self) -> bool {
        self.0 == 0.0
    }

    /// Return a distribution for sampling.
    pub fn into_distr(self) -> Bernoulli {
        self.into()
    }
}

/// Degree to adjust probability towards random value
/// when mutating.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "f64"))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
pub struct MutationAdjustRate(f64);

impl Default for MutationAdjustRate {
    fn default() -> Self {
        Self(0.05)
    }
}

impl LowerBounded for MutationAdjustRate {
    fn min_value() -> Self {
        Self(EPSILON)
    }
}

impl UpperBounded for MutationAdjustRate {
    fn max_value() -> Self {
        Self(1.)
    }
}

derive_new_from_bounded_float!(MutationAdjustRate(f64));
derive_into_inner!(MutationAdjustRate(f64));
derive_try_from_from_new!(MutationAdjustRate(f64));
derive_from_str_from_try_into!(MutationAdjustRate(f64));

impl Eq for MutationAdjustRate {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for MutationAdjustRate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `f64` has total ordering for the the range of values allowed by this type.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

/// Probability for a sampled bit to be true.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "f64"))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
pub struct Probability(f64);

impl Probability {
    /// # Safety
    ///
    /// This function is safe
    /// if the given value
    /// is within range `[0,1]`.
    pub const unsafe fn new_unchecked(x: f64) -> Self {
        Self(x)
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self(0.5)
    }
}

impl LowerBounded for Probability {
    fn min_value() -> Self {
        Self(0.)
    }
}

impl UpperBounded for Probability {
    fn max_value() -> Self {
        Self(1.)
    }
}

derive_new_from_bounded_float!(Probability(f64));
derive_into_inner!(Probability(f64));
derive_try_from_from_new!(Probability(f64));
derive_from_str_from_try_into!(Probability(f64));

impl Eq for Probability {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for Probability {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `f64` has total ordering for the the range of values allowed by this type.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

/// PBIL can be considered done
/// when all probabilities are above this threshold
/// or below the inverse.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "Probability"))]
#[cfg_attr(feature = "serde", serde(try_from = "Probability"))]
pub struct ProbabilityThreshold {
    ub: Probability,
    lb: Probability,
}

/// Error returned when 'ConvergedThreshold' is given an invalid value.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum InvalidProbabilityThresholdError {
    /// Value is below the lower bound.
    TooLow,
    /// Value is above the upper bound.
    TooHigh,
}

impl ProbabilityThreshold {
    /// Return a new 'ConvergedThreshold' if given a valid value.
    pub fn new(value: Probability) -> Result<Self, InvalidProbabilityThresholdError> {
        if value < Self::min_value().into() {
            Err(InvalidProbabilityThresholdError::TooLow)
        } else if value > Self::max_value().into() {
            Err(InvalidProbabilityThresholdError::TooHigh)
        } else {
            Ok(Self {
                ub: value,
                lb: Probability(1. - f64::from(value)),
            })
        }
    }

    /// Unwrap 'ConvergedThreshold' into inner value.
    pub fn into_inner(self) -> Probability {
        self.ub
    }

    /// Return the threshold upper bound.
    pub fn upper_bound(&self) -> Probability {
        self.ub
    }

    /// Return the threshold lower bound.
    pub fn lower_bound(&self) -> Probability {
        self.lb
    }
}

impl LowerBounded for ProbabilityThreshold {
    fn min_value() -> Self {
        Self {
            ub: Probability(0.5 + EPSILON),
            lb: Probability(0.5 - EPSILON),
        }
    }
}

impl UpperBounded for ProbabilityThreshold {
    fn max_value() -> Self {
        Self {
            ub: Probability(1. - EPSILON),
            lb: Probability(EPSILON),
        }
    }
}

impl Default for ProbabilityThreshold {
    fn default() -> Self {
        Self {
            ub: Probability(0.75),
            lb: Probability(0.25),
        }
    }
}

// `lb` is fully dependent on `ub`.
impl PartialEq for ProbabilityThreshold {
    fn eq(&self, other: &Self) -> bool {
        self.ub.eq(&other.ub)
    }
}
impl Eq for ProbabilityThreshold {}
impl PartialOrd for ProbabilityThreshold {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ub.partial_cmp(&other.ub)
    }
}
impl Ord for ProbabilityThreshold {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `f64` has total ordering for the the range of values allowed by this type.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

impl From<ProbabilityThreshold> for Probability {
    fn from(x: ProbabilityThreshold) -> Self {
        x.ub
    }
}

impl TryFrom<Probability> for ProbabilityThreshold {
    type Error = InvalidProbabilityThresholdError;
    fn try_from(value: Probability) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_samples_from_str_returns_correct_result() {
        assert_eq!("10".parse::<NumSamples>().unwrap(), NumSamples(10));
    }

    #[test]
    fn adjust_rate_from_str_returns_correct_result() {
        assert_eq!("0.2".parse::<AdjustRate>().unwrap(), AdjustRate(0.2));
    }

    #[test]
    fn mutation_chance_from_str_returns_correct_result() {
        assert_eq!(
            "0.2".parse::<MutationChance>().unwrap(),
            MutationChance::new(0.2).unwrap()
        );
    }

    #[test]
    fn mutation_adjust_rate_from_str_returns_correct_result() {
        assert_eq!(
            "0.2".parse::<MutationAdjustRate>().unwrap(),
            MutationAdjustRate(0.2)
        );
    }
}
