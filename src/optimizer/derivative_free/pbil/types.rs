use core::convert::TryFrom;
use derive_more::{Display, From, FromStr, Into};
use num_traits::bounds::{LowerBounded, UpperBounded};
use rand::distributions::Bernoulli;
use std::f64::EPSILON;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::derive::{
    derive_from_str_from_try_into, derive_try_from_bounded_float, derive_try_from_lower_bounded,
};

/// Number of bits in generated points
/// and probabilities in PBIL.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, From, FromStr, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NumBits(usize);

/// Number of samples generated
/// during steps.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

derive_try_from_lower_bounded!(usize, NumSamples);

derive_from_str_from_try_into!(usize, NumSamples);

/// Degree to adjust probabilities towards best point
/// during steps.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

derive_try_from_bounded_float!(f64, AdjustRate);

derive_from_str_from_try_into!(f64, AdjustRate);

/// Probability for each probability to mutate,
/// independently.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "f64"))]
pub struct MutationChance(f64);

impl MutationChance {
    /// Return recommended default mutation chance,
    /// average of one mutation per step.
    pub fn default(num_bits: NumBits) -> Self {
        if num_bits == 0.into() {
            Self(1.)
        } else {
            Self(1. / usize::from(num_bits) as f64)
        }
    }
}

impl LowerBounded for MutationChance {
    fn min_value() -> Self {
        Self(0.)
    }
}

impl UpperBounded for MutationChance {
    fn max_value() -> Self {
        Self(1.)
    }
}

impl From<MutationChance> for Bernoulli {
    fn from(x: MutationChance) -> Self {
        Bernoulli::new(x.into()).unwrap()
    }
}

derive_try_from_bounded_float!(f64, MutationChance);

derive_from_str_from_try_into!(f64, MutationChance);

/// Degree to adjust probability towards random value
/// when mutating.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

derive_try_from_bounded_float!(f64, MutationAdjustRate);

derive_from_str_from_try_into!(f64, MutationAdjustRate);

/// Probability for a sampled bit to be true.
#[derive(Clone, Copy, Debug, Display, PartialEq, PartialOrd, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

derive_try_from_bounded_float!(f64, Probability);

derive_from_str_from_try_into!(f64, Probability);

/// PBIL can be considered done
/// when all probabilities are above this threshold
/// or below the inverse.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergedThreshold {
    ub: Probability,
    lb: Probability,
}

impl LowerBounded for ConvergedThreshold {
    fn min_value() -> Self {
        Self {
            ub: Probability(0.5 + EPSILON),
            lb: Probability(0.5 - EPSILON),
        }
    }
}

impl UpperBounded for ConvergedThreshold {
    fn max_value() -> Self {
        Self {
            ub: Probability(1. - EPSILON),
            lb: Probability(EPSILON),
        }
    }
}

impl From<ConvergedThreshold> for Probability {
    fn from(x: ConvergedThreshold) -> Self {
        x.ub
    }
}

impl TryFrom<Probability> for ConvergedThreshold {
    type Error = ConvergedThresholdTryFromError;

    fn try_from(value: Probability) -> Result<Self, Self::Error> {
        if value < Self::min_value().into() {
            Err(Self::Error::TooLow)
        } else if value > Self::max_value().into() {
            Err(Self::Error::TooHigh)
        } else {
            Ok(Self {
                ub: value,
                lb: Probability(1. - f64::from(value)),
            })
        }
    }
}

/// Error returned when 'ConvergedThreshold' is given an invalid value.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum ConvergedThresholdTryFromError {
    /// Value is below the lower bound.
    TooLow,
    /// Value is above the upper bound.
    TooHigh,
}

impl ConvergedThreshold {
    /// Return the threshold upper bound.
    pub fn upper_bound(&self) -> Probability {
        self.ub
    }

    /// Return the threshold lower bound.
    pub fn lower_bound(&self) -> Probability {
        self.lb
    }
}

impl Default for ConvergedThreshold {
    fn default() -> Self {
        Self {
            ub: Probability(0.75),
            lb: Probability(0.25),
        }
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
            MutationChance(0.2)
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
