use core::convert::TryFrom;
use std::{f64::EPSILON, fmt};

use derive_more::{Display, Into};
use derive_num_bounded::{
    derive_from_str_from_try_into, derive_into_inner, derive_new_from_bounded_float,
    derive_new_from_lower_bounded, derive_try_from_from_new,
};
use num_traits::bounds::{LowerBounded, UpperBounded};
use rand::{distributions::Bernoulli, prelude::Distribution};

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
#[derive(Clone)]
pub struct MutationChance {
    chance: f64,
    distribution: Bernoulli,
}

/// Error returned when [`MutationChance`] is given an invalid value.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
pub enum InvalidMutationChanceError {
    /// Value is NaN.
    #[error("{0} is NaN")]
    IsNan(f64),
    /// Value is below lower bound.
    #[error("{0} is below lower bound ({})", MutationChance::min_value())]
    TooLow(f64),
    /// Value is above upper bound.
    #[error("{0} is above upper bound ({})", MutationChance::max_value())]
    TooHigh(f64),
}

impl fmt::Debug for MutationChance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MutationChance").field(&self.chance).finish()
    }
}

impl fmt::Display for MutationChance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.chance.fmt(f)
    }
}

impl Eq for MutationChance {}

impl Ord for MutationChance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `f64` has total ordering for the the range of values allowed by this type.
        unsafe { self.partial_cmp(other).unwrap_unchecked() }
    }
}

impl PartialEq for MutationChance {
    fn eq(&self, other: &Self) -> bool {
        self.chance.eq(&other.chance)
    }
}

impl PartialOrd for MutationChance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.chance.partial_cmp(&other.chance)
    }
}

impl From<MutationChance> for f64 {
    fn from(value: MutationChance) -> Self {
        value.chance
    }
}

impl LowerBounded for MutationChance {
    fn min_value() -> Self {
        unsafe { Self::new_unchecked(0.0) }
    }
}

impl UpperBounded for MutationChance {
    fn max_value() -> Self {
        unsafe { Self::new_unchecked(1.0) }
    }
}

impl Distribution<bool> for MutationChance {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
        self.distribution.sample(rng)
    }
}

#[cfg(any(serde, test))]
impl<'de> serde::Deserialize<'de> for MutationChance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let chance = f64::deserialize(deserializer)?;
        match MutationChance::new(chance) {
            Ok(x) => Ok(x),
            Err(e) => Err(<D::Error as serde::de::Error>::custom(e)),
        }
    }
}

#[cfg(any(serde, test))]
impl serde::Serialize for MutationChance {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.chance)
    }
}

derive_try_from_from_new!(MutationChance(f64));
derive_from_str_from_try_into!(MutationChance(f64));

impl MutationChance {
    /// Return a new [`MutationChance`] if given a valid value.
    pub fn new(value: f64) -> Result<Self, InvalidMutationChanceError> {
        match (
            value.partial_cmp(&Self::min_value().chance),
            value.partial_cmp(&Self::max_value().chance),
        ) {
            (None, _) | (_, None) => Err(InvalidMutationChanceError::IsNan(value)),
            (Some(std::cmp::Ordering::Less), _) => Err(InvalidMutationChanceError::TooLow(value)),
            (_, Some(std::cmp::Ordering::Greater)) => {
                Err(InvalidMutationChanceError::TooHigh(value))
            }
            _ => Ok(unsafe { Self::new_unchecked(value) }),
        }
    }

    /// Return recommended default mutation chance,
    /// average of one mutation per step.
    pub fn default_for(len: usize) -> Self {
        if len == 0 {
            unsafe { Self::new_unchecked(1.0) }
        } else {
            unsafe { Self::new_unchecked(1. / len as f64) }
        }
    }

    /// # Safety
    ///
    /// Value must be within range.
    unsafe fn new_unchecked(value: f64) -> Self {
        Self {
            chance: value,
            distribution: Bernoulli::new(value).unwrap(),
        }
    }

    /// Unwrap [`MutationChance`] into inner value.
    pub fn into_inner(self) -> f64 {
        self.chance
    }

    /// Return whether no chance to mutate.
    pub fn is_zero(&self) -> bool {
        self.chance == 0.0
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
    use proptest::prelude::*;
    use test_strategy::proptest;

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

    #[proptest()]
    fn mutation_chance_serializes_correctly(chance: MutationChance) {
        prop_assert!(
            (serde_json::from_str::<MutationChance>(&serde_json::to_string(&chance).unwrap())
                .unwrap()
                .into_inner()
                - chance.into_inner())
                < 1e10
        )
    }

    #[test]
    fn mutation_adjust_rate_from_str_returns_correct_result() {
        assert_eq!(
            "0.2".parse::<MutationAdjustRate>().unwrap(),
            MutationAdjustRate(0.2)
        );
    }
}
