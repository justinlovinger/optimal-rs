use derive_getters::Getters;
use optimal_core::prelude::*;

use crate::{types::*, Pbil};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A type containing an array of probabilities.
pub trait Probabilities {
    /// Return probabilities.
    fn probabilities(&self) -> &[Probability];
}

impl<B, F> Probabilities for Pbil<B, F> {
    fn probabilities(&self) -> &[Probability] {
        self.state().probabilities()
    }
}

/// PBIL runner
/// to check for converged probabilities.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UntilProbabilitiesConverged<I> {
    config: UntilProbabilitiesConvergedConfig,
    it: I,
}

/// Config for PBIL runner
/// to check for converged probabilities.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UntilProbabilitiesConvergedConfig {
    /// Probability convergence parameter.
    pub threshold: ProbabilityThreshold,
}

impl UntilProbabilitiesConvergedConfig {
    /// Return this runner
    /// wrapping the given iterator.
    pub fn start<I>(self, it: I) -> UntilProbabilitiesConverged<I> {
        UntilProbabilitiesConverged { config: self, it }
    }
}

impl<I> UntilProbabilitiesConverged<I> {
    /// Return configuration and iterator.
    pub fn into_inner(self) -> (UntilProbabilitiesConvergedConfig, I) {
        (self.config, self.it)
    }
}

impl<I> StreamingIterator for UntilProbabilitiesConverged<I>
where
    I: StreamingIterator + Probabilities,
{
    type Item = I::Item;

    fn advance(&mut self) {
        self.it.advance()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.it.get()
    }

    fn is_done(&self) -> bool {
        self.it.is_done()
            || self.it.probabilities().iter().all(|p| {
                p > &self.config.threshold.upper_bound() || p < &self.config.threshold.lower_bound()
            })
    }
}
