use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use rand::prelude::*;

use crate::{low_level::*, types::*};

/// PBIL independent of problem.
#[derive(Clone, Debug, PartialEq, PartialOrd, Builder)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pbil {
    /// See [`NumSamples`].
    #[builder(default)]
    pub num_samples: NumSamples,
    /// See [`AdjustRate`].
    #[builder(default)]
    pub adjust_rate: AdjustRate,
    /// See [`MutationChance`].
    pub mutation_chance: MutationChance,
    /// See [`MutationAdjustRate`].
    #[builder(default)]
    pub mutation_adjust_rate: MutationAdjustRate,
    /// See [`PbilStoppingCriteria`].
    #[builder(default)]
    pub stopping_criteria: PbilStoppingCriteria,
}

impl Pbil {
    /// Prepare PBIL for a specific problem.
    pub fn for_<F, V>(self, len: usize, obj_func: F) -> PbilFor<F>
    where
        F: Fn(&[bool]) -> V,
    {
        PbilFor {
            agnostic: self,
            len,
            obj_func,
        }
    }
}

impl PbilBuilder {
    /// Prepare PBIL for a specific problem.
    pub fn for_<F, V>(&mut self, len: usize, obj_func: F) -> PbilFor<F>
    where
        F: Fn(&[bool]) -> V,
    {
        if self.mutation_chance.is_none() {
            self.mutation_chance = Some(MutationChance::default_for(len))
        }
        self.build().unwrap().for_(len, obj_func)
    }
}

/// Options for stopping a PBIL optimization-loop.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PbilStoppingCriteria {
    /// Stop when the given iteration is reached.
    Iteration(usize),
    /// Stop when all probabilities reach the given threshold.
    Threshold(ProbabilityThreshold),
}

impl Default for PbilStoppingCriteria {
    fn default() -> Self {
        Self::Threshold(ProbabilityThreshold::default())
    }
}

/// PBIL for a specific problem.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PbilFor<F> {
    /// Problem-agnostic variables.
    pub agnostic: Pbil,
    /// Length of each point in problem.
    pub len: usize,
    /// Objective function to minimize.
    pub obj_func: F,
}

impl<F> PbilFor<F> {
    /// Prepare PBIL with state.
    pub fn with<R>(self, rng: R) -> PbilWith<F, R>
    where
        R: Rng,
    {
        PbilWith {
            rng,
            probabilities: std::iter::repeat(Probability::default())
                .take(self.len)
                .collect(),
            problem: self,
        }
    }

    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin<V>(self) -> Vec<bool>
    where
        F: Fn(&[bool]) -> V,
        V: PartialOrd,
    {
        self.with(SmallRng::from_entropy()).argmin()
    }
}

/// PBIL with state.
#[derive(Clone, Debug, Dissolve, Getters)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct PbilWith<F, R> {
    problem: PbilFor<F>,
    /// Source of randomness.
    pub rng: R,
    probabilities: Vec<Probability>,
}

impl<F, R> PbilWith<F, R> {
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin<V>(mut self) -> Vec<bool>
    where
        F: Fn(&[bool]) -> V,
        V: PartialOrd,
        R: Rng,
    {
        match self.problem.agnostic.stopping_criteria {
            PbilStoppingCriteria::Iteration(i) => {
                for _ in 0..i {
                    self = self.step();
                }
            }
            PbilStoppingCriteria::Threshold(threshold) => {
                while !converged(threshold, self.probabilities.iter().copied()) {
                    self = self.step();
                }
            }
        }
        point_from(self.probabilities).collect()
    }

    fn step<V>(mut self) -> Self
    where
        F: Fn(&[bool]) -> V,
        V: PartialOrd,
        R: Rng,
    {
        let probabilities = adjust_probabilities(
            self.problem.agnostic.adjust_rate,
            Sampleable::new(&self.probabilities).best_sample(
                self.problem.agnostic.num_samples,
                &self.problem.obj_func,
                &mut self.rng,
            ),
            self.probabilities,
        );
        let probabilities = if !self.problem.agnostic.mutation_chance.is_zero() {
            mutate_probabilities(
                &self.problem.agnostic.mutation_chance,
                self.problem.agnostic.mutation_adjust_rate,
                probabilities,
                &mut self.rng,
            )
            .collect()
        } else {
            probabilities.collect()
        };

        Self {
            problem: self.problem,
            rng: self.rng,
            probabilities,
        }
    }
}
