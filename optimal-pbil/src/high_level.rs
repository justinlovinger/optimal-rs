use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use optimal_compute_core::{
    arg, arg1, argvals,
    peano::Zero,
    run::{ArgVal, Value},
    val, val1, Computation, Run,
};
use rand::prelude::*;

use crate::{low_level::*, types::*};

/// PBIL independent of problem.
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Builder)]
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
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
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
        PbilWith { problem: self, rng }
    }

    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin<V>(self) -> Vec<bool>
    where
        F: Fn(&[bool]) -> V,
        V: 'static + Clone + ArgVal + PartialOrd,
    {
        self.with(SmallRng::from_entropy()).argmin()
    }
}

/// PBIL with state.
#[derive(Clone, Debug, Dissolve, Getters)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct PbilWith<F, R> {
    /// Problem-specific variables.
    pub problem: PbilFor<F>,
    /// Source of randomness.
    pub rng: R,
}

impl<F, R> PbilWith<F, R> {
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin<V>(self) -> Vec<bool>
    where
        F: Fn(&[bool]) -> V,
        V: 'static + Clone + ArgVal + PartialOrd,
        R: 'static + Clone + ArgVal + Rng,
    {
        let probabilities = std::iter::repeat(Probability::default())
            .take(self.problem.len)
            .collect::<Vec<_>>();
        match self.problem.agnostic.stopping_criteria {
            PbilStoppingCriteria::Iteration(i) => PointFrom::new(
                val!(0)
                    .zip(val!(self.rng).zip(val1!(probabilities)))
                    .loop_while(
                        ("i", ("rng", "probabilities")),
                        (arg!("i", usize) + val!(1)).zip(
                            arg1!("probabilities", Probability)
                                .zip(BestSample::new(
                                    val!(self.problem.agnostic.num_samples),
                                    arg1!("sample").black_box::<_, Zero, V>(|sample: Vec<bool>| {
                                        Value((self.problem.obj_func)(&sample))
                                    }),
                                    arg1!("probabilities", Probability),
                                    arg!("rng", R),
                                ))
                                .then(
                                    ("probabilities", ("rng", "sample")),
                                    Mutate::new(
                                        val!(self.problem.agnostic.mutation_chance),
                                        val!(self.problem.agnostic.mutation_adjust_rate),
                                        Adjust::new(
                                            val!(self.problem.agnostic.adjust_rate),
                                            arg1!("probabilities"),
                                            arg1!("sample"),
                                        ),
                                        arg!("rng"),
                                    ),
                                ),
                        ),
                        arg!("i", usize).lt(val!(i)),
                    )
                    .then(
                        ("i", ("rng", "probabilities")),
                        arg1!("probabilities", Probability),
                    ),
            )
            .run(argvals![]),
            PbilStoppingCriteria::Threshold(threshold) => PointFrom::new(
                val!(self.rng)
                    .zip(val1!(probabilities))
                    .loop_while(
                        ("rng", "probabilities"),
                        arg1!("probabilities", Probability)
                            .zip(BestSample::new(
                                val!(self.problem.agnostic.num_samples),
                                arg1!("sample").black_box::<_, Zero, V>(|sample: Vec<bool>| {
                                    Value((self.problem.obj_func)(&sample))
                                }),
                                arg1!("probabilities", Probability),
                                arg!("rng", R),
                            ))
                            .then(
                                ("probabilities", ("rng", "sample")),
                                Mutate::new(
                                    val!(self.problem.agnostic.mutation_chance),
                                    val!(self.problem.agnostic.mutation_adjust_rate),
                                    Adjust::new(
                                        val!(self.problem.agnostic.adjust_rate),
                                        arg1!("probabilities"),
                                        arg1!("sample"),
                                    ),
                                    arg!("rng", R),
                                ),
                            ),
                        Converged::new(val!(threshold), arg1!("probabilities", Probability)).not(),
                    )
                    .then(
                        ("rng", "probabilities"),
                        arg1!("probabilities", Probability),
                    ),
            )
            .run(argvals![]),
        }
    }
}
