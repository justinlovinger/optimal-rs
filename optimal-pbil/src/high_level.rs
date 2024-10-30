use core::fmt;

use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use optimal_compute_core::{
    arg, arg1,
    cmp::{Lt, Not},
    control_flow::{LoopWhile, Then},
    math::Add,
    named_args,
    peano::{One, Zero},
    run::NamedArgs,
    val, val1,
    zip::Zip,
    Arg, Computation, ComputationFn, Names, Run, Val,
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
    ///
    /// Arguments:
    ///
    /// - `obj_func`: a computation-function taking a vector of `bool` called `sample`
    ///               and returning a scalar.
    pub fn for_<F>(self, len: usize, obj_func: F) -> PbilFor<F>
    where
        F: ComputationFn<Dim = Zero>,
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
    ///
    /// Arguments:
    ///
    /// - `obj_func`: a computation-function taking a vector of `bool` called `sample`
    ///               and returning a scalar.
    pub fn for_<F>(&mut self, len: usize, obj_func: F) -> PbilFor<F>
    where
        F: ComputationFn<Dim = Zero>,
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
    pub fn argmin(self) -> Vec<bool>
    where
        PbilComputation<F, SmallRng>: Run<Output = Vec<bool>>,
        F: Clone + ComputationFn<Dim = Zero>,
        F::Item: Clone + fmt::Debug + PartialOrd,
    {
        self.with(SmallRng::from_entropy()).argmin()
    }

    /// Return a computation representing this algorithm.
    pub fn computation(self) -> PbilComputation<F, SmallRng>
    where
        F: Clone + ComputationFn<Dim = Zero>,
        F::Item: Clone + fmt::Debug + PartialOrd,
    {
        self.with(SmallRng::from_entropy()).computation()
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

impl<F, R> PbilWith<F, R>
where
    F: Clone + ComputationFn<Dim = Zero>,
    F::Item: Clone + fmt::Debug + PartialOrd,
    R: Rng,
{
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<bool>
    where
        PbilComputation<F, R>: Run<Output = Vec<bool>>,
    {
        self.computation().run(named_args![])
    }

    /// Return a computation representing this algorithm.
    pub fn computation(self) -> PbilComputation<F, R> {
        match self.problem.agnostic.stopping_criteria {
            PbilStoppingCriteria::Iteration(i) => {
                PbilComputation::Iteration(self.computation_iteration(i))
            }
            PbilStoppingCriteria::Threshold(threshold) => {
                PbilComputation::Threshold(self.computation_threshold(threshold))
            }
        }
    }

    fn computation_iteration(self, i: usize) -> PbilIteration<F, R> {
        let probabilities = self.initial_probabilities();
        PointFrom::new(
            val!(0)
                .zip(val!(self.rng).zip(val1!(probabilities)))
                .loop_while(
                    ("i", ("rng", "probabilities")),
                    (arg!("i", usize) + val!(1)).zip(
                        arg1!("probabilities", Probability)
                            .zip(best_sample(
                                val!(self.problem.agnostic.num_samples),
                                self.problem.obj_func,
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
    }

    fn computation_threshold(self, threshold: ProbabilityThreshold) -> PbilThreshold<F, R> {
        let probabilities = self.initial_probabilities();
        PointFrom::new(
            val!(self.rng)
                .zip(val1!(probabilities))
                .loop_while(
                    ("rng", "probabilities"),
                    arg1!("probabilities", Probability)
                        .zip(best_sample(
                            val!(self.problem.agnostic.num_samples),
                            self.problem.obj_func,
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
    }

    fn initial_probabilities(&self) -> Vec<Probability> {
        std::iter::repeat(Probability::default())
            .take(self.problem.len)
            .collect()
    }
}

/// A computation representing PBIL.
#[derive(Clone, Debug)]
pub enum PbilComputation<F, R>
where
    Self: Computation,
    F: ComputationFn<Dim = Zero>,
    F::Item: Clone + fmt::Debug + PartialOrd,
    R: Rng,
{
    /// See [`PbilIteration`].
    Iteration(PbilIteration<F, R>),
    /// See [`PbilThreshold`].
    Threshold(PbilThreshold<F, R>),
}

impl<F, R> Computation for PbilComputation<F, R>
where
    R: Rng,
    F: ComputationFn<Dim = Zero>,
    F::Item: Clone + fmt::Debug + PartialOrd,
    PbilIteration<F, R>: Computation<Dim = One, Item = bool>,
    PbilThreshold<F, R>: Computation<Dim = One, Item = bool>,
{
    type Dim = One;
    type Item = bool;
}

impl<F, R> ComputationFn for PbilComputation<F, R>
where
    Self: Computation,
    F: ComputationFn<Dim = Zero>,
    F::Item: Clone + fmt::Debug + PartialOrd,
    R: Rng,
    PbilIteration<F, R>: ComputationFn,
    PbilThreshold<F, R>: ComputationFn,
{
    fn arg_names(&self) -> Names {
        match self {
            PbilComputation::Iteration(x) => x.arg_names(),
            PbilComputation::Threshold(x) => x.arg_names(),
        }
    }
}

impl<F, R> fmt::Display for PbilComputation<F, R>
where
    Self: Computation,
    F: ComputationFn<Dim = Zero>,
    F::Item: Clone + fmt::Debug + PartialOrd,
    R: Rng,
    PbilIteration<F, R>: fmt::Display,
    PbilThreshold<F, R>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PbilComputation::Iteration(x) => x.fmt(f),
            PbilComputation::Threshold(x) => x.fmt(f),
        }
    }
}

impl<F, R> Run for PbilComputation<F, R>
where
    Self: Computation,
    F: ComputationFn<Dim = Zero>,
    F::Item: Clone + fmt::Debug + PartialOrd,
    R: Rng,
    PbilIteration<F, R>: Run<Output = Vec<bool>>,
    PbilThreshold<F, R>: Run<Output = Vec<bool>>,
{
    type Output = Vec<bool>;

    fn run(self, args: NamedArgs) -> Self::Output {
        match self {
            PbilComputation::Iteration(x) => x.run(args),
            PbilComputation::Threshold(x) => x.run(args),
        }
    }
}

/// A computation representing PBIL with iteration-count as a stopping criteria.
pub type PbilIteration<F, R> = PointFrom<
    Then<
        LoopWhile<
            Zip<Val<Zero, usize>, Zip<Val<Zero, R>, Val<One, Vec<Probability>>>>,
            (&'static str, (&'static str, &'static str)),
            Zip<
                Add<Arg<Zero, usize>, Val<Zero, usize>>,
                Then<
                    Zip<
                        Arg<One, Probability>,
                        BestSample<Val<Zero, NumSamples>, F, Arg<One, Probability>, Arg<Zero, R>>,
                    >,
                    (&'static str, (&'static str, &'static str)),
                    Mutate<
                        Val<Zero, MutationChance>,
                        Val<Zero, MutationAdjustRate>,
                        Adjust<Val<Zero, AdjustRate>, Arg<One, Probability>, Arg<One, bool>>,
                        Arg<Zero, R>,
                    >,
                >,
            >,
            Lt<Arg<Zero, usize>, Val<Zero, usize>>,
        >,
        (&'static str, (&'static str, &'static str)),
        Arg<One, Probability>,
    >,
>;

/// A computation representing PBIL with probability-threshold as a stopping criteria.
pub type PbilThreshold<F, R> = PointFrom<
    Then<
        LoopWhile<
            Zip<Val<Zero, R>, Val<One, Vec<Probability>>>,
            (&'static str, &'static str),
            Then<
                Zip<
                    Arg<One, Probability>,
                    BestSample<Val<Zero, NumSamples>, F, Arg<One, Probability>, Arg<Zero, R>>,
                >,
                (&'static str, (&'static str, &'static str)),
                Mutate<
                    Val<Zero, MutationChance>,
                    Val<Zero, MutationAdjustRate>,
                    Adjust<Val<Zero, AdjustRate>, Arg<One, Probability>, Arg<One, bool>>,
                    Arg<Zero, R>,
                >,
            >,
            Not<Converged<Val<Zero, ProbabilityThreshold>, Arg<One, Probability>>>,
        >,
        (&'static str, &'static str),
        Arg<One, Probability>,
    >,
>;
