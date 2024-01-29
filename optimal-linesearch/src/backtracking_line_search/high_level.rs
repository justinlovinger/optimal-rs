use std::{
    iter::Sum,
    ops::{Add, Mul, Neg, RangeInclusive},
};

use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use num_traits::{AsPrimitive, Float};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    prelude::*,
};

use crate::{initial_step_size::IncrRate, step_direction::steepest_descent, StepSize};

use super::{
    low_level::BacktrackingSearcher,
    types::{BacktrackingRate, SufficientDecreaseParameter},
};

type DynObjFuncAndD<A> = Box<dyn Fn(&[A]) -> (A, Vec<A>)>;

/// Backtracking line-search independent of problem.
#[derive(Clone, Debug, PartialEq, PartialOrd, Builder)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[builder(build_fn(skip))]
pub struct BacktrackingLineSearch<A> {
    /// See [`SufficientDecreaseParameter`].
    #[builder(default)]
    pub c_1: SufficientDecreaseParameter<A>,
    /// See [`BacktrackingRate`].
    #[builder(default)]
    pub backtracking_rate: BacktrackingRate<A>,
    /// See [`StepSize`].
    #[builder(default)]
    pub initial_step_size: StepSize<A>,
    /// See [`StepDirection`].
    #[builder(default)]
    pub direction: StepDirection,
    /// See [`StepSizeUpdate`].
    #[builder(default)]
    pub step_size_update: StepSizeUpdate<A>,
    /// See [`BacktrackingLineSearchStoppingCriteria`].
    #[builder(default)]
    pub stopping_criteria: BacktrackingLineSearchStoppingCriteria,
}

impl<A> BacktrackingLineSearch<A> {
    /// Prepare backtracking line-search for a specific problem.
    pub fn for_<F, FD>(
        self,
        obj_func: F,
        obj_func_d: FD,
    ) -> BacktrackingLineSearchFor<A, F, DynObjFuncAndD<A>>
    where
        F: Fn(&[A]) -> A + Clone + 'static,
        FD: Fn(&[A]) -> Vec<A> + 'static,
    {
        let obj_func_ = obj_func.clone();
        BacktrackingLineSearchFor {
            agnostic: self,
            obj_func,
            obj_func_and_d: Box::new(move |point| (obj_func_(point), obj_func_d(point))),
        }
    }

    /// Prepare backtracking line-search for a specific problem
    /// where value and derivatives can be efficiently calculated together.
    pub fn for_combined<F, FFD>(
        self,
        obj_func: F,
        obj_func_and_d: FFD,
    ) -> BacktrackingLineSearchFor<A, F, FFD>
    where
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        BacktrackingLineSearchFor {
            agnostic: self,
            obj_func,
            obj_func_and_d,
        }
    }
}

impl<A> BacktrackingLineSearchBuilder<A> {
    /// Builds a new [`BacktrackingLineSearch`].
    pub fn build(&mut self) -> BacktrackingLineSearch<A>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
    {
        let backtracking_rate = self.backtracking_rate.unwrap_or_default();
        BacktrackingLineSearch {
            c_1: self.c_1.unwrap_or_default(),
            backtracking_rate,
            initial_step_size: self
                .initial_step_size
                .unwrap_or_else(|| StepSize::new(A::one()).unwrap()),
            direction: self.direction.clone().unwrap_or_default(),
            step_size_update: self.step_size_update.clone().unwrap_or_else(|| {
                StepSizeUpdate::IncrPrev(IncrRate::from_backtracking_rate(backtracking_rate))
            }),
            stopping_criteria: self.stopping_criteria.clone().unwrap_or_default(),
        }
    }

    // `len` will be useful when more directions are available,
    // so we can decide between BFGS and L-BFGS,
    // for example.

    /// Prepare backtracking line-search for a specific problem.
    pub fn for_<F, FD>(
        &mut self,
        _len: usize,
        obj_func: F,
        obj_func_d: FD,
    ) -> BacktrackingLineSearchFor<A, F, DynObjFuncAndD<A>>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
        F: Fn(&[A]) -> A + Clone + 'static,
        FD: Fn(&[A]) -> Vec<A> + 'static,
    {
        self.build().for_(obj_func, obj_func_d)
    }

    /// Prepare backtracking line-search for a specific problem
    /// where value and derivatives can be efficiently calculated together.
    pub fn for_combined<F, FFD>(
        &mut self,
        _len: usize,
        obj_func: F,
        obj_func_and_d: FFD,
    ) -> BacktrackingLineSearchFor<A, F, FFD>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        self.build().for_combined(obj_func, obj_func_and_d)
    }
}

/// Options for step-direction.
#[derive(Clone, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum StepDirection {
    #[default]
    /// Steepest descent.
    Steepest,
}

/// Options for getting new initial step-size after each iteration.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum StepSizeUpdate<A> {
    /// Increment last step-size by a fixed factor.
    IncrPrev(IncrRate<A>),
}

/// Options for stopping a backtracking line-search optimization-loop.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BacktrackingLineSearchStoppingCriteria {
    /// Stop when the given iteration is reached.
    Iteration(usize),
}

impl Default for BacktrackingLineSearchStoppingCriteria {
    fn default() -> Self {
        Self::Iteration(100)
    }
}

/// Backtracking line-search for a specific problem.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BacktrackingLineSearchFor<A, F, FFD> {
    /// Problem-agnostic variables.
    pub agnostic: BacktrackingLineSearch<A>,
    /// Objective function to minimize.
    pub obj_func: F,
    /// Derivative of objective function to minimize.
    pub obj_func_and_d: FFD,
}

impl<A, F, FFD> BacktrackingLineSearchFor<A, F, FFD> {
    /// Prepare backtracking line-search with a random point.
    pub fn random(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
    ) -> BacktrackingLineSearchWith<A, F, FFD>
    where
        A: Clone + SampleUniform,
    {
        self.random_using(initial_bounds, SmallRng::from_entropy())
    }

    /// Prepare backtracking line-search with a random point
    /// using a specific RNG.
    pub fn random_using<R>(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        mut rng: R,
    ) -> BacktrackingLineSearchWith<A, F, FFD>
    where
        A: Clone + SampleUniform,
        R: Rng,
    {
        self.point(
            initial_bounds
                .into_iter()
                .map(|range| {
                    let (start, end) = range.into_inner();
                    Uniform::new_inclusive(start, end).sample(&mut rng)
                })
                .collect(),
        )
    }

    /// Prepare backtracking line-search with a specific point.
    pub fn point(self, point: Vec<A>) -> BacktrackingLineSearchWith<A, F, FFD>
    where
        A: Clone,
    {
        BacktrackingLineSearchWith {
            step_size: self.agnostic.initial_step_size.clone(),
            problem: self,
            point,
        }
    }
}

/// Backtracking line-search with state.
#[derive(Clone, Debug, Dissolve, Getters)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct BacktrackingLineSearchWith<A, F, FFD> {
    /// Problem-specific variables.
    pub problem: BacktrackingLineSearchFor<A, F, FFD>,
    /// Initial step-size for next search.
    pub step_size: StepSize<A>,
    /// Best point found so far.
    pub point: Vec<A>,
}

impl<A, F, FFD> BacktrackingLineSearchWith<A, F, FFD> {
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(mut self) -> Vec<A>
    where
        A: Clone + PartialOrd + Neg<Output = A> + Add<Output = A> + Mul<Output = A> + Sum,
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        match self.problem.agnostic.stopping_criteria {
            BacktrackingLineSearchStoppingCriteria::Iteration(i) => {
                for _ in 0..i {
                    self = self.step();
                }
            }
        }
        self.point
    }

    fn step(mut self) -> Self
    where
        A: Clone + PartialOrd + Neg<Output = A> + Add<Output = A> + Mul<Output = A> + Sum,
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        let (value, derivatives) = (self.problem.obj_func_and_d)(&self.point);
        let direction = match self.problem.agnostic.direction {
            StepDirection::Steepest => steepest_descent(derivatives.iter().cloned()).collect(),
        };
        (self.step_size, self.point) = BacktrackingSearcher::new(
            self.problem.agnostic.c_1.clone(),
            self.point,
            value,
            derivatives,
            direction,
        )
        .search(
            self.problem.agnostic.backtracking_rate.clone(),
            &self.problem.obj_func,
            self.step_size.clone(),
        );

        self.step_size = match &self.problem.agnostic.step_size_update {
            StepSizeUpdate::IncrPrev(rate) => rate.clone() * self.step_size,
        };

        Self {
            problem: self.problem,
            step_size: self.step_size,
            point: self.point,
        }
    }
}
