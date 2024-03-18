use std::{
    iter::Sum,
    ops::{RangeInclusive, Sub},
};

use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand};
use num_traits::{AsPrimitive, Float, Signed};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    prelude::*,
};

use crate::{
    initial_step_size::IncrRate,
    is_near_minima,
    step_direction::{
        bfgs::{
            approx_inv_snd_derivatives, bfgs_direction, initial_approx_inv_snd_derivatives_gamma,
            initial_approx_inv_snd_derivatives_identity,
        },
        steepest_descent,
    },
    StepSize,
};

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
    pub direction: StepDirection,
    /// See [`StepSizeUpdate`].
    #[builder(default)]
    pub step_size_update: StepSizeUpdate<A>,
    /// See [`BacktrackingLineSearchStoppingCriteria`].
    #[builder(default)]
    pub stopping_criteria: BacktrackingLineSearchStoppingCriteria,
}

/// Options for step-direction.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum StepDirection {
    /// Steepest descent.
    Steepest,
    /// BFGS
    Bfgs {
        /// How to handle the first few iterations of BFGS,
        /// before it has enough information to approximate second-derivatives.
        initializer: BfgsInitializer,
    },
}

/// Options for BFGS initialization.
#[derive(Clone, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BfgsInitializer {
    /// Initialize using [`initial_approx_inv_snd_derivatives_identity`].
    Identity,
    /// Initialize using [`initial_approx_inv_snd_derivatives_gamma`].
    #[default]
    Gamma,
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
#[derive(Clone, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BacktrackingLineSearchStoppingCriteria {
    /// Stop when the given iteration is reached.
    Iteration(usize),
    /// Stop when the point is near a minima.
    #[default]
    NearMinima,
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
    /// Prepare backtracking line-search for a specific problem.
    pub fn for_<F, FD>(
        &mut self,
        len: usize,
        obj_func: F,
        obj_func_d: FD,
    ) -> BacktrackingLineSearchFor<A, F, DynObjFuncAndD<A>>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
        F: Fn(&[A]) -> A + Clone + 'static,
        FD: Fn(&[A]) -> Vec<A> + 'static,
    {
        self.build(len).for_(obj_func, obj_func_d)
    }

    /// Prepare backtracking line-search for a specific problem
    /// where value and derivatives can be efficiently calculated together.
    pub fn for_combined<F, FFD>(
        &mut self,
        len: usize,
        obj_func: F,
        obj_func_and_d: FFD,
    ) -> BacktrackingLineSearchFor<A, F, FFD>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        self.build(len).for_combined(obj_func, obj_func_and_d)
    }

    /// Builds a new [`BacktrackingLineSearch`].
    fn build(&mut self, len: usize) -> BacktrackingLineSearch<A>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
    {
        let backtracking_rate = self.backtracking_rate.unwrap_or_default();

        // Memory and time for BFGS scales quadratically
        // with length of points,
        // so it is only appropriate for small problems.
        // Note,
        // this cutoff is conservative
        // and could use more testing.
        let direction = if len <= 20 {
            StepDirection::Bfgs {
                initializer: Default::default(),
            }
        } else {
            StepDirection::Steepest
        };

        BacktrackingLineSearch {
            c_1: self.c_1.unwrap_or_default(),
            backtracking_rate,
            initial_step_size: self
                .initial_step_size
                .unwrap_or_else(|| StepSize::new(A::one()).unwrap()),
            direction,
            step_size_update: self.step_size_update.clone().unwrap_or_else(|| {
                StepSizeUpdate::IncrPrev(IncrRate::from_backtracking_rate(backtracking_rate))
            }),
            stopping_criteria: self.stopping_criteria.clone().unwrap_or_default(),
        }
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
    pub fn with_random_point(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
    ) -> BacktrackingLineSearchWith<A, F, FFD>
    where
        A: Clone + SampleUniform,
    {
        self.with_random_point_using(initial_bounds, SmallRng::from_entropy())
    }

    /// Prepare backtracking line-search with a random point
    /// using a specific RNG.
    pub fn with_random_point_using<R>(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        mut rng: R,
    ) -> BacktrackingLineSearchWith<A, F, FFD>
    where
        A: Clone + SampleUniform,
        R: Rng,
    {
        self.with_point(
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
    pub fn with_point(self, initial_point: Vec<A>) -> BacktrackingLineSearchWith<A, F, FFD>
    where
        A: Clone,
    {
        BacktrackingLineSearchWith {
            problem: self,
            initial_point,
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
    /// Initial point to search from.
    pub initial_point: Vec<A>,
}

impl<A, F, FFD> BacktrackingLineSearchWith<A, F, FFD> {
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<A>
    where
        A: Sum + Signed + Float + ScalarOperand + LinalgScalar,
        f64: AsPrimitive<A>,
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        // The compiler should remove these `clone()`s.
        // We use them so we can call `self.step`.
        let mut step_dir_state = match &self.problem.agnostic.direction {
            StepDirection::Steepest => StepDirState::Steepest,
            StepDirection::Bfgs { .. } => StepDirState::Bfgs(BfgsIteration::First),
        };
        let mut step_size = self.problem.agnostic.initial_step_size;
        let mut point = Array1::from_vec(self.initial_point.clone());
        match self.problem.agnostic.stopping_criteria {
            BacktrackingLineSearchStoppingCriteria::Iteration(i) => {
                for _ in 0..i {
                    let (value, derivatives) =
                        (self.problem.obj_func_and_d)(point.as_slice().unwrap());
                    (step_dir_state, step_size, point) =
                        self.step(step_dir_state, step_size, point, value, derivatives);
                }
            }
            BacktrackingLineSearchStoppingCriteria::NearMinima => loop {
                let (value, derivatives) = (self.problem.obj_func_and_d)(point.as_slice().unwrap());
                if is_near_minima(value, derivatives.iter().copied()) {
                    break;
                }
                (step_dir_state, step_size, point) =
                    self.step(step_dir_state, step_size, point, value, derivatives);
            },
        }
        point.to_vec()
    }

    fn step(
        &self,
        step_dir_state: StepDirState<A>,
        step_size: StepSize<A>,
        point: Array1<A>,
        value: A,
        derivatives: Vec<A>,
    ) -> (StepDirState<A>, StepSize<A>, Array1<A>)
    where
        A: Sum + Float + ScalarOperand + LinalgScalar,
        F: Fn(&[A]) -> A,
        FFD: Fn(&[A]) -> (A, Vec<A>),
    {
        let derivatives = Array1::from_vec(derivatives);

        let (direction, step_dir_partial_state) = match step_dir_state {
            StepDirState::Steepest => (
                steepest_descent(derivatives.iter().cloned()).collect(),
                StepDirPartialState::Steepest,
            ),
            StepDirState::Bfgs(bfgs_state) => {
                let (direction, partial_state) = match bfgs_state {
                    BfgsIteration::First => {
                        let approx_inv_snd_derivatives =
                            initial_approx_inv_snd_derivatives_identity(point.len());
                        let direction =
                            bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
                        (
                            direction,
                            match &self.problem.agnostic.direction {
                                StepDirection::Bfgs { initializer } => match initializer {
                                    BfgsInitializer::Identity => BfgsPartialIteration::Other {
                                        approx_inv_snd_derivatives,
                                    },
                                    BfgsInitializer::Gamma => BfgsPartialIteration::Second,
                                },
                                _ => panic!(),
                            },
                        )
                    }
                    BfgsIteration::Second {
                        prev_derivatives,
                        prev_step,
                    } => {
                        let approx_inv_snd_derivatives = initial_approx_inv_snd_derivatives_gamma(
                            prev_derivatives,
                            prev_step,
                            derivatives.view(),
                        );
                        let direction =
                            bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
                        (
                            direction,
                            BfgsPartialIteration::Other {
                                approx_inv_snd_derivatives,
                            },
                        )
                    }
                    BfgsIteration::Other {
                        prev_derivatives,
                        prev_approx_inv_snd_derivatives,
                        prev_step,
                    } => {
                        let approx_inv_snd_derivatives = approx_inv_snd_derivatives(
                            prev_approx_inv_snd_derivatives,
                            prev_step,
                            prev_derivatives,
                            derivatives.view(),
                        );
                        let direction =
                            bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
                        (
                            direction,
                            BfgsPartialIteration::Other {
                                approx_inv_snd_derivatives,
                            },
                        )
                    }
                };
                (direction.to_vec(), StepDirPartialState::Bfgs(partial_state))
            }
        };

        // The compiler should remove these clones
        // if they are not necessary
        // for the type of step-direction used.
        let (step_size, new_point) = BacktrackingSearcher::new(
            self.problem.agnostic.c_1,
            point.clone().to_vec(),
            value,
            derivatives.clone(),
            direction,
        )
        .search(
            self.problem.agnostic.backtracking_rate,
            &self.problem.obj_func,
            step_size,
        );
        let new_point = Array1::from_vec(new_point);

        let step_dir_state = match step_dir_partial_state {
            StepDirPartialState::Steepest => StepDirState::Steepest,
            StepDirPartialState::Bfgs(partial_state) => {
                let prev_derivatives = derivatives;
                // We could theoretically get `prev_step` directly from line-search,
                // but then we would need to calculate each point of line-search
                // less efficiently,
                // calculating step and point separately.
                let prev_step = new_point.view().sub(point);
                StepDirState::Bfgs(match partial_state {
                    BfgsPartialIteration::Second => BfgsIteration::Second {
                        prev_derivatives,
                        prev_step,
                    },
                    BfgsPartialIteration::Other {
                        approx_inv_snd_derivatives,
                    } => BfgsIteration::Other {
                        prev_derivatives,
                        prev_approx_inv_snd_derivatives: approx_inv_snd_derivatives,
                        prev_step,
                    },
                })
            }
        };

        let step_size = match &self.problem.agnostic.step_size_update {
            StepSizeUpdate::IncrPrev(rate) => *rate * step_size,
        };

        (step_dir_state, step_size, new_point)
    }
}

enum StepDirState<A> {
    Steepest,
    Bfgs(BfgsIteration<A>),
}

enum BfgsIteration<A> {
    First,
    Second {
        prev_derivatives: Array1<A>,
        prev_step: Array1<A>,
    },
    Other {
        prev_derivatives: Array1<A>,
        prev_approx_inv_snd_derivatives: Array2<A>,
        prev_step: Array1<A>,
    },
}

enum StepDirPartialState<A> {
    Steepest,
    Bfgs(BfgsPartialIteration<A>),
}

enum BfgsPartialIteration<A> {
    Second,
    Other {
        approx_inv_snd_derivatives: Array2<A>,
    },
}
