use core::fmt;
use std::{iter::Sum, ops::RangeInclusive};

use computation_types::{
    arg, arg1, arg2,
    cmp::{Eq, Ge, Le, Lt, Max, Not},
    control_flow::{If, LoopWhile, Then},
    linalg::{FromDiagElem, IdentityMatrix, MatMul, MulCol, MulOut, ScalarProduct},
    math::{Abs, Add, Div, Mul, Neg, Sub},
    named_args,
    peano::{One, Two, Zero},
    val, val1,
    zip::{Zip, Zip3, Zip4, Zip5, Zip6, Zip7, Zip8},
    AnyArg, Arg, Computation, ComputationFn, Function, Len, NamedArgs, Run, Val,
};
use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use ndarray::{LinalgScalar, ScalarOperand};
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
    low_level::search,
    types::{BacktrackingRate, SufficientDecreaseParameter},
};

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
    /// Initialize using [`crate::step_direction::bfgs::initial_approx_inv_snd_derivatives_identity`].
    Identity,
    /// Initialize using [`crate::step_direction::bfgs::initial_approx_inv_snd_derivatives_gamma`].
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
    ///
    /// Arguments:
    ///
    /// - `obj_func`: a computation-function taking a vector of `A` called `point`
    ///               and returning a scalar of `A`.
    /// - `obj_func_d`: a computation-function taking a vector of `A` called `point`
    ///                 and returning a vector of `A`.
    pub fn for_<F, FD>(
        self,
        obj_func: F,
        obj_func_d: FD,
    ) -> BacktrackingLineSearchFor<A, F, Zip<F, FD>>
    where
        F: Clone + ComputationFn<Dim = Zero, Item = A>,
        F::Filled: Computation<Dim = Zero, Item = A>,
        FD: Clone + ComputationFn<Dim = One, Item = A>,
        FD::Filled: Computation<Dim = One, Item = A>,
    {
        self.for_combined(obj_func.clone(), obj_func.zip(obj_func_d))
    }

    /// Prepare backtracking line-search for a specific problem
    /// where value and derivatives can be efficiently calculated together.
    ///
    /// Arguments:
    ///
    /// - `obj_func`: a computation-function taking a vector of `A` called `point`
    ///               and returning a scalar of `A`.
    /// - `obj_func_and_d`: a computation-function taking a vector of `A` called `point`
    ///                     and returning a tuple of (scalar `A`, vector `A`).
    pub fn for_combined<F, FFD>(
        self,
        obj_func: F,
        obj_func_and_d: FFD,
    ) -> BacktrackingLineSearchFor<A, F, FFD>
    where
        F: Clone + ComputationFn<Dim = Zero, Item = A>,
        F::Filled: Computation<Dim = Zero, Item = A>,
        FFD: Clone + ComputationFn<Dim = (Zero, One), Item = (A, A)>,
        FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
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
    ///
    /// Arguments:
    ///
    /// - `obj_func`: a computation-function taking a vector of `A` called `point`
    ///               and returning a scalar of `A`.
    /// - `obj_func_d`: a computation-function taking a vector of `A` called `point`
    ///                 and returning a vector of `A`.
    pub fn for_<F, FD>(
        &mut self,
        len: usize,
        obj_func: F,
        obj_func_d: FD,
    ) -> BacktrackingLineSearchFor<A, F, Zip<F, FD>>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
        F: Clone + ComputationFn<Dim = Zero, Item = A>,
        F::Filled: Computation<Dim = Zero, Item = A>,
        FD: Clone + ComputationFn<Dim = One, Item = A>,
        FD::Filled: Computation<Dim = One, Item = A>,
    {
        self.build(len).for_(obj_func, obj_func_d)
    }

    /// Prepare backtracking line-search for a specific problem
    /// where value and derivatives can be efficiently calculated together.
    ///
    /// Arguments:
    ///
    /// - `obj_func`: a computation-function taking a vector of `A` called `point`
    ///               and returning a scalar of `A`.
    /// - `obj_func_and_d`: a computation-function taking a vector of `A` called `point`
    ///                     and returning a tuple of (scalar `A`, vector `A`).
    pub fn for_combined<F, FFD>(
        &mut self,
        len: usize,
        obj_func: F,
        obj_func_and_d: FFD,
    ) -> BacktrackingLineSearchFor<A, F, FFD>
    where
        A: 'static + Copy + std::fmt::Debug + Float,
        f64: AsPrimitive<A>,
        F: Clone + ComputationFn<Dim = Zero, Item = A>,
        F::Filled: Computation<Dim = Zero, Item = A>,
        FFD: Clone + ComputationFn<Dim = (Zero, One), Item = (A, A)>,
        FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
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
        let direction = self.direction.clone().unwrap_or_else(|| {
            if len <= 20 {
                StepDirection::Bfgs {
                    initializer: Default::default(),
                }
            } else {
                StepDirection::Steepest
            }
        });

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

macro_rules! bfgs_loop {
    ( $c_1:expr, $backtracking_rate:expr, $obj_func:expr, $incr_rate:expr $( , )? ) => {
        Zip7(
            arg1!("prev_derivatives", A),
            arg1!("prev_step", A),
            arg!("step_size", StepSize<A>),
            arg1!("point", A),
            arg!("value", A),
            arg1!("derivatives", A),
            approx_inv_snd_derivatives(
                arg2!("prev_approx_inv_snd_derivatives", A),
                arg1!("prev_derivatives", A),
                arg1!("prev_step", A),
                arg1!("derivatives", A),
            ),
        )
        .then(Function::anonymous(
            (
                "prev_derivatives",
                "prev_step",
                "step_size",
                "point",
                "value",
                "derivatives",
                "approx_inv_snd_derivatives",
            ),
            Zip4(
                arg1!("point", A),
                arg1!("derivatives", A),
                arg2!("approx_inv_snd_derivatives", A),
                search(
                    // Note,
                    // these variables only work here
                    // if the computations do not use any unexpected arguments.
                    $c_1,
                    $backtracking_rate,
                    $obj_func,
                    arg!("step_size", StepSize<A>),
                    arg1!("point", A),
                    arg!("value", A),
                    arg1!("derivatives", A),
                    bfgs_direction(
                        arg2!("approx_inv_snd_derivatives", A),
                        arg1!("derivatives", A),
                    ),
                ),
            ),
        ))
        .then(Function::anonymous(
            (
                "prev_point",
                "prev_derivatives",
                "prev_approx_inv_snd_derivatives",
                ("step_size", "point"),
            ),
            Zip4(
                arg1!("prev_point", A),
                arg1!("prev_derivatives", A),
                arg2!("prev_approx_inv_snd_derivatives", A),
                (val!($incr_rate) * arg!("step_size", StepSize<A>)).zip(arg1!("point", A)),
            ),
        ))
    };
}

impl<A, F, FFD> BacktrackingLineSearchWith<A, F, FFD>
where
    A: Sum + Signed + Float + ScalarOperand + LinalgScalar + AnyArg,
    f64: AsPrimitive<A>,
    F: Clone + ComputationFn<Dim = Zero, Item = A>,
    F::Filled: Computation<Dim = Zero, Item = A>,
    FFD: Clone + ComputationFn<Dim = (Zero, One), Item = (A, A)>,
    FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
{
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<A>
    where
        BacktrackingLineSearchComputation<A, F, FFD>: Run<Output = Vec<A>>,
    {
        self.computation().run(named_args![])
    }

    /// Return a computation representing this algorithm.
    pub fn computation(self) -> BacktrackingLineSearchComputation<A, F, FFD> {
        match (
            self.problem.agnostic.direction.clone(),
            self.problem.agnostic.step_size_update.clone(),
            self.problem.agnostic.stopping_criteria.clone(),
        ) {
            (
                StepDirection::Steepest,
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::Iteration(i),
            ) => BacktrackingLineSearchComputation::SteepestIncrPrevIteration(
                self.steepest_incr_prev_iteration(incr_rate, i),
            ),
            (
                StepDirection::Steepest,
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::NearMinima,
            ) => BacktrackingLineSearchComputation::SteepestIncrPrevNearMinima(
                self.steepest_incr_prev_near_minima(incr_rate),
            ),
            (
                StepDirection::Bfgs { initializer },
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::Iteration(i),
            ) => match initializer {
                BfgsInitializer::Identity => {
                    BacktrackingLineSearchComputation::BfgsIdIncrPrevIteration(
                        self.bfgs_id_incr_prev_iteration(incr_rate, i),
                    )
                }
                BfgsInitializer::Gamma => {
                    BacktrackingLineSearchComputation::BfgsGammaIncrPrevIteration(
                        self.bfgs_gamma_incr_prev_iteration(incr_rate, i),
                    )
                }
            },
            (
                StepDirection::Bfgs { initializer },
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::NearMinima,
            ) => match initializer {
                BfgsInitializer::Identity => {
                    BacktrackingLineSearchComputation::BfgsIdIncrPrevNearMinima(
                        self.bfgs_id_incr_prev_near_minima(incr_rate),
                    )
                }
                BfgsInitializer::Gamma => {
                    BacktrackingLineSearchComputation::BfgsGammaIncrPrevNearMinima(
                        self.bfgs_gamma_incr_prev_near_minima(incr_rate),
                    )
                }
            },
        }
    }

    fn steepest_incr_prev_iteration(
        self,
        incr_rate: IncrRate<A>,
        i: usize,
    ) -> BacktrackingLineSearchSteepestIncrPrevIteration<A, F, FFD> {
        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        Zip(
            val!(0_usize),
            initial_step_size.zip(val1!(self.initial_point)),
        )
        .loop_while(
            ("i", ("step_size", "point")),
            Zip(
                arg!("i", usize) + val!(1_usize),
                Zip3(
                    arg!("step_size", StepSize<A>),
                    arg1!("point", A),
                    self.problem.obj_func_and_d,
                )
                .then(Function::anonymous(
                    ("step_size", "point", ("value", "derivatives")),
                    search(
                        c_1,
                        backtracking_rate,
                        self.problem.obj_func,
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        arg!("value", A),
                        arg1!("derivatives", A),
                        steepest_descent(arg1!("derivatives", A)),
                    ),
                ))
                .then(Function::anonymous(
                    ("step_size", "point"),
                    (val!(incr_rate) * arg!("step_size", StepSize<A>)).zip(arg1!("point", A)),
                )),
            ),
            arg!("i", usize).lt(val!(i)),
        )
        .then(Function::anonymous(
            ("i", ("step_size", "point")),
            arg1!("point", A),
        ))
    }

    fn steepest_incr_prev_near_minima(
        self,
        incr_rate: IncrRate<A>,
    ) -> BacktrackingLineSearchSteepestIncrPrevNearMinima<A, F, FFD> {
        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        initial_step_size
            .zip(val1!(self.initial_point))
            .then(Function::anonymous(
                ("step_size", "point"),
                Zip3(
                    arg!("step_size", StepSize<A>),
                    arg1!("point", A),
                    self.problem.obj_func_and_d.clone(),
                ),
            ))
            .loop_while(
                ("step_size", "point", ("value", "derivatives")),
                search(
                    c_1,
                    backtracking_rate,
                    self.problem.obj_func,
                    arg!("step_size", StepSize<A>),
                    arg1!("point", A),
                    arg!("value", A),
                    arg1!("derivatives", A),
                    steepest_descent(arg1!("derivatives", A)),
                )
                .then(Function::anonymous(
                    ("step_size", "point"),
                    Zip3(
                        val!(incr_rate) * arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        self.problem.obj_func_and_d,
                    ),
                )),
                is_near_minima(arg!("value", A), arg1!("derivatives", A)).not(),
            )
            .then(Function::anonymous(
                ("step_size", "point", ("value", "derivatives")),
                arg1!("point", A),
            ))
    }

    fn bfgs_id_incr_prev_iteration(
        self,
        incr_rate: IncrRate<A>,
        i: usize,
    ) -> BacktrackingLineSearchBfgsIdIncrPrevIteration<A, F, FFD> {
        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        let len = self.initial_point.len();
        val!(0_usize).zip(val1!(self.initial_point)).if_(
            ("i", "point"),
            arg!("i", usize).ge(val!(i)),
            arg1!("point", A),
            Zip(
                arg!("i", usize) + val!(1_usize),
                arg1!("point", A)
                    .then(Function::anonymous(
                        "point",
                        arg1!("point", A).zip(self.problem.obj_func_and_d.clone()),
                    ))
                    .then(Function::anonymous(
                        ("point", ("value", "derivatives")),
                        Zip3(
                            arg1!("point", A),
                            arg1!("derivatives", A),
                            search(
                                c_1,
                                backtracking_rate,
                                self.problem.obj_func.clone(),
                                initial_step_size,
                                arg1!("point", A),
                                arg!("value", A),
                                arg1!("derivatives", A),
                                bfgs_direction(
                                    initial_approx_inv_snd_derivatives_identity::<_, A>(val!(len)),
                                    arg1!("derivatives", A),
                                ),
                            ),
                        )
                        .then(Function::anonymous(
                            ("prev_point", "prev_derivatives", ("step_size", "point")),
                            Zip4(
                                arg1!("prev_point", A),
                                arg1!("prev_derivatives", A),
                                initial_approx_inv_snd_derivatives_identity::<_, A>(val!(len)),
                                (val!(incr_rate) * arg!("step_size", StepSize<A>))
                                    .zip(arg1!("point", A)),
                            ),
                        )),
                    )),
            )
            .loop_while(
                (
                    "i",
                    (
                        "prev_point",
                        "prev_derivatives",
                        "prev_approx_inv_snd_derivatives",
                        ("step_size", "point"),
                    ),
                ),
                Zip(
                    arg!("i", usize) + val!(1_usize),
                    Zip6(
                        arg1!("prev_derivatives", A),
                        arg2!("prev_approx_inv_snd_derivatives", A),
                        arg1!("point", A) - arg1!("prev_point", A),
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        self.problem.obj_func_and_d,
                    )
                    .then(Function::anonymous(
                        (
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            "prev_step",
                            "step_size",
                            "point",
                            ("value", "derivatives"),
                        ),
                        bfgs_loop!(c_1, backtracking_rate, self.problem.obj_func, incr_rate),
                    )),
                ),
                arg!("i", usize).lt(val!(i)),
            )
            .then(Function::anonymous(
                (
                    "i",
                    (
                        "prev_point",
                        "prev_derivatives",
                        "prev_approx_inv_snd_derivatives",
                        ("step_size", "point"),
                    ),
                ),
                arg1!("point", A),
            )),
        )
    }

    fn bfgs_gamma_incr_prev_iteration(
        self,
        incr_rate: IncrRate<A>,
        i: usize,
    ) -> BacktrackingLineSearchBfgsGammaIncrPrevIteration<A, F, FFD> {
        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        let len = self.initial_point.len();
        val!(0_usize).zip(val1!(self.initial_point)).if_(
            ("i", "point"),
            arg!("i", usize).ge(val!(i)),
            arg1!("point", A),
            Zip(
                arg!("i", usize) + val!(1_usize),
                arg1!("point", A)
                    .then(Function::anonymous(
                        "point",
                        arg1!("point", A).zip(self.problem.obj_func_and_d.clone()),
                    ))
                    .then(Function::anonymous(
                        ("point", ("value", "derivatives")),
                        Zip3(
                            arg1!("point", A),
                            arg1!("derivatives", A),
                            search(
                                c_1,
                                backtracking_rate,
                                self.problem.obj_func.clone(),
                                initial_step_size,
                                arg1!("point", A),
                                arg!("value", A),
                                arg1!("derivatives", A),
                                bfgs_direction(
                                    initial_approx_inv_snd_derivatives_identity::<_, A>(val!(len)),
                                    arg1!("derivatives", A),
                                ),
                            ),
                        )
                        .then(Function::anonymous(
                            ("prev_point", "prev_derivatives", ("step_size", "point")),
                            Zip3(
                                arg1!("prev_point", A),
                                arg1!("prev_derivatives", A),
                                (val!(incr_rate) * arg!("step_size", StepSize<A>))
                                    .zip(arg1!("point", A)),
                            ),
                        )),
                    )),
            )
            .if_(
                (
                    "i",
                    ("prev_point", "prev_derivatives", ("step_size", "point")),
                ),
                arg!("i", usize).ge(val!(i)),
                arg1!("point", A),
                Zip(
                    arg!("i", usize) + val!(1_usize),
                    Zip5(
                        arg1!("prev_derivatives", A),
                        arg1!("point", A) - arg1!("prev_point", A),
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        self.problem.obj_func_and_d.clone(),
                    )
                    .then(Function::anonymous(
                        (
                            "prev_derivatives",
                            "prev_step",
                            "step_size",
                            "point",
                            ("value", "derivatives"),
                        ),
                        Zip7(
                            arg1!("prev_derivatives", A),
                            arg1!("prev_step", A),
                            arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            arg!("value", A),
                            arg1!("derivatives", A),
                            initial_approx_inv_snd_derivatives_gamma(
                                arg1!("prev_derivatives", A),
                                arg1!("prev_step", A),
                                arg1!("derivatives", A),
                            ),
                        ),
                    ))
                    .then(Function::anonymous(
                        (
                            "prev_derivatives",
                            "prev_step",
                            "step_size",
                            "point",
                            "value",
                            "derivatives",
                            "approx_inv_snd_derivatives",
                        ),
                        Zip4(
                            arg1!("point", A),
                            arg1!("derivatives", A),
                            arg2!("approx_inv_snd_derivatives", A),
                            search(
                                c_1,
                                backtracking_rate,
                                self.problem.obj_func.clone(),
                                arg!("step_size", StepSize<A>),
                                arg1!("point", A),
                                arg!("value", A),
                                arg1!("derivatives", A),
                                bfgs_direction(
                                    arg2!("approx_inv_snd_derivatives", A),
                                    arg1!("derivatives", A),
                                ),
                            ),
                        )
                        .then(Function::anonymous(
                            (
                                "prev_point",
                                "prev_derivatives",
                                "prev_approx_inv_snd_derivatives",
                                ("step_size", "point"),
                            ),
                            Zip4(
                                arg1!("prev_point", A),
                                arg1!("prev_derivatives", A),
                                arg2!("prev_approx_inv_snd_derivatives", A),
                                (val!(incr_rate) * arg!("step_size", StepSize<A>))
                                    .zip(arg1!("point", A)),
                            ),
                        )),
                    )),
                )
                .loop_while(
                    (
                        "i",
                        (
                            "prev_point",
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            ("step_size", "point"),
                        ),
                    ),
                    Zip(
                        arg!("i", usize) + val!(1_usize),
                        Zip6(
                            arg1!("prev_derivatives", A),
                            arg2!("prev_approx_inv_snd_derivatives", A),
                            arg1!("point", A) - arg1!("prev_point", A),
                            arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            self.problem.obj_func_and_d,
                        )
                        .then(Function::anonymous(
                            (
                                "prev_derivatives",
                                "prev_approx_inv_snd_derivatives",
                                "prev_step",
                                "step_size",
                                "point",
                                ("value", "derivatives"),
                            ),
                            bfgs_loop!(c_1, backtracking_rate, self.problem.obj_func, incr_rate),
                        )),
                    ),
                    arg!("i", usize).lt(val!(i)),
                )
                .then(Function::anonymous(
                    (
                        "i",
                        (
                            "prev_point",
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            ("step_size", "point"),
                        ),
                    ),
                    arg1!("point", A),
                )),
            ),
        )
    }

    fn bfgs_id_incr_prev_near_minima(
        self,
        incr_rate: IncrRate<A>,
    ) -> BacktrackingLineSearchBfgsIdIncrPrevNearMinima<A, F, FFD> {
        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        let len = self.initial_point.len();
        val1!(self.initial_point)
            .then(Function::anonymous(
                "point",
                arg1!("point", A).zip(self.problem.obj_func_and_d.clone()),
            ))
            .if_(
                ("point", ("value", "derivatives")),
                is_near_minima(arg!("value", A), arg1!("derivatives", A)),
                arg1!("point", A),
                Zip4(
                    arg1!("point", A),
                    arg!("value", A),
                    arg1!("derivatives", A),
                    initial_approx_inv_snd_derivatives_identity::<_, A>(val!(len)),
                )
                .then(Function::anonymous(
                    (
                        "point",
                        "value",
                        "derivatives",
                        "approx_inv_snd_derivatives",
                    ),
                    Zip4(
                        arg1!("point", A),
                        arg1!("derivatives", A),
                        arg2!("approx_inv_snd_derivatives", A),
                        search(
                            c_1,
                            backtracking_rate,
                            self.problem.obj_func.clone(),
                            initial_step_size,
                            arg1!("point", A),
                            arg!("value", A),
                            arg1!("derivatives", A),
                            bfgs_direction(
                                arg2!("approx_inv_snd_derivatives", A),
                                arg1!("derivatives", A),
                            ),
                        ),
                    ),
                ))
                .then(Function::anonymous(
                    (
                        "prev_point",
                        "prev_derivatives",
                        "prev_approx_inv_snd_derivatives",
                        ("step_size", "point"),
                    ),
                    Zip6(
                        arg1!("prev_point", A),
                        arg1!("prev_derivatives", A),
                        arg2!("prev_approx_inv_snd_derivatives", A),
                        val!(incr_rate) * arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        self.problem.obj_func_and_d.clone(),
                    ),
                ))
                .loop_while(
                    (
                        "prev_point",
                        "prev_derivatives",
                        "prev_approx_inv_snd_derivatives",
                        "step_size",
                        "point",
                        ("value", "derivatives"),
                    ),
                    Zip7(
                        arg1!("prev_derivatives", A),
                        arg2!("prev_approx_inv_snd_derivatives", A),
                        arg1!("point", A) - arg1!("prev_point", A),
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        arg!("value", A),
                        arg1!("derivatives", A),
                    )
                    .then(Function::anonymous(
                        (
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            "prev_step",
                            "step_size",
                            "point",
                            "value",
                            "derivatives",
                        ),
                        bfgs_loop!(c_1, backtracking_rate, self.problem.obj_func, incr_rate),
                    ))
                    .then(Function::anonymous(
                        (
                            "prev_point",
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            ("step_size", "point"),
                        ),
                        Zip6(
                            arg1!("prev_point", A),
                            arg1!("prev_derivatives", A),
                            arg2!("prev_approx_inv_snd_derivatives", A),
                            arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            self.problem.obj_func_and_d,
                        ),
                    )),
                    is_near_minima(arg!("value", A), arg1!("derivatives", A)).not(),
                )
                .then(Function::anonymous(
                    (
                        "prev_point",
                        "prev_derivatives",
                        "prev_approx_inv_snd_derivatives",
                        "step_size",
                        "point",
                        ("value", "derivatives"),
                    ),
                    arg1!("point", A),
                )),
            )
    }

    fn bfgs_gamma_incr_prev_near_minima(
        self,
        incr_rate: IncrRate<A>,
    ) -> BacktrackingLineSearchBfgsGammaIncrPrevNearMinima<A, F, FFD> {
        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        let len = self.initial_point.len();
        val1!(self.initial_point)
            .then(Function::anonymous(
                "point",
                arg1!("point", A).zip(self.problem.obj_func_and_d.clone()),
            ))
            .if_(
                ("point", ("value", "derivatives")),
                is_near_minima(arg!("value", A), arg1!("derivatives", A)),
                arg1!("point", A),
                Zip3(
                    arg1!("point", A),
                    arg1!("derivatives", A),
                    search(
                        c_1,
                        backtracking_rate,
                        self.problem.obj_func.clone(),
                        initial_step_size,
                        arg1!("point", A),
                        arg!("value", A),
                        arg1!("derivatives", A),
                        bfgs_direction(
                            initial_approx_inv_snd_derivatives_identity::<_, A>(val!(len)),
                            arg1!("derivatives", A),
                        ),
                    ),
                )
                .then(Function::anonymous(
                    ("prev_point", "prev_derivatives", ("step_size", "point")),
                    Zip5(
                        arg1!("prev_point", A),
                        arg1!("prev_derivatives", A),
                        val!(incr_rate) * arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        self.problem.obj_func_and_d.clone(),
                    ),
                ))
                .if_(
                    (
                        "prev_point",
                        "prev_derivatives",
                        "step_size",
                        "point",
                        ("value", "derivatives"),
                    ),
                    is_near_minima(arg!("value", A), arg1!("derivatives", A)),
                    arg1!("point", A),
                    Zip6(
                        arg1!("prev_derivatives", A),
                        arg1!("point", A) - arg1!("prev_point", A),
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        arg!("value", A),
                        arg1!("derivatives", A),
                    )
                    .then(Function::anonymous(
                        (
                            "prev_derivatives",
                            "prev_step",
                            "step_size",
                            "point",
                            "value",
                            "derivatives",
                        ),
                        Zip7(
                            arg1!("prev_derivatives", A),
                            arg1!("prev_step", A),
                            arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            arg!("value", A),
                            arg1!("derivatives", A),
                            initial_approx_inv_snd_derivatives_gamma(
                                arg1!("prev_derivatives", A),
                                arg1!("prev_step", A),
                                arg1!("derivatives", A),
                            ),
                        ),
                    ))
                    .then(Function::anonymous(
                        (
                            "prev_derivatives",
                            "prev_step",
                            "step_size",
                            "point",
                            "value",
                            "derivatives",
                            "approx_inv_snd_derivatives",
                        ),
                        Zip4(
                            arg1!("point", A),
                            arg1!("derivatives", A),
                            arg2!("approx_inv_snd_derivatives", A),
                            search(
                                c_1,
                                backtracking_rate,
                                self.problem.obj_func.clone(),
                                arg!("step_size", StepSize<A>),
                                arg1!("point", A),
                                arg!("value", A),
                                arg1!("derivatives", A),
                                bfgs_direction(
                                    arg2!("approx_inv_snd_derivatives", A),
                                    arg1!("derivatives", A),
                                ),
                            ),
                        ),
                    ))
                    .then(Function::anonymous(
                        (
                            "prev_point",
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            ("step_size", "point"),
                        ),
                        Zip6(
                            arg1!("prev_point", A),
                            arg1!("prev_derivatives", A),
                            arg2!("prev_approx_inv_snd_derivatives", A),
                            val!(incr_rate) * arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            self.problem.obj_func_and_d.clone(),
                        ),
                    ))
                    .loop_while(
                        (
                            "prev_point",
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            "step_size",
                            "point",
                            ("value", "derivatives"),
                        ),
                        Zip7(
                            arg1!("prev_derivatives", A),
                            arg2!("prev_approx_inv_snd_derivatives", A),
                            arg1!("point", A) - arg1!("prev_point", A),
                            arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            arg!("value", A),
                            arg1!("derivatives", A),
                        )
                        .then(Function::anonymous(
                            (
                                "prev_derivatives",
                                "prev_approx_inv_snd_derivatives",
                                "prev_step",
                                "step_size",
                                "point",
                                "value",
                                "derivatives",
                            ),
                            bfgs_loop!(c_1, backtracking_rate, self.problem.obj_func, incr_rate),
                        ))
                        .then(Function::anonymous(
                            (
                                "prev_point",
                                "prev_derivatives",
                                "prev_approx_inv_snd_derivatives",
                                ("step_size", "point"),
                            ),
                            Zip6(
                                arg1!("prev_point", A),
                                arg1!("prev_derivatives", A),
                                arg2!("prev_approx_inv_snd_derivatives", A),
                                arg!("step_size", StepSize<A>),
                                arg1!("point", A),
                                self.problem.obj_func_and_d,
                            ),
                        )),
                        is_near_minima(arg!("value", A), arg1!("derivatives", A)).not(),
                    )
                    .then(Function::anonymous(
                        (
                            "prev_point",
                            "prev_derivatives",
                            "prev_approx_inv_snd_derivatives",
                            "step_size",
                            "point",
                            ("value", "derivatives"),
                        ),
                        arg1!("point", A),
                    )),
                ),
            )
    }
}

/// A computation representing backtracking line-search.
#[derive(Clone, Debug)]
pub enum BacktrackingLineSearchComputation<A, F, FFD>
where
    A: Sum + Signed + Float + ScalarOperand + LinalgScalar + AnyArg,
    f64: AsPrimitive<A>,
    F: ComputationFn<Dim = Zero, Item = A>,
    F::Filled: Computation<Dim = Zero, Item = A>,
    FFD: ComputationFn<Dim = (Zero, One), Item = (A, A)>,
    FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
{
    /// See [`BacktrackingLineSearchSteepestIncrPrevIteration`].
    SteepestIncrPrevIteration(BacktrackingLineSearchSteepestIncrPrevIteration<A, F, FFD>),
    /// See [`BacktrackingLineSearchSteepestIncrPrevNearMinima`].
    SteepestIncrPrevNearMinima(BacktrackingLineSearchSteepestIncrPrevNearMinima<A, F, FFD>),
    /// See [`BacktrackingLineSearchBfgsIdIncrPrevIteration`].
    BfgsIdIncrPrevIteration(BacktrackingLineSearchBfgsIdIncrPrevIteration<A, F, FFD>),
    /// See [`BacktrackingLineSearchBfgsGammaIncrPrevIteration`].
    BfgsGammaIncrPrevIteration(BacktrackingLineSearchBfgsGammaIncrPrevIteration<A, F, FFD>),
    /// See [`BacktrackingLineSearchBfgsIdIncrPrevNearMinima`].
    BfgsIdIncrPrevNearMinima(BacktrackingLineSearchBfgsIdIncrPrevNearMinima<A, F, FFD>),
    /// See [`BacktrackingLineSearchBfgsGammaIncrPrevNearMinima`].
    BfgsGammaIncrPrevNearMinima(BacktrackingLineSearchBfgsGammaIncrPrevNearMinima<A, F, FFD>),
}

impl<A, F, FFD> Computation for BacktrackingLineSearchComputation<A, F, FFD>
where
    A: Sum + Signed + Float + ScalarOperand + LinalgScalar + AnyArg,
    f64: AsPrimitive<A>,
    F: ComputationFn<Dim = Zero, Item = A>,
    F::Filled: Computation<Dim = Zero, Item = A>,
    FFD: ComputationFn<Dim = (Zero, One), Item = (A, A)>,
    FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
    BacktrackingLineSearchSteepestIncrPrevIteration<A, F, FFD>: Computation<Dim = One, Item = A>,
    BacktrackingLineSearchSteepestIncrPrevNearMinima<A, F, FFD>: Computation<Dim = One, Item = A>,
    BacktrackingLineSearchBfgsIdIncrPrevIteration<A, F, FFD>: Computation<Dim = One, Item = A>,
    BacktrackingLineSearchBfgsGammaIncrPrevIteration<A, F, FFD>: Computation<Dim = One, Item = A>,
    BacktrackingLineSearchBfgsIdIncrPrevNearMinima<A, F, FFD>: Computation<Dim = One, Item = A>,
    BacktrackingLineSearchBfgsGammaIncrPrevNearMinima<A, F, FFD>: Computation<Dim = One, Item = A>,
{
    type Dim = One;
    type Item = bool;
}

impl<A, F, FFD> fmt::Display for BacktrackingLineSearchComputation<A, F, FFD>
where
    Self: Computation,
    A: Sum + Signed + Float + ScalarOperand + LinalgScalar + AnyArg,
    f64: AsPrimitive<A>,
    F: ComputationFn<Dim = Zero, Item = A>,
    F::Filled: Computation<Dim = Zero, Item = A>,
    FFD: ComputationFn<Dim = (Zero, One), Item = (A, A)>,
    FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
    BacktrackingLineSearchSteepestIncrPrevIteration<A, F, FFD>: fmt::Display,
    BacktrackingLineSearchSteepestIncrPrevNearMinima<A, F, FFD>: fmt::Display,
    BacktrackingLineSearchBfgsIdIncrPrevIteration<A, F, FFD>: fmt::Display,
    BacktrackingLineSearchBfgsGammaIncrPrevIteration<A, F, FFD>: fmt::Display,
    BacktrackingLineSearchBfgsIdIncrPrevNearMinima<A, F, FFD>: fmt::Display,
    BacktrackingLineSearchBfgsGammaIncrPrevNearMinima<A, F, FFD>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BacktrackingLineSearchComputation::SteepestIncrPrevIteration(x) => x.fmt(f),
            BacktrackingLineSearchComputation::SteepestIncrPrevNearMinima(x) => x.fmt(f),
            BacktrackingLineSearchComputation::BfgsIdIncrPrevIteration(x) => x.fmt(f),
            BacktrackingLineSearchComputation::BfgsGammaIncrPrevIteration(x) => x.fmt(f),
            BacktrackingLineSearchComputation::BfgsIdIncrPrevNearMinima(x) => x.fmt(f),
            BacktrackingLineSearchComputation::BfgsGammaIncrPrevNearMinima(x) => x.fmt(f),
        }
    }
}

impl<A, F, FFD> Run for BacktrackingLineSearchComputation<A, F, FFD>
where
    Self: Computation,
    A: Sum + Signed + Float + ScalarOperand + LinalgScalar + AnyArg,
    f64: AsPrimitive<A>,
    F: ComputationFn<Dim = Zero, Item = A>,
    F::Filled: Computation<Dim = Zero, Item = A>,
    FFD: ComputationFn<Dim = (Zero, One), Item = (A, A)>,
    FFD::Filled: Computation<Dim = (Zero, One), Item = (A, A)>,
    BacktrackingLineSearchSteepestIncrPrevIteration<A, F, FFD>: Run<Output = Vec<A>>,
    BacktrackingLineSearchSteepestIncrPrevNearMinima<A, F, FFD>: Run<Output = Vec<A>>,
    BacktrackingLineSearchBfgsIdIncrPrevIteration<A, F, FFD>: Run<Output = Vec<A>>,
    BacktrackingLineSearchBfgsGammaIncrPrevIteration<A, F, FFD>: Run<Output = Vec<A>>,
    BacktrackingLineSearchBfgsIdIncrPrevNearMinima<A, F, FFD>: Run<Output = Vec<A>>,
    BacktrackingLineSearchBfgsGammaIncrPrevNearMinima<A, F, FFD>: Run<Output = Vec<A>>,
{
    type Output = Vec<A>;

    fn run(self, args: NamedArgs) -> Self::Output {
        match self {
            BacktrackingLineSearchComputation::SteepestIncrPrevIteration(x) => x.run(args),
            BacktrackingLineSearchComputation::SteepestIncrPrevNearMinima(x) => x.run(args),
            BacktrackingLineSearchComputation::BfgsIdIncrPrevIteration(x) => x.run(args),
            BacktrackingLineSearchComputation::BfgsGammaIncrPrevIteration(x) => x.run(args),
            BacktrackingLineSearchComputation::BfgsIdIncrPrevNearMinima(x) => x.run(args),
            BacktrackingLineSearchComputation::BfgsGammaIncrPrevNearMinima(x) => x.run(args),
        }
    }
}

/// A computation representing backtracking line-search
/// with steepest-descent direction,
/// increment-previous step-size update,
/// and max-iteration stopping criteria.
pub type BacktrackingLineSearchSteepestIncrPrevIteration<A, F, FFD> = Then<
    LoopWhile<
        Zip<Val<Zero, usize>, Zip<Val<Zero, StepSize<A>>, Val<One, Vec<A>>>>,
        (&'static str, (&'static str, &'static str)),
        Zip<
            Add<Arg<Zero, usize>, Val<Zero, usize>>,
            Then<
                Then<
                    Zip3<Arg<Zero, StepSize<A>>, Arg<One, A>, FFD>,
                    (&'static str, &'static str, (&'static str, &'static str)),
                    Then<
                        LoopWhile<
                            Then<
                                Zip7<
                                    Val<Zero, SufficientDecreaseParameter<A>>,
                                    Val<Zero, BacktrackingRate<A>>,
                                    Arg<Zero, StepSize<A>>,
                                    Arg<One, A>,
                                    Arg<Zero, A>,
                                    Arg<One, A>,
                                    Neg<Arg<One, A>>,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip8<
                                    Mul<
                                        Arg<Zero, SufficientDecreaseParameter<A>>,
                                        ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                    >,
                                    Arg<Zero, BacktrackingRate<A>>,
                                    Arg<One, A>,
                                    Arg<Zero, A>,
                                    Arg<One, A>,
                                    Arg<Zero, StepSize<A>>,
                                    Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                    Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip8<
                                Arg<Zero, A>,
                                Arg<Zero, BacktrackingRate<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                Arg<Zero, StepSize<A>>,
                                Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                            >,
                            Not<
                                Le<F, Add<Arg<Zero, A>, Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>>>,
                            >,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                        ),
                        Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                    >,
                >,
                (&'static str, &'static str),
                Zip<Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>, Arg<One, A>>,
            >,
        >,
        Lt<Arg<Zero, usize>, Val<Zero, usize>>,
    >,
    (&'static str, (&'static str, &'static str)),
    Arg<One, A>,
>;

/// A computation representing backtracking line-search
/// with steepest-descent direction,
/// increment-previous step-size update,
/// and near-minima stopping criteria.
pub type BacktrackingLineSearchSteepestIncrPrevNearMinima<A, F, FFD> = Then<
    LoopWhile<
        Then<
            Zip<Val<Zero, StepSize<A>>, Val<One, Vec<A>>>,
            (&'static str, &'static str),
            Zip3<Arg<Zero, StepSize<A>>, Arg<One, A>, FFD>,
        >,
        (&'static str, &'static str, (&'static str, &'static str)),
        Then<
            Then<
                LoopWhile<
                    Then<
                        Zip7<
                            Val<Zero, SufficientDecreaseParameter<A>>,
                            Val<Zero, BacktrackingRate<A>>,
                            Arg<Zero, StepSize<A>>,
                            Arg<One, A>,
                            Arg<Zero, A>,
                            Arg<One, A>,
                            Neg<Arg<One, A>>,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                        ),
                        Zip8<
                            Mul<
                                Arg<Zero, SufficientDecreaseParameter<A>>,
                                ScalarProduct<Arg<One, A>, Arg<One, A>>,
                            >,
                            Arg<Zero, BacktrackingRate<A>>,
                            Arg<One, A>,
                            Arg<Zero, A>,
                            Arg<One, A>,
                            Arg<Zero, StepSize<A>>,
                            Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                            Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                        >,
                    >,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                    ),
                    Zip8<
                        Arg<Zero, A>,
                        Arg<Zero, BacktrackingRate<A>>,
                        Arg<One, A>,
                        Arg<Zero, A>,
                        Arg<One, A>,
                        Arg<Zero, StepSize<A>>,
                        Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                        Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                    >,
                    Not<Le<F, Add<Arg<Zero, A>, Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>>>>,
                >,
                (
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                ),
                Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
            >,
            (&'static str, &'static str),
            Zip3<Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>, Arg<One, A>, FFD>,
        >,
        Not<Lt<Max<Abs<Arg<One, A>>>, Mul<Val<Zero, A>, Add<Val<Zero, A>, Abs<Arg<Zero, A>>>>>>,
    >,
    (&'static str, &'static str, (&'static str, &'static str)),
    Arg<One, A>,
>;

/// A computation representing backtracking line-search
/// with BFGS direction,
/// identity initialization for BFGS,
/// increment-previous step-size update,
/// and max-iteration stopping criteria.
pub type BacktrackingLineSearchBfgsIdIncrPrevIteration<A, F, FFD> = If<
    Zip<Val<Zero, usize>, Val<One, Vec<A>>>,
    (&'static str, &'static str),
    Ge<Arg<Zero, usize>, Val<Zero, usize>>,
    Arg<One, A>,
    Then<
        LoopWhile<
            Zip<
                Add<Arg<Zero, usize>, Val<Zero, usize>>,
                Then<
                    Then<Arg<One, A>, &'static str, Zip<Arg<One, A>, FFD>>,
                    (&'static str, (&'static str, &'static str)),
                    Then<
                        Zip3<
                            Arg<One, A>,
                            Arg<One, A>,
                            Then<
                                LoopWhile<
                                    Then<
                                        Zip7<
                                            Val<Zero, SufficientDecreaseParameter<A>>,
                                            Val<Zero, BacktrackingRate<A>>,
                                            Val<Zero, StepSize<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Neg<
                                                MulCol<
                                                    IdentityMatrix<Val<Zero, usize>, A>,
                                                    Arg<One, A>,
                                                >,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip8<
                                            Mul<
                                                Arg<Zero, SufficientDecreaseParameter<A>>,
                                                ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                            >,
                                            Arg<Zero, BacktrackingRate<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Arg<Zero, StepSize<A>>,
                                            Add<
                                                Arg<One, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                            >,
                                            Mul<
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                            >,
                                        >,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip8<
                                        Arg<Zero, A>,
                                        Arg<Zero, BacktrackingRate<A>>,
                                        Arg<One, A>,
                                        Arg<Zero, A>,
                                        Arg<One, A>,
                                        Arg<Zero, StepSize<A>>,
                                        Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                        Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                    >,
                                    Not<
                                        Le<
                                            F,
                                            Add<
                                                Arg<Zero, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                            >,
                                        >,
                                    >,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                            >,
                        >,
                        (&'static str, &'static str, (&'static str, &'static str)),
                        Zip4<
                            Arg<One, A>,
                            Arg<One, A>,
                            IdentityMatrix<Val<Zero, usize>, A>,
                            Zip<Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>, Arg<One, A>>,
                        >,
                    >,
                >,
            >,
            (
                &'static str,
                (
                    &'static str,
                    &'static str,
                    &'static str,
                    (&'static str, &'static str),
                ),
            ),
            Zip<
                Add<Arg<Zero, usize>, Val<Zero, usize>>,
                Then<
                    Zip6<
                        Arg<One, A>,
                        Arg<Two, A>,
                        Sub<Arg<One, A>, Arg<One, A>>,
                        Arg<Zero, StepSize<A>>,
                        Arg<One, A>,
                        FFD,
                    >,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        (&'static str, &'static str),
                    ),
                    Then<
                        Then<
                            Zip7<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Zero, StepSize<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                If<
                                    Then<
                                        Zip3<
                                            Arg<Two, A>,
                                            Arg<One, A>,
                                            Sub<Arg<One, A>, Arg<One, A>>,
                                        >,
                                        (&'static str, &'static str, &'static str),
                                        Zip4<
                                            Arg<Two, A>,
                                            Arg<One, A>,
                                            Arg<One, A>,
                                            ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                        >,
                                    >,
                                    (&'static str, &'static str, &'static str, &'static str),
                                    Eq<Arg<Zero, A>, Val<Zero, A>>,
                                    Arg<Two, A>,
                                    Then<
                                        Then<
                                            Zip5<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Div<Val<Zero, A>, Arg<Zero, A>>,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip7<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<Zero, A>,
                                                Mul<Arg<One, A>, Arg<Zero, A>>,
                                                IdentityMatrix<Len<Arg<One, A>>, A>,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Add<
                                            MatMul<
                                                MatMul<
                                                    Sub<
                                                        Arg<Two, A>,
                                                        MulOut<Arg<One, A>, Arg<One, A>>,
                                                    >,
                                                    Arg<Two, A>,
                                                >,
                                                Sub<
                                                    Arg<Two, A>,
                                                    MulOut<
                                                        Mul<Arg<One, A>, Arg<Zero, A>>,
                                                        Arg<One, A>,
                                                    >,
                                                >,
                                            >,
                                            MulOut<Arg<One, A>, Arg<One, A>>,
                                        >,
                                    >,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip4<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Two, A>,
                                Then<
                                    LoopWhile<
                                        Then<
                                            Zip7<
                                                Val<Zero, SufficientDecreaseParameter<A>>,
                                                Val<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip8<
                                                Mul<
                                                    Arg<Zero, SufficientDecreaseParameter<A>>,
                                                    ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                                >,
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Arg<Zero, StepSize<A>>,
                                                Add<
                                                    Arg<One, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                >,
                                                Mul<
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                >,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip8<
                                            Arg<Zero, A>,
                                            Arg<Zero, BacktrackingRate<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Arg<Zero, StepSize<A>>,
                                            Add<
                                                Arg<One, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                            >,
                                            Mul<
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                            >,
                                        >,
                                        Not<
                                            Le<
                                                F,
                                                Add<
                                                    Arg<Zero, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                                >,
                                            >,
                                        >,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                >,
                            >,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            (&'static str, &'static str),
                        ),
                        Zip4<
                            Arg<One, A>,
                            Arg<One, A>,
                            Arg<Two, A>,
                            Zip<Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>, Arg<One, A>>,
                        >,
                    >,
                >,
            >,
            Lt<Arg<Zero, usize>, Val<Zero, usize>>,
        >,
        (
            &'static str,
            (
                &'static str,
                &'static str,
                &'static str,
                (&'static str, &'static str),
            ),
        ),
        Arg<One, A>,
    >,
>;

/// A computation representing backtracking line-search
/// with BFGS direction,
/// gamma initialization for BFGS,
/// increment-previous step-size update,
/// and max-iteration stopping criteria.
pub type BacktrackingLineSearchBfgsGammaIncrPrevIteration<A, F, FFD> = If<
    Zip<Val<Zero, usize>, Val<One, Vec<A>>>,
    (&'static str, &'static str),
    Ge<Arg<Zero, usize>, Val<Zero, usize>>,
    Arg<One, A>,
    If<
        Zip<
            Add<Arg<Zero, usize>, Val<Zero, usize>>,
            Then<
                Then<Arg<One, A>, &'static str, Zip<Arg<One, A>, FFD>>,
                (&'static str, (&'static str, &'static str)),
                Then<
                    Zip3<
                        Arg<One, A>,
                        Arg<One, A>,
                        Then<
                            LoopWhile<
                                Then<
                                    Zip7<
                                        Val<Zero, SufficientDecreaseParameter<A>>,
                                        Val<Zero, BacktrackingRate<A>>,
                                        Val<Zero, StepSize<A>>,
                                        Arg<One, A>,
                                        Arg<Zero, A>,
                                        Arg<One, A>,
                                        Neg<
                                            MulCol<
                                                IdentityMatrix<Val<Zero, usize>, A>,
                                                Arg<One, A>,
                                            >,
                                        >,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip8<
                                        Mul<
                                            Arg<Zero, SufficientDecreaseParameter<A>>,
                                            ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                        >,
                                        Arg<Zero, BacktrackingRate<A>>,
                                        Arg<One, A>,
                                        Arg<Zero, A>,
                                        Arg<One, A>,
                                        Arg<Zero, StepSize<A>>,
                                        Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                        Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                    >,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip8<
                                    Arg<Zero, A>,
                                    Arg<Zero, BacktrackingRate<A>>,
                                    Arg<One, A>,
                                    Arg<Zero, A>,
                                    Arg<One, A>,
                                    Arg<Zero, StepSize<A>>,
                                    Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                    Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                >,
                                Not<
                                    Le<
                                        F,
                                        Add<
                                            Arg<Zero, A>,
                                            Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                        >,
                                    >,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                        >,
                    >,
                    (&'static str, &'static str, (&'static str, &'static str)),
                    Zip3<
                        Arg<One, A>,
                        Arg<One, A>,
                        Zip<Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>, Arg<One, A>>,
                    >,
                >,
            >,
        >,
        (
            &'static str,
            (&'static str, &'static str, (&'static str, &'static str)),
        ),
        Ge<Arg<Zero, usize>, Val<Zero, usize>>,
        Arg<One, A>,
        Then<
            LoopWhile<
                Zip<
                    Add<Arg<Zero, usize>, Val<Zero, usize>>,
                    Then<
                        Then<
                            Zip5<
                                Arg<One, A>,
                                Sub<Arg<One, A>, Arg<One, A>>,
                                Arg<Zero, StepSize<A>>,
                                Arg<One, A>,
                                FFD,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                (&'static str, &'static str),
                            ),
                            Zip7<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Zero, StepSize<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                If<
                                    Then<
                                        Zip<Arg<One, A>, Sub<Arg<One, A>, Arg<One, A>>>,
                                        (&'static str, &'static str),
                                        Zip3<
                                            Arg<One, A>,
                                            Arg<One, A>,
                                            ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                        >,
                                    >,
                                    (&'static str, &'static str, &'static str),
                                    Eq<Arg<Zero, A>, Val<Zero, A>>,
                                    IdentityMatrix<Len<Arg<One, A>>, A>,
                                    FromDiagElem<
                                        Len<Arg<One, A>>,
                                        Div<ScalarProduct<Arg<One, A>, Arg<One, A>>, Arg<Zero, A>>,
                                    >,
                                >,
                            >,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                        ),
                        Then<
                            Zip4<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Two, A>,
                                Then<
                                    LoopWhile<
                                        Then<
                                            Zip7<
                                                Val<Zero, SufficientDecreaseParameter<A>>,
                                                Val<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip8<
                                                Mul<
                                                    Arg<Zero, SufficientDecreaseParameter<A>>,
                                                    ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                                >,
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Arg<Zero, StepSize<A>>,
                                                Add<
                                                    Arg<One, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                >,
                                                Mul<
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                >,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip8<
                                            Arg<Zero, A>,
                                            Arg<Zero, BacktrackingRate<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Arg<Zero, StepSize<A>>,
                                            Add<
                                                Arg<One, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                            >,
                                            Mul<
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                            >,
                                        >,
                                        Not<
                                            Le<
                                                F,
                                                Add<
                                                    Arg<Zero, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                                >,
                                            >,
                                        >,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                (&'static str, &'static str),
                            ),
                            Zip4<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Two, A>,
                                Zip<
                                    Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>,
                                    Arg<One, A>,
                                >,
                            >,
                        >,
                    >,
                >,
                (
                    &'static str,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        (&'static str, &'static str),
                    ),
                ),
                Zip<
                    Add<Arg<Zero, usize>, Val<Zero, usize>>,
                    Then<
                        Zip6<
                            Arg<One, A>,
                            Arg<Two, A>,
                            Sub<Arg<One, A>, Arg<One, A>>,
                            Arg<Zero, StepSize<A>>,
                            Arg<One, A>,
                            FFD,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            (&'static str, &'static str),
                        ),
                        Then<
                            Then<
                                Zip7<
                                    Arg<One, A>,
                                    Arg<One, A>,
                                    Arg<Zero, StepSize<A>>,
                                    Arg<One, A>,
                                    Arg<Zero, A>,
                                    Arg<One, A>,
                                    If<
                                        Then<
                                            Zip3<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Sub<Arg<One, A>, Arg<One, A>>,
                                            >,
                                            (&'static str, &'static str, &'static str),
                                            Zip4<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Arg<One, A>,
                                                ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                            >,
                                        >,
                                        (&'static str, &'static str, &'static str, &'static str),
                                        Eq<Arg<Zero, A>, Val<Zero, A>>,
                                        Arg<Two, A>,
                                        Then<
                                            Then<
                                                Zip5<
                                                    Arg<Two, A>,
                                                    Arg<One, A>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Div<Val<Zero, A>, Arg<Zero, A>>,
                                                >,
                                                (
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                ),
                                                Zip7<
                                                    Arg<Two, A>,
                                                    Arg<One, A>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Arg<Zero, A>,
                                                    Mul<Arg<One, A>, Arg<Zero, A>>,
                                                    IdentityMatrix<Len<Arg<One, A>>, A>,
                                                >,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Add<
                                                MatMul<
                                                    MatMul<
                                                        Sub<
                                                            Arg<Two, A>,
                                                            MulOut<Arg<One, A>, Arg<One, A>>,
                                                        >,
                                                        Arg<Two, A>,
                                                    >,
                                                    Sub<
                                                        Arg<Two, A>,
                                                        MulOut<
                                                            Mul<Arg<One, A>, Arg<Zero, A>>,
                                                            Arg<One, A>,
                                                        >,
                                                    >,
                                                >,
                                                MulOut<Arg<One, A>, Arg<One, A>>,
                                            >,
                                        >,
                                    >,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip4<
                                    Arg<One, A>,
                                    Arg<One, A>,
                                    Arg<Two, A>,
                                    Then<
                                        LoopWhile<
                                            Then<
                                                Zip7<
                                                    Val<Zero, SufficientDecreaseParameter<A>>,
                                                    Val<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Arg<One, A>,
                                                    Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                                >,
                                                (
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                ),
                                                Zip8<
                                                    Mul<
                                                        Arg<Zero, SufficientDecreaseParameter<A>>,
                                                        ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                                    >,
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Arg<One, A>,
                                                    Arg<Zero, StepSize<A>>,
                                                    Add<
                                                        Arg<One, A>,
                                                        Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                    >,
                                                    Mul<
                                                        Arg<Zero, BacktrackingRate<A>>,
                                                        Arg<Zero, StepSize<A>>,
                                                    >,
                                                >,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip8<
                                                Arg<Zero, A>,
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Arg<Zero, StepSize<A>>,
                                                Add<
                                                    Arg<One, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                >,
                                                Mul<
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                >,
                                            >,
                                            Not<
                                                Le<
                                                    F,
                                                    Add<
                                                        Arg<Zero, A>,
                                                        Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                                    >,
                                                >,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                    >,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                (&'static str, &'static str),
                            ),
                            Zip4<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Two, A>,
                                Zip<
                                    Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>,
                                    Arg<One, A>,
                                >,
                            >,
                        >,
                    >,
                >,
                Lt<Arg<Zero, usize>, Val<Zero, usize>>,
            >,
            (
                &'static str,
                (
                    &'static str,
                    &'static str,
                    &'static str,
                    (&'static str, &'static str),
                ),
            ),
            Arg<One, A>,
        >,
    >,
>;

/// A computation representing backtracking line-search
/// with BFGS direction,
/// identity initialization for BFGS,
/// increment-previous step-size update,
/// and near-minima stopping criteria.
pub type BacktrackingLineSearchBfgsIdIncrPrevNearMinima<A, F, FFD> = If<
    Then<Val<One, Vec<A>>, &'static str, Zip<Arg<One, A>, FFD>>,
    (&'static str, (&'static str, &'static str)),
    Lt<Max<Abs<Arg<One, A>>>, Mul<Val<Zero, A>, Add<Val<Zero, A>, Abs<Arg<Zero, A>>>>>,
    Arg<One, A>,
    Then<
        LoopWhile<
            Then<
                Then<
                    Zip4<
                        Arg<One, A>,
                        Arg<Zero, A>,
                        Arg<One, A>,
                        IdentityMatrix<Val<Zero, usize>, A>,
                    >,
                    (&'static str, &'static str, &'static str, &'static str),
                    Zip4<
                        Arg<One, A>,
                        Arg<One, A>,
                        Arg<Two, A>,
                        Then<
                            LoopWhile<
                                Then<
                                    Zip7<
                                        Val<Zero, SufficientDecreaseParameter<A>>,
                                        Val<Zero, BacktrackingRate<A>>,
                                        Val<Zero, StepSize<A>>,
                                        Arg<One, A>,
                                        Arg<Zero, A>,
                                        Arg<One, A>,
                                        Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip8<
                                        Mul<
                                            Arg<Zero, SufficientDecreaseParameter<A>>,
                                            ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                        >,
                                        Arg<Zero, BacktrackingRate<A>>,
                                        Arg<One, A>,
                                        Arg<Zero, A>,
                                        Arg<One, A>,
                                        Arg<Zero, StepSize<A>>,
                                        Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                        Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                    >,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip8<
                                    Arg<Zero, A>,
                                    Arg<Zero, BacktrackingRate<A>>,
                                    Arg<One, A>,
                                    Arg<Zero, A>,
                                    Arg<One, A>,
                                    Arg<Zero, StepSize<A>>,
                                    Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                    Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                >,
                                Not<
                                    Le<
                                        F,
                                        Add<
                                            Arg<Zero, A>,
                                            Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                        >,
                                    >,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                        >,
                    >,
                >,
                (
                    &'static str,
                    &'static str,
                    &'static str,
                    (&'static str, &'static str),
                ),
                Zip6<
                    Arg<One, A>,
                    Arg<One, A>,
                    Arg<Two, A>,
                    Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>,
                    Arg<One, A>,
                    FFD,
                >,
            >,
            (
                &'static str,
                &'static str,
                &'static str,
                &'static str,
                &'static str,
                (&'static str, &'static str),
            ),
            Then<
                Then<
                    Zip7<
                        Arg<One, A>,
                        Arg<Two, A>,
                        Sub<Arg<One, A>, Arg<One, A>>,
                        Arg<Zero, StepSize<A>>,
                        Arg<One, A>,
                        Arg<Zero, A>,
                        Arg<One, A>,
                    >,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                    ),
                    Then<
                        Then<
                            Zip7<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Zero, StepSize<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                If<
                                    Then<
                                        Zip3<
                                            Arg<Two, A>,
                                            Arg<One, A>,
                                            Sub<Arg<One, A>, Arg<One, A>>,
                                        >,
                                        (&'static str, &'static str, &'static str),
                                        Zip4<
                                            Arg<Two, A>,
                                            Arg<One, A>,
                                            Arg<One, A>,
                                            ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                        >,
                                    >,
                                    (&'static str, &'static str, &'static str, &'static str),
                                    Eq<Arg<Zero, A>, Val<Zero, A>>,
                                    Arg<Two, A>,
                                    Then<
                                        Then<
                                            Zip5<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Div<Val<Zero, A>, Arg<Zero, A>>,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip7<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<Zero, A>,
                                                Mul<Arg<One, A>, Arg<Zero, A>>,
                                                IdentityMatrix<Len<Arg<One, A>>, A>,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Add<
                                            MatMul<
                                                MatMul<
                                                    Sub<
                                                        Arg<Two, A>,
                                                        MulOut<Arg<One, A>, Arg<One, A>>,
                                                    >,
                                                    Arg<Two, A>,
                                                >,
                                                Sub<
                                                    Arg<Two, A>,
                                                    MulOut<
                                                        Mul<Arg<One, A>, Arg<Zero, A>>,
                                                        Arg<One, A>,
                                                    >,
                                                >,
                                            >,
                                            MulOut<Arg<One, A>, Arg<One, A>>,
                                        >,
                                    >,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip4<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Two, A>,
                                Then<
                                    LoopWhile<
                                        Then<
                                            Zip7<
                                                Val<Zero, SufficientDecreaseParameter<A>>,
                                                Val<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip8<
                                                Mul<
                                                    Arg<Zero, SufficientDecreaseParameter<A>>,
                                                    ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                                >,
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Arg<Zero, StepSize<A>>,
                                                Add<
                                                    Arg<One, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                >,
                                                Mul<
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                >,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip8<
                                            Arg<Zero, A>,
                                            Arg<Zero, BacktrackingRate<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Arg<Zero, StepSize<A>>,
                                            Add<
                                                Arg<One, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                            >,
                                            Mul<
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                            >,
                                        >,
                                        Not<
                                            Le<
                                                F,
                                                Add<
                                                    Arg<Zero, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                                >,
                                            >,
                                        >,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                >,
                            >,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            (&'static str, &'static str),
                        ),
                        Zip4<
                            Arg<One, A>,
                            Arg<One, A>,
                            Arg<Two, A>,
                            Zip<Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>, Arg<One, A>>,
                        >,
                    >,
                >,
                (
                    &'static str,
                    &'static str,
                    &'static str,
                    (&'static str, &'static str),
                ),
                Zip6<
                    Arg<One, A>,
                    Arg<One, A>,
                    Arg<Two, A>,
                    Arg<Zero, StepSize<A>>,
                    Arg<One, A>,
                    FFD,
                >,
            >,
            Not<Lt<Max<Abs<Arg<One, A>>>, Mul<Val<Zero, A>, Add<Val<Zero, A>, Abs<Arg<Zero, A>>>>>>,
        >,
        (
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            (&'static str, &'static str),
        ),
        Arg<One, A>,
    >,
>;

/// A computation representing backtracking line-search
/// with BFGS direction,
/// gamma initialization for BFGS,
/// increment-previous step-size update,
/// and near-minima stopping criteria.
pub type BacktrackingLineSearchBfgsGammaIncrPrevNearMinima<A, F, FFD> = If<
    Then<Val<One, Vec<A>>, &'static str, Zip<Arg<One, A>, FFD>>,
    (&'static str, (&'static str, &'static str)),
    Lt<Max<Abs<Arg<One, A>>>, Mul<Val<Zero, A>, Add<Val<Zero, A>, Abs<Arg<Zero, A>>>>>,
    Arg<One, A>,
    If<
        Then<
            Zip3<
                Arg<One, A>,
                Arg<One, A>,
                Then<
                    LoopWhile<
                        Then<
                            Zip7<
                                Val<Zero, SufficientDecreaseParameter<A>>,
                                Val<Zero, BacktrackingRate<A>>,
                                Val<Zero, StepSize<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                Neg<MulCol<IdentityMatrix<Val<Zero, usize>, A>, Arg<One, A>>>,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip8<
                                Mul<
                                    Arg<Zero, SufficientDecreaseParameter<A>>,
                                    ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                >,
                                Arg<Zero, BacktrackingRate<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                Arg<Zero, StepSize<A>>,
                                Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                            >,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                        ),
                        Zip8<
                            Arg<Zero, A>,
                            Arg<Zero, BacktrackingRate<A>>,
                            Arg<One, A>,
                            Arg<Zero, A>,
                            Arg<One, A>,
                            Arg<Zero, StepSize<A>>,
                            Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                            Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                        >,
                        Not<Le<F, Add<Arg<Zero, A>, Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>>>>,
                    >,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                        &'static str,
                    ),
                    Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                >,
            >,
            (&'static str, &'static str, (&'static str, &'static str)),
            Zip5<
                Arg<One, A>,
                Arg<One, A>,
                Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>,
                Arg<One, A>,
                FFD,
            >,
        >,
        (
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            (&'static str, &'static str),
        ),
        Lt<Max<Abs<Arg<One, A>>>, Mul<Val<Zero, A>, Add<Val<Zero, A>, Abs<Arg<Zero, A>>>>>,
        Arg<One, A>,
        Then<
            LoopWhile<
                Then<
                    Then<
                        Then<
                            Zip6<
                                Arg<One, A>,
                                Sub<Arg<One, A>, Arg<One, A>>,
                                Arg<Zero, StepSize<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                                &'static str,
                            ),
                            Zip7<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Zero, StepSize<A>>,
                                Arg<One, A>,
                                Arg<Zero, A>,
                                Arg<One, A>,
                                If<
                                    Then<
                                        Zip<Arg<One, A>, Sub<Arg<One, A>, Arg<One, A>>>,
                                        (&'static str, &'static str),
                                        Zip3<
                                            Arg<One, A>,
                                            Arg<One, A>,
                                            ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                        >,
                                    >,
                                    (&'static str, &'static str, &'static str),
                                    Eq<Arg<Zero, A>, Val<Zero, A>>,
                                    IdentityMatrix<Len<Arg<One, A>>, A>,
                                    FromDiagElem<
                                        Len<Arg<One, A>>,
                                        Div<ScalarProduct<Arg<One, A>, Arg<One, A>>, Arg<Zero, A>>,
                                    >,
                                >,
                            >,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                        ),
                        Zip4<
                            Arg<One, A>,
                            Arg<One, A>,
                            Arg<Two, A>,
                            Then<
                                LoopWhile<
                                    Then<
                                        Zip7<
                                            Val<Zero, SufficientDecreaseParameter<A>>,
                                            Val<Zero, BacktrackingRate<A>>,
                                            Arg<Zero, StepSize<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip8<
                                            Mul<
                                                Arg<Zero, SufficientDecreaseParameter<A>>,
                                                ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                            >,
                                            Arg<Zero, BacktrackingRate<A>>,
                                            Arg<One, A>,
                                            Arg<Zero, A>,
                                            Arg<One, A>,
                                            Arg<Zero, StepSize<A>>,
                                            Add<
                                                Arg<One, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                            >,
                                            Mul<
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<Zero, StepSize<A>>,
                                            >,
                                        >,
                                    >,
                                    (
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                        &'static str,
                                    ),
                                    Zip8<
                                        Arg<Zero, A>,
                                        Arg<Zero, BacktrackingRate<A>>,
                                        Arg<One, A>,
                                        Arg<Zero, A>,
                                        Arg<One, A>,
                                        Arg<Zero, StepSize<A>>,
                                        Add<Arg<One, A>, Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>>,
                                        Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
                                    >,
                                    Not<
                                        Le<
                                            F,
                                            Add<
                                                Arg<Zero, A>,
                                                Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                            >,
                                        >,
                                    >,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                            >,
                        >,
                    >,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        (&'static str, &'static str),
                    ),
                    Zip6<
                        Arg<One, A>,
                        Arg<One, A>,
                        Arg<Two, A>,
                        Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>,
                        Arg<One, A>,
                        FFD,
                    >,
                >,
                (
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                    &'static str,
                    (&'static str, &'static str),
                ),
                Then<
                    Then<
                        Zip7<
                            Arg<One, A>,
                            Arg<Two, A>,
                            Sub<Arg<One, A>, Arg<One, A>>,
                            Arg<Zero, StepSize<A>>,
                            Arg<One, A>,
                            Arg<Zero, A>,
                            Arg<One, A>,
                        >,
                        (
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                            &'static str,
                        ),
                        Then<
                            Then<
                                Zip7<
                                    Arg<One, A>,
                                    Arg<One, A>,
                                    Arg<Zero, StepSize<A>>,
                                    Arg<One, A>,
                                    Arg<Zero, A>,
                                    Arg<One, A>,
                                    If<
                                        Then<
                                            Zip3<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Sub<Arg<One, A>, Arg<One, A>>,
                                            >,
                                            (&'static str, &'static str, &'static str),
                                            Zip4<
                                                Arg<Two, A>,
                                                Arg<One, A>,
                                                Arg<One, A>,
                                                ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                            >,
                                        >,
                                        (&'static str, &'static str, &'static str, &'static str),
                                        Eq<Arg<Zero, A>, Val<Zero, A>>,
                                        Arg<Two, A>,
                                        Then<
                                            Then<
                                                Zip5<
                                                    Arg<Two, A>,
                                                    Arg<One, A>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Div<Val<Zero, A>, Arg<Zero, A>>,
                                                >,
                                                (
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                ),
                                                Zip7<
                                                    Arg<Two, A>,
                                                    Arg<One, A>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Arg<Zero, A>,
                                                    Mul<Arg<One, A>, Arg<Zero, A>>,
                                                    IdentityMatrix<Len<Arg<One, A>>, A>,
                                                >,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Add<
                                                MatMul<
                                                    MatMul<
                                                        Sub<
                                                            Arg<Two, A>,
                                                            MulOut<Arg<One, A>, Arg<One, A>>,
                                                        >,
                                                        Arg<Two, A>,
                                                    >,
                                                    Sub<
                                                        Arg<Two, A>,
                                                        MulOut<
                                                            Mul<Arg<One, A>, Arg<Zero, A>>,
                                                            Arg<One, A>,
                                                        >,
                                                    >,
                                                >,
                                                MulOut<Arg<One, A>, Arg<One, A>>,
                                            >,
                                        >,
                                    >,
                                >,
                                (
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                    &'static str,
                                ),
                                Zip4<
                                    Arg<One, A>,
                                    Arg<One, A>,
                                    Arg<Two, A>,
                                    Then<
                                        LoopWhile<
                                            Then<
                                                Zip7<
                                                    Val<Zero, SufficientDecreaseParameter<A>>,
                                                    Val<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Arg<One, A>,
                                                    Neg<MulCol<Arg<Two, A>, Arg<One, A>>>,
                                                >,
                                                (
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                    &'static str,
                                                ),
                                                Zip8<
                                                    Mul<
                                                        Arg<Zero, SufficientDecreaseParameter<A>>,
                                                        ScalarProduct<Arg<One, A>, Arg<One, A>>,
                                                    >,
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<One, A>,
                                                    Arg<Zero, A>,
                                                    Arg<One, A>,
                                                    Arg<Zero, StepSize<A>>,
                                                    Add<
                                                        Arg<One, A>,
                                                        Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                    >,
                                                    Mul<
                                                        Arg<Zero, BacktrackingRate<A>>,
                                                        Arg<Zero, StepSize<A>>,
                                                    >,
                                                >,
                                            >,
                                            (
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                                &'static str,
                                            ),
                                            Zip8<
                                                Arg<Zero, A>,
                                                Arg<Zero, BacktrackingRate<A>>,
                                                Arg<One, A>,
                                                Arg<Zero, A>,
                                                Arg<One, A>,
                                                Arg<Zero, StepSize<A>>,
                                                Add<
                                                    Arg<One, A>,
                                                    Mul<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                                >,
                                                Mul<
                                                    Arg<Zero, BacktrackingRate<A>>,
                                                    Arg<Zero, StepSize<A>>,
                                                >,
                                            >,
                                            Not<
                                                Le<
                                                    F,
                                                    Add<
                                                        Arg<Zero, A>,
                                                        Mul<Arg<Zero, StepSize<A>>, Arg<Zero, A>>,
                                                    >,
                                                >,
                                            >,
                                        >,
                                        (
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                            &'static str,
                                        ),
                                        Zip<Arg<Zero, StepSize<A>>, Arg<One, A>>,
                                    >,
                                >,
                            >,
                            (
                                &'static str,
                                &'static str,
                                &'static str,
                                (&'static str, &'static str),
                            ),
                            Zip4<
                                Arg<One, A>,
                                Arg<One, A>,
                                Arg<Two, A>,
                                Zip<
                                    Mul<Val<Zero, IncrRate<A>>, Arg<Zero, StepSize<A>>>,
                                    Arg<One, A>,
                                >,
                            >,
                        >,
                    >,
                    (
                        &'static str,
                        &'static str,
                        &'static str,
                        (&'static str, &'static str),
                    ),
                    Zip6<
                        Arg<One, A>,
                        Arg<One, A>,
                        Arg<Two, A>,
                        Arg<Zero, StepSize<A>>,
                        Arg<One, A>,
                        FFD,
                    >,
                >,
                Not<
                    Lt<
                        Max<Abs<Arg<One, A>>>,
                        Mul<Val<Zero, A>, Add<Val<Zero, A>, Abs<Arg<Zero, A>>>>,
                    >,
                >,
            >,
            (
                &'static str,
                &'static str,
                &'static str,
                &'static str,
                &'static str,
                (&'static str, &'static str),
            ),
            Arg<One, A>,
        >,
    >,
>;
