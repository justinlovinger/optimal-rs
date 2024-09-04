use std::{iter::Sum, ops::RangeInclusive};

use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::{AsPrimitive, Float, Signed};
use optimal_compute_core::{
    arg, arg1, arg2, argvals,
    peano::{One, Zero},
    run::Value,
    val, val1,
    zip::{Zip3, Zip4, Zip5, Zip6, Zip7},
    Computation, Run,
};
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
        Zip7::new(
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
        .then(
            (
                "prev_derivatives",
                "prev_step",
                "step_size",
                "point",
                "value",
                "derivatives",
                "approx_inv_snd_derivatives",
            ),
            Zip4::new(
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
        )
        .then(
            (
                "prev_point",
                "prev_derivatives",
                "prev_approx_inv_snd_derivatives",
                ("step_size", "point"),
            ),
            Zip4::new(
                arg1!("prev_point", A),
                arg1!("prev_derivatives", A),
                arg2!("prev_approx_inv_snd_derivatives", A),
                (val!($incr_rate) * arg!("step_size", StepSize<A>)).zip(arg1!("point", A)),
            ),
        )
    };
}

impl<A, F, FFD> BacktrackingLineSearchWith<A, F, FFD>
where
    A: std::fmt::Debug + Sum + Signed + Float + ScalarOperand + LinalgScalar,
    f64: AsPrimitive<A>,
    F: Fn(&[A]) -> A,
    FFD: Fn(&[A]) -> (A, Vec<A>),
{
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<A> {
        // This code is a bit of a mess.
        // It is written the way it is
        // in part because we want it to eventually return a computation,
        // not the result of running a computation.

        let c_1 = val!(self.problem.agnostic.c_1);
        let backtracking_rate = val!(self.problem.agnostic.backtracking_rate);
        let initial_step_size = val!(self.problem.agnostic.initial_step_size);
        let obj_func = arg1!("point", A)
            .black_box::<_, Zero, A>(|point: Vec<A>| Value((self.problem.obj_func)(&point)));
        let obj_func_and_d =
            arg1!("point", A).black_box::<_, (Zero, One), (A, A)>(|point: Vec<A>| {
                let (value, derivatives) = (self.problem.obj_func_and_d)(&point);
                (Value(value), Value(derivatives))
            });
        match (
            self.problem.agnostic.direction,
            self.problem.agnostic.step_size_update,
            self.problem.agnostic.stopping_criteria,
        ) {
            (
                StepDirection::Steepest,
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::Iteration(i),
            ) => val!(0)
                .zip(initial_step_size.zip(val1!(self.initial_point)))
                .loop_while(
                    ("i", ("step_size", "point")),
                    (arg!("i", usize) + val!(1)).zip(
                        Zip3::new(
                            arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            obj_func_and_d,
                        )
                        .then(
                            ("step_size", "point", ("value", "derivatives")),
                            search(
                                c_1,
                                backtracking_rate,
                                obj_func,
                                arg!("step_size", StepSize<A>),
                                arg1!("point", A),
                                arg!("value", A),
                                arg1!("derivatives", A),
                                steepest_descent(arg1!("derivatives", A)),
                            ),
                        )
                        .then(
                            ("step_size", "point"),
                            (val!(incr_rate) * arg!("step_size", StepSize<A>))
                                .zip(arg1!("point", A)),
                        ),
                    ),
                    arg!("i", usize).lt(val!(i)),
                )
                .then(("i", ("step_size", "point")), arg1!("point", A))
                .run(argvals![]),
            (
                StepDirection::Steepest,
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::NearMinima,
            ) => initial_step_size
                .zip(val1!(self.initial_point))
                .then(
                    ("step_size", "point"),
                    Zip3::new(
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        obj_func_and_d,
                    ),
                )
                .loop_while(
                    ("step_size", "point", ("value", "derivatives")),
                    search(
                        c_1,
                        backtracking_rate,
                        obj_func,
                        arg!("step_size", StepSize<A>),
                        arg1!("point", A),
                        arg!("value", A),
                        arg1!("derivatives", A),
                        steepest_descent(arg1!("derivatives", A)),
                    )
                    .then(
                        ("step_size", "point"),
                        Zip3::new(
                            val!(incr_rate) * arg!("step_size", StepSize<A>),
                            arg1!("point", A),
                            obj_func_and_d,
                        ),
                    ),
                    is_near_minima(arg!("value", A), arg1!("derivatives", A)).not(),
                )
                .then(
                    ("step_size", "point", ("value", "derivatives")),
                    arg1!("point", A),
                )
                .run(argvals![]),
            (
                StepDirection::Bfgs { initializer },
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::Iteration(i),
            ) => {
                if i == 0 {
                    self.initial_point
                } else {
                    let len = self.initial_point.len();
                    let first_iteration = val1!(self.initial_point)
                        .then("point", arg1!("point", A).zip(obj_func_and_d))
                        .then(
                            ("point", ("value", "derivatives")),
                            Zip3::new(
                                arg1!("point", A),
                                arg1!("derivatives", A),
                                search(
                                    c_1,
                                    backtracking_rate,
                                    obj_func,
                                    initial_step_size,
                                    arg1!("point", A),
                                    arg!("value", A),
                                    arg1!("derivatives", A),
                                    bfgs_direction(
                                        initial_approx_inv_snd_derivatives_identity(val!(len)),
                                        arg1!("derivatives", A),
                                    ),
                                ),
                            )
                            .then(
                                ("prev_point", "prev_derivatives", ("step_size", "point")),
                                Zip3::new(
                                    arg1!("prev_point", A),
                                    arg1!("prev_derivatives", A),
                                    (val!(incr_rate) * arg!("step_size", StepSize<A>))
                                        .zip(arg1!("point", A)),
                                ),
                            ),
                        );
                    if i == 1 {
                        first_iteration
                            .then(
                                ("prev_point", "prev_derivatives", ("step_size", "point")),
                                arg1!("point", A),
                            )
                            .run(argvals![])
                    } else {
                        match initializer {
                            BfgsInitializer::Identity => first_iteration
                                .then(
                                    ("prev_point", "prev_derivatives", ("step_size", "point")),
                                    val!(1).zip(Zip4::new(
                                        arg1!("prev_point", A),
                                        arg1!("prev_derivatives", A),
                                        initial_approx_inv_snd_derivatives_identity(val!(len)),
                                        arg!("step_size", StepSize<A>).zip(arg1!("point", A)),
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
                                    (arg!("i", usize) + val!(1_usize)).zip(
                                        Zip6::new(
                                            arg1!("prev_derivatives", A),
                                            arg2!("prev_approx_inv_snd_derivatives", A),
                                            arg1!("point", A) - arg1!("prev_point", A),
                                            arg!("step_size", StepSize<A>),
                                            arg1!("point", A),
                                            obj_func_and_d,
                                        )
                                        .then(
                                            (
                                                "prev_derivatives",
                                                "prev_approx_inv_snd_derivatives",
                                                "prev_step",
                                                "step_size",
                                                "point",
                                                ("value", "derivatives"),
                                            ),
                                            bfgs_loop!(c_1, backtracking_rate, obj_func, incr_rate),
                                        ),
                                    ),
                                    arg!("i", usize).lt(val!(i)),
                                )
                                .then(
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
                                )
                                .run(argvals![]),
                            BfgsInitializer::Gamma => {
                                let second_iteration = first_iteration
                                    .then(
                                        ("prev_point", "prev_derivatives", ("step_size", "point")),
                                        Zip5::new(
                                            arg1!("prev_derivatives", A),
                                            arg1!("point", A) - arg1!("prev_point", A),
                                            arg!("step_size", StepSize<A>),
                                            arg1!("point", A),
                                            obj_func_and_d,
                                        ),
                                    )
                                    .then(
                                        (
                                            "prev_derivatives",
                                            "prev_step",
                                            "step_size",
                                            "point",
                                            ("value", "derivatives"),
                                        ),
                                        Zip7::new(
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
                                    )
                                    .then(
                                        (
                                            "prev_derivatives",
                                            "prev_step",
                                            "step_size",
                                            "point",
                                            "value",
                                            "derivatives",
                                            "approx_inv_snd_derivatives",
                                        ),
                                        val!(2_usize).zip(
                                            Zip4::new(
                                                arg1!("point", A),
                                                arg1!("derivatives", A),
                                                arg2!("approx_inv_snd_derivatives", A),
                                                search(
                                                    c_1,
                                                    backtracking_rate,
                                                    obj_func,
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
                                            .then(
                                                (
                                                    "prev_point",
                                                    "prev_derivatives",
                                                    "prev_approx_inv_snd_derivatives",
                                                    ("step_size", "point"),
                                                ),
                                                Zip4::new(
                                                    arg1!("prev_point", A),
                                                    arg1!("prev_derivatives", A),
                                                    arg2!("prev_approx_inv_snd_derivatives", A),
                                                    (val!(incr_rate)
                                                        * arg!("step_size", StepSize<A>))
                                                    .zip(arg1!("point", A)),
                                                ),
                                            ),
                                        ),
                                    );

                                if i == 2 {
                                    second_iteration
                                        .then(
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
                                        )
                                        .run(argvals![])
                                } else {
                                    second_iteration
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
                                            (arg!("i", usize) + val!(1_usize)).zip(
                                                Zip6::new(
                                                    arg1!("prev_derivatives", A),
                                                    arg2!("prev_approx_inv_snd_derivatives", A),
                                                    arg1!("point", A) - arg1!("prev_point", A),
                                                    arg!("step_size", StepSize<A>),
                                                    arg1!("point", A),
                                                    obj_func_and_d,
                                                )
                                                .then(
                                                    (
                                                        "prev_derivatives",
                                                        "prev_approx_inv_snd_derivatives",
                                                        "prev_step",
                                                        "step_size",
                                                        "point",
                                                        ("value", "derivatives"),
                                                    ),
                                                    bfgs_loop!(
                                                        c_1,
                                                        backtracking_rate,
                                                        obj_func,
                                                        incr_rate,
                                                    ),
                                                ),
                                            ),
                                            arg!("i", usize).lt(val!(i)),
                                        )
                                        .then(
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
                                        )
                                        .run(argvals![])
                                }
                            }
                        }
                    }
                }
            }
            (
                StepDirection::Bfgs { initializer },
                StepSizeUpdate::IncrPrev(incr_rate),
                BacktrackingLineSearchStoppingCriteria::NearMinima,
            ) => {
                let len = self.initial_point.len();
                match initializer {
                    BfgsInitializer::Identity => val1!(self.initial_point)
                        .then("point", arg1!("point", A).zip(obj_func_and_d))
                        .if_(
                            ("point", ("value", "derivatives")),
                            is_near_minima(arg!("value", A), arg1!("derivatives", A)),
                            arg1!("point", A),
                            Zip4::new(
                                arg1!("point", A),
                                arg!("value", A),
                                arg1!("derivatives", A),
                                initial_approx_inv_snd_derivatives_identity::<_, A>(val!(len)),
                            )
                            .then(
                                (
                                    "point",
                                    "value",
                                    "derivatives",
                                    "approx_inv_snd_derivatives",
                                ),
                                Zip4::new(
                                    arg1!("point", A),
                                    arg1!("derivatives", A),
                                    arg2!("approx_inv_snd_derivatives", A),
                                    search(
                                        c_1,
                                        backtracking_rate,
                                        obj_func,
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
                            )
                            .then(
                                (
                                    "prev_point",
                                    "prev_derivatives",
                                    "prev_approx_inv_snd_derivatives",
                                    ("step_size", "point"),
                                ),
                                Zip6::new(
                                    arg1!("prev_point", A),
                                    arg1!("prev_derivatives", A),
                                    arg2!("prev_approx_inv_snd_derivatives", A),
                                    val!(incr_rate) * arg!("step_size", StepSize<A>),
                                    arg1!("point", A),
                                    obj_func_and_d,
                                ),
                            )
                            .loop_while(
                                (
                                    "prev_point",
                                    "prev_derivatives",
                                    "prev_approx_inv_snd_derivatives",
                                    "step_size",
                                    "point",
                                    ("value", "derivatives"),
                                ),
                                Zip7::new(
                                    arg1!("prev_derivatives", A),
                                    arg2!("prev_approx_inv_snd_derivatives", A),
                                    arg1!("point", A) - arg1!("prev_point", A),
                                    arg!("step_size", StepSize<A>),
                                    arg1!("point", A),
                                    arg!("value", A),
                                    arg1!("derivatives", A),
                                )
                                .then(
                                    (
                                        "prev_derivatives",
                                        "prev_approx_inv_snd_derivatives",
                                        "prev_step",
                                        "step_size",
                                        "point",
                                        "value",
                                        "derivatives",
                                    ),
                                    bfgs_loop!(c_1, backtracking_rate, obj_func, incr_rate),
                                )
                                .then(
                                    (
                                        "prev_point",
                                        "prev_derivatives",
                                        "prev_approx_inv_snd_derivatives",
                                        ("step_size", "point"),
                                    ),
                                    Zip6::new(
                                        arg1!("prev_point", A),
                                        arg1!("prev_derivatives", A),
                                        arg2!("prev_approx_inv_snd_derivatives"),
                                        arg!("step_size", StepSize<A>),
                                        arg1!("point", A),
                                        obj_func_and_d,
                                    ),
                                ),
                                is_near_minima(arg!("value", A), arg1!("derivatives", A)).not(),
                            )
                            .then(
                                (
                                    "prev_point",
                                    "prev_derivatives",
                                    "prev_approx_inv_snd_derivatives",
                                    "step_size",
                                    "point",
                                    ("value", "derivatives"),
                                ),
                                arg1!("point", A),
                            ),
                        )
                        .run(argvals![]),
                    BfgsInitializer::Gamma => val1!(self.initial_point)
                        .then("point", arg1!("point", A).zip(obj_func_and_d))
                        .if_(
                            ("point", ("value", "derivatives")),
                            is_near_minima(arg!("value", A), arg1!("derivatives", A)),
                            arg1!("point", A),
                            Zip3::new(
                                arg1!("point", A),
                                arg1!("derivatives", A),
                                search(
                                    c_1,
                                    backtracking_rate,
                                    obj_func,
                                    initial_step_size,
                                    arg1!("point", A),
                                    arg!("value", A),
                                    arg1!("derivatives", A),
                                    bfgs_direction(
                                        initial_approx_inv_snd_derivatives_identity(val!(len)),
                                        arg1!("derivatives", A),
                                    ),
                                ),
                            )
                            .then(
                                ("prev_point", "prev_derivatives", ("step_size", "point")),
                                Zip5::new(
                                    arg1!("prev_point", A),
                                    arg1!("prev_derivatives", A),
                                    val!(incr_rate) * arg!("step_size", StepSize<A>),
                                    arg1!("point", A),
                                    obj_func_and_d,
                                ),
                            )
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
                                Zip6::new(
                                    arg1!("prev_derivatives", A),
                                    arg1!("point", A) - arg1!("prev_point", A),
                                    arg!("step_size", StepSize<A>),
                                    arg1!("point", A),
                                    arg!("value", A),
                                    arg1!("derivatives", A),
                                )
                                .then(
                                    (
                                        "prev_derivatives",
                                        "prev_step",
                                        "step_size",
                                        "point",
                                        "value",
                                        "derivatives",
                                    ),
                                    Zip7::new(
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
                                )
                                .then(
                                    (
                                        "prev_derivatives",
                                        "prev_step",
                                        "step_size",
                                        "point",
                                        "value",
                                        "derivatives",
                                        "approx_inv_snd_derivatives",
                                    ),
                                    Zip4::new(
                                        arg1!("point", A),
                                        arg1!("derivatives", A),
                                        arg2!("approx_inv_snd_derivatives", A),
                                        search(
                                            c_1,
                                            backtracking_rate,
                                            obj_func,
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
                                )
                                .then(
                                    (
                                        "prev_point",
                                        "prev_derivatives",
                                        "prev_approx_inv_snd_derivatives",
                                        ("step_size", "point"),
                                    ),
                                    Zip6::new(
                                        arg1!("prev_point", A),
                                        arg1!("prev_derivatives", A),
                                        arg2!("prev_approx_inv_snd_derivatives", A),
                                        val!(incr_rate) * arg!("step_size", StepSize<A>),
                                        arg1!("point", A),
                                        obj_func_and_d,
                                    ),
                                )
                                .loop_while(
                                    (
                                        "prev_point",
                                        "prev_derivatives",
                                        "prev_approx_inv_snd_derivatives",
                                        "step_size",
                                        "point",
                                        ("value", "derivatives"),
                                    ),
                                    Zip7::new(
                                        arg1!("prev_derivatives", A),
                                        arg2!("prev_approx_inv_snd_derivatives", A),
                                        arg1!("point", A) - arg1!("prev_point", A),
                                        arg!("step_size", StepSize<A>),
                                        arg1!("point", A),
                                        arg!("value", A),
                                        arg1!("derivatives", A),
                                    )
                                    .then(
                                        (
                                            "prev_derivatives",
                                            "prev_approx_inv_snd_derivatives",
                                            "prev_step",
                                            "step_size",
                                            "point",
                                            "value",
                                            "derivatives",
                                        ),
                                        bfgs_loop!(c_1, backtracking_rate, obj_func, incr_rate),
                                    )
                                    .then(
                                        (
                                            "prev_point",
                                            "prev_derivatives",
                                            "prev_approx_inv_snd_derivatives",
                                            ("step_size", "point"),
                                        ),
                                        Zip6::new(
                                            arg1!("prev_point", A),
                                            arg1!("prev_derivatives", A),
                                            arg2!("prev_approx_inv_snd_derivatives"),
                                            arg!("step_size", StepSize<A>),
                                            arg1!("point", A),
                                            obj_func_and_d,
                                        ),
                                    ),
                                    is_near_minima(arg!("value", A), arg1!("derivatives", A)).not(),
                                )
                                .then(
                                    (
                                        "prev_point",
                                        "prev_derivatives",
                                        "prev_approx_inv_snd_derivatives",
                                        "step_size",
                                        "point",
                                        ("value", "derivatives"),
                                    ),
                                    arg1!("point", A),
                                ),
                            ),
                        )
                        .run(argvals![]),
                }
            }
        }
    }
}
