//! Low-level functions offering greater flexibility.
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{peano::*, run::Value, zip::*, *};
//! use optimal_linesearch::{
//!     backtracking_line_search::{low_level::*, types::*},
//!     initial_step_size::IncrRate,
//!     step_direction::steepest_descent,
//!     StepSize,
//! };
//!
//! fn main() {
//!     let c_1 = SufficientDecreaseParameter::default();
//!     let backtracking_rate = BacktrackingRate::default();
//!     let incr_rate = IncrRate::from_backtracking_rate(backtracking_rate);
//!     let initial_step_size = StepSize::new(1.0).unwrap();
//!     let initial_point = vec![10.0, 10.0];
//!
//!     let line_search = val!(0)
//!         .zip(val!(initial_step_size).zip(val1!(initial_point)))
//!         .loop_while(
//!             ("i", ("step_size", "point")),
//!             (arg!("i", usize) + val!(1)).zip(
//!                 Zip3::new(
//!                     arg!("step_size", StepSize<f64>),
//!                     arg1!("point", f64),
//!                     arg1!("point", f64).black_box::<_, (Zero, One), (f64, f64)>(
//!                         |point: Vec<f64>| (Value(obj_func(&point)), Value(obj_func_d(&point))),
//!                     ),
//!                 )
//!                 .then(
//!                     ("step_size", "point", ("value", "derivatives")),
//!                     search(
//!                         val!(c_1),
//!                         val!(backtracking_rate),
//!                         arg1!("point", f64)
//!                             .black_box::<_, Zero, f64>(|point: Vec<f64>| Value(obj_func(&point))),
//!                         arg!("step_size", StepSize<f64>),
//!                         arg1!("point", f64),
//!                         arg!("value", f64),
//!                         arg1!("derivatives", f64),
//!                         steepest_descent(arg1!("derivatives", f64)),
//!                     ),
//!                 )
//!                 .then(
//!                     ("step_size", "point"),
//!                     (val!(incr_rate) * arg!("step_size", StepSize<f64>)).zip(arg1!("point", f64)),
//!                 ),
//!             ),
//!             arg!("i", usize).lt(val!(100)),
//!         )
//!         .then(("i", ("step_size", "point")), arg1!("point", f64));
//!
//!     println!("{:?}", line_search.run(argvals![]));
//! }
//!
//! fn obj_func(point: &[f64]) -> f64 {
//!     point.iter().map(|x| x.powi(2)).sum()
//! }
//!
//! fn obj_func_d(point: &[f64]) -> Vec<f64> {
//!     point.iter().map(|x| 2.0 * x).collect()
//! }
//! ```

use core::ops;

use optimal_compute_core::{
    arg, arg1,
    cmp::{Le, Not},
    control_flow::{LoopWhile, Then},
    linalg::ScalarProduct,
    math::{Add, Mul},
    peano::{One, Zero},
    zip::{Zip, Zip7, Zip8},
    Arg, Computation, ComputationFn,
};

use crate::{descend, Descend, StepSize};

use super::types::*;

type Search<C1, B, F, S, P, V, DE, DI, A> = Then<
    LoopWhile<
        Then<
            Zip7<C1, B, S, P, V, DE, DI>,
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
                C1TimesDerivativesDotDirection<
                    Arg<Zero, SufficientDecreaseParameter<A>>,
                    Arg<One, A>,
                    Arg<One, A>,
                >,
                Arg<Zero, BacktrackingRate<A>>,
                Arg<One, A>,
                Arg<Zero, A>,
                Arg<One, A>,
                Arg<Zero, StepSize<A>>,
                Descend<Arg<Zero, StepSize<A>>, Arg<One, A>, Arg<One, A>>,
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
            Descend<Arg<Zero, StepSize<A>>, Arg<One, A>, Arg<One, A>>,
            Mul<Arg<Zero, BacktrackingRate<A>>, Arg<Zero, StepSize<A>>>,
        >,
        Not<IsSufficientDecrease<Arg<Zero, A>, Arg<Zero, A>, Arg<Zero, StepSize<A>>, F>>,
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
>;

/// Return the step-size
/// and corresponding point
/// that minimizes the objective function.
#[allow(clippy::too_many_arguments)]
pub fn search<C1, B, F, S, P, V, DE, DI, A>(
    c_1: C1,
    backtracking_rate: B,
    obj_func: F,
    initial_step_size: S,
    point: P,
    value: V,
    derivatives: DE,
    direction: DI,
) -> Search<C1, B, F, S, P, V, DE, DI, A>
where
    C1: Computation<Dim = Zero, Item = SufficientDecreaseParameter<A>>,
    B: Computation<Dim = Zero, Item = BacktrackingRate<A>>,
    F: ComputationFn<Dim = Zero, Item = A>,
    S: Computation<Dim = Zero, Item = StepSize<A>>,
    P: Computation<Dim = One, Item = A>,
    V: Computation<Dim = Zero, Item = A>,
    DE: Computation<Dim = One, Item = A>,
    DI: Computation<Dim = One, Item = A>,
    A: Clone + PartialOrd + ops::Add<Output = A> + ops::Mul<Output = A>,
{
    Zip7::new(
        c_1,
        backtracking_rate,
        initial_step_size,
        point,
        value,
        derivatives,
        direction,
    )
    .then(
        (
            "c_1",
            "backtracking_rate",
            "initial_step_size",
            "point",
            "value",
            "derivatives",
            "direction",
        ),
        Zip8::new(
            c_1_times_derivatives_dot_direction(
                arg!("c_1", SufficientDecreaseParameter<A>),
                arg1!("derivatives", A),
                arg1!("direction", A),
            ),
            arg!("backtracking_rate", BacktrackingRate<A>),
            arg1!("point", A),
            arg!("value", A),
            arg1!("direction", A),
            arg!("initial_step_size", StepSize<A>),
            descend(
                arg!("initial_step_size", StepSize<A>),
                arg1!("direction", A),
                arg1!("point", A),
            ),
            arg!("backtracking_rate", BacktrackingRate<A>) * arg!("initial_step_size", StepSize<A>),
        ),
    )
    .loop_while(
        (
            "c1tddd",
            "backtracking_rate",
            "initial_point",
            "value",
            "direction",
            "step_size",
            "point",
            "next_step_size",
        ),
        Zip8::new(
            arg!("c1tddd"),
            arg!("backtracking_rate", BacktrackingRate<A>),
            arg1!("initial_point", A),
            arg!("value", A),
            arg1!("direction", A),
            arg!("next_step_size", StepSize<A>),
            descend(
                arg!("next_step_size", StepSize<A>),
                arg1!("direction", A),
                arg1!("initial_point", A),
            ),
            arg!("backtracking_rate", BacktrackingRate<A>) * arg!("next_step_size", StepSize<A>),
        ),
        is_sufficient_decrease(
            arg!("c1tddd", A),
            arg!("value", A),
            arg!("step_size", StepSize<A>),
            obj_func,
        )
        .not(),
    )
    .then(
        (
            "c1tddd",
            "backtracking_rate",
            "initial_point",
            "value",
            "direction",
            "step_size",
            "point",
            "next_step_size",
        ),
        arg!("step_size", StepSize<A>).zip(arg1!("point", A)),
    )
}

/// See [`c_1_times_derivatives_dot_direction`].
pub type C1TimesDerivativesDotDirection<C1, DE, DI> = Mul<C1, ScalarProduct<DE, DI>>;

/// Prepare a value to check for sufficient decrease.
pub fn c_1_times_derivatives_dot_direction<C1, DE, DI, A>(
    c_1: C1,
    derivatives: DE,
    direction: DI,
) -> C1TimesDerivativesDotDirection<C1, DE, DI>
where
    C1: Computation<Dim = Zero, Item = SufficientDecreaseParameter<A>>,
    DE: Computation<Dim = One, Item = A>,
    DI: Computation<Dim = One, Item = A>,
    A: ops::Add<Output = A> + ops::Mul<Output = A>,
{
    c_1.mul(derivatives.scalar_product(direction))
}

/// See [`is_sufficient_decrease`].
pub type IsSufficientDecrease<C1TDDD, V, S, VAS> = Le<VAS, Add<V, Mul<S, C1TDDD>>>;

/// Return whether `point_at_step` is sufficient.
///
/// The sufficient decrease condition,
/// also known as the Armijo rule,
/// mathematically written as:
/// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
pub fn is_sufficient_decrease<C1TDDD, V, S, VAS, A>(
    c_1_times_derivatives_dot_direction: C1TDDD,
    value: V,
    step_size: S,
    value_at_step: VAS,
) -> IsSufficientDecrease<C1TDDD, V, S, VAS>
where
    C1TDDD: Computation<Dim = Zero, Item = A>,
    V: Computation<Dim = Zero, Item = A>,
    S: Computation<Dim = Zero, Item = StepSize<A>>,
    VAS: Computation<Dim = Zero, Item = A>,
    A: PartialOrd + ops::Add<Output = A> + ops::Mul<Output = A>,
{
    value_at_step.le(value.add(step_size.mul(c_1_times_derivatives_dot_direction)))
}
