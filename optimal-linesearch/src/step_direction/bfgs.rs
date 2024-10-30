//! Functions for the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm,
//! an algorithm to approximate a step-direction
//! using second-derivatives
//! without directly calculating second-derivatives.
//!
//! Note,
//! this requires `n^2` memory and time,
//! where `n` is the length of a point.
//! It is not suitable for problems with large points.
//! See L-BFGS for a limited-memory alternative
//! more suitable for problems with large points
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{peano::*, run::Value, zip::*, *};
//! use optimal_linesearch::{step_direction::bfgs::*, StepSize};
//!
//! fn main() {
//!     let step_size = val!(StepSize::new(0.5).unwrap());
//!     let initial_point = vec![10.0, 10.0];
//!     let len = initial_point.len();
//!     let obj_func_d =
//!         arg1!("point", f64).black_box::<_, One, f64>(|point: Vec<f64>| Value(obj_func_d(&point)));
//!
//!     let bfgs = val!(1)
//!         .zip(
//!             val1!(initial_point)
//!                 .then(
//!                     "point",
//!                     Zip3(
//!                         arg1!("point", f64),
//!                         obj_func_d,
//!                         initial_approx_inv_snd_derivatives_identity::<_, f64>(val!(len)),
//!                     ),
//!                 )
//!                 .then(
//!                     ("point", "derivatives", "approx_inv_snd_derivatives"),
//!                     Zip4(
//!                         arg1!("point", f64),
//!                         arg1!("derivatives", f64),
//!                         arg2!("approx_inv_snd_derivatives", f64),
//!                         step_size
//!                             * bfgs_direction(
//!                                 arg2!("approx_inv_snd_derivatives", f64),
//!                                 arg1!("derivatives", f64),
//!                             ),
//!                     ),
//!                 )
//!                 .then(
//!                     ("point", "derivatives", "approx_inv_snd_derivatives", "step"),
//!                     Zip4(
//!                         arg1!("derivatives", f64),
//!                         arg2!("approx_inv_snd_derivatives", f64),
//!                         arg1!("step", f64),
//!                         arg1!("point", f64) + arg1!("step", f64),
//!                     ),
//!                 ),
//!         )
//!         .loop_while(
//!             (
//!                 "i",
//!                 (
//!                     "prev_derivatives",
//!                     "prev_approx_inv_snd_derivatives",
//!                     "prev_step",
//!                     "point",
//!                 ),
//!             ),
//!             (arg!("i", usize) + val!(1_usize)).zip(
//!                 Zip5(
//!                     arg1!("prev_derivatives", f64),
//!                     arg2!("prev_approx_inv_snd_derivatives", f64),
//!                     arg1!("prev_step", f64),
//!                     arg1!("point", f64),
//!                     obj_func_d,
//!                 )
//!                 .then(
//!                     (
//!                         "prev_derivatives",
//!                         "prev_approx_inv_snd_derivatives",
//!                         "prev_step",
//!                         "point",
//!                         "derivatives",
//!                     ),
//!                     Zip3(
//!                         arg1!("point", f64),
//!                         arg1!("derivatives", f64),
//!                         approx_inv_snd_derivatives(
//!                             arg2!("prev_approx_inv_snd_derivatives", f64),
//!                             arg1!("prev_derivatives", f64),
//!                             arg1!("prev_step", f64),
//!                             arg1!("derivatives", f64),
//!                         ),
//!                     ),
//!                 )
//!                 .then(
//!                     ("point", "derivatives", "approx_inv_snd_derivatives"),
//!                     Zip4(
//!                         arg1!("point", f64),
//!                         arg1!("derivatives", f64),
//!                         arg2!("approx_inv_snd_derivatives", f64),
//!                         step_size
//!                             * bfgs_direction(
//!                                 arg2!("approx_inv_snd_derivatives", f64),
//!                                 arg1!("derivatives", f64),
//!                             ),
//!                     ),
//!                 )
//!                 .then(
//!                     ("point", "derivatives", "approx_inv_snd_derivatives", "step"),
//!                     Zip4(
//!                         arg1!("derivatives", f64),
//!                         arg2!("approx_inv_snd_derivatives", f64),
//!                         arg1!("step", f64),
//!                         arg1!("point", f64) + arg1!("step", f64),
//!                     ),
//!                 ),
//!             ),
//!             arg!("i", usize).lt(val!(10)),
//!         )
//!         .then(
//!             (
//!                 "i",
//!                 (
//!                     "prev_derivatives",
//!                     "prev_approx_inv_snd_derivatives",
//!                     "prev_step",
//!                     "point",
//!                 ),
//!             ),
//!             arg1!("point", f64),
//!         );
//!
//!     println!("{:?}", bfgs.run(named_args![]));
//! }
//!
//! fn obj_func_d(point: &[f64]) -> Vec<f64> {
//!     point.iter().copied().map(|x| 2.0 * x).collect()
//! }
//! ```

use std::ops;

use ndarray::{LinalgScalar, ScalarOperand};
use optimal_compute_core::{
    arg, arg1, arg2,
    cmp::Eq,
    control_flow::{If, Then},
    linalg::{FromDiagElem, IdentityMatrix, MatMul, MulOut, ScalarProduct},
    math::{Add, Div, Mul, Sub},
    peano::{One, Two, Zero},
    val,
    zip::{Zip, Zip3, Zip4, Zip5, Zip7},
    Arg, Computation, ComputationFn, Len, Val,
};

/// Return a placeholder for approximate inverse second-derivatives,
/// useful for starting BFGS.
pub fn initial_approx_inv_snd_derivatives_identity<L, T>(len: L) -> IdentityMatrix<L, T>
where
    L: Computation<Dim = Zero, Item = usize>,
    T: Clone + num_traits::Zero + num_traits::One,
{
    len.identity_matrix()
}

/// See [`initial_approx_inv_snd_derivatives_gamma`].
pub type InitialApproxInvSndDerivativesGamma<PD, PS, D, A> = If<
    Then<
        Zip<PS, Sub<D, PD>>,
        (&'static str, &'static str),
        Zip3<Arg<One, A>, Arg<One, A>, ScalarProduct<Arg<One, A>, Arg<One, A>>>,
    >,
    (&'static str, &'static str, &'static str),
    Eq<Arg<Zero, A>, Val<Zero, A>>,
    IdentityMatrix<Len<Arg<One, A>>, A>,
    FromDiagElem<Len<Arg<One, A>>, Div<ScalarProduct<Arg<One, A>, Arg<One, A>>, Arg<Zero, A>>>,
>;

/// Return a placeholder for approximate inverse second-derivatives,
/// informed by the previous step,
/// but still not as good as actual approximate inverse second-derivatives.
///
/// The value along the diagonal,
/// `gamma`,
/// is defined as
/// `gamma = (\vec{s}^T \vec{y}) / (\vec{y}^T \vec{y})`,
/// where
/// `\vec{s} = \vec{x}_k - \vec{x}_{k - 1}`;
/// `\vec{y} = \vec{dx}_k - \vec{dx}_{k - 1}`;
/// `\vec{x}` is a point;
/// and `\vec{dx}` is the derivatives, Jacobian, of a point.
///
/// `\vec{s}` is the previous step,
/// and `\vec{y}` is the difference in derivatives between this step and the last.
/// All vectors are column-vectors,
/// so `\vec{x}^T` is a row-vector.
/// The current iteration is `k`,
/// and `k - 1` is the previous iteration.
pub fn initial_approx_inv_snd_derivatives_gamma<PD, PS, D, A>(
    prev_derivatives: PD,
    prev_step: PS,
    derivatives: D,
) -> InitialApproxInvSndDerivativesGamma<PD, PS, D, A>
where
    PD: Computation<Dim = One, Item = A>,
    PS: ComputationFn<Dim = One, Item = A>,
    D: Computation<Dim = One, Item = A>,
    A: LinalgScalar,
{
    prev_step
        .zip(derivatives.sub(prev_derivatives))
        .then(
            ("prev_step", "derivatives_diff"),
            Zip3(
                arg1!("prev_step", A),
                arg1!("derivatives_diff", A),
                arg1!("derivatives_diff", A).scalar_product(arg1!("derivatives_diff", A)),
            ),
        )
        // Default to identity if we cannot get gamma.
        .if_(
            ("prev_step", "derivatives_diff", "derivatives_diff_dotted"),
            arg!("derivatives_diff_dotted", A).eq(val!(A::zero())),
            arg1!("derivatives_diff", A).len().identity_matrix(),
            FromDiagElem::new(
                arg1!("derivatives_diff", A).len(),
                arg1!("prev_step", A).scalar_product(arg1!("derivatives_diff"))
                    / arg!("derivatives_diff_dotted"),
            ),
        )
}

/// See [`bfgs_direction`].
pub type BFGSDirection<AISD, D> =
    optimal_compute_core::math::Neg<optimal_compute_core::linalg::MulCol<AISD, D>>;

/// Return a direction of descent informed by approximate second-derivative information.
pub fn bfgs_direction<AISD, D, A>(
    approx_inv_snd_derivatives: AISD,
    derivatives: D,
) -> BFGSDirection<AISD, D>
where
    AISD: Computation<Dim = Two, Item = A>,
    D: Computation<Dim = One, Item = A>,
    A: ops::Neg<Output = A> + LinalgScalar,
{
    approx_inv_snd_derivatives.mul_col(derivatives).neg()
}

/// See [`approx_inv_snd_derivatives`].
pub type ApproxInvSndDerivatives<PISD, PD, PS, D, A> = If<
    Then<
        Zip3<PISD, PS, Sub<D, PD>>,
        (&'static str, &'static str, &'static str),
        Zip4<Arg<Two, A>, Arg<One, A>, Arg<One, A>, ScalarProduct<Arg<One, A>, Arg<One, A>>>,
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
            optimal_compute_core::zip::Zip7<
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
                MatMul<Sub<Arg<Two, A>, MulOut<Arg<One, A>, Arg<One, A>>>, Arg<Two, A>>,
                Sub<Arg<Two, A>, MulOut<Mul<Arg<One, A>, Arg<Zero, A>>, Arg<One, A>>>,
            >,
            MulOut<Arg<One, A>, Arg<One, A>>,
        >,
    >,
>;

/// Apply the BFGS update-rule to obtain the next approximate inverse-second-derivatives.
///
/// Mathematically,
/// the next set of approximate inverse-second-derivatives,
/// the approximate inverse-Hessian,
/// `H_{k+1}`,
/// are defined as,
/// `H_{k+1} = (I - p \vec{s} \vec{y}^T) H_k (I - p \vec{y} \vec{s}^T) + p \vec{s} \vec{s}^T`
/// where
/// `p = 1 / (\vec{y}^T \vec{s})`;
/// `\vec{s} = \vec{x}_k - \vec{x}_{k - 1}`;
/// `\vec{y} = \vec{dx}_k - \vec{dx}_{k - 1}`;
/// `\vec{x}` is a point;
/// and `\vec{dx}` is the derivatives, Jacobian, of a point.
///
/// `H_k` is the previous set of approximate inverse-second-derivatives,
/// `\vec{s}` is the previous step,
/// and `\vec{y}` is the difference in derivatives between this step and the last.
/// All vectors are column-vectors,
/// so `\vec{x}^T` is a row-vector.
/// The current iteration is `k`,
/// and `k - 1` is the previous iteration.
pub fn approx_inv_snd_derivatives<PISD, PD, PS, D, A>(
    prev_inv_snd_derivatives: PISD,
    prev_derivatives: PD,
    prev_step: PS,
    derivatives: D,
) -> ApproxInvSndDerivatives<PISD, PD, PS, D, A>
where
    PISD: ComputationFn<Dim = Two, Item = A>,
    PD: Computation<Dim = One, Item = A>,
    PS: Computation<Dim = One, Item = A>,
    D: Computation<Dim = One, Item = A>,
    A: ScalarOperand + LinalgScalar,
{
    Zip3(
        prev_inv_snd_derivatives,
        prev_step,
        derivatives.sub(prev_derivatives),
    )
    .then(
        ("prev_inv_snd_derivatives", "prev_step", "derivatives_diff"),
        Zip4(
            arg2!("prev_inv_snd_derivatives", A),
            arg1!("prev_step", A),
            arg1!("derivatives_diff", A),
            arg1!("derivatives_diff", A).scalar_product(arg1!("prev_step", A)),
        ),
    )
    // Default to reusing the previous approximate inverse-second-derivatives
    // if we cannot apply an update.
    .if_(
        (
            "prev_inv_snd_derivatives",
            "prev_step",
            "derivatives_diff",
            "derivatives_diff_dot_prev_step",
        ),
        arg!("derivatives_diff_dot_prev_step", A).eq(val!(A::zero())),
        arg2!("prev_inv_snd_derivatives"),
        Zip5(
            arg2!("prev_inv_snd_derivatives", A),
            arg1!("prev_step", A),
            arg1!("derivatives_diff", A),
            arg!("derivatives_diff_dot_prev_step", A),
            val!(A::one()) / arg!("derivatives_diff_dot_prev_step"),
        )
        .then(
            (
                "prev_inv_snd_derivatives",
                "prev_step",
                "derivatives_diff",
                "derivatives_diff_dot_prev_step",
                "p",
            ),
            Zip7(
                arg2!("prev_inv_snd_derivatives", A),
                arg1!("prev_step", A),
                arg1!("derivatives_diff", A),
                arg!("derivatives_diff_dot_prev_step", A),
                arg!("p", A),
                arg1!("prev_step", A) * arg!("p"),
                arg1!("prev_step", A).len().identity_matrix::<A>(),
            ),
        )
        .then(
            (
                "prev_inv_snd_derivatives",
                "prev_step",
                "derivatives_diff",
                "derivatives_diff_dot_prev_step",
                "p",
                "p_times_prev_step",
                "identity",
            ),
            (arg2!("identity", A)
                - arg1!("p_times_prev_step", A).mul_out(arg1!("derivatives_diff", A)))
            .mat_mul(arg2!("prev_inv_snd_derivatives", A))
            .mat_mul(
                arg2!("identity", A)
                    - (arg1!("derivatives_diff", A) * arg!("p", A)).mul_out(arg1!("prev_step", A)),
            ) + arg1!("p_times_prev_step", A).mul_out(arg1!("prev_step", A)),
        ),
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use optimal_compute_core::{named_args, val, val1, val2, Run};

    use crate::{descend, step_direction::steepest_descent, StepSize};

    use super::*;

    #[test]
    fn bfgs_should_equal_steepest_descent_if_second_derivatives_are_identity() {
        let iterations = 10;
        let step_size = StepSize::new(0.05).unwrap();
        let initial_point = vec![10.0, 10.0];
        // At time of writing,
        // `assert_ulps_eq` only supports slices,
        // not `Vec`.
        // See <https://github.com/brendanzab/approx/pull/69>.
        assert_ulps_eq!(
            steepest(iterations, step_size, flat_d, initial_point.clone())
                .iter()
                .map(|xs| xs.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
            bfgs(iterations, step_size, flat_d, initial_point)
                .iter()
                .map(|xs| xs.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn steepest<FD>(
        iterations: usize,
        step_size: StepSize<f64>,
        obj_func_d: FD,
        initial_point: Vec<f64>,
    ) -> Vec<Vec<f64>>
    where
        FD: Fn(&[f64]) -> Vec<f64>,
    {
        let mut points = Vec::new();

        let mut point = initial_point;
        for _ in 0..iterations {
            point = descend(
                val!(step_size),
                steepest_descent(val1!(obj_func_d(&point))),
                val1!(point),
            )
            .run(named_args![]);

            points.push(point.clone());
        }

        points
    }

    fn bfgs<FD>(
        iterations: usize,
        step_size: StepSize<f64>,
        obj_func_d: FD,
        initial_point: Vec<f64>,
    ) -> Vec<Vec<f64>>
    where
        FD: Fn(&[f64]) -> Vec<f64>,
    {
        let mut points = Vec::new();

        let mut point = initial_point;

        let (mut prev_approx_inv_snd_derivatives, mut prev_derivatives, mut prev_step) = {
            let derivatives = obj_func_d(&point);
            let approx_inv_snd_derivatives =
                initial_approx_inv_snd_derivatives_identity(val!(point.len())).run(named_args![]);
            let step = (val!(step_size)
                * bfgs_direction(
                    val2!(approx_inv_snd_derivatives.clone()),
                    val1!(derivatives.iter().cloned()),
                ))
            .run(named_args![]);
            point = (val1!(point) + val1!(step.iter().cloned())).run(named_args![]);

            points.push(point.clone());

            (approx_inv_snd_derivatives, derivatives, step)
        };
        for _ in 1..iterations {
            let derivatives = obj_func_d(&point);
            let approx_inv_snd_derivatives = approx_inv_snd_derivatives(
                val2!(prev_approx_inv_snd_derivatives),
                val1!(prev_derivatives),
                val1!(prev_step),
                val1!(derivatives.iter().cloned()),
            )
            .run(named_args![]);
            let step = (val!(step_size)
                * bfgs_direction(
                    val2!(approx_inv_snd_derivatives.clone()),
                    val1!(derivatives.iter().cloned()),
                ))
            .run(named_args![]);
            point = (val1!(point) + val1!(step.iter().cloned())).run(named_args![]);

            points.push(point.clone());

            (prev_approx_inv_snd_derivatives, prev_derivatives, prev_step) =
                (approx_inv_snd_derivatives, derivatives, step);
        }

        points
    }

    fn flat_d(point: &[f64]) -> Vec<f64> {
        point.to_vec()
    }
}
