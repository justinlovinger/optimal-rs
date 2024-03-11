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
//! use ndarray::{Array1, Array2, ArrayView1};
//!
//! use optimal_linesearch::{
//!     step_direction::bfgs::{
//!         approx_inv_snd_derivatives, bfgs_direction, initial_approx_inv_snd_derivatives_gamma,
//!         initial_approx_inv_snd_derivatives_identity,
//!     },
//!     StepSize,
//! };
//!
//! enum BfgsIteration<A> {
//!     First,
//!     Second {
//!         prev_derivatives: Array1<A>,
//!         prev_step: Array1<A>,
//!     },
//!     Other {
//!         prev_derivatives: Array1<A>,
//!         prev_approx_inv_snd_derivatives: Array2<A>,
//!         prev_step: Array1<A>,
//!     },
//! }
//!
//! fn main() {
//!     let step_size = StepSize::new(0.5).unwrap();
//!
//!     let mut point = Array1::from_vec(vec![10.0, 10.0]);
//!     let mut bfgs_state = BfgsIteration::First;
//!     for _ in 0..10 {
//!         let derivatives = obj_func_d(point.view());
//!         bfgs_state = match bfgs_state {
//!             BfgsIteration::First => {
//!                 let approx_inv_snd_derivatives =
//!                     initial_approx_inv_snd_derivatives_identity(point.len());
//!                 let step = step_size.into_inner()
//!                     * bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
//!                 point = point + step.view();
//!                 BfgsIteration::Second {
//!                     prev_derivatives: derivatives,
//!                     prev_step: step,
//!                 }
//!             }
//!             BfgsIteration::Second {
//!                 prev_derivatives,
//!                 prev_step,
//!             } => {
//!                 let approx_inv_snd_derivatives = initial_approx_inv_snd_derivatives_gamma(
//!                     prev_derivatives,
//!                     prev_step,
//!                     derivatives.view(),
//!                 );
//!                 let step = step_size.into_inner()
//!                     * bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
//!                 point = point + step.view();
//!                 BfgsIteration::Other {
//!                     prev_derivatives: derivatives,
//!                     prev_approx_inv_snd_derivatives: approx_inv_snd_derivatives,
//!                     prev_step: step,
//!                 }
//!             }
//!             BfgsIteration::Other {
//!                 prev_derivatives,
//!                 prev_approx_inv_snd_derivatives,
//!                 prev_step,
//!             } => {
//!                 let approx_inv_snd_derivatives = approx_inv_snd_derivatives(
//!                     prev_approx_inv_snd_derivatives,
//!                     prev_step,
//!                     prev_derivatives,
//!                     derivatives.view(),
//!                 );
//!                 let step = step_size.into_inner()
//!                     * bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
//!                 point = point + step.view();
//!                 BfgsIteration::Other {
//!                     prev_derivatives: derivatives,
//!                     prev_approx_inv_snd_derivatives: approx_inv_snd_derivatives,
//!                     prev_step: step,
//!                 }
//!             }
//!         };
//!     }
//!     println!("{:?}", point);
//! }
//!
//! fn obj_func_d(point: ArrayView1<f64>) -> Array1<f64> {
//!     point.iter().copied().map(|x| 2.0 * x).collect()
//! }
//! ```

use std::ops::{Mul, Neg, Sub};

use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Ix1, Ix2, LinalgScalar, RawData,
    ScalarOperand,
};
use num_traits::{One, Zero};

/// Return a placeholder for approximate inverse second-derivatives,
/// useful for starting BFGS.
pub fn initial_approx_inv_snd_derivatives_identity<A>(len: usize) -> Array2<A>
where
    A: Clone + Zero + One,
{
    Array2::from_diag_elem(len, A::one())
}

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
pub fn initial_approx_inv_snd_derivatives_gamma<A>(
    prev_derivatives: Array1<A>,
    prev_step: Array1<A>,
    derivatives: ArrayView1<A>,
) -> Array2<A>
where
    A: LinalgScalar,
{
    let derivatives_diff = derivatives.sub(prev_derivatives);
    let derivatives_diff_dotted = derivatives_diff.dot(&derivatives_diff);
    // Default to identity if we cannot get gamma.
    let elem = if derivatives_diff_dotted.is_zero() {
        A::one()
    } else {
        prev_step.dot(&derivatives_diff) / derivatives_diff_dotted
    };
    Array2::from_diag_elem(prev_step.len(), elem)
}

/// Return a direction of descent informed by approximate second-derivative information.
pub fn bfgs_direction<A>(
    approx_inv_snd_derivatives: ArrayView2<A>,
    derivatives: ArrayView1<A>,
) -> Array1<A>
where
    A: Neg<Output = A> + LinalgScalar,
{
    approx_inv_snd_derivatives
        .dot(&derivatives)
        .mapv_into(|x| -x)
}

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
pub fn approx_inv_snd_derivatives<A>(
    prev_inv_snd_derivatives: Array2<A>,
    prev_derivatives: Array1<A>,
    prev_step: Array1<A>,
    derivatives: ArrayView1<A>,
) -> Array2<A>
where
    A: ScalarOperand + LinalgScalar,
{
    let derivatives_diff = derivatives.sub(prev_derivatives);
    let derivatives_diff_dot_prev_step = derivatives_diff.dot(&prev_step);
    // Default to reusing the previous approximate inverse-second-derivatives
    // if we cannot apply an update.
    if derivatives_diff_dot_prev_step.is_zero() {
        prev_inv_snd_derivatives
    } else {
        let p = A::one() / derivatives_diff_dot_prev_step;
        let p_times_prev_step = prev_step.view().mul(p);
        let identity = Array2::from_diag_elem(prev_step.len(), A::one());
        (identity.view().sub(
            p_times_prev_step
                .view()
                .into_column_vector()
                .dot(&derivatives_diff.view().into_row_vector()),
        ))
        .dot(&prev_inv_snd_derivatives)
        .dot(
            &(identity
                - (derivatives_diff * p)
                    .into_column_vector()
                    .dot(&prev_step.view().into_row_vector())),
        ) + p_times_prev_step
            .into_column_vector()
            .dot(&prev_step.into_row_vector())
    }
}

trait IntoRowVector<S>
where
    S: RawData,
{
    fn into_row_vector(self) -> ArrayBase<S, Ix2>;
}

impl<S> IntoRowVector<S> for ArrayBase<S, Ix1>
where
    S: RawData,
{
    fn into_row_vector(self) -> ArrayBase<S, Ix2> {
        let len = self.len();
        self.into_shape((1, len)).unwrap()
    }
}

trait IntoColumnVector<S>
where
    S: RawData,
{
    fn into_column_vector(self) -> ArrayBase<S, Ix2>;
}

impl<S> IntoColumnVector<S> for ArrayBase<S, Ix1>
where
    S: RawData,
{
    fn into_column_vector(self) -> ArrayBase<S, Ix2> {
        let len = self.len();
        self.into_shape((len, 1)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{descend, step_direction::steepest_descent, StepSize};

    use super::*;

    #[test]
    fn bfgs_should_equal_steepest_descent_if_second_derivatives_are_identity() {
        let iterations = 10;
        let step_size = StepSize::new(0.05).unwrap();
        let initial_point = Array1::from_elem(2, 10.0);
        assert_eq!(
            steepest(iterations, step_size, flat_d, initial_point.clone()),
            bfgs(iterations, step_size, flat_d, initial_point)
        )
    }

    fn steepest<FD>(
        iterations: usize,
        step_size: StepSize<f64>,
        obj_func_d: FD,
        initial_point: Array1<f64>,
    ) -> Vec<Array1<f64>>
    where
        FD: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        let mut points = Vec::new();

        let mut point = initial_point;
        for _ in 0..iterations {
            point = descend(
                step_size,
                steepest_descent(obj_func_d(point.view()).to_vec()),
                point,
            )
            .collect();

            points.push(point.clone());
        }

        points
    }

    fn bfgs<FD>(
        iterations: usize,
        step_size: StepSize<f64>,
        obj_func_d: FD,
        initial_point: Array1<f64>,
    ) -> Vec<Array1<f64>>
    where
        FD: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        let mut points = Vec::new();

        let mut point = initial_point;
        let mut bfgs_state = BfgsIteration::First;
        for _ in 0..iterations {
            let derivatives = obj_func_d(point.view());
            bfgs_state = match bfgs_state {
                BfgsIteration::First => {
                    let approx_inv_snd_derivatives =
                        initial_approx_inv_snd_derivatives_identity(point.len());
                    let step = step_size.into_inner()
                        * bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
                    point = point + step.view();
                    BfgsIteration::Second {
                        prev_derivatives: derivatives,
                        prev_step: step,
                    }
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
                    let step = step_size.into_inner()
                        * bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
                    point = point + step.view();
                    BfgsIteration::Other {
                        prev_derivatives: derivatives,
                        prev_approx_inv_snd_derivatives: approx_inv_snd_derivatives,
                        prev_step: step,
                    }
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
                    let step = step_size.into_inner()
                        * bfgs_direction(approx_inv_snd_derivatives.view(), derivatives.view());
                    point = point + step.view();
                    BfgsIteration::Other {
                        prev_derivatives: derivatives,
                        prev_approx_inv_snd_derivatives: approx_inv_snd_derivatives,
                        prev_step: step,
                    }
                }
            };

            points.push(point.clone());
        }

        points
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

    fn flat_d(point: ArrayView1<f64>) -> Array1<f64> {
        point.iter().copied().collect()
    }
}
