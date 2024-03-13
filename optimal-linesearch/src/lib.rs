#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Line-search optimizers.
//!
//! Fixed step-size optimization can also be performed
//! using this package:
//!
//! ```
//! use optimal_linesearch::{descend, step_direction::steepest_descent, StepSize};
//!
//! fn main() {
//!     let step_size = StepSize::new(0.5).unwrap();
//!     let mut point = vec![10.0, 10.0];
//!     for _ in 0..10 {
//!         point = descend(step_size, steepest_descent(obj_func_d(&point)), point).collect();
//!     }
//!     println!("{:?}", point);
//! }
//!
//! fn obj_func_d(point: &[f64]) -> Vec<f64> {
//!     point.iter().copied().map(|x| 2.0 * x).collect()
//! }
//! ```
//!
//! See [`backtracking_line_search`] for more sophisticated and effective optimizers.

pub mod backtracking_line_search;
pub mod initial_step_size;
pub mod step_direction;

use std::ops::{Add, Mul};

use num_traits::{AsPrimitive, One, Signed};

pub use self::types::*;

/// Descend in step-direction
/// by moving `point` `step_size` length in `direction`.
pub fn descend<A>(
    step_size: StepSize<A>,
    direction: impl IntoIterator<Item = A>,
    point: impl IntoIterator<Item = A>,
) -> impl Iterator<Item = A>
where
    A: Clone + Add<Output = A> + Mul<Output = A>,
{
    point
        .into_iter()
        .zip(direction)
        .map(move |(x, d)| x + step_size.clone() * d)
}

/// Return whether a point is sufficiently close to a minima.
///
/// Also known as the derivative-norm stopping-criteria.
/// This is mathematically defined as,
/// `||\vec{dx}||_inf < 10^-5 (1 + |fx|)`,
/// where `fx` is the value of a point
/// and `\vec{dx}` is the derivative of the same point.
///
/// Returns true for empty derivatives.
pub fn is_near_minima<A>(value: A, derivatives: impl IntoIterator<Item = A>) -> bool
where
    A: Copy + PartialOrd + Signed + Mul<Output = A> + One + 'static,
    f64: AsPrimitive<A>,
{
    const COEFF: f64 = 0.00001; // 10^-5
    infinite_norm(derivatives)
        .map(|inf_norm_ds| inf_norm_ds < COEFF.as_() * (A::one() + value.abs()))
        .unwrap_or(true)
}

fn infinite_norm<A>(xs: impl IntoIterator<Item = A>) -> Option<A>
where
    A: PartialOrd + Signed,
{
    xs.into_iter()
        .map(|x| x.abs())
        .reduce(|acc, x| if x > acc { x } else { acc })
}

mod types {
    use std::ops::Mul;

    use derive_more::Display;
    use derive_num_bounded::{derive_into_inner, derive_new_from_lower_bounded_partial_ord};
    use num_traits::{bounds::LowerBounded, real::Real};

    /// Multiplier for each component of a step-direction
    /// in derivative optimization.
    #[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct StepSize<A>(pub(crate) A);

    derive_new_from_lower_bounded_partial_ord!(StepSize<A: Real>);
    derive_into_inner!(StepSize<A>);

    impl<A> LowerBounded for StepSize<A>
    where
        A: Real,
    {
        fn min_value() -> Self {
            Self(A::zero() + A::epsilon())
        }
    }

    impl<A> Mul<A> for StepSize<A>
    where
        A: Mul<Output = A>,
    {
        type Output = A;

        fn mul(self, rhs: A) -> Self::Output {
            self.0 * rhs
        }
    }
}
