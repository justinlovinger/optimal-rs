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

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use rand::{rngs::SmallRng, SeedableRng};

    use crate::backtracking_line_search::{
        BacktrackingLineSearchBuilder, BfgsInitializer, StepDirection,
    };

    // Theoretically,
    // these optimizers should always solve any convex problem.
    // In practice,
    // with the numerical-stability issues of floating-point values,
    // an optimizer may either get stuck approaching an optimal value.
    // We use static seeds to avoid getting a random point
    // that results in such an issue.

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_steepest_descent() {
        test_can_solve_with(StepDirection::Steepest);
    }

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_bfgs_id() {
        test_can_solve_with(StepDirection::Bfgs {
            initializer: BfgsInitializer::Identity,
        });
    }

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_bfgs_gamma() {
        test_can_solve_with(StepDirection::Bfgs {
            initializer: BfgsInitializer::Gamma,
        });
    }

    fn test_can_solve_with(direction: StepDirection) {
        for seed in 0..10 {
            assert!(run(seed, 100, direction.clone(), sphere, sphere_d) <= 0.00001);
            assert!(run(seed, 10, direction.clone(), skewed_sphere, skewed_sphere_d) <= 0.00001);
        }
    }

    fn run<F, FD>(
        seed: u64,
        len: usize,
        direction: StepDirection,
        obj_func: F,
        obj_func_d: FD,
    ) -> f64
    where
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        let point = BacktrackingLineSearchBuilder::default()
            .direction(direction)
            .for_(len, obj_func.clone(), obj_func_d)
            .with_random_point_using(initial_bounds(len), SmallRng::seed_from_u64(seed))
            .argmin();
        (obj_func)(&point)
    }

    fn initial_bounds(len: usize) -> impl Iterator<Item = RangeInclusive<f64>> {
        std::iter::repeat(0.0..=1.0).take(len)
    }

    fn sphere(point: &[f64]) -> f64 {
        point.iter().map(|x| x.powi(2)).sum()
    }
    fn sphere_d(point: &[f64]) -> Vec<f64> {
        point.iter().map(|x| 2.0 * x).collect()
    }

    // Note,
    // this is only defined for non-negative numbers.
    fn skewed_sphere(point: &[f64]) -> f64 {
        let len = point.len() as f64;
        point
            .iter()
            .enumerate()
            .map(|(i, x)| x.powf(1.0 + ((i + 1) as f64) / len))
            .sum()
    }
    fn skewed_sphere_d(point: &[f64]) -> Vec<f64> {
        let len = point.len() as f64;
        point
            .iter()
            .enumerate()
            .map(|(i, x)| (1.0 + ((i + 1) as f64) / len) * x)
            .collect()
    }
}
