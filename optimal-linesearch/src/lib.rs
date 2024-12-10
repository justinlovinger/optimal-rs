#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Line-search optimizers.
//!
//! Fixed step-size optimization can also be performed
//! using this package:
//!
//! ```
//! use computation_types::*;
//! use optimal_linesearch::{descend, step_direction::steepest_descent, StepSize};
//!
//! println!(
//!     "{:?}",
//!     val!(0)
//!         .zip(val1!(vec![10.0, 10.0]))
//!         .loop_while(
//!             ("i", "point"),
//!             (arg!("i", usize) + val!(1)).zip(descend(
//!                 val!(StepSize::new(0.5).unwrap()),
//!                 steepest_descent(val!(2.0) * arg1!("point", f64)),
//!                 arg1!("point", f64),
//!             )),
//!             arg!("i", usize).lt(val!(10)),
//!         )
//!         .snd()
//!         .run()
//! )
//! ```
//!
//! See [`backtracking_line_search`] for more sophisticated and effective optimizers.

pub mod backtracking_line_search;
pub mod initial_step_size;
pub mod step_direction;

use core::ops;

use computation_types::{
    cmp::{Lt, Max},
    math::{Abs, Add, Mul},
    peano, val, Computation, Val0,
};
use num_traits::{AsPrimitive, One, Signed};

pub use self::types::*;

/// See [`descend`].
pub type Descend<S, D, P> = Add<P, Mul<S, D>>;

/// Descend in step-direction
/// by moving `point` `step_size` length in `direction`.
#[allow(clippy::type_complexity)]
pub fn descend<S, D, P, SA, DPDim>(step_size: S, direction: D, point: P) -> Descend<S, D, P>
where
    S: Computation<Dim = peano::Zero, Item = StepSize<SA>>,
    D: Computation<Dim = peano::Suc<DPDim>>,
    P: Computation<Dim = peano::Suc<DPDim>>,
    StepSize<SA>: ops::Mul<D::Item>,
    P::Item: ops::Add<<Mul<S, D> as Computation>::Item>,
{
    point.add(step_size.mul(direction))
}

/// See [`is_near_minima`].
pub type IsNearMinima<V, D> = Lt<
    InfiniteNorm<D>,
    Mul<Val0<<V as Computation>::Item>, Add<Val0<<V as Computation>::Item>, Abs<V>>>,
>;

/// Return whether a point is sufficiently close to a minima.
///
/// Also known as the derivative-norm stopping-criteria.
/// This is mathematically defined as,
/// `||\vec{dx}||_inf < 10^-5 (1 + |fx|)`,
/// where `fx` is the value of a point
/// and `\vec{dx}` is the derivative of the same point.
///
/// Returns true for empty derivatives.
pub fn is_near_minima<V, D>(value: V, derivatives: D) -> IsNearMinima<V, D>
where
    V: Computation<Dim = peano::Zero>,
    V::Item: Copy + PartialOrd + Signed + ops::Mul<Output = V::Item> + One + 'static,
    f64: AsPrimitive<V::Item>,
    D: Computation<Dim = peano::One>,
    D::Item: PartialOrd + PartialOrd<V::Item> + Signed,
{
    const COEFF: f64 = 0.00001; // 10^-5
    infinite_norm(derivatives).lt(val!(COEFF.as_()) * (val!(<V::Item as One>::one()) + value.abs()))
}

/// See [`infinite_norm`].
pub type InfiniteNorm<A> = Max<Abs<A>>;

fn infinite_norm<A>(xs: A) -> InfiniteNorm<A>
where
    A: Computation<Dim = peano::One>,
    A::Item: PartialOrd + Signed,
{
    xs.abs().max()
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

    use computation_types::{
        arg1,
        peano::{One, Zero},
        Computation,
    };
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::backtracking_line_search::{
        BacktrackingLineSearchBuilder, BacktrackingLineSearchStoppingCriteria, BfgsInitializer,
        StepDirection,
    };

    // Theoretically,
    // these optimizers should always solve any convex problem.
    // In practice,
    // with the numerical-stability issues of floating-point values,
    // an optimizer may get stuck approaching an optimal value.
    // We use static seeds to avoid getting a random point
    // that results in such an issue.

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_steepest_descent() {
        test_can_solve_with(
            StepDirection::Steepest,
            BacktrackingLineSearchStoppingCriteria::NearMinima,
        );
    }

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_bfgs_id() {
        test_can_solve_with(
            StepDirection::Bfgs {
                initializer: BfgsInitializer::Identity,
            },
            BacktrackingLineSearchStoppingCriteria::NearMinima,
        );
    }

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_bfgs_gamma() {
        test_can_solve_with(
            StepDirection::Bfgs {
                initializer: BfgsInitializer::Gamma,
            },
            BacktrackingLineSearchStoppingCriteria::NearMinima,
        );
    }

    // Technically,
    // backtracking line-search is not guarenteed to converge
    // in any particular number of iterations.
    // However,
    // if we set the number high enough,
    // it should in practice.

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_steepest_descent_and_iterations()
    {
        test_can_solve_with(
            StepDirection::Steepest,
            BacktrackingLineSearchStoppingCriteria::Iteration(100),
        );
    }

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_bfgs_id_and_iterations() {
        test_can_solve_with(
            StepDirection::Bfgs {
                initializer: BfgsInitializer::Identity,
            },
            BacktrackingLineSearchStoppingCriteria::Iteration(100),
        );
    }

    #[test]
    fn backtracking_line_search_should_solve_convex_problems_with_bfgs_gamma_and_iterations() {
        test_can_solve_with(
            StepDirection::Bfgs {
                initializer: BfgsInitializer::Gamma,
            },
            BacktrackingLineSearchStoppingCriteria::Iteration(100),
        );
    }

    fn test_can_solve_with(
        direction: StepDirection,
        stopping_criteria: BacktrackingLineSearchStoppingCriteria,
    ) {
        for seed in 0..10 {
            assert!(
                run(
                    seed,
                    100,
                    direction.clone(),
                    stopping_criteria.clone(),
                    sphere,
                    sphere_d
                ) <= 0.00001
            );
            assert!(
                run(
                    seed,
                    10,
                    direction.clone(),
                    stopping_criteria.clone(),
                    skewed_sphere,
                    skewed_sphere_d
                ) <= 0.00001
            );
        }
    }

    fn run<F, FD>(
        seed: u64,
        len: usize,
        direction: StepDirection,
        stopping_criteria: BacktrackingLineSearchStoppingCriteria,
        obj_func: F,
        obj_func_d: FD,
    ) -> f64
    where
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        let point = BacktrackingLineSearchBuilder::default()
            .direction(direction)
            .stopping_criteria(stopping_criteria)
            .for_(
                len,
                arg1!("point").black_box::<_, Zero, f64>(|point: Vec<f64>| obj_func(&point)),
                arg1!("point").black_box::<_, One, f64>(|point: Vec<f64>| obj_func_d(&point)),
            )
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
