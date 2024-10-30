//! Methods to get direction for line-search.

pub mod bfgs;

use core::ops;

use computation_types::{math::Neg, peano, Computation};

/// Return the direction of steepest descent.
pub fn steepest_descent<D, Dim>(derivatives: D) -> Neg<D>
where
    D: Computation<Dim = peano::Suc<Dim>>,
    D::Item: ops::Neg,
{
    derivatives.neg()
}
