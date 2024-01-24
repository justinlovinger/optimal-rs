//! Methods to get direction for line-search.

use std::ops::Neg;

/// Return the direction of steepest descent.
pub fn steepest_descent<A>(derivatives: &[A]) -> Vec<A>
where
    A: Clone + Neg<Output = A>,
{
    derivatives.iter().cloned().map(|x| x.neg()).collect()
}
