//! Methods to get direction for line-search.

use std::ops::Neg;

/// Return the direction of steepest descent.
pub fn steepest_descent<A>(derivatives: impl IntoIterator<Item = A>) -> impl Iterator<Item = A>
where
    A: Clone + Neg<Output = A>,
{
    derivatives.into_iter().map(|x| x.neg())
}
