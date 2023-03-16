use std::ops::RangeInclusive;

use ndarray::{prelude::*, Data, RawData, RemoveAxis};

// When stable,
// add default implementations for both methods
// and use `rustc_must_implement_one_of`,
// <https://github.com/rust-lang/rust/pull/92164>.
/// An optimization problem,
/// as defined by having an objective function.
pub trait Problem<A, B> {
    /// Return objective values of points.
    fn evaluate_all<S, D>(&self, points: ArrayBase<S, D>) -> Array<B, D::Smaller>
    where
        S: RawData<Elem = A> + Data,
        D: Dimension + RemoveAxis,
    {
        points.map_axis(Axis(points.ndim() - 1), |point| self.evaluate(point))
    }

    /// Return the objective value of a point.
    ///
    /// This can be implemented as
    /// `self.evaluate_all(point).into_scalar()`
    /// if `evaluate_all` is implemented.
    fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> B
    where
        S: RawData<Elem = A> + Data;
}

impl<A, B, F> Problem<A, B> for F
where
    F: Fn(ArrayView1<A>) -> B,
{
    fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> B
    where
        S: RawData<Elem = A> + Data,
    {
        (self)(point.view())
    }
}

/// An optimization problem
/// with a differentiable objective function.
pub trait Differentiable<A, B>: Problem<A, B> {
    /// Return objective value and partial derivatives of a point.
    ///
    /// Override for objective functions
    /// capable of more efficiently calculating both
    /// simultaneously.
    fn evaluate_differentiate<S>(&self, point: ArrayBase<S, Ix1>) -> (B, Array1<B>)
    where
        S: RawData<Elem = A> + Data,
    {
        (self.evaluate(point.view()), self.differentiate(point))
    }

    // As of 2023-03-13,
    // `ndarray` lacks a method to map an axis to an axis,
    // making implementation of a row-polymorphic method difficult.
    /// Return partial derivatives of a point.
    fn differentiate<S>(&self, point: ArrayBase<S, Ix1>) -> Array1<B>
    where
        S: RawData<Elem = A> + Data;
}

/// An optimization problem with a fixed length
/// for every point
/// in its problem space.
#[allow(clippy::len_without_is_empty)] // Problems should never have no length.
pub trait FixedLength {
    /// Return length of points for this problem.
    fn len(&self) -> usize;
}

/// An optimization problem with fixed bounds
/// for every element of every point
/// in its problem space.
pub trait Bounded<A>
where
    A: PartialOrd,
{
    /// Return whether this problem contains the given point.
    fn contains<S>(&self, point: ArrayBase<S, Ix1>) -> bool
    where
        S: RawData<Elem = A> + Data,
    {
        point
            .iter()
            .zip(self.bounds())
            .all(|(x, range)| range.contains(x))
    }

    /// Return bounds for this problem.
    fn bounds(&self) -> Box<dyn Iterator<Item = RangeInclusive<A>>>;
}
