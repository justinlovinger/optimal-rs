use std::ops::RangeInclusive;

use ndarray::{prelude::*, Data, RawData, RemoveAxis};

// When stable,
// add default implementations for both methods
// and use `rustc_must_implement_one_of`,
// <https://github.com/rust-lang/rust/pull/92164>.
/// An optimization problem,
/// as defined by having an objective function.
pub trait Problem {
    /// Elements in points.
    type PointElem;

    /// Value returned by problem
    /// when point is evaluated.
    type PointValue;

    /// Return objective values of points.
    fn evaluate_all<S, D>(&self, points: ArrayBase<S, D>) -> Array<Self::PointValue, D::Smaller>
    where
        S: RawData<Elem = Self::PointElem> + Data,
        D: Dimension + RemoveAxis,
    {
        points.map_axis(Axis(points.ndim() - 1), |point| self.evaluate(point))
    }

    /// Return the objective value of a point.
    ///
    /// This can be implemented as
    /// `self.evaluate_all(point).into_scalar()`
    /// if `evaluate_all` is implemented.
    fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> Self::PointValue
    where
        S: RawData<Elem = Self::PointElem> + Data;
}

/// An optimization problem
/// with a differentiable objective function.
pub trait Differentiable: Problem {
    /// Return objective value and partial derivatives of a point.
    ///
    /// Override for objective functions
    /// capable of more efficiently calculating both
    /// simultaneously.
    fn evaluate_differentiate<S>(
        &self,
        point: ArrayBase<S, Ix1>,
    ) -> (Self::PointValue, Array1<Self::PointElem>)
    where
        S: RawData<Elem = Self::PointElem> + Data,
    {
        (self.evaluate(point.view()), self.differentiate(point))
    }

    // As of 2023-03-13,
    // `ndarray` lacks a method to map an axis to an axis,
    // making implementation of a row-polymorphic method difficult.
    /// Return partial derivatives of a point.
    fn differentiate<S>(&self, point: ArrayBase<S, Ix1>) -> Array1<Self::PointElem>
    where
        S: RawData<Elem = Self::PointElem> + Data;
}

/// An optimization problem with a fixed length
/// for every point
/// in its problem space.
#[allow(clippy::len_without_is_empty)] // Problems should never have no length.
pub trait FixedLength: Problem {
    /// Return length of points for this problem.
    fn len(&self) -> usize;
}

/// An optimization problem with fixed bounds
/// for every element of every point
/// in its problem space.
pub trait Bounded: Problem {
    /// Return whether this problem contains the given point.
    fn contains<S>(&self, point: ArrayBase<S, Ix1>) -> bool
    where
        S: RawData<Elem = Self::PointElem> + Data,
        Self::PointElem: PartialOrd,
    {
        point
            .iter()
            .zip(self.bounds())
            .all(|(x, range)| range.contains(x))
    }

    /// Return bounds for this problem.
    fn bounds(&self) -> Box<dyn Iterator<Item = RangeInclusive<Self::PointElem>>>;
}
