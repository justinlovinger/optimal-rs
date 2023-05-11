use std::ops::RangeInclusive;

use blanket::blanket;
use ndarray::prelude::*;

// When stable,
// add default implementations for both methods
// and use `rustc_must_implement_one_of`,
// <https://github.com/rust-lang/rust/pull/92164>.
/// An optimization problem,
/// as defined by having an objective function.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Problem {
    /// Elements in points.
    type PointElem;

    /// Value returned by problem
    /// when point is evaluated.
    type PointValue;

    /// Return objective values of points.
    fn evaluate_population(
        &self,
        points: CowArray<Self::PointElem, Ix2>,
    ) -> Array1<Self::PointValue> {
        points.map_axis(Axis(points.ndim() - 1), |point| self.evaluate(point.into()))
    }

    /// Return the objective value of a point.
    ///
    /// This can be implemented as
    /// `self.evaluate_all(point).into_scalar()`
    /// if `evaluate_all` is implemented.
    fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue;
}

/// An optimization problem
/// with a differentiable objective function.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Differentiable: Problem {
    /// Return objective value and partial derivatives of a point.
    ///
    /// Override for objective functions
    /// capable of more efficiently calculating both
    /// simultaneously.
    fn evaluate_differentiate(
        &self,
        point: CowArray<Self::PointElem, Ix1>,
    ) -> (Self::PointValue, Array1<Self::PointElem>) {
        (
            self.evaluate(point.view().into()),
            self.differentiate(point),
        )
    }

    // As of 2023-03-13,
    // `ndarray` lacks a method to map an axis to an axis,
    // making implementation of a row-polymorphic method difficult.
    /// Return partial derivatives of a point.
    fn differentiate(&self, point: CowArray<Self::PointElem, Ix1>) -> Array1<Self::PointElem>;
}

/// An optimization problem with a fixed length
/// for every point
/// in its problem space.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
#[allow(clippy::len_without_is_empty)] // Problems should never have no length.
pub trait FixedLength: Problem {
    /// Return length of points for this problem.
    fn len(&self) -> usize;
}

/// An optimization problem with fixed bounds
/// for every element of every point
/// in its problem space.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Bounded: Problem {
    /// Return whether this problem contains the given point.
    fn contains(&self, point: ArrayView1<Self::PointElem>) -> bool
    where
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

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use super::*;

    assert_obj_safe!(Problem<PointElem = f64, PointValue = f64>);
    assert_obj_safe!(Differentiable<PointElem = f64, PointValue = f64>);
    assert_obj_safe!(FixedLength<PointElem = f64, PointValue = f64>);
    assert_obj_safe!(Bounded<PointElem = f64, PointValue = f64>);
}
