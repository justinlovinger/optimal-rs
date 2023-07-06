use blanket::blanket;
use ndarray::prelude::*;

/// An optimization problem,
/// as defined by having an objective function.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Problem {
    /// A point in this problem space.
    type Point<'a>
    where
        Self: 'a;

    /// Value of a point in this problem space.
    type Value;

    /// Return the objective value of a point in this problem space.
    fn evaluate<'a>(&'a self, point: Self::Point<'a>) -> Self::Value;
}

/// An extension to `Problem` for populations of points,
/// automatically implemented for `CowArray` points.
///
/// Override for objective functions
/// capable of more efficiently evaluating multiple points
/// simultaneously.
pub trait ProblemPop<Elem> {
    /// A population of points in this problem space.
    type Population<'a>
    where
        Self: 'a,
        Elem: 'a;

    /// A collection of values of points in this problem space.
    type Values;

    /// Return objective values of points.
    fn evaluate_population(&self, points: Self::Population<'_>) -> Self::Values;
}

impl<Elem, T> ProblemPop<Elem> for T
where
    for<'a> T: Problem<Point<'a> = CowArray<'a, Elem, Ix1>> + 'a,
{
    type Population<'a> = CowArray<'a, Elem, Ix2>
    where
        Elem: 'a;

    type Values = Array1<<Self as Problem>::Value>;

    fn evaluate_population(&self, points: Self::Population<'_>) -> Self::Values {
        points.map_axis(Axis(points.ndim() - 1), |point| self.evaluate(point.into()))
    }
}

/// An optimization problem
/// with a differentiable objective function.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Differentiable: Problem {
    /// Derivative,
    /// or partial derivatives,
    /// of a point in this problem space.
    type Derivative;

    /// Return partial derivatives of a point.
    fn differentiate<'a>(&'a self, point: Self::Point<'a>) -> Self::Derivative;
}

/// An extension to `Differentiable` for simultaneously evaluating and differentiating points,
/// automatically implemented for `CowArray` points.
///
/// Override for objective functions
/// capable of more efficiently calculating both
/// simultaneously.
pub trait EvaluateDifferentiate<Elem>: Differentiable + Problem {
    /// Return objective value and partial derivatives of a point.
    fn evaluate_differentiate<'a>(
        &'a self,
        point: Self::Point<'a>,
    ) -> (Self::Value, Self::Derivative);
}

impl<T, Elem> EvaluateDifferentiate<Elem> for T
where
    for<'a> T: Differentiable<Point<'a> = CowArray<'a, Elem, Ix1>> + 'a,
{
    fn evaluate_differentiate<'a>(
        &'a self,
        point: Self::Point<'a>,
    ) -> (Self::Value, Self::Derivative) {
        (
            self.evaluate(point.view().into()),
            self.differentiate(point),
        )
    }
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
    /// Bounds for points in this problem space.
    type Bounds;

    /// Return whether this problem contains the given point.
    fn contains<A, I>(&self, point: I) -> bool
    where
        Self::Bounds: Iterator<Item = std::ops::RangeInclusive<A>>,
        I: IntoIterator<Item = A>,
        A: PartialOrd,
    {
        point
            .into_iter()
            .zip(self.bounds())
            .all(|(x, range)| range.contains(&x))
    }

    /// Return bounds for this problem.
    fn bounds(&self) -> Self::Bounds;
}

// #[cfg(test)]
// mod tests {
//     use static_assertions::assert_obj_safe;

//     use super::*;

//     assert_obj_safe!(Problem<Point = f64, Value = f64>);
//     assert_obj_safe!(Differentiable<Point = f64, Value = f64, Derivative = f64>);
//     assert_obj_safe!(FixedLength<Point = f64, Value = f64>);
//     assert_obj_safe!(Bounded<Point = f64, Value = f64, Bounds = Vec<f64>>);
// }
