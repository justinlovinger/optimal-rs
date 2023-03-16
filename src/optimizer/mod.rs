//! Mathematical optimization framework.
//!
//! An optimizer is split between a static configuration
//! and a dynamic state.
//! Traits are defined on the configuration
//! and may take a state.

pub mod derivative;
pub mod derivative_free;
mod iterator;

use ndarray::prelude::*;
use rand::Rng;

pub use self::iterator::*;

/// An optimizer configuration
/// qualified to initialize an optimizer
/// using a 'Rng'.
pub trait InitializeUsing<'a, F, O> {
    /// Return an optimizer
    /// initialized using `rng`.
    fn initialize_using<R>(&'a self, objective: &'a F, rng: &mut R) -> O
    where
        R: Rng;
}

/// An optimizer configuration
/// qualified to initialize an optimizer.
pub trait Initialize<'a, F, O> {
    /// Return an initialized optimizer.
    fn initialize(&'a self, objective: &'a F) -> O;
}

/// The core of an optimizer,
/// step from one state to another,
/// improving objective value.
pub trait Step {
    /// Perform an optimization step.
    fn step(&mut self);
}

/// An optimizer able to efficiently provide a view
/// of a point to be evaluated.
/// For optimizers evaluating at most one point per step.
pub trait Point<A> {
    /// Return point to be evaluated.
    fn point(&self) -> Option<ArrayView1<A>>;
}

/// An optimizer able to efficiently provide a view
/// of points to be evaluated.
/// For optimizers evaluating more than one point per step.
pub trait Points<A> {
    /// Return points to be evaluated.
    fn points(&self) -> ArrayView2<A>;
}

/// Indicate whether or not an optimizer is done.
pub trait IsDone {
    /// Return if optimizer is done.
    fn is_done(&self) -> bool;
}

/// An optimizer able to return the best point discovered.
pub trait BestPoint<A> {
    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<A, Ix1>;
}

/// An optimizer able to return the value of the best point discovered.
pub trait BestPointValue<A> {
    /// Return the value of the best point discovered.
    fn best_point_value(&self) -> Option<A>;
}

impl<'a, F, O, T> InitializeUsing<'a, F, O> for Box<T>
where
    T: ?Sized + InitializeUsing<'a, F, O>,
{
    fn initialize_using<R>(&'a self, objective: &'a F, rng: &mut R) -> O
    where
        R: Rng,
    {
        self.as_ref().initialize_using(objective, rng)
    }
}

impl<'a, F, O, T> Initialize<'a, F, O> for Box<T>
where
    T: ?Sized + Initialize<'a, F, O>,
{
    fn initialize(&'a self, objective: &'a F) -> O {
        self.as_ref().initialize(objective)
    }
}

impl<T> Step for Box<T>
where
    T: ?Sized + Step,
{
    fn step(&mut self) {
        self.as_mut().step()
    }
}

impl<T> IsDone for Box<T>
where
    T: ?Sized + IsDone,
{
    fn is_done(&self) -> bool {
        self.as_ref().is_done()
    }
}

impl<A, T> Point<A> for Box<T>
where
    T: ?Sized + Point<A>,
{
    fn point(&self) -> Option<ArrayView1<A>> {
        self.as_ref().point()
    }
}

impl<A, T> Points<A> for Box<T>
where
    T: ?Sized + Points<A>,
{
    fn points(&self) -> ArrayView2<A> {
        self.as_ref().points()
    }
}

impl<A, T> BestPoint<A> for Box<T>
where
    T: ?Sized + BestPoint<A>,
{
    fn best_point(&self) -> CowArray<A, Ix1> {
        self.as_ref().best_point()
    }
}

impl<A, T> BestPointValue<A> for Box<T>
where
    T: ?Sized + BestPointValue<A>,
{
    fn best_point_value(&self) -> Option<A> {
        self.as_ref().best_point_value()
    }
}
