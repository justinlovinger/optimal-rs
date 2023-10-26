//! Step-direction along the gradient of steepest descent.

use std::ops::Neg;

use crate::{Derivatives, DoneStepDirection, StepDirection};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A component for step-direction
/// along the gradient of steepest descent.
#[derive(Clone, Debug, derive_getters::Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SteepestDescent<A, P, V, FD> {
    state: State<A, P, V>,

    /// Function returning value and partial derivatives
    /// of objective function to minimize.
    obj_func_d: FD,
}

/// Steepest descent state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State<A, P, V> {
    /// Run has not started.
    UnStarted,
    /// Run just started.
    Started {
        /// Point to evaluate.
        point: P,
    },
    /// Point evaluated.
    Evaluated {
        /// Value of point.
        value: V,
    },
    /// Step-direction calculated.
    Calculated(DoneStepDirection<A, V>),
    /// Run finished.
    Finished,
}

impl<A, P, V, FD> SteepestDescent<A, P, V, FD> {
    /// Return a new 'SteepestDescent'.
    pub fn new(obj_func_d: FD) -> Self
    where
        FD: Fn(&[A]) -> V,
    {
        Self {
            state: State::UnStarted,
            obj_func_d,
        }
    }
}

impl<A, P, V, FD> StepDirection for SteepestDescent<A, P, V, FD>
where
    A: Copy + Neg<Output = A>,
    P: AsRef<Vec<A>>,
    V: Derivatives<A>,
    FD: Fn(&[A]) -> V,
{
    type Elem = A;
    type Point = P;
    type Value = V;

    fn start_iteration(&mut self, point: Self::Point) {
        self.state = State::Started { point };
    }

    fn step(&mut self) -> Option<DoneStepDirection<A, V>> {
        let mut ret = None;
        replace_with::replace_with_or_abort(&mut self.state, |state| match state {
            State::UnStarted => State::UnStarted,
            State::Started { point } => State::Evaluated {
                value: (self.obj_func_d)(point.as_ref()),
            },
            State::Evaluated { value } => State::Calculated(DoneStepDirection {
                step_direction: value.derivatives().iter().copied().map(|x| -x).collect(),
                value,
            }),
            State::Calculated(x) => {
                ret = Some(x);
                State::Finished
            }
            State::Finished => State::Finished,
        });
        ret
    }
}
