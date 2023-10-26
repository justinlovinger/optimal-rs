use crate::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A component for getting line-search step-direction.
pub trait StepDirection {
    /// Elements of a point.
    type Elem;

    /// Point of elements.
    type Point;

    /// Result of an objective-function evaluation.
    type Value;

    /// Start a new iteration of this component.
    fn start_iteration(&mut self, point: Self::Point);

    /// Return step-direction if ready.
    fn step(&mut self) -> Option<DoneStepDirection<Self::Elem, Self::Value>>;
}

/// Result of a 'StepDirection' run.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DoneStepDirection<A, V> {
    /// Result of objective-function evaluation.
    pub value: V,
    /// Line-search step-direction.
    pub step_direction: Vec<A>,
}

/// A component for getting line-search initial step-size.
pub trait InitialStepSize {
    /// Elements of a point.
    type Elem;

    /// Start a new iteration of this component.
    fn start_iteration(&mut self, last_step_size: Option<StepSize<Self::Elem>>);

    /// Return step-size if ready.
    fn step(&mut self) -> Option<StepSize<Self::Elem>>;
}

/// A component for getting line-search step-direction
/// and initial step-size.
pub trait StepDirectionInitialStepSize {
    /// Elements of a point.
    type Elem;

    /// Point of elements.
    type Point;

    /// Result of an objective-function evaluation.
    type Value;

    /// Start a new iteration of this component.
    fn start_iteration(&mut self, point: Self::Point, last_step_size: Option<StepSize<Self::Elem>>);

    /// Return step-direction and initial step-size if ready.
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
    ) -> Option<(
        DoneStepDirection<Self::Elem, Self::Value>,
        StepSize<Self::Elem>,
    )>;
}

/// Wrapper for step-direction and initial step-size components
/// with no dependence on each other.
#[derive(Clone, Debug, PartialEq, Eq, Hash, derive_getters::Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D: Serialize, I: Serialize, D::Elem: Serialize, D::Value: Serialize",
        deserialize = "D: Deserialize<'de>, I: Deserialize<'de>, D::Elem: Deserialize<'de>, D::Value: Deserialize<'de>"
    ))
)]
pub struct IndependentStepDirectionInitialStepSize<D, I>
where
    D: StepDirection,
{
    step_direction: D,
    initial_step_size: I,

    #[getter(skip)]
    done_step_direction: Option<DoneStepDirection<D::Elem, D::Value>>,
}

impl<D, I> IndependentStepDirectionInitialStepSize<D, I>
where
    D: StepDirection,
{
    /// Return a new 'IndependentStepDirectionInitialStepSize'.
    pub fn new(step_direction: D, initial_step_size: I) -> Self {
        Self {
            step_direction,
            initial_step_size,
            done_step_direction: None,
        }
    }
}

impl<D, I> StepDirectionInitialStepSize for IndependentStepDirectionInitialStepSize<D, I>
where
    D: StepDirection<Elem = I::Elem>,
    I: InitialStepSize,
{
    type Elem = I::Elem;
    type Point = D::Point;
    type Value = D::Value;

    fn start_iteration(
        &mut self,
        point: Self::Point,
        last_step_size: Option<StepSize<Self::Elem>>,
    ) {
        self.step_direction.start_iteration(point);
        self.initial_step_size.start_iteration(last_step_size);
    }

    fn step(
        &mut self,
    ) -> Option<(
        DoneStepDirection<Self::Elem, Self::Value>,
        StepSize<Self::Elem>,
    )> {
        if self.done_step_direction.is_some() {
            match self.initial_step_size.step() {
                Some(y) => Some((self.done_step_direction.take().unwrap(), y)),
                None => None,
            }
        } else {
            match self.step_direction.step() {
                Some(x) => {
                    self.done_step_direction = Some(x);
                    None
                }
                None => None,
            }
        }
    }
}
