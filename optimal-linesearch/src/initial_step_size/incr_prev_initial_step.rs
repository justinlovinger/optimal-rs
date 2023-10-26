//! Initial step-size by incrementing previous step-size.

use std::{fmt, ops::Mul};

use num_traits::{real::Real, AsPrimitive};

use crate::{backtracking_line_search::BacktrackingRate, InitialStepSize, StepSize};

pub use self::types::IncrRate;

/// A component for initial step-size
/// by incrementing previous step-size.
#[derive(Clone, Debug, derive_getters::Getters)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IncrPrevStep<A> {
    config: Config<A>,
    state: State<A>,
}

/// Config for incrementing previous step-size.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config<A> {
    /// Step-size to use when no previous step-size.
    pub default_step_size: StepSize<A>,
    /// Rate to increase previous step-size.
    pub incr_rate: IncrRate<A>,
}

/// State for incrementing previous step-size.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum State<A> {
    /// Run has not started.
    UnStarted,
    /// Run just started.
    Started {
        /// Step-size of last iteration.
        last_step_size: Option<StepSize<A>>,
    },
    /// Used default step-size.
    UsedDefault {
        /// Step-size to return.
        new_step_size: StepSize<A>,
    },
    /// Incremented previous step-size.
    IncrementedPrevious {
        /// Step-size to return.
        new_step_size: StepSize<A>,
    },
    /// Run finished.
    Finished,
}

impl<A> Config<A> {
    /// Return default config from backtracking-rate.
    pub fn from_backtracking_rate(backtracking_rate: BacktrackingRate<A>) -> Self
    where
        A: fmt::Debug + Real + 'static,
        f64: AsPrimitive<A>,
    {
        Self {
            default_step_size: StepSize::new(1.0.as_()).unwrap(),
            incr_rate: IncrRate::from_backtracking_rate(backtracking_rate),
        }
    }

    /// Return a new 'IncrPrevStep'.
    pub fn build(self) -> IncrPrevStep<A> {
        IncrPrevStep {
            config: self,
            state: State::UnStarted,
        }
    }
}

impl<A> InitialStepSize for IncrPrevStep<A>
where
    A: Copy + Mul<Output = A>,
{
    type Elem = A;

    fn start_iteration(&mut self, last_step_size: Option<StepSize<A>>) {
        self.state = State::Started { last_step_size }
    }

    fn step(&mut self) -> Option<StepSize<A>> {
        let mut ret = None;
        replace_with::replace_with_or_abort(&mut self.state, |state| match state {
            State::UnStarted => State::UnStarted,
            State::Started { last_step_size } => match last_step_size {
                Some(x) => State::IncrementedPrevious {
                    new_step_size: self.config.incr_rate * x,
                },
                None => State::UsedDefault {
                    new_step_size: self.config.default_step_size,
                },
            },
            State::UsedDefault { new_step_size } => {
                ret = Some(new_step_size);
                State::Finished
            }
            State::IncrementedPrevious { new_step_size } => {
                ret = Some(new_step_size);
                State::Finished
            }
            State::Finished => State::Finished,
        });
        ret
    }
}

mod types {
    use std::ops::{Div, Mul, Sub};

    use derive_more::Display;
    use derive_num_bounded::{derive_into_inner, derive_new_from_lower_bounded_partial_ord};
    use num_traits::{bounds::LowerBounded, real::Real, AsPrimitive, One};

    use crate::{backtracking_line_search::BacktrackingRate, StepSize};

    /// Rate to increase step size before starting each line search.
    #[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    pub struct IncrRate<A>(A);

    derive_new_from_lower_bounded_partial_ord!(IncrRate<A: Real>);
    derive_into_inner!(IncrRate<A>);

    impl<A> IncrRate<A>
    where
        A: 'static + Copy + One + Sub<Output = A> + Div<Output = A>,
        f64: AsPrimitive<A>,
    {
        /// Return increase rate slightly more than one step up from backtracking rate.
        pub fn from_backtracking_rate(x: BacktrackingRate<A>) -> IncrRate<A> {
            Self(2.0.as_() / x.into_inner() - A::one())
        }
    }

    impl<A> LowerBounded for IncrRate<A>
    where
        A: Real,
    {
        fn min_value() -> Self {
            Self(A::one() + A::epsilon())
        }
    }

    impl<A> Mul<StepSize<A>> for IncrRate<A>
    where
        A: Mul<Output = A>,
    {
        type Output = StepSize<A>;

        fn mul(self, rhs: StepSize<A>) -> Self::Output {
            StepSize(self.0 * rhs.into_inner())
        }
    }
}
