use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use derive_getters::Getters;
use num_traits::AsPrimitive;
pub use optimal_core::prelude::*;

use crate::StepSize;

use self::{line_search::LineSearch, valued::Valued};

use super::types::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynState<A> {
    Started(Started<A>),
    Evaluated(Evaluated<A>),
    InitializedSearching(InitializedSearching<A>),
    FakeStepped(FakeStepped<A>),
    FakeStepEvaluated(FakeStepEvaluated<A>),
    StepSizeDecremented(StepSizeDecremented<A>),
    Stepped(Stepped<A>),
    Finished(Finished<A>),
    StepSizeIncremented(StepSizeIncremented<A>),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Started<A> {
    pub point: Vec<A>,
    pub step_size: StepSize<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Evaluated<A> {
    pub point: Valued<Vec<A>, (A, Vec<A>)>,
    pub step_size: StepSize<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InitializedSearching<A> {
    pub line_search: LineSearch<A>,
    pub step_size: StepSize<A>,
}

#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FakeStepped<A> {
    line_search: LineSearch<A>,
    step_size: StepSize<A>,
    point_at_step: Vec<A>,
}

#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FakeStepEvaluated<A> {
    line_search: LineSearch<A>,
    step_size: StepSize<A>,
    point_at_step: Valued<Vec<A>, A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StepSizeDecremented<A> {
    pub line_search: LineSearch<A>,
    pub step_size: StepSize<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Stepped<A> {
    pub point: Vec<A>,
    pub last_step_size: StepSize<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Finished<A> {
    pub point: Vec<A>,
    pub last_step_size: StepSize<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StepSizeIncremented<A> {
    pub point: Vec<A>,
    pub step_size: StepSize<A>,
}

impl<A> DynState<A> {
    /// Return an initial state.
    pub fn new(point: Vec<A>, initial_step_size: StepSize<A>) -> Self {
        DynState::Started(Started::new(point, initial_step_size))
    }
}

impl<A> Started<A> {
    pub fn new(point: Vec<A>, step_size: StepSize<A>) -> Self {
        Self { point, step_size }
    }

    pub fn into_evaluated(self, f: impl FnOnce(&[A]) -> (A, Vec<A>)) -> Evaluated<A> {
        Evaluated::new(self.point, self.step_size, f)
    }
}

impl<A> Evaluated<A> {
    pub fn new(point: Vec<A>, step_size: StepSize<A>, f: impl FnOnce(&[A]) -> (A, Vec<A>)) -> Self {
        Self {
            point: Valued::new(point, f),
            step_size,
        }
    }

    pub fn into_initialized_searching(
        self,
        c_1: SufficientDecreaseParameter<A>,
    ) -> InitializedSearching<A>
    where
        A: 'static + Copy + Neg<Output = A> + Mul<Output = A> + Sum,
        f64: AsPrimitive<A>,
    {
        InitializedSearching::new(c_1, self.point, self.step_size)
    }
}

impl<A> InitializedSearching<A> {
    pub fn new(
        c_1: SufficientDecreaseParameter<A>,
        point: Valued<Vec<A>, (A, Vec<A>)>,
        step_size: StepSize<A>,
    ) -> Self
    where
        A: 'static + Copy + Neg<Output = A> + Mul<Output = A> + Sum,
        f64: AsPrimitive<A>,
    {
        Self {
            line_search: LineSearch::new(c_1, point),
            step_size,
        }
    }

    pub fn into_fake_stepped(self) -> FakeStepped<A>
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        FakeStepped::new(self.line_search, self.step_size)
    }
}

impl<A> FakeStepped<A> {
    pub fn new(line_search: LineSearch<A>, step_size: StepSize<A>) -> Self
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        Self {
            point_at_step: descend(
                line_search.point(),
                step_size.clone(),
                line_search.step_direction(),
            ),
            line_search,
            step_size,
        }
    }

    pub fn into_fake_step_evaluated(self, f: impl FnOnce(&[A]) -> A) -> FakeStepEvaluated<A> {
        FakeStepEvaluated::new(self.line_search, self.step_size, self.point_at_step, f)
    }
}

impl<A> FakeStepEvaluated<A> {
    // `step_size` and `point_at_step` cannot be safely passed
    // independent of `line_search`.
    fn new(
        line_search: LineSearch<A>,
        step_size: StepSize<A>,
        point_at_step: Vec<A>,
        f: impl FnOnce(&[A]) -> A,
    ) -> Self {
        Self {
            line_search,
            step_size,
            point_at_step: Valued::new(point_at_step, f),
        }
    }

    pub fn is_sufficient_decrease(&self) -> bool
    where
        A: Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        is_sufficient_decrease(
            *self.line_search.value(),
            self.step_size,
            *self
                .line_search
                .c_1_times_point_derivatives_dot_step_direction(),
            *self.point_at_step.value(),
        )
    }

    pub fn into_step_size_decremented(
        self,
        backtracking_rate: BacktrackingRate<A>,
    ) -> StepSizeDecremented<A>
    where
        A: Mul<Output = A>,
    {
        StepSizeDecremented::new(backtracking_rate, self.line_search, self.step_size)
    }

    pub fn into_stepped(self) -> Stepped<A> {
        Stepped::new(self.point_at_step.into_parts().0, self.step_size)
    }
}

impl<A> StepSizeDecremented<A> {
    pub fn new(
        backtracking_rate: BacktrackingRate<A>,
        line_search: LineSearch<A>,
        step_size: StepSize<A>,
    ) -> Self
    where
        A: Mul<Output = A>,
    {
        Self {
            line_search,
            step_size: StepSize(backtracking_rate.into_inner() * step_size.into_inner()),
        }
    }

    pub fn into_fake_stepped(self) -> FakeStepped<A>
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        FakeStepped::new(self.line_search, self.step_size)
    }
}

impl<A> Stepped<A> {
    pub fn new(point_at_step: Vec<A>, step_size: StepSize<A>) -> Self {
        Self {
            point: point_at_step,
            last_step_size: step_size,
        }
    }

    pub fn into_finished(self) -> Finished<A> {
        Finished::new(self.point, self.last_step_size)
    }
}

impl<A> Finished<A> {
    pub fn new(point: Vec<A>, last_step_size: StepSize<A>) -> Self {
        Self {
            point,
            last_step_size,
        }
    }

    pub fn into_step_size_incremented(
        self,
        initial_step_size_incr_rate: IncrRate<A>,
    ) -> StepSizeIncremented<A>
    where
        A: Mul<Output = A>,
    {
        StepSizeIncremented::new(initial_step_size_incr_rate, self.point, self.last_step_size)
    }
}

impl<A> StepSizeIncremented<A> {
    pub fn new(
        initial_step_size_incr_rate: IncrRate<A>,
        point: Vec<A>,
        last_step_size: StepSize<A>,
    ) -> Self
    where
        A: Mul<Output = A>,
    {
        Self {
            point,
            step_size: StepSize(initial_step_size_incr_rate * last_step_size.into_inner()),
        }
    }

    pub fn into_started(self) -> Started<A> {
        Started::new(self.point, self.step_size)
    }
}

fn descend<A>(point: &[A], step_size: StepSize<A>, step_direction: &[A]) -> Vec<A>
where
    A: Clone + Add<Output = A> + Mul<Output = A>,
{
    point
        .iter()
        .zip(step_direction)
        .map(|(x, d)| x.clone() + step_size.clone() * d.clone())
        .collect()
}

/// The sufficient decrease condition,
/// also known as the Armijo rule,
/// mathematically written as:
/// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
fn is_sufficient_decrease<A>(
    value: A,
    step_size: StepSize<A>,
    c_1_times_point_derivatives_dot_step_direction: A,
    value_at_step: A,
) -> bool
where
    A: PartialOrd + Add<Output = A> + Mul<Output = A>,
{
    value_at_step <= value + step_size * c_1_times_point_derivatives_dot_step_direction
}

mod valued {
    use std::borrow::Borrow;

    use derive_getters::{Dissolve, Getters};

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, PartialEq, Eq, Dissolve, Getters)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[dissolve(rename = "into_parts")]
    pub struct Valued<T, B> {
        x: T,
        value: B,
    }

    impl<T, B> Valued<T, B> {
        pub fn new<Borrowed, F>(x: T, f: F) -> Self
        where
            Borrowed: ?Sized,
            T: Borrow<Borrowed>,
            F: FnOnce(&Borrowed) -> B,
        {
            Self {
                value: f(x.borrow()),
                x,
            }
        }
    }
}

mod line_search {
    use std::{
        fmt::Debug,
        iter::Sum,
        ops::{Mul, Neg},
    };

    use derive_getters::{Dissolve, Getters};
    use num_traits::AsPrimitive;

    use crate::backtracking_steepest::SufficientDecreaseParameter;

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    use super::valued::Valued;

    #[derive(Clone, Debug, Dissolve, Getters)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[dissolve(rename = "into_parts")]
    pub struct LineSearch<A> {
        point: Vec<A>,
        value: A,
        step_direction: Vec<A>,
        c_1_times_point_derivatives_dot_step_direction: A,
    }

    impl<A> LineSearch<A> {
        pub fn new(c_1: SufficientDecreaseParameter<A>, point: Valued<Vec<A>, (A, Vec<A>)>) -> Self
        where
            A: 'static + Copy + Neg<Output = A> + Mul<Output = A> + Sum,
            f64: AsPrimitive<A>,
        {
            let (point, (value, derivatives)) = point.into_parts();
            let step_direction = derivatives.iter().map(|x| -*x).collect::<Vec<_>>();
            Self {
                point,
                value,
                c_1_times_point_derivatives_dot_step_direction: c_1.into_inner()
                    * derivatives
                        .into_iter()
                        .zip(step_direction.iter().copied())
                        .map(|(x, y)| x * y)
                        .sum(),
                step_direction,
            }
        }
    }
}
