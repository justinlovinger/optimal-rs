use std::ops::{Mul, SubAssign};

use crate::StepSize;

use self::valued::Valued;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynState<A> {
    Started(Started<A>),
    Evaluated(Evaluated<A>),
    Stepped(Stepped<A>),
    Finished(Finished<A>),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Started<A> {
    pub point: Vec<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Evaluated<A> {
    pub point: Valued<Vec<A>, Vec<A>>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Stepped<A> {
    pub point: Vec<A>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Finished<A> {
    pub point: Vec<A>,
}

impl<A> DynState<A> {
    pub fn new(point: Vec<A>) -> Self {
        Self::Started(Started::new(point))
    }
}

impl<A> Started<A> {
    pub fn new(point: Vec<A>) -> Self {
        Self { point }
    }

    pub fn into_evaluated(self, f: impl FnOnce(&[A]) -> Vec<A>) -> Evaluated<A> {
        Evaluated::new(self.point, f)
    }
}

impl<A> Evaluated<A> {
    pub fn new(point: Vec<A>, f: impl FnOnce(&[A]) -> Vec<A>) -> Self {
        Self {
            point: Valued::new(point, f),
        }
    }

    pub fn into_stepped(self, step_size: StepSize<A>) -> Stepped<A>
    where
        A: Clone + SubAssign + Mul<Output = A>,
    {
        Stepped::new(step_size, self.point)
    }
}

impl<A> Stepped<A> {
    pub fn new(step_size: StepSize<A>, point: Valued<Vec<A>, Vec<A>>) -> Self
    where
        A: Clone + SubAssign + Mul<Output = A>,
    {
        let (mut point, derivatives) = point.into_parts();
        point
            .iter_mut()
            .zip(derivatives)
            .for_each(|(x, d)| *x -= step_size.clone() * d);
        Self { point }
    }

    pub fn into_finished(self) -> Finished<A> {
        Finished::new(self.point)
    }
}

impl<A> Finished<A> {
    pub fn new(point: Vec<A>) -> Self {
        Self { point }
    }

    pub fn into_started(self) -> Started<A> {
        Started::new(self.point)
    }
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
