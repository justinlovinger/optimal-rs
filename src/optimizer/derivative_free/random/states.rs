use ndarray::{prelude::*, Data, RawData};
use rand::{distributions::Distribution, Rng};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Initial state,
/// ready to evaluate points.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Init<A> {
    points: Array2<A>,
}

impl<A> Init<A> {
    /// Return a new initial state.
    pub fn new<D, R>(num_points: usize, distributions: &Array1<D>, mut rng: R) -> Self
    where
        D: Distribution<A>,
        R: Rng,
    {
        Self {
            points: random_points(num_points, distributions.view(), &mut rng),
        }
    }
}

impl<A> Init<A>
where
    A: Clone,
{
    /// Step to 'Done' state.
    pub fn to_done<B, S>(self, point_values: ArrayBase<S, Ix1>) -> Done<A, B>
    where
        B: Clone + PartialOrd,
        S: Data<Elem = B>,
    {
        let (best_point_value, best_point) = point_values
            .into_iter()
            .zip(self.points.rows())
            .min_by(|(x, _), (y, _)| x.partial_cmp(y).expect("point values should be comparable"))
            .expect("should have at least one point");
        Done {
            // Ideally,
            // this should take an owned slice
            // of `self.points`,
            // instead of cloning.
            best_point: best_point.into_owned(),
            best_point_value: best_point_value.clone(),
        }
    }
}

impl<A> Init<A> {
    /// Return points to evaluate.
    pub fn points(&self) -> &Array2<A> {
        &self.points
    }
}

/// Final state,
/// with points evaluated.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Done<A, B> {
    best_point: Array1<A>,
    best_point_value: B,
}

impl<A, B> Done<A, B> {
    /// Return best point.
    pub fn best_point(&self) -> &Array1<A> {
        &self.best_point
    }

    /// Return value of best point.
    pub fn best_point_value(&self) -> &B {
        &self.best_point_value
    }
}

fn random_points<A, D, S, R>(
    num_points: usize,
    distributions: ArrayBase<S, Ix1>,
    rng: &mut R,
) -> Array2<A>
where
    D: Distribution<A>,
    S: RawData<Elem = D> + Data,
    R: Rng,
{
    distributions
        .broadcast((num_points, distributions.len()))
        .unwrap()
        .map(|distr| rng.sample(distr))
}
