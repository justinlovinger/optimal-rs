pub use self::{rand::*, seeded_rand::*};

#[allow(clippy::module_inception)]
mod rand {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, Args, Computation, ComputationFn};

    #[derive(Clone, Copy, Debug)]
    pub struct Rand<Dist, T>
    where
        Self: Computation,
    {
        pub distribution: Dist,
        ty: PhantomData<T>,
    }

    impl<Dist, T> Rand<Dist, T>
    where
        Self: Computation,
    {
        pub fn new(distribution: Dist) -> Self {
            Self {
                distribution,
                ty: PhantomData,
            }
        }
    }

    impl<Dist, T> Computation for Rand<Dist, T>
    where
        Dist: Computation,
    {
        type Dim = Dist::Dim;
        type Item = T;
    }

    impl<Dist, T> ComputationFn for Rand<Dist, T>
    where
        Self: Computation,
        Dist: ComputationFn,
    {
        fn args(&self) -> Args {
            self.distribution.args()
        }
    }

    impl_core_ops!(Rand<Dist, T>);

    impl<Dist, T> fmt::Display for Rand<Dist, T>
    where
        Self: Computation,
        Dist: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "rand({})", self.distribution)
        }
    }
}

mod seeded_rand {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, peano::Zero, Args, Computation, ComputationFn};

    #[derive(Clone, Copy, Debug)]
    pub struct SeededRand<R, Dist, T>
    where
        Self: Computation,
    {
        pub rng: R,
        pub distribution: Dist,
        ty: PhantomData<T>,
    }

    impl<R, Dist, T> SeededRand<R, Dist, T>
    where
        Self: Computation,
    {
        pub fn new(rng: R, distribution: Dist) -> Self {
            Self {
                rng,
                distribution,
                ty: PhantomData,
            }
        }
    }

    impl<R, Dist, T> Computation for SeededRand<R, Dist, T>
    where
        R: Computation,
        Dist: Computation,
    {
        type Dim = (Zero, Dist::Dim);
        type Item = (R::Item, T);
    }

    impl<R, Dist, T> ComputationFn for SeededRand<R, Dist, T>
    where
        Self: Computation,
        R: ComputationFn,
        Dist: ComputationFn,
    {
        fn args(&self) -> Args {
            self.rng.args().union(self.distribution.args())
        }
    }

    impl_core_ops!(SeededRand<R, Dist, T>);

    impl<R, Dist, T> fmt::Display for SeededRand<R, Dist, T>
    where
        Self: Computation,
        R: fmt::Display,
        Dist: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "seeded_rand({}, {})", self.rng, self.distribution)
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::{distributions::Standard, rngs::StdRng, SeedableRng};
    use test_strategy::proptest;

    use crate::{
        rand::{rand::Rand, seeded_rand::SeededRand},
        run::Matrix,
        val, val1, val2,
    };

    #[test]
    fn rands_should_display() {
        let dist = val!(Standard);
        assert_eq!(
            Rand::<_, i32>::new(dist).to_string(),
            format!("rand({})", dist)
        );
    }

    #[proptest]
    fn rands_should_display_1d(#[strategy(1_usize..10)] x: usize) {
        let dist = val1!(std::iter::repeat(Standard).take(x).collect::<Vec<_>>());
        prop_assert_eq!(
            Rand::<_, i32>::new(dist.clone()).to_string(),
            format!("rand({})", dist)
        );
    }

    #[proptest]
    fn rands_should_display_2d(
        #[strategy(1_usize..10)] x: usize,
        #[strategy(1_usize..10)] y: usize,
    ) {
        let dist = val2!(Matrix::from_vec(
            (x, y),
            std::iter::repeat(Standard).take(x * y).collect::<Vec<_>>()
        )
        .unwrap());
        prop_assert_eq!(
            Rand::<_, i32>::new(dist.clone()).to_string(),
            format!("rand({})", dist)
        );
    }

    #[proptest]
    fn seededrands_should_display(seed: u64) {
        let rng = val!(StdRng::seed_from_u64(seed));
        let dist = val!(Standard);
        prop_assert_eq!(
            SeededRand::<_, _, i32>::new(rng.clone(), dist).to_string(),
            format!("seeded_rand({}, {})", rng, dist)
        );
    }

    #[proptest]
    fn seededrands_should_display_1d(seed: u64, #[strategy(1_usize..10)] x: usize) {
        let rng = val!(StdRng::seed_from_u64(seed));
        let dist = val1!(std::iter::repeat(Standard).take(x).collect::<Vec<_>>());
        prop_assert_eq!(
            SeededRand::<_, _, i32>::new(rng.clone(), dist.clone()).to_string(),
            format!("seeded_rand({}, {})", rng, dist)
        );
    }

    #[proptest]
    fn seededrands_should_display_2d(
        seed: u64,
        #[strategy(1_usize..10)] x: usize,
        #[strategy(1_usize..10)] y: usize,
    ) {
        let rng = val!(StdRng::seed_from_u64(seed));
        let dist = val2!(Matrix::from_vec(
            (x, y),
            std::iter::repeat(Standard).take(x * y).collect::<Vec<_>>()
        )
        .unwrap());
        prop_assert_eq!(
            SeededRand::<_, _, i32>::new(rng.clone(), dist.clone()).to_string(),
            format!("seeded_rand({}, {})", rng, dist)
        );
    }
}
