pub use self::{rand::*, seeded_rand::*};

#[allow(clippy::module_inception)]
mod rand {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, Computation, ComputationFn, NamedArgs, Names};

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
        Rand<Dist::Filled, T>: Computation,
    {
        type Filled = Rand<Dist::Filled, T>;

        fn fill(self, named_args: NamedArgs) -> Self::Filled {
            Rand {
                distribution: self.distribution.fill(named_args),
                ty: self.ty,
            }
        }

        fn arg_names(&self) -> Names {
            self.distribution.arg_names()
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

    use crate::{impl_core_ops, peano::Zero, Computation, ComputationFn, NamedArgs, Names};

    #[derive(Clone, Copy, Debug)]
    pub struct SeededRand<Dist, T, R>
    where
        Self: Computation,
    {
        pub distribution: Dist,
        ty: PhantomData<T>,
        pub rng: R,
    }

    impl<Dist, T, R> SeededRand<Dist, T, R>
    where
        Self: Computation,
    {
        pub fn new(distribution: Dist, rng: R) -> Self {
            Self {
                distribution,
                ty: PhantomData,
                rng,
            }
        }
    }

    impl<Dist, T, R> Computation for SeededRand<Dist, T, R>
    where
        Dist: Computation,
        R: Computation,
    {
        type Dim = (Dist::Dim, Zero);
        type Item = (T, R::Item);
    }

    impl<Dist, T, R> ComputationFn for SeededRand<Dist, T, R>
    where
        Self: Computation,
        Dist: ComputationFn,
        R: ComputationFn,
        SeededRand<Dist::Filled, T, R::Filled>: Computation,
    {
        type Filled = SeededRand<Dist::Filled, T, R::Filled>;

        fn fill(self, named_args: NamedArgs) -> Self::Filled {
            let (args_0, args_1) = named_args
                .partition(&self.rng.arg_names(), &self.distribution.arg_names())
                .unwrap_or_else(|e| panic!("{}", e,));
            SeededRand {
                distribution: self.distribution.fill(args_1),
                ty: self.ty,
                rng: self.rng.fill(args_0),
            }
        }

        fn arg_names(&self) -> Names {
            self.rng.arg_names().union(self.distribution.arg_names())
        }
    }

    impl_core_ops!(SeededRand<Dist, T, R>);

    impl<Dist, T, R> fmt::Display for SeededRand<Dist, T, R>
    where
        Self: Computation,
        Dist: fmt::Display,
        R: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "seeded_rand({}, {})", self.distribution, self.rng)
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
        let dist = val!(Standard);
        let rng = val!(StdRng::seed_from_u64(seed));
        prop_assert_eq!(
            SeededRand::<_, i32, _>::new(dist, rng.clone()).to_string(),
            format!("seeded_rand({}, {})", dist, rng)
        );
    }

    #[proptest]
    fn seededrands_should_display_1d(seed: u64, #[strategy(1_usize..10)] x: usize) {
        let dist = val1!(std::iter::repeat(Standard).take(x).collect::<Vec<_>>());
        let rng = val!(StdRng::seed_from_u64(seed));
        prop_assert_eq!(
            SeededRand::<_, i32, _>::new(dist.clone(), rng.clone()).to_string(),
            format!("seeded_rand({}, {})", dist, rng)
        );
    }

    #[proptest]
    fn seededrands_should_display_2d(
        seed: u64,
        #[strategy(1_usize..10)] x: usize,
        #[strategy(1_usize..10)] y: usize,
    ) {
        let dist = val2!(Matrix::from_vec(
            (x, y),
            std::iter::repeat(Standard).take(x * y).collect::<Vec<_>>()
        )
        .unwrap());
        let rng = val!(StdRng::seed_from_u64(seed));
        prop_assert_eq!(
            SeededRand::<_, i32, _>::new(dist.clone(), rng.clone()).to_string(),
            format!("seeded_rand({}, {})", dist, rng)
        );
    }
}
