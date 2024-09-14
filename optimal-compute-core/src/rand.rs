pub use self::{rand::*, rands::*, seeded_rand::*, seeded_rands::*, tuple_len::*};

#[allow(clippy::module_inception)]
mod rand {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, Args, Computation, ComputationFn};

    use super::TupleLen;

    #[derive(Clone, Copy, Debug)]
    pub struct Rand<Sh, Dist, T>
    where
        Self: Computation,
    {
        pub shape: Sh,
        pub distribution: Dist,
        ty: PhantomData<T>,
    }

    impl<Sh, Dist, T> Rand<Sh, Dist, T>
    where
        Self: Computation,
    {
        pub fn new(shape: Sh, distribution: Dist) -> Self {
            Self {
                shape,
                distribution,
                ty: PhantomData,
            }
        }
    }

    impl<Sh, Dist, T> Computation for Rand<Sh, Dist, T>
    where
        Sh: TupleLen,
    {
        type Dim = Sh::Len;
        type Item = T;
    }

    impl<Sh, Dist, T> ComputationFn for Rand<Sh, Dist, T>
    where
        Self: Computation,
    {
        fn args(&self) -> Args {
            Args::new()
        }
    }

    impl_core_ops!(Rand<Sh, Dist, T>);

    impl<Sh, Dist, T> fmt::Display for Rand<Sh, Dist, T>
    where
        Self: Computation,
        Sh: fmt::Debug,
        Dist: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "random({:?}, {:?})", self.shape, self.distribution)
        }
    }
}

mod seeded_rand {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, peano::Zero, Args, Computation, ComputationFn};

    use super::TupleLen;

    #[derive(Clone, Copy, Debug)]
    pub struct SeededRand<R, Sh, Dist, T>
    where
        Self: Computation,
    {
        pub rng_comp: R,
        pub shape: Sh,
        pub distribution: Dist,
        ty: PhantomData<T>,
    }

    impl<R, Sh, Dist, T> SeededRand<R, Sh, Dist, T>
    where
        Self: Computation,
    {
        pub fn new(rng_comp: R, shape: Sh, distribution: Dist) -> Self {
            Self {
                rng_comp,
                shape,
                distribution,
                ty: PhantomData,
            }
        }
    }

    impl<R, Sh, Dist, T> Computation for SeededRand<R, Sh, Dist, T>
    where
        R: Computation<Dim = Zero>,
        Sh: TupleLen,
    {
        type Dim = (Zero, Sh::Len);
        type Item = (R::Item, T);
    }

    impl<R, Sh, Dist, T> ComputationFn for SeededRand<R, Sh, Dist, T>
    where
        Self: Computation,
        R: ComputationFn,
    {
        fn args(&self) -> Args {
            self.rng_comp.args()
        }
    }

    impl_core_ops!(SeededRand<R, Sh, Dist, T>);

    impl<R, Sh, Dist, T> fmt::Display for SeededRand<R, Sh, Dist, T>
    where
        Self: Computation,
        R: fmt::Debug,
        Sh: fmt::Debug,
        Dist: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "seeded_random({:?}, {:?}, {:?})",
                self.rng_comp, self.shape, self.distribution
            )
        }
    }
}

mod tuple_len {
    use crate::peano::{One, Two, Zero};

    pub trait TupleLen {
        type Len;
    }

    impl TupleLen for () {
        type Len = Zero;
    }

    impl<T0> TupleLen for (T0,) {
        type Len = One;
    }

    impl<T0, T1> TupleLen for (T0, T1) {
        type Len = Two;
    }
}

mod rands {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, Args, Computation, ComputationFn};

    #[derive(Clone, Copy, Debug)]
    pub struct Rands<Dist, T>
    where
        Self: Computation,
    {
        pub distribution_comp: Dist,
        ty: PhantomData<T>,
    }

    impl<Dist, T> Rands<Dist, T>
    where
        Self: Computation,
    {
        pub fn new(distribution_comp: Dist) -> Self {
            Self {
                distribution_comp,
                ty: PhantomData,
            }
        }
    }

    impl<Dist, T> Computation for Rands<Dist, T>
    where
        Dist: Computation,
    {
        type Dim = Dist::Dim;
        type Item = T;
    }

    impl<Dist, T> ComputationFn for Rands<Dist, T>
    where
        Self: Computation,
        Dist: ComputationFn,
    {
        fn args(&self) -> Args {
            self.distribution_comp.args()
        }
    }

    impl_core_ops!(Rands<Dist, T>);

    impl<Dist, T> fmt::Display for Rands<Dist, T>
    where
        Self: Computation,
        Dist: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "randoms({})", self.distribution_comp)
        }
    }
}

mod seeded_rands {
    use core::fmt;
    use std::marker::PhantomData;

    use crate::{impl_core_ops, peano::Zero, Args, Computation, ComputationFn};

    #[derive(Clone, Copy, Debug)]
    pub struct SeededRands<R, Dist, T>
    where
        Self: Computation,
    {
        pub rng_comp: R,
        pub distribution_comp: Dist,
        ty: PhantomData<T>,
    }

    impl<R, Dist, T> SeededRands<R, Dist, T>
    where
        Self: Computation,
    {
        pub fn new(rng_comp: R, distribution_comp: Dist) -> Self {
            Self {
                rng_comp,
                distribution_comp,
                ty: PhantomData,
            }
        }
    }

    impl<R, Dist, T> Computation for SeededRands<R, Dist, T>
    where
        R: Computation,
        Dist: Computation,
    {
        type Dim = (Zero, Dist::Dim);
        type Item = (R::Item, T);
    }

    impl<R, Dist, T> ComputationFn for SeededRands<R, Dist, T>
    where
        Self: Computation,
        R: ComputationFn,
        Dist: ComputationFn,
    {
        fn args(&self) -> Args {
            self.rng_comp.args().union(self.distribution_comp.args())
        }
    }

    impl_core_ops!(SeededRands<R, Dist, T>);

    impl<R, Dist, T> fmt::Display for SeededRands<R, Dist, T>
    where
        Self: Computation,
        R: fmt::Display,
        Dist: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "seeded_randoms({}, {})",
                self.rng_comp, self.distribution_comp
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::{distributions::Standard, rngs::StdRng, SeedableRng};
    use test_strategy::proptest;

    use crate::{
        rand::{rands::Rands, seeded_rands::SeededRands, Rand, SeededRand},
        run::Matrix,
        val, val1, val2,
    };

    #[test]
    fn rand_should_display() {
        assert_eq!(
            Rand::<_, _, i32>::new((), Standard).to_string(),
            "random((), Standard)"
        );
    }

    #[proptest]
    fn rand_should_display_1d(x: usize) {
        prop_assert_eq!(
            Rand::<_, _, i32>::new((x,), Standard).to_string(),
            format!("random(({},), Standard)", x)
        );
    }

    #[proptest]
    fn rand_should_display_2d(x: usize, y: usize) {
        prop_assert_eq!(
            Rand::<_, _, i32>::new((x, y), Standard).to_string(),
            format!("random(({}, {}), Standard)", x, y)
        );
    }

    #[proptest]
    fn seededrand_should_display(seed: u64) {
        let rng = val!(StdRng::seed_from_u64(seed));
        prop_assert_eq!(
            SeededRand::<_, _, _, i32>::new(rng.clone(), (), Standard).to_string(),
            format!("seeded_random({:?}, (), Standard)", rng)
        );
    }

    #[proptest]
    fn seededrand_should_display_1d(seed: u64, x: usize) {
        let rng = val!(StdRng::seed_from_u64(seed));
        prop_assert_eq!(
            SeededRand::<_, _, _, i32>::new(rng.clone(), (x,), Standard).to_string(),
            format!("seeded_random({:?}, ({},), Standard)", rng, x)
        );
    }

    #[proptest]
    fn seededrand_should_display_2d(seed: u64, x: usize, y: usize) {
        let rng = val!(StdRng::seed_from_u64(seed));
        prop_assert_eq!(
            SeededRand::<_, _, _, i32>::new(rng.clone(), (x, y), Standard).to_string(),
            format!("seeded_random({:?}, ({}, {}), Standard)", rng, x, y)
        );
    }

    #[test]
    fn rands_should_display() {
        let dist = val!(Standard);
        assert_eq!(
            Rands::<_, i32>::new(dist).to_string(),
            format!("randoms({})", dist)
        );
    }

    #[proptest]
    fn rands_should_display_1d(#[strategy(1_usize..10)] x: usize) {
        let dist = val1!(std::iter::repeat(Standard).take(x).collect::<Vec<_>>());
        prop_assert_eq!(
            Rands::<_, i32>::new(dist.clone()).to_string(),
            format!("randoms({})", dist)
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
            Rands::<_, i32>::new(dist.clone()).to_string(),
            format!("randoms({})", dist)
        );
    }

    #[proptest]
    fn seededrands_should_display(seed: u64) {
        let rng = val!(StdRng::seed_from_u64(seed));
        let dist = val!(Standard);
        prop_assert_eq!(
            SeededRands::<_, _, i32>::new(rng.clone(), dist).to_string(),
            format!("seeded_randoms({}, {})", rng, dist)
        );
    }

    #[proptest]
    fn seededrands_should_display_1d(seed: u64, #[strategy(1_usize..10)] x: usize) {
        let rng = val!(StdRng::seed_from_u64(seed));
        let dist = val1!(std::iter::repeat(Standard).take(x).collect::<Vec<_>>());
        prop_assert_eq!(
            SeededRands::<_, _, i32>::new(rng.clone(), dist.clone()).to_string(),
            format!("seeded_randoms({}, {})", rng, dist)
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
            SeededRands::<_, _, i32>::new(rng.clone(), dist.clone()).to_string(),
            format!("seeded_randoms({}, {})", rng, dist)
        );
    }
}
