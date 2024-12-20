mod rands {
    use rand::distributions::Distribution;

    use crate::{
        peano::{One, Two, Zero},
        rand::Rand,
        run::{Matrix, RunCore},
        Computation,
    };

    impl<Dist, T, Out> RunCore for Rand<Dist, T>
    where
        Dist: Computation + RunCore<Output = Out>,
        Out: BroadcastRands<Dist::Dim, T>,
    {
        type Output = Out::Output;

        fn run_core(self) -> Self::Output {
            self.distribution.run_core().broadcast()
        }
    }

    pub trait BroadcastRands<Dim, T> {
        type Output;

        fn broadcast(self) -> Self::Output;
    }

    impl<Dist, T> BroadcastRands<Zero, T> for Dist
    where
        Dist: Distribution<T>,
    {
        type Output = T;

        fn broadcast(self) -> Self::Output {
            self.sample(&mut rand::thread_rng())
        }
    }

    impl<Dists, T> BroadcastRands<One, T> for Dists
    where
        Dists: IntoIterator,
        Dists::Item: Distribution<T>,
    {
        type Output = Vec<T>;

        fn broadcast(self) -> Self::Output {
            self.into_iter()
                .map(|dist| dist.sample(&mut rand::thread_rng()))
                .collect()
        }
    }

    impl<Dists, T> BroadcastRands<Two, T> for Matrix<Dists>
    where
        Dists: IntoIterator,
        Dists::Item: Distribution<T>,
    {
        type Output = Matrix<Vec<T>>;

        fn broadcast(self) -> Self::Output {
            // Neither shape nor the length of `inner` will change,
            // so they should still be fine.
            unsafe {
                Matrix::new_unchecked(
                    self.shape(),
                    self.into_inner()
                        .into_iter()
                        .map(|dist| dist.sample(&mut rand::thread_rng()))
                        .collect(),
                )
            }
        }
    }
}

mod seeded_rands {
    use rand::{distributions::Distribution, Rng};

    use crate::{
        peano::{One, Two, Zero},
        rand::SeededRand,
        run::{Matrix, RunCore},
        Computation,
    };

    impl<Dist, DistOut, T, R> RunCore for SeededRand<Dist, T, R>
    where
        Self: Computation,
        Dist: Computation + RunCore<Output = DistOut>,
        DistOut: BroadcastSeededRands<Dist::Dim, T>,
        R: RunCore,
        R::Output: Rng,
    {
        type Output = (DistOut::Output, R::Output);

        fn run_core(self) -> Self::Output {
            let mut rng = self.rng.run_core();
            let out = self.distribution.run_core().broadcast(&mut rng);
            (out, rng)
        }
    }

    pub trait BroadcastSeededRands<Dim, T> {
        type Output;

        fn broadcast<R>(self, rng: &mut R) -> Self::Output
        where
            R: Rng;
    }

    impl<Dist, T> BroadcastSeededRands<Zero, T> for Dist
    where
        Dist: Distribution<T>,
    {
        type Output = T;

        fn broadcast<R>(self, rng: &mut R) -> Self::Output
        where
            R: Rng,
        {
            self.sample(rng)
        }
    }

    impl<Dists, T> BroadcastSeededRands<One, T> for Dists
    where
        Dists: IntoIterator,
        Dists::Item: Distribution<T>,
    {
        type Output = Vec<T>;

        fn broadcast<R>(self, rng: &mut R) -> Self::Output
        where
            R: Rng,
        {
            self.into_iter().map(|dist| dist.sample(rng)).collect()
        }
    }

    impl<Dists, T> BroadcastSeededRands<Two, T> for Matrix<Dists>
    where
        Dists: IntoIterator,
        Dists::Item: Distribution<T>,
    {
        type Output = Matrix<Vec<T>>;

        fn broadcast<R>(self, rng: &mut R) -> Self::Output
        where
            R: Rng,
        {
            // Neither shape nor the length of `inner` will change,
            // so they should still be fine.
            unsafe {
                Matrix::new_unchecked(
                    self.shape(),
                    self.into_inner()
                        .into_iter()
                        .map(|dist| dist.sample(rng))
                        .collect(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::{distributions::Standard, rngs::StdRng, SeedableRng};
    use test_strategy::proptest;

    use crate::{
        arg,
        rand::{Rand, SeededRand},
        run::Matrix,
        val, val1, val2,
        zip::Zip,
        Computation, Run,
    };

    #[test]
    fn rands_should_generate_scalars() {
        let x = Rand::<_, f64>::new(val!(Standard)).run();

        assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn rands_should_generate_vectors(#[strategy(1_usize..10)] len: usize) {
        let xs = Rand::<_, f64>::new(val1!(std::iter::repeat(Standard)
            .take(len)
            .collect::<Vec<_>>()))
        .run();

        prop_assert_eq!(xs.len(), len);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn rands_should_generate_matrices(
        #[strategy(1_usize..10)] x_len: usize,
        #[strategy(1_usize..10)] y_len: usize,
    ) {
        let shape = (x_len, y_len);
        let xs = Rand::<_, f64>::new(val2!(Matrix::from_vec(
            (x_len, y_len),
            std::iter::repeat(Standard)
                .take(x_len * y_len)
                .collect::<Vec<_>>()
        )
        .unwrap()))
        .run();

        prop_assert_eq!(xs.shape(), shape);
        let xs = xs.into_inner();
        prop_assert_eq!(xs.len(), shape.0 * shape.1);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrands_should_generate_scalars(seed: u64) {
        let (x, _rng) =
            SeededRand::<_, f64, _>::new(val!(Standard), val!(StdRng::seed_from_u64(seed))).run();

        prop_assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn seededrands_should_generate_vectors(seed: u64, #[strategy(1_usize..10)] len: usize) {
        let (xs, _rng) = SeededRand::<_, f64, _>::new(
            val1!(std::iter::repeat(Standard).take(len).collect::<Vec<_>>()),
            val!(StdRng::seed_from_u64(seed)),
        )
        .run();

        prop_assert_eq!(xs.len(), len);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrands_should_generate_matrices(
        seed: u64,
        #[strategy(1_usize..10)] x_len: usize,
        #[strategy(1_usize..10)] y_len: usize,
    ) {
        let shape = (x_len, y_len);
        let (xs, _rng) = SeededRand::<_, f64, _>::new(
            val2!(Matrix::from_vec(
                (x_len, y_len),
                std::iter::repeat(Standard)
                    .take(x_len * y_len)
                    .collect::<Vec<_>>()
            )
            .unwrap()),
            val!(StdRng::seed_from_u64(seed)),
        )
        .run();

        prop_assert_eq!(xs.shape(), shape);
        let xs = xs.into_inner();
        prop_assert_eq!(xs.len(), shape.0 * shape.1);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrands_should_loop(seed: u64) {
        let (x, _rng) = Zip(val!(0.0), val!(StdRng::seed_from_u64(seed)))
            .loop_while(
                ("x", "rng"),
                SeededRand::<_, f64, _>::new(val!(Standard), arg!("rng", StdRng)),
                arg!("x", f64).gt(val!(0.5)).not(),
            )
            .run();
        prop_assert!(x > 0.5);
    }
}
