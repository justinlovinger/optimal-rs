mod rands {
    use rand::distributions::Distribution;

    use crate::{
        peano::{One, Two, Zero},
        rand::Rand,
        run::{ArgVals, Matrix, RunCore, Value},
        Computation,
    };

    impl<DistComp, T, Dist, Out> RunCore for Rand<DistComp, T>
    where
        DistComp: Computation + RunCore<Output = Value<Dist>>,
        Dist: BroadcastRands<DistComp::Dim, T, Output = Out>,
    {
        type Output = Value<Out>;

        fn run_core(self, args: ArgVals) -> Self::Output {
            Value(self.distribution.run_core(args).0.broadcast())
        }
    }

    trait BroadcastRands<Dim, T> {
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
        run::{ArgVals, DistributeArgs, Matrix, RunCore, Value},
        Computation,
    };

    impl<RComp, DistComp, T, R, Dist, Out> RunCore for SeededRand<RComp, DistComp, T>
    where
        Self: Computation,
        DistComp: Computation,
        (RComp, DistComp): DistributeArgs<Output = (Value<R>, Value<Dist>)>,
        R: Rng,
        Dist: BroadcastSeededRands<DistComp::Dim, T, Output = Out>,
    {
        type Output = (Value<R>, Value<Out>);

        fn run_core(self, args: ArgVals) -> Self::Output {
            let (mut rng, dist) = (self.rng, self.distribution).distribute(args);
            let out = Value(dist.0.broadcast(&mut rng.0));
            (rng, out)
        }
    }

    trait BroadcastSeededRands<Dim, T> {
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
        arg, argvals,
        rand::{Rand, SeededRand},
        run::Matrix,
        val, val1, val2, Computation, Run,
    };

    #[test]
    fn rands_should_generate_scalars() {
        let x = Rand::<_, f64>::new(val!(Standard)).run(argvals![]);

        assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn rands_should_generate_vectors(#[strategy(1_usize..10)] len: usize) {
        let xs = Rand::<_, f64>::new(val1!(std::iter::repeat(Standard)
            .take(len)
            .collect::<Vec<_>>()))
        .run(argvals![]);

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
        .run(argvals![]);

        prop_assert_eq!(xs.shape(), shape);
        let xs = xs.into_inner();
        prop_assert_eq!(xs.len(), shape.0 * shape.1);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrands_should_generate_scalars(seed: u64) {
        let (_rng, x) =
            SeededRand::<_, _, f64>::new(val!(StdRng::seed_from_u64(seed)), val!(Standard))
                .run(argvals![]);

        prop_assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn seededrands_should_generate_vectors(seed: u64, #[strategy(1_usize..10)] len: usize) {
        let (_rng, xs) = SeededRand::<_, _, f64>::new(
            val!(StdRng::seed_from_u64(seed)),
            val1!(std::iter::repeat(Standard).take(len).collect::<Vec<_>>()),
        )
        .run(argvals![]);

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
        let (_rng, xs) = SeededRand::<_, _, f64>::new(
            val!(StdRng::seed_from_u64(seed)),
            val2!(Matrix::from_vec(
                (x_len, y_len),
                std::iter::repeat(Standard)
                    .take(x_len * y_len)
                    .collect::<Vec<_>>()
            )
            .unwrap()),
        )
        .run(argvals![]);

        prop_assert_eq!(xs.shape(), shape);
        let xs = xs.into_inner();
        prop_assert_eq!(xs.len(), shape.0 * shape.1);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrands_should_loop(seed: u64) {
        let (_rng, x) = val!(StdRng::seed_from_u64(seed))
            .zip(val!(0.0))
            .loop_while(
                ("rng", "x"),
                SeededRand::<_, _, f64>::new(arg!("rng", StdRng), val!(Standard)),
                arg!("x", f64).gt(val!(0.5)).not(),
            )
            .run(argvals![]);
        prop_assert!(x > 0.5);
    }
}
