#[allow(clippy::module_inception)]
mod rand {
    use rand::distributions::Distribution;

    use crate::{
        rand::Rand,
        run::{ArgVals, Matrix, RunCore, Value},
    };

    impl<Dist, T> RunCore for Rand<(), Dist, T>
    where
        Dist: Distribution<T>,
    {
        type Output = Value<T>;

        fn run_core(self, _args: ArgVals) -> Self::Output {
            Value(self.distribution.sample(&mut rand::thread_rng()))
        }
    }

    impl<Dist, T> RunCore for Rand<(usize,), Dist, T>
    where
        Dist: Distribution<T>,
    {
        type Output =
            Value<std::iter::Take<rand::distributions::DistIter<Dist, rand::rngs::ThreadRng, T>>>;

        fn run_core(self, _args: ArgVals) -> Self::Output {
            Value(
                self.distribution
                    .sample_iter(rand::thread_rng())
                    .take(self.shape.0),
            )
        }
    }

    impl<Dist, T> RunCore for Rand<(usize, usize), Dist, T>
    where
        Dist: Distribution<T>,
    {
        type Output = Value<
            Matrix<std::iter::Take<rand::distributions::DistIter<Dist, rand::rngs::ThreadRng, T>>>,
        >;

        fn run_core(self, _args: ArgVals) -> Self::Output {
            // We `take` the right number of elements for the shape,
            // so the resulting `Matrix` should be correct.
            Value(unsafe {
                Matrix::new_unchecked(
                    self.shape,
                    self.distribution
                        .sample_iter(rand::thread_rng())
                        .take(self.shape.0 * self.shape.1),
                )
            })
        }
    }
}

mod seeded_rand {
    use rand::{distributions::Distribution, Rng};

    use crate::{
        rand::SeededRand,
        run::{ArgVals, Matrix, RunCore, Value},
        Computation,
    };

    impl<RComp, Dist, T, R> RunCore for SeededRand<RComp, (), Dist, T>
    where
        Self: Computation,
        RComp: RunCore<Output = Value<R>>,
        R: Rng,
        Dist: Distribution<T>,
    {
        type Output = (Value<R>, Value<T>);

        fn run_core(self, args: ArgVals) -> Self::Output {
            let mut rng = self.rng_comp.run_core(args);
            let out = Value(self.distribution.sample(&mut rng.0));
            (rng, out)
        }
    }

    impl<RComp, Dist, T, R> RunCore for SeededRand<RComp, (usize,), Dist, T>
    where
        Self: Computation,
        RComp: RunCore<Output = Value<R>>,
        R: Rng,
        Dist: Distribution<T>,
    {
        type Output = (Value<R>, Value<Vec<T>>);

        fn run_core(self, args: ArgVals) -> Self::Output {
            let mut rng = self.rng_comp.run_core(args);
            let out = Value(
                self.distribution
                    .sample_iter(&mut rng.0)
                    .take(self.shape.0)
                    .collect(),
            );
            (rng, out)
        }
    }

    impl<RComp, Dist, T, R> RunCore for SeededRand<RComp, (usize, usize), Dist, T>
    where
        Self: Computation,
        RComp: RunCore<Output = Value<R>>,
        R: Rng,
        Dist: Distribution<T>,
    {
        type Output = (Value<R>, Value<Matrix<Vec<T>>>);

        fn run_core(self, args: ArgVals) -> Self::Output {
            let mut rng = self.rng_comp.run_core(args);
            // We `take` the right number of elements for the shape,
            // so the resulting `Matrix` should be correct.
            let out = Value(unsafe {
                Matrix::new_unchecked(
                    self.shape,
                    self.distribution
                        .sample_iter(&mut rng.0)
                        .take(self.shape.0 * self.shape.1)
                        .collect(),
                )
            });
            (rng, out)
        }
    }
}

mod rands {
    use rand::distributions::Distribution;

    use crate::{
        peano::{One, Two, Zero},
        rand::Rands,
        run::{ArgVals, Matrix, RunCore, Value},
        Computation,
    };

    impl<DistComp, T, Dist, Out> RunCore for Rands<DistComp, T>
    where
        DistComp: Computation + RunCore<Output = Value<Dist>>,
        Dist: BroadcastRands<DistComp::Dim, T, Output = Out>,
    {
        type Output = Value<Out>;

        fn run_core(self, args: ArgVals) -> Self::Output {
            Value(self.distribution_comp.run_core(args).0.broadcast())
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
        rand::SeededRands,
        run::{ArgVals, DistributeArgs, Matrix, RunCore, Value},
        Computation,
    };

    impl<RComp, DistComp, T, R, Dist, Out> RunCore for SeededRands<RComp, DistComp, T>
    where
        Self: Computation,
        DistComp: Computation,
        (RComp, DistComp): DistributeArgs<Output = (Value<R>, Value<Dist>)>,
        R: Rng,
        Dist: BroadcastSeededRands<DistComp::Dim, T, Output = Out>,
    {
        type Output = (Value<R>, Value<Out>);

        fn run_core(self, args: ArgVals) -> Self::Output {
            let (mut rng, dist) = (self.rng_comp, self.distribution_comp).distribute(args);
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
        rand::{Rand, Rands, SeededRand, SeededRands},
        run::Matrix,
        val, val1, val2, Computation, Run,
    };

    #[test]
    fn rand_should_generate_scalars() {
        let x = Rand::<_, _, f64>::new((), Standard).run(argvals![]);

        assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn rand_should_generate_vectors(#[strategy(1_usize..10)] len: usize) {
        let xs = Rand::<_, _, f64>::new((len,), Standard).run(argvals![]);

        prop_assert_eq!(xs.len(), len);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn rand_should_generate_matrices(
        #[strategy(1_usize..10)] x_len: usize,
        #[strategy(1_usize..10)] y_len: usize,
    ) {
        let shape = (x_len, y_len);
        let xs = Rand::<_, _, f64>::new(shape, Standard).run(argvals![]);

        prop_assert_eq!(xs.shape(), shape);
        let xs = xs.into_inner();
        prop_assert_eq!(xs.len(), shape.0 * shape.1);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrand_should_generate_scalars(seed: u64) {
        let (_rng, x) =
            SeededRand::<_, _, _, f64>::new(val!(StdRng::seed_from_u64(seed)), (), Standard)
                .run(argvals![]);

        prop_assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn seededrand_should_generate_vectors(seed: u64, #[strategy(1_usize..10)] len: usize) {
        let (_rng, xs) =
            SeededRand::<_, _, _, f64>::new(val!(StdRng::seed_from_u64(seed)), (len,), Standard)
                .run(argvals![]);

        prop_assert_eq!(xs.len(), len);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrand_should_generate_matrices(
        seed: u64,
        #[strategy(1_usize..10)] x_len: usize,
        #[strategy(1_usize..10)] y_len: usize,
    ) {
        let shape = (x_len, y_len);
        let (_rng, xs) =
            SeededRand::<_, _, _, f64>::new(val!(StdRng::seed_from_u64(seed)), shape, Standard)
                .run(argvals![]);

        prop_assert_eq!(xs.shape(), shape);
        let xs = xs.into_inner();
        prop_assert_eq!(xs.len(), shape.0 * shape.1);
        for x in xs {
            prop_assert!((0.0..1.0).contains(&x));
        }
    }

    #[proptest]
    fn seededrand_should_loop(seed: u64) {
        let (_rng, x) = val!(StdRng::seed_from_u64(seed))
            .zip(val!(0.0))
            .loop_while(
                ("rng", "x"),
                SeededRand::<_, _, _, f64>::new(arg!("rng", StdRng), (), Standard),
                arg!("x", f64).gt(val!(0.5)).not(),
            )
            .run(argvals![]);
        prop_assert!(x > 0.5);
    }

    #[test]
    fn rands_should_generate_scalars() {
        let x = Rands::<_, f64>::new(val!(Standard)).run(argvals![]);

        assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn rands_should_generate_vectors(#[strategy(1_usize..10)] len: usize) {
        let xs = Rands::<_, f64>::new(val1!(std::iter::repeat(Standard)
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
        let xs = Rands::<_, f64>::new(val2!(Matrix::from_vec(
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
            SeededRands::<_, _, f64>::new(val!(StdRng::seed_from_u64(seed)), val!(Standard))
                .run(argvals![]);

        prop_assert!((0.0..1.0).contains(&x));
    }

    #[proptest]
    fn seededrands_should_generate_vectors(seed: u64, #[strategy(1_usize..10)] len: usize) {
        let (_rng, xs) = SeededRands::<_, _, f64>::new(
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
        let (_rng, xs) = SeededRands::<_, _, f64>::new(
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
                SeededRands::<_, _, f64>::new(arg!("rng", StdRng), val!(Standard)),
                arg!("x", f64).gt(val!(0.5)).not(),
            )
            .run(argvals![]);
        prop_assert!(x > 0.5);
    }
}
