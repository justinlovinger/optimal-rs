use core::ops;

use paste::paste;

use crate::{
    math::*,
    peano::{One, Two, Zero},
    run::{ArgVals, DistributeArgs, Matrix, RunCore, Unwrap, Value},
    Computation,
};

macro_rules! impl_run_core_for_binary_op {
    ( $op:ident ) => {
        impl_run_core_for_binary_op!($op, ops);
    };
    ( $op:ident, $package:ident ) => {
        paste! {
            impl<A, B, OutA, OutB> RunCore for $op<A, B>
            where
                A: Computation,
                B: Computation,
                (A, B): DistributeArgs<Output = (Value<OutA>, Value<OutB>)>,
                OutA: [<Broadcast $op>]<OutB, A::Dim, B::Dim>
            {
                type Output = Value<OutA::Output>;

                fn run_core(self, args: ArgVals) -> Self::Output {
                    let (x, y) = (self.0, self.1).distribute(args).unwrap();
                    Value(x.[<broadcast_ $op:lower>](y))
                }
            }

            pub trait [<Broadcast $op>]<Rhs, LhsDim, RhsDim> {
                type Output;

                fn [<broadcast_ $op:lower>](self, rhs: Rhs) -> Self::Output;
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Rhs, Zero, Zero> for Lhs
            where
                Lhs: $package::$op<Rhs>,
            {
                type Output = Lhs::Output;

                fn [<broadcast_ $op:lower>](self, rhs: Rhs) -> Self::Output {
                    $package::$op::[<$op:lower>](self, rhs)
                }
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Rhs, One, One> for Lhs
            where
                Lhs: IntoIterator,
                Lhs::Item: $package::$op<Rhs::Item>,
                Rhs: IntoIterator,
            {
                type Output = std::iter::Map<std::iter::Zip<Lhs::IntoIter, Rhs::IntoIter>, fn((Lhs::Item, Rhs::Item)) -> <Lhs::Item as $package::$op<Rhs::Item>>::Output>;

                fn [<broadcast_ $op:lower>](self, rhs: Rhs) -> Self::Output {
                    self.into_iter().zip(rhs).map(|(x, y)| $package::$op::[<$op:lower>](x, y))
                }
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Matrix<Rhs>, Two, Two> for Matrix<Lhs>
            where
                Lhs: IntoIterator,
                Lhs::Item: $package::$op<Rhs::Item>,
                Rhs: IntoIterator,
            {
                type Output = Matrix<std::iter::Map<std::iter::Zip<Lhs::IntoIter, Rhs::IntoIter>, fn((Lhs::Item, Rhs::Item)) -> <Lhs::Item as $package::$op<Rhs::Item>>::Output>>;

                fn [<broadcast_ $op:lower>](self, rhs: Matrix<Rhs>) -> Self::Output {
                    debug_assert_eq!(self.shape(), rhs.shape());
                    // Assuming the above assert passes,
                    // neither shape nor the length of `inner` will change,
                    // so they should still be fine.
                    unsafe {
                        Matrix::new_unchecked(
                            self.shape(),
                            self.into_inner()
                                .into_iter()
                                .zip(rhs.into_inner())
                                .map(|(x, y)| $package::$op::[<$op:lower>](x, y)),
                        )
                    }
                }
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Rhs, One, Zero> for Lhs
            where
                Lhs: IntoIterator,
                Lhs::Item: $package::$op<Rhs>,
                Rhs: Clone,
            {
                type Output = std::iter::Map<std::iter::Zip<Lhs::IntoIter, std::iter::Repeat<Rhs>>, fn((Lhs::Item, Rhs)) -> <Lhs::Item as $package::$op<Rhs>>::Output>;

                fn [<broadcast_ $op:lower>](self, rhs: Rhs) -> Self::Output {
                    self.into_iter().zip(std::iter::repeat(rhs)).map(|(x, y)| $package::$op::[<$op:lower>](x, y))
                }
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Rhs, Zero, One> for Lhs
            where
                Lhs: Clone + $package::$op<Rhs::Item>,
                Rhs: IntoIterator,
            {
                type Output = std::iter::Map<std::iter::Zip<std::iter::Repeat<Lhs>, Rhs::IntoIter>, fn((Lhs, Rhs::Item)) -> <Lhs as $package::$op<Rhs::Item>>::Output>;

                fn [<broadcast_ $op:lower>](self, rhs: Rhs) -> Self::Output {
                    std::iter::repeat(self).zip(rhs).map(|(x, y)| $package::$op::[<$op:lower>](x, y))
                }
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Rhs, Two, Zero> for Matrix<Lhs>
            where
                Lhs: IntoIterator,
                Lhs::Item: $package::$op<Rhs>,
                Rhs: Clone,
            {
                type Output = Matrix<std::iter::Map<std::iter::Zip<Lhs::IntoIter, std::iter::Repeat<Rhs>>, fn((Lhs::Item, Rhs)) -> <Lhs::Item as $package::$op<Rhs>>::Output>>;

                fn [<broadcast_ $op:lower>](self, rhs: Rhs) -> Self::Output {
                    // Neither shape nor the length of `inner` will change,
                    // so they should still be fine.
                    unsafe {
                        Matrix::new_unchecked(
                            self.shape(),
                            self.into_inner()
                                .into_iter()
                                .zip(std::iter::repeat(rhs))
                                .map(|(x, y)| $package::$op::[<$op:lower>](x, y)),
                        )
                    }
                }
            }

            impl<Lhs, Rhs> [<Broadcast $op>]<Matrix<Rhs>, Zero, Two> for Lhs
            where
                Lhs: Clone + $package::$op<Rhs::Item>,
                Rhs: IntoIterator,
            {
                type Output = Matrix<std::iter::Map<std::iter::Zip<std::iter::Repeat<Lhs>, Rhs::IntoIter>, fn((Lhs, Rhs::Item)) -> <Lhs as $package::$op<Rhs::Item>>::Output>>;

                fn [<broadcast_ $op:lower>](self, rhs: Matrix<Rhs>) -> Self::Output {
                    // Neither shape nor the length of `inner` will change,
                    // so they should still be fine.
                    unsafe {
                        Matrix::new_unchecked(
                            rhs.shape(),
                            std::iter::repeat(self)
                                .zip(rhs.into_inner())
                                .map(|(x, y)| $package::$op::[<$op:lower>](x, y)),
                        )
                    }
                }
            }
        }
    };
}

macro_rules! impl_run_for_unary_op {
    ( $op:ident ) => {
        impl_run_for_unary_op!($op, ops);
    };
    ( $op:ident, $package:ident ) => {
        paste! {
            impl<A, Out> RunCore for $op<A>
            where
                A: Computation + RunCore<Output = Value<Out>>,
                Out: [<Broadcast $op>]<A::Dim>
            {
                type Output = Value<<Out as [<Broadcast $op>]<A::Dim>>::Output>;

                fn run_core(self, args: ArgVals) -> Self::Output {
                    Value(self.0.run_core(args).unwrap().[<broadcast_ $op:lower>]())
                }
            }

            pub trait [<Broadcast $op>]<LhsDim> {
                type Output;

                fn [<broadcast_ $op:lower>](self) -> Self::Output;
            }

            impl<Lhs> [<Broadcast $op>]<Zero> for Lhs
            where
                Lhs: $package::$op,
            {
                type Output = Lhs::Output;

                fn [<broadcast_ $op:lower>](self) -> Self::Output {
                    $package::$op::[<$op:lower>](self)
                }
            }

            impl<Lhs> [<Broadcast $op>]<One> for Lhs
            where
                Lhs: IntoIterator,
                Lhs::Item: $package::$op,
            {
                type Output = std::iter::Map<Lhs::IntoIter, fn(Lhs::Item) -> <Lhs::Item as $package::$op>::Output>;

                fn [<broadcast_ $op:lower>](self) -> Self::Output {
                    self.into_iter().map($package::$op::[<$op:lower>])
                }
            }

            impl<Lhs> [<Broadcast $op>]<Two> for Matrix<Lhs>
            where
                Lhs: IntoIterator,
                Lhs::Item: $package::$op,
            {
                type Output = Matrix<std::iter::Map<Lhs::IntoIter, fn(Lhs::Item) -> <Lhs::Item as $package::$op>::Output>>;

                fn [<broadcast_ $op:lower>](self) -> Self::Output {
                    // Neither shape nor the length of `inner` will change,
                    // so they should still be fine.
                    unsafe {
                        Matrix::new_unchecked(
                            self.shape(),
                            self.into_inner()
                                .into_iter()
                                .map($package::$op::[<$op:lower>])
                        )
                    }
                }
            }
        }
    };
}

impl_run_core_for_binary_op!(Add);
impl_run_core_for_binary_op!(Sub);
impl_run_core_for_binary_op!(Mul);
impl_run_core_for_binary_op!(Div);
impl_run_core_for_binary_op!(Pow, num_traits);
impl_run_for_unary_op!(Neg);

mod abs {
    use num_traits::Signed;

    use super::*;

    impl<A, Out> RunCore for Abs<A>
    where
        A: Computation + RunCore<Output = Value<Out>>,
        Out: BroadcastAbs<A::Dim>,
    {
        type Output = Value<<Out as BroadcastAbs<A::Dim>>::Output>;

        fn run_core(self, args: ArgVals) -> Self::Output {
            Value(self.0.run_core(args).unwrap().broadcast_abs())
        }
    }

    pub trait BroadcastAbs<LhsDim> {
        type Output;

        fn broadcast_abs(self) -> Self::Output;
    }

    impl<Lhs> BroadcastAbs<Zero> for Lhs
    where
        Lhs: Signed,
    {
        type Output = Lhs;

        fn broadcast_abs(self) -> Self::Output {
            self.abs()
        }
    }

    impl<Lhs> BroadcastAbs<One> for Lhs
    where
        Lhs: IntoIterator,
        Lhs::Item: Signed,
    {
        type Output = std::iter::Map<Lhs::IntoIter, fn(Lhs::Item) -> Lhs::Item>;

        fn broadcast_abs(self) -> Self::Output {
            self.into_iter().map(|x| x.abs())
        }
    }

    impl<Lhs> BroadcastAbs<Two> for Matrix<Lhs>
    where
        Lhs: IntoIterator,
        Lhs::Item: Signed,
    {
        type Output = Matrix<std::iter::Map<Lhs::IntoIter, fn(Lhs::Item) -> Lhs::Item>>;

        fn broadcast_abs(self) -> Self::Output {
            // Neither shape nor the length of `inner` will change,
            // so they should still be fine.
            unsafe {
                Matrix::new_unchecked(self.shape(), self.into_inner().into_iter().map(|x| x.abs()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::{Add, Div, Mul, Neg, Sub};

    use paste::paste;
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{argvals, run::Matrix, val, val1, val2, Computation, Run};

    macro_rules! test_binary_op {
        ( $op:ident ) => {
            test_binary_op!($op, (-1000..1000), i32);
        };
        ( $op:ident, $range:expr, $ty:ty ) => {
            paste! {
                #[proptest]
                fn [<$op _should_ $op _scalars>](
                    #[strategy($range)] x: $ty,
                    #[strategy($range)] y: $ty,
                ) {
                    prop_assert_eq!(Computation::$op(val!(x), (val!(y))).run(argvals![]), x.$op(y));
                }

                #[proptest]
                fn [<$op _should_ $op _vectors>](
                    #[strategy($range)] x1: $ty,
                    #[strategy($range)] x2: $ty,
                    #[strategy($range)] y1: $ty,
                    #[strategy($range)] y2: $ty,
                ) {
                    prop_assert_eq!(
                        Computation::$op(val1!([x1, x2]), val1!([y1, y2])).run(argvals![]),
                        [x1.$op(y1), x2.$op(y2)]
                    );
                }

                #[proptest]
                fn [<$op _should_ $op _matrices>](
                    #[strategy($range)] x1: $ty,
                    #[strategy($range)] x2: $ty,
                    #[strategy($range)] x3: $ty,
                    #[strategy($range)] x4: $ty,
                    #[strategy($range)] y1: $ty,
                    #[strategy($range)] y2: $ty,
                    #[strategy($range)] y3: $ty,
                    #[strategy($range)] y4: $ty,
                ) {
                    prop_assert_eq!(
                        Computation::$op(
                            val2!(Matrix::from_vec((2, 2), vec![x1, x2, x3, x4]).unwrap()),
                            val2!(Matrix::from_vec((2, 2), vec![y1, y2, y3, y4]).unwrap()),
                        ).run(argvals![]),
                        Matrix::from_vec((2, 2), vec![x1.$op(y1), x2.$op(y2), x3.$op(y3), x4.$op(y4)]).unwrap()
                    );
                }

                #[proptest]
                fn [<$op _should_broadcast_scalars_to_vectors>](
                    #[strategy($range)] x: $ty,
                    #[strategy($range)] y: $ty,
                    #[strategy($range)] z: $ty,
                ) {
                    prop_assert_eq!(
                        Computation::$op(val1!([x, y]), val!(z)).run(argvals![]),
                        [x.$op(z), y.$op(z)]
                    );
                    prop_assert_eq!(
                        Computation::$op(val!(x), val1!([y, z])).run(argvals![]),
                        [x.$op(y), x.$op(z)]
                    );
                }

                #[proptest]
                fn [<$op _should_broadcast_scalars_to_matrices>](
                    #[strategy($range)] x: $ty,
                    #[strategy($range)] y: $ty,
                    #[strategy($range)] z: $ty,
                    #[strategy($range)] q: $ty,
                    #[strategy($range)] r: $ty,
                ) {
                    prop_assert_eq!(
                        Computation::$op(
                            val2!(Matrix::from_vec((2, 2), vec![x, y, z, q]).unwrap()),
                            val!(r),
                        ).run(argvals![]),
                        Matrix::from_vec((2, 2), vec![x.$op(r), y.$op(r), z.$op(r), q.$op(r)]).unwrap()
                    );
                    prop_assert_eq!(
                        Computation::$op(
                            val!(x),
                            val2!(Matrix::from_vec((2, 2), vec![y, z, q, r]).unwrap()),
                        ).run(argvals![]),
                        Matrix::from_vec((2, 2), vec![x.$op(y), x.$op(z), x.$op(q), x.$op(r)]).unwrap()
                    );
                }
            }
        };
    }

    macro_rules! test_unary_op {
        ( $op:ident ) => {
            test_unary_op!($op, (-1000..1000), i32);
        };
        ( $op:ident, $range:expr, $ty:ty ) => {
            paste! {
                #[proptest]
                fn [<$op _should_ $op _scalars>](#[strategy($range)] x: $ty) {
                    prop_assert_eq!(Computation::$op(val!(x)).run(argvals![]), x.$op());
                }

                #[proptest]
                fn [<$op _should_ $op _vectors>](
                    #[strategy($range)] x1: $ty,
                    #[strategy($range)] x2: $ty,
                ) {
                    prop_assert_eq!(
                        Computation::$op(val1!([x1, x2])).run(argvals![]),
                        [x1.$op(), x2.$op()]
                    );
                }

                #[proptest]
                fn [<$op _should_ $op _matrices>](
                    #[strategy($range)] x1: $ty,
                    #[strategy($range)] x2: $ty,
                    #[strategy($range)] x3: $ty,
                    #[strategy($range)] x4: $ty,
                ) {
                    prop_assert_eq!(
                        Computation::$op(val2!(Matrix::from_vec((2, 2), vec![x1, x2, x3, x4]).unwrap())).run(argvals![]),
                        Matrix::from_vec((2, 2), vec![x1.$op(), x2.$op(), x3.$op(), x4.$op()]).unwrap()
                    );
                }
            }
        };
    }

    test_binary_op!(add);
    test_binary_op!(sub);
    test_binary_op!(mul);
    test_binary_op!(div, (1_u32..1000), u32);
    test_binary_op!(pow, (0_u32..10), u32);
    test_unary_op!(neg);
    test_unary_op!(abs);
}
