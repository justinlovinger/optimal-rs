use core::fmt;

use computation_types::{impl_core_ops, peano::Zero, Computation, ComputationFn, NamedArgs, Names};

use super::{AdjustRate, Probability};

/// Adjust each probability towards corresponding `sample` bit
/// at `rate`.
#[derive(Clone, Copy, Debug)]
pub struct Adjust<R, P, B>
where
    Self: Computation,
{
    /// Computation representing [`AdjustRate`].
    pub rate: R,
    /// Computation representing probability to adjust.
    pub probability: P,
    /// Computation representing sample to adjust towards.
    pub sample: B,
}

impl<R, P, B> Adjust<R, P, B>
where
    Self: Computation,
{
    #[allow(missing_docs)]
    pub fn new(rate: R, probability: P, sample: B) -> Self {
        Self {
            rate,
            probability,
            sample,
        }
    }
}

impl<R, P, B> Computation for Adjust<R, P, B>
where
    R: Computation<Dim = Zero, Item = AdjustRate>,
    P: Computation<Item = Probability>,
    B: Computation<Dim = P::Dim, Item = bool>,
{
    type Dim = P::Dim;
    type Item = Probability;
}

impl<R, P, B> ComputationFn for Adjust<R, P, B>
where
    Self: Computation,
    R: ComputationFn,
    P: ComputationFn,
    B: ComputationFn,
    Adjust<R::Filled, P::Filled, B::Filled>: Computation,
{
    type Filled = Adjust<R::Filled, P::Filled, B::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        let (args_0, args_1, args_2) = named_args
            .partition3(
                &self.rate.arg_names(),
                &self.probability.arg_names(),
                &self.sample.arg_names(),
            )
            .unwrap_or_else(|e| panic!("{}", e,));
        Adjust {
            rate: self.rate.fill(args_0),
            probability: self.probability.fill(args_1),
            sample: self.sample.fill(args_2),
        }
    }

    fn arg_names(&self) -> Names {
        self.rate
            .arg_names()
            .union(self.probability.arg_names())
            .union(self.sample.arg_names())
    }
}

impl_core_ops!(Adjust<R, P, B>);

impl<R, P, B> fmt::Display for Adjust<R, P, B>
where
    Self: Computation,
    R: fmt::Display,
    P: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "adjust({}, {}, {})",
            self.rate, self.probability, self.sample
        )
    }
}

mod run {
    use computation_types::{
        peano::{One, Two, Zero},
        run::{Matrix, RunCore},
        Computation, Unwrap, Value,
    };

    use crate::low_level::{adjust, AdjustRate, Probability};

    use super::Adjust;

    impl<R, P, B, OutP, OutB> RunCore for Adjust<R, P, B>
    where
        Self: Computation,
        R: Computation + RunCore<Output = Value<AdjustRate>>,
        P: Computation + RunCore<Output = Value<OutP>>,
        B: Computation + RunCore<Output = Value<OutB>>,
        OutP: BroadcastAdjust<OutB, P::Dim, B::Dim>,
    {
        type Output = Value<OutP::Output>;

        fn run_core(self) -> Self::Output {
            Value(self.probability.run_core().unwrap().broadcast_adjust(
                self.rate.run_core().unwrap(),
                self.sample.run_core().unwrap(),
            ))
        }
    }

    pub trait BroadcastAdjust<Rhs, LhsDim, RhsDim> {
        type Output;

        fn broadcast_adjust(self, rate: AdjustRate, rhs: Rhs) -> Self::Output;
    }

    impl BroadcastAdjust<bool, Zero, Zero> for Probability {
        type Output = Probability;

        fn broadcast_adjust(self, rate: AdjustRate, rhs: bool) -> Self::Output {
            adjust_probability(rate, self, rhs)
        }
    }

    impl<Lhs, Rhs> BroadcastAdjust<Rhs, One, One> for Lhs
    where
        Lhs: IntoIterator<Item = Probability>,
        Rhs: IntoIterator<Item = bool>,
    {
        type Output = std::iter::Map<
            std::iter::Zip<
                std::iter::Zip<std::iter::Repeat<AdjustRate>, Lhs::IntoIter>,
                Rhs::IntoIter,
            >,
            fn(((AdjustRate, Lhs::Item), Rhs::Item)) -> Probability,
        >;

        fn broadcast_adjust(self, rate: AdjustRate, rhs: Rhs) -> Self::Output {
            std::iter::repeat(rate)
                .zip(self)
                .zip(rhs)
                .map(|((rate, p), b)| adjust_probability(rate, p, b))
        }
    }

    impl<Lhs, Rhs> BroadcastAdjust<Matrix<Rhs>, Two, Two> for Matrix<Lhs>
    where
        Lhs: IntoIterator<Item = Probability>,
        Rhs: IntoIterator<Item = bool>,
    {
        type Output = Matrix<
            std::iter::Map<
                std::iter::Zip<
                    std::iter::Zip<std::iter::Repeat<AdjustRate>, Lhs::IntoIter>,
                    Rhs::IntoIter,
                >,
                fn(((AdjustRate, Lhs::Item), Rhs::Item)) -> Probability,
            >,
        >;

        fn broadcast_adjust(self, rate: AdjustRate, rhs: Matrix<Rhs>) -> Self::Output {
            debug_assert_eq!(self.shape(), rhs.shape());
            // Assuming the above assert passes,
            // neither shape nor the length of `inner` will change,
            // so they should still be fine.
            unsafe {
                Matrix::new_unchecked(
                    self.shape(),
                    std::iter::repeat(rate)
                        .zip(self.into_inner())
                        .zip(rhs)
                        .map(|((rate, p), b)| adjust_probability(rate, p, b)),
                )
            }
        }
    }

    /// Adjust a probability from `p` to `b`
    /// at given rate.
    fn adjust_probability(rate: AdjustRate, p: Probability, b: bool) -> Probability {
        // This operation is safe
        // because `Probability` is closed under `adjust`
        // with rate in [0,1].
        unsafe { Probability::new_unchecked(adjust(rate.into(), p.into(), b as u8 as f64)) }
    }
}
