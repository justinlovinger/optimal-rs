use crate::{
    linalg::{FromDiagElem, IdentityMatrix, MatMul, MulCol, MulOut, ScalarProduct},
    math::Mul,
    peano::{One, Two},
    run::{Collect, DistributeArgs, Matrix},
    sum::Sum,
    Computation, NamedArgs, Unwrap, Value,
};

use super::RunCore;

impl<Len, T> RunCore for IdentityMatrix<Len, T>
where
    Self: Computation,
    Len: RunCore<Output = Value<usize>>,
    T: Clone + num_traits::Zero + num_traits::One,
{
    type Output = Value<Matrix<Vec<T>>>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        let len = self.len.run_core(args).unwrap();
        let matrix = ndarray::Array2::from_diag_elem(len, T::one());
        Value(Matrix::from_vec((len, len), matrix.into_raw_vec()).unwrap())
    }
}

impl<Len, Elem, T> RunCore for FromDiagElem<Len, Elem>
where
    Self: Computation,
    (Len, Elem): DistributeArgs<Output = (Value<usize>, Value<T>)>,
    T: Clone + num_traits::Zero,
{
    type Output = Value<Matrix<Vec<T>>>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        let (len, elem) = (self.len, self.elem).distribute(args).unwrap();
        let matrix = ndarray::Array2::from_diag_elem(len, elem);
        Value(Matrix::from_vec((len, len), matrix.into_raw_vec()).unwrap())
    }
}

impl<A, B> RunCore for ScalarProduct<A, B>
where
    Self: Computation,
    Mul<A, B>: Computation,
    Sum<Mul<A, B>>: Computation + RunCore,
{
    type Output = <Sum<Mul<A, B>> as RunCore>::Output;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        Sum(Mul(self.0, self.1)).run_core(args)
    }
}

impl<A, B, OutA, OutB, Elem> RunCore for MatMul<A, B>
where
    Self: Computation,
    (A, B): DistributeArgs<Output = (OutA, OutB)>,
    OutA: Collect<Two, Collected = Value<Matrix<Vec<Elem>>>>,
    OutB: Collect<Two, Collected = Value<Matrix<Vec<Elem>>>>,
    Elem: ndarray::LinalgScalar,
{
    type Output = Value<Matrix<Vec<Elem>>>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        let (xs, ys) = (self.0, self.1).distribute(args).collect().unwrap();
        Value(mat_mul(xs, ys))
    }
}

impl<A, B, OutA, OutB, Elem> RunCore for MulOut<A, B>
where
    Self: Computation,
    (A, B): DistributeArgs<Output = (OutA, OutB)>,
    OutA: Collect<One, Collected = Value<Vec<Elem>>>,
    OutB: Collect<One, Collected = Value<Vec<Elem>>>,
    Elem: ndarray::LinalgScalar,
{
    type Output = Value<Matrix<Vec<Elem>>>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        let (xs, ys) = (self.0, self.1).distribute(args).collect().unwrap();
        // `xs` and `ys` are 1d, so shapes will match its lengths if at least one dimension is `1`.
        Value(mat_mul(
            unsafe { Matrix::new_unchecked((xs.len(), 1), xs) },
            unsafe { Matrix::new_unchecked((1, ys.len()), ys) },
        ))
    }
}

impl<A, B, OutA, OutB, Elem> RunCore for MulCol<A, B>
where
    Self: Computation,
    (A, B): DistributeArgs<Output = (OutA, OutB)>,
    OutA: Collect<Two, Collected = Value<Matrix<Vec<Elem>>>>,
    OutB: Collect<One, Collected = Value<Vec<Elem>>>,
    Elem: ndarray::LinalgScalar,
{
    type Output = Value<Vec<Elem>>;

    fn run_core(self, args: NamedArgs) -> Self::Output {
        let (xs, ys) = (self.0, self.1).distribute(args).collect().unwrap();
        // `ys` is 1d, so the shape will match its length if at least one dimension is `1`.
        let ys = unsafe { Matrix::new_unchecked((ys.len(), 1), ys) };
        Value(mat_mul(xs, ys).into_inner())
    }
}

fn mat_mul<A>(xs: Matrix<Vec<A>>, ys: Matrix<Vec<A>>) -> Matrix<Vec<A>>
where
    A: ndarray::LinalgScalar,
{
    // Matrix multiplication is hard to perform efficiently,
    // so we rely on `ndarray` for now.
    // Theoretically,
    // we may be able to return an iterator
    // if we perform the operation ourselves.
    let out = ndarray::Array::from_shape_vec(xs.shape(), xs.into_inner())
        .unwrap()
        .dot(&ndarray::Array::from_shape_vec(ys.shape(), ys.into_inner()).unwrap());
    let shape = out.shape();
    // We assume `ndarray` got the shape right.
    unsafe { Matrix::new_unchecked((shape[0], shape[1]), out.into_raw_vec()) }
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{linalg::FromDiagElem, named_args, run::Matrix, val, val1, val2, Computation, Run};

    #[proptest]
    fn identity_matrix_should_return_identity_matrix(#[strategy(1_usize..=10)] len: usize) {
        let actual = val!(len).identity_matrix::<i32>().run(named_args![]);
        let expected = ndarray::Array2::from_diag_elem(len, i32::one());
        prop_assert_eq!(actual.shape(), (expected.shape()[0], expected.shape()[1]));
        prop_assert_eq!(actual.into_inner(), expected.into_raw_vec());
    }

    #[proptest]
    fn from_diag_elem_should_return_square_matrix_with_elem_at_diagonal(
        #[strategy(1_usize..=10)] len: usize,
        elem: i32,
    ) {
        let actual = FromDiagElem::new(val!(len), val!(elem)).run(named_args![]);
        let expected = ndarray::Array2::from_diag_elem(len, elem);
        prop_assert_eq!(actual.shape(), (expected.shape()[0], expected.shape()[1]));
        prop_assert_eq!(actual.into_inner(), expected.into_raw_vec());
    }

    #[proptest]
    fn scalar_product_should_mul_then_sum(
        #[strategy(-1000..1000)] x1: i32,
        #[strategy(-1000..1000)] x2: i32,
        #[strategy(-1000..1000)] y1: i32,
        #[strategy(-1000..1000)] y2: i32,
    ) {
        let lhs = val1!([x1, x2]);
        let rhs = val1!([y1, y2]);
        prop_assert_eq!(
            lhs.scalar_product(rhs).run(named_args![]),
            lhs.mul(rhs).sum().run(named_args![])
        );
    }

    #[proptest]
    fn mat_mul_should_perform_matrix_multiplication(
        #[strategy(-1000..1000)] x11: i32,
        #[strategy(-1000..1000)] x21: i32,
        #[strategy(-1000..1000)] x12: i32,
        #[strategy(-1000..1000)] x22: i32,
        #[strategy(-1000..1000)] y11: i32,
        #[strategy(-1000..1000)] y21: i32,
        #[strategy(-1000..1000)] y12: i32,
        #[strategy(-1000..1000)] y22: i32,
        #[strategy(-1000..1000)] y13: i32,
        #[strategy(-1000..1000)] y23: i32,
    ) {
        let lhs_m = Matrix::from_vec((2, 2), vec![x11, x21, x12, x22]).unwrap();
        let rhs_m = Matrix::from_vec((2, 3), vec![y11, y21, y12, y22, y13, y23]).unwrap();
        prop_assert_eq!(
            val2!(lhs_m.clone())
                .mat_mul(val2!(rhs_m.clone()))
                .run(named_args![])
                .into_inner(),
            ndarray::Array::from_shape_vec(lhs_m.shape(), lhs_m.into_inner())
                .unwrap()
                .dot(&ndarray::Array::from_shape_vec(rhs_m.shape(), rhs_m.into_inner()).unwrap())
                .into_raw_vec()
        );
    }

    #[proptest]
    fn mul_out_should_equal_col_matrix_mat_mul_row_matrix(
        #[strategy(-1000..1000)] x1: i32,
        #[strategy(-1000..1000)] x2: i32,
        #[strategy(-1000..1000)] y1: i32,
        #[strategy(-1000..1000)] y2: i32,
        #[strategy(-1000..1000)] y3: i32,
    ) {
        let out = val1!([x1, x2])
            .mul_out(val1!([y1, y2, y3]))
            .run(named_args![]);
        prop_assert_eq!(out.shape(), (2, 3));
        prop_assert_eq!(
            out,
            val2!(Matrix::from_vec((2, 1), vec![x1, x2]).unwrap())
                .mat_mul(val2!(Matrix::from_vec((1, 3), vec![y1, y2, y3]).unwrap()))
                .run(named_args![])
        );
    }

    #[proptest]
    fn mul_col_should_equal_mat_mul_by_col_matrix(
        #[strategy(-1000..1000)] x11: i32,
        #[strategy(-1000..1000)] x21: i32,
        #[strategy(-1000..1000)] x31: i32,
        #[strategy(-1000..1000)] x12: i32,
        #[strategy(-1000..1000)] x22: i32,
        #[strategy(-1000..1000)] x32: i32,
        #[strategy(-1000..1000)] y1: i32,
        #[strategy(-1000..1000)] y2: i32,
    ) {
        let lhs = val2!(Matrix::from_vec((3, 2), vec![x11, x21, x31, x12, x22, x32]).unwrap());
        let out = lhs.clone().mul_col(val1!([y1, y2])).run(named_args![]);
        prop_assert_eq!(out.len(), 3);
        prop_assert_eq!(
            out,
            lhs.mat_mul(val2!(Matrix::from_vec((2, 1), vec![y1, y2]).unwrap()))
                .run(named_args![])
                .into_inner()
        );
    }
}
