use core::{fmt, ops};
use std::marker::PhantomData;

use crate::{
    impl_core_ops,
    peano::{One, Two, Zero},
    Computation, ComputationFn,
};

/// See [`Computation::identity_matrix`].
#[derive(Clone, Copy, Debug)]
pub struct IdentityMatrix<A, T> {
    pub(crate) inner: A,
    pub(crate) ty: PhantomData<T>,
}

impl<A, T> Computation for IdentityMatrix<A, T>
where
    A: Computation<Dim = Zero, Item = usize>,
{
    type Dim = Two;
    type Item = T;
}

impl<A, T> ComputationFn for IdentityMatrix<A, T>
where
    Self: Computation,
    A: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.inner.args()
    }
}

impl_core_ops!(IdentityMatrix<A, T>);

impl<A, T> fmt::Display for IdentityMatrix<A, T>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "identity_matrix({})", self.inner)
    }
}

/// A computation representing a diagonal matrix
/// from an element.
#[derive(Clone, Copy, Debug)]
pub struct FromDiagElem<Len, Elem> {
    pub(crate) len: Len,
    pub(crate) elem: Elem,
}

impl<Len, Elem> FromDiagElem<Len, Elem> {
    #[allow(missing_docs)]
    pub fn new(len: Len, elem: Elem) -> Self
    where
        Self: Computation,
    {
        Self { len, elem }
    }
}

impl<Len, Elem> Computation for FromDiagElem<Len, Elem>
where
    Len: Computation<Dim = Zero, Item = usize>,
    Elem: Computation<Dim = Zero>,
{
    type Dim = Two;
    type Item = Elem::Item;
}

impl<Len, Elem> ComputationFn for FromDiagElem<Len, Elem>
where
    Self: Computation,
    Len: ComputationFn,
    Elem: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.len.args().union(self.elem.args())
    }
}

impl_core_ops!(FromDiagElem<Len, Elem>);

impl<Len, Elem> fmt::Display for FromDiagElem<Len, Elem>
where
    Len: fmt::Display,
    Elem: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "from_diag_elem({}, {})", self.len, self.elem)
    }
}

/// See [`Computation::scalar_product`].
#[derive(Clone, Copy, Debug)]
pub struct ScalarProduct<A, B>(pub(crate) A, pub(crate) B);

impl<A, B> Computation for ScalarProduct<A, B>
where
    A: Computation<Dim = One>,
    B: Computation<Dim = One>,
    A::Item: ops::Mul<B::Item>,
    <A::Item as ops::Mul<B::Item>>::Output: ops::Add,
{
    type Dim = Zero;
    type Item = <<A::Item as ops::Mul<B::Item>>::Output as ops::Add>::Output;
}

impl<A, B> ComputationFn for ScalarProduct<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.0.args().union(self.1.args())
    }
}

impl_core_ops!(ScalarProduct<A, B>);

/// See [`Computation::mat_mul`].
#[derive(Clone, Copy, Debug)]
pub struct MatMul<A, B>(pub(crate) A, pub(crate) B);

impl<A, B> Computation for MatMul<A, B>
where
    A: Computation<Dim = Two>,
    B: Computation<Dim = Two>,
    A::Item: ops::Mul<B::Item>,
    <A::Item as ops::Mul<B::Item>>::Output: ops::Add,
{
    type Dim = Two;
    type Item = <<A::Item as ops::Mul<B::Item>>::Output as ops::Add>::Output;
}

impl<A, B> ComputationFn for MatMul<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.0.args().union(self.1.args())
    }
}

impl_core_ops!(MatMul<A, B>);

/// See [`Computation::mul_out`].
#[derive(Clone, Copy, Debug)]
pub struct MulOut<A, B>(pub(crate) A, pub(crate) B);

impl<A, B> Computation for MulOut<A, B>
where
    A: Computation<Dim = One>,
    B: Computation<Dim = One>,
    A::Item: ops::Mul<B::Item>,
{
    type Dim = Two;
    type Item = <A::Item as ops::Mul<B::Item>>::Output;
}

impl<A, B> ComputationFn for MulOut<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.0.args().union(self.1.args())
    }
}

impl_core_ops!(MulOut<A, B>);

/// See [`Computation::mul_col`].
#[derive(Clone, Copy, Debug)]
pub struct MulCol<A, B>(pub(crate) A, pub(crate) B);

impl<A, B> Computation for MulCol<A, B>
where
    A: Computation<Dim = Two>,
    B: Computation<Dim = One>,
    A::Item: ops::Mul<B::Item>,
    <A::Item as ops::Mul<B::Item>>::Output: ops::Add,
{
    type Dim = One;
    type Item = <<A::Item as ops::Mul<B::Item>>::Output as ops::Add>::Output;
}

impl<A, B> ComputationFn for MulCol<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
{
    fn args(&self) -> crate::Args {
        self.0.args().union(self.1.args())
    }
}

impl_core_ops!(MulCol<A, B>);

impl<A, B> fmt::Display for ScalarProduct<A, B>
where
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} . {})", self.0, self.1)
    }
}

impl<A, B> fmt::Display for MatMul<A, B>
where
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} x {})", self.0, self.1)
    }
}

impl<A, B> fmt::Display for MulOut<A, B>
where
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(col({}) x row({}))", self.0, self.1)
    }
}

impl<A, B> fmt::Display for MulCol<A, B>
where
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} x col({}))", self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{linalg::FromDiagElem, run::Matrix, val, val1, val2, Computation};

    #[proptest]
    fn identity_matrix_should_display(x: usize) {
        let inp = val!(x);
        prop_assert_eq!(
            inp.identity_matrix::<i32>().to_string(),
            format!("identity_matrix({})", inp)
        );
    }

    #[proptest]
    fn from_diag_elem_should_display(len: usize, elem: i32) {
        let len = val!(len);
        let elem = val!(elem);
        prop_assert_eq!(
            FromDiagElem::new(len, elem).to_string(),
            format!("from_diag_elem({}, {})", len, elem)
        );
    }

    #[proptest]
    fn scalar_product_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val1!([x, y]);
        let rhs = val1!([z, q]);
        prop_assert_eq!(
            lhs.scalar_product(rhs).to_string(),
            format!("({} . {})", lhs, rhs)
        );
    }

    #[proptest]
    fn mat_mul_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val2!(Matrix::from_vec((2, 1), vec![x, y]).unwrap());
        let rhs = val2!(Matrix::from_vec((1, 2), vec![z, q]).unwrap());
        prop_assert_eq!(
            lhs.clone().mat_mul(rhs.clone()).to_string(),
            format!("({} x {})", lhs, rhs)
        );
    }

    #[proptest]
    fn mul_out_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val1!([x, y]);
        let rhs = val1!([z, q]);
        prop_assert_eq!(
            lhs.mul_out(rhs).to_string(),
            format!("(col({}) x row({}))", lhs, rhs)
        );
    }

    #[proptest]
    fn mul_col_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val2!(Matrix::from_vec((2, 1), vec![x, y]).unwrap());
        let rhs = val1!([z, q]);
        prop_assert_eq!(
            lhs.clone().mul_col(rhs).to_string(),
            format!("({} x col({}))", lhs, rhs)
        );
    }
}
